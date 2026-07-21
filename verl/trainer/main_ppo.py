# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)  # used for multimodal LLM, could be none

        from agent_system.environments import make_envs
        envs, val_envs = make_envs(config, tokenizer=tokenizer, processor=processor)

        # vllm early verify
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == 'episode':
            from agent_system.reward_manager import EpisodeRewardManager
            reward_fn = EpisodeRewardManager(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
            # Note that we always use function-based RM for validation
            val_reward_fn = EpisodeRewardManager(tokenizer=tokenizer, num_examine=1, normalize_by_length=False)
        elif reward_manager_name == 'rrg':
            # RRG trajectory/macro reward = generative answer-recovery recall (the env supplies the
            # step/micro action-recovery margin). The blind 8B reader assembles the gold-schema
            # answer from the policy's reasonings; recall in [0,1] is the macro reward.
            from agent_system.reward_manager import RRGTrajectoryRewardManager
            rcfg = config.env.rrg
            # Guard: the "gen" step-reward mode drops the explicit content-free control and relies
            # on GiGPO's step (micro) group-mean normalization to remove the screenshot-obviousness
            # baseline. That only happens when the micro channel is active (step_advantage_w>0). If
            # it's off, raw mean-SSR leaks that baseline straight into the reward.
            if rcfg.get('step_reward_mode', 'mc') == 'gen':
                _saw = config.algorithm.get('gigpo', {}).get('step_advantage_w', 0.0) \
                    if hasattr(config.algorithm, 'get') else getattr(
                        getattr(config.algorithm, 'gigpo', None), 'step_advantage_w', 0.0)
                if not _saw or float(_saw) <= 0.0:
                    print("[rrg][WARN] step_reward_mode='gen' relies on GiGPO micro-channel "
                          "group-norm to remove the screenshot baseline, but "
                          f"algorithm.gigpo.step_advantage_w={_saw} (<=0). The gen reward will "
                          "leak the screenshot-obviousness baseline. Set step_advantage_w>0 or "
                          "use step_reward_mode='mc' with subtract_control.", flush=True)
            common = dict(concurrency=rcfg.get('concurrency', 64),
                          data_kind=rcfg.get('data_kind', 'amex'),
                          train_task_root=rcfg.get('train_task_root', None),
                          val_task_root=rcfg.get('val_task_root', None),
                          answer_prompt_path=rcfg.get('answer_prompt_path', None))
            # Train macro reward: 8B reader (logprobs path shares this host), bumped token
            # budget, and the completeness shaping (#3) applied to the reward written to the
            # tensor. Raw recall/correct are still logged unchanged.
            # Prefer the key from the environment (RRG_READER_KEY) so a real train-side API key
            # never enters the Hydra config tree -- which verl prints to stdout AND uploads to
            # swanlab. Mirrors the val-side RRG_VAL_READER_KEY handling below.
            train_key = os.environ.get('RRG_READER_KEY') or rcfg.get('reader_key', 'sk-dummy')
            reward_fn = RRGTrajectoryRewardManager(
                tokenizer=tokenizer, num_examine=0, is_val=False,
                reader_url=rcfg.reader_url, reader_model=rcfg.reader_model,
                reader_key=train_key,
                answer_max_tokens=rcfg.get('answer_max_tokens', 2048),
                traj_reward_shaping=rcfg.get('traj_reward_shaping', 'none'),
                shaping_power=rcfg.get('shaping_power', 2.0),
                correct_bonus_lambda=rcfg.get('correct_bonus_lambda', 0.5),
                recall_threshold=rcfg.get('recall_threshold', 0.9),
                threshold_bonus=rcfg.get('threshold_bonus', 0.5),
                answer_step_credit=rcfg.get('answer_step_credit', False),
                step_credit_w=rcfg.get('step_credit_w', 1.0),
                step_credit_mode=rcfg.get('step_credit_mode', 'first_appearance'),
                step_credit_combine=rcfg.get('step_credit_combine', 'add'),
                max_prefixes=rcfg.get('max_prefixes', 8),
                clamp_negative=rcfg.get('clamp_negative', True),
                repetition_penalty=rcfg.get('repetition_penalty', False),
                repetition_penalty_w=rcfg.get('repetition_penalty_w', 1.0),
                repetition_lookback=rcfg.get('repetition_lookback', 15),
                repetition_threshold=rcfg.get('repetition_threshold', 0.97),
                traj_reward_weight=rcfg.get('traj_reward_weight', 1.0),
                processor=processor, **common)
            # Val/eval reward: optional stronger reader (doubao) for an accurate test_score, and
            # shaping FORCED OFF + weight FORCED to 1.0 so val/rrg/test_score stays raw recall (a
            # comparable metric) regardless of the train-side traj_reward_weight/shaping knobs.
            v_url = rcfg.get('val_reader_url', None) or rcfg.reader_url
            v_model = rcfg.get('val_reader_model', None) or rcfg.reader_model
            # Prefer the key from the environment (RRG_VAL_READER_KEY) so the secret never
            # enters the Hydra config tree -- which verl prints to stdout AND uploads to swanlab.
            # Falls back to the train-side env key/config before the plain-text yaml default, so
            # a train-side RRG_READER_KEY is honored here too when val doesn't set its own.
            v_key = (os.environ.get('RRG_VAL_READER_KEY')
                     or rcfg.get('val_reader_key', None) or train_key)
            v_max = rcfg.get('val_answer_max_tokens', None) or rcfg.get('answer_max_tokens', 2048)
            val_reward_fn = RRGTrajectoryRewardManager(
                tokenizer=tokenizer, num_examine=1, is_val=True,
                reader_url=v_url, reader_model=v_model, reader_key=v_key,
                answer_max_tokens=v_max, traj_reward_shaping='none',
                processor=processor, **common)
        else:
            raise NotImplementedError

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        assert config.actor_rollout_ref.rollout.n == 1, "In verl, actor_rollout_ref.rollout.n>1 is for GRPO. In verl+env, we keep n=1, and achieve GRPO by env.rollout.n"

        from agent_system.multi_turn_rollout import TrajectoryCollector
        traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
        )
        trainer.init_workers()
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset.

    Arguments:
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # use sampler for better ckpt resume
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
