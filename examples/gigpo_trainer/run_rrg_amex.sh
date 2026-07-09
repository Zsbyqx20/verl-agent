set -x
# ---------------------------------------------------------------------------
# RRG reverse-reasoning RL smoke test (AMEX navigation toy).
# Two-channel GiGPO reward: STEP = action-recovery margin (env), TRAJ = completion judge
# (reward_manager=rrg). gamma=0 so step_rewards = immediate per-step margin. Frozen reader =
# Qwen3-VL-8B served remotely (env.rrg.reader_url). Policy init: SFT-4B (set POLICY_MODEL).
#
#   bash examples/gigpo_trainer/run_rrg_amex.sh
# ---------------------------------------------------------------------------
RRG_DATA_ROOT=${RRG_DATA_ROOT:-/data/liuguohong/workspace/ReverseReasoningGenerator}
# Policy = SFT'd Qwen3-VL-4B. Falls back to the base 4B for a pure-plumbing run.
POLICY_MODEL=${POLICY_MODEL:-/data/liuguohong/.cache/modelscope/hub/models/Qwen/Qwen3-VL-4B-Instruct}
SYSTEM_PROMPT_FILE=${SYSTEM_PROMPT_FILE:-$RRG_DATA_ROOT/prompts/system_amex.txt}

num_cpus_per_env_worker=0.1
train_data_size=4      # number of distinct episodes per batch (env_num); must be divisible by n_gpus
val_data_size=4
group_size=8           # G rollouts per task (env.rollout.n) -> GiGPO groups
mode="mean_norm"

# Dummy parquet (count + modality only; the replay env serves the real frames).
python3 -m examples.data_preprocess.prepare_rrg \
    --train_data_size $train_data_size --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    algorithm.gamma=0.0 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    algorithm.use_kl_in_reward=False \
    data.train_files=$HOME/data/verl-agent/rrg/train.parquet \
    data.val_files=$HOME/data/verl-agent/rrg/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=8192 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.image_key=images \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$POLICY_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    reward_model.reward_manager=rrg \
    reward_model.launch_reward_fn_async=False \
    env.env_name=rrg \
    env.seed=0 \
    env.max_steps=16 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.rrg.train_jsonl=$RRG_DATA_ROOT/data/amex_reason_test.jsonl \
    env.rrg.train_image_root=$RRG_DATA_ROOT/data/sft/amex_test_images \
    env.rrg.num_episodes=$train_data_size \
    env.rrg.val_num_episodes=$val_data_size \
    env.rrg.system_prompt_file=$SYSTEM_PROMPT_FILE \
    env.rrg.concurrency=64 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_agent_rrg' \
    trainer.experiment_name='rrg_amex_smoke' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=2 \
    trainer.val_before_train=False $@
