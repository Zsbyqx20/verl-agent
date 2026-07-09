set -x
# ---------------------------------------------------------------------------
# RRG reverse-reasoning RL — FULL AMEX train set, 8 GPUs.
# Two-channel GiGPO: STEP = action-recovery margin (env), TRAJ = completion judge
# (reward_manager=rrg). gamma=0 -> step_rewards = immediate per-step margin.
# Frozen reader = Qwen3-VL-8B (env.rrg.reader_url). Policy init = patched SFT-4B.
# Train pool = all 2133 train episodes (num_episodes=null); val = AMEX test split.
#
#   bash examples/gigpo_trainer/run_rrg_amex_full.sh
# Optional SwanLab: prepend  SWANLAB_MODE=local  (or  SWANLAB_API_KEY=...).
# ---------------------------------------------------------------------------
RRG_DATA_ROOT=${RRG_DATA_ROOT:-/data/liuguohong/workspace/ReverseReasoningGenerator}
POLICY_MODEL=${POLICY_MODEL:-/data/liuguohong/workspace/rrpo/patched_policy}
SYSTEM_PROMPT_FILE=${SYSTEM_PROMPT_FILE:-$RRG_DATA_ROOT/prompts/system_amex.txt}

num_cpus_per_env_worker=0.1
train_data_size=16     # distinct episodes sampled per batch (env_num); MUST be divisible by n_gpus=8
val_data_size=16       # val episodes per batch; also divisible by 8
group_size=8           # G rollouts per task (env.rollout.n) -> GiGPO group size
mode="mean_norm"

# steps_per_epoch = train_rows / train_data_size. Full train pool = 2133 episodes -> ~134
# steps/epoch covers each distinct task ~once per epoch. (Row content is dummy; the env
# re-samples real episodes every batch via num_episodes=null. ONLY the row COUNT matters here.)
steps_per_epoch=134
train_rows=$(( steps_per_epoch * train_data_size ))   # 2144
val_rows=$val_data_size                                # 1 val batch per test_freq

# Dummy parquet (row COUNT sets steps/epoch + modality; the replay env serves the real frames).
python3 -m examples.data_preprocess.prepare_rrg \
    --train_data_size $train_data_size --val_data_size $val_data_size \
    --train_rows $train_rows --val_rows $val_rows

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
    data.max_prompt_length=12288 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.image_key=images \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$POLICY_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    reward_model.reward_manager=rrg \
    reward_model.launch_reward_fn_async=False \
    env.env_name=rrg \
    env.seed=0 \
    env.max_steps=40 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.rrg.train_jsonl=$RRG_DATA_ROOT/data/amex_reason_train.jsonl \
    env.rrg.train_image_root=$RRG_DATA_ROOT/data/sft/amex_train_images \
    env.rrg.num_episodes=null \
    env.rrg.val_jsonl=$RRG_DATA_ROOT/data/amex_reason_test.jsonl \
    env.rrg.val_image_root=$RRG_DATA_ROOT/data/sft/amex_test_images \
    env.rrg.val_num_episodes=$val_data_size \
    env.rrg.system_prompt_file=$SYSTEM_PROMPT_FILE \
    env.rrg.concurrency=96 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='RRG-RL' \
    trainer.experiment_name='gigpo-4b-g8-bs16-lr1e6' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.total_epochs=1 \
    trainer.val_before_train=True $@
