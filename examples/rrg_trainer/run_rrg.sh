set -x

# =============================================
# RRG (Reverse Reasoning Generator) Training
# =============================================

TRAJECTORY_DATA=/data/liuguohong/workspace/rrg/data/boot/traces_20260413.json
DATA_DIR=$HOME/data/verl-agent/rrg

# Step 1: Prepare parquet dataset
python3 -m examples.data_preprocess.prepare_rrg \
    --input $TRAJECTORY_DATA \
    --local_dir $DATA_DIR \
    --train_data_size 16 \
    --val_data_size 16

# Step 2: Run RRG training
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=rrg \
    algorithm.rrg.w_fact=1.0 \
    algorithm.rrg.w_reason=1.0 \
    algorithm.rrg.w_final=1.0 \
    algorithm.rrg.token_scale=16.0 \
    algorithm.rrg.alpha_fact_penalty=0.2 \
    algorithm.rrg.reason_length_penalty=0.3 \
    algorithm.rrg.final_judge_model=doubao-seed-2-0-pro-260215 \
    algorithm.rrg.rank_judge_model=doubao-seed-2-0-pro-260215 \
    algorithm.rrg.max_judge_workers=16 \
    algorithm.rrg.max_retries=3 \
    reward_model.reward_manager=rrg \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_prompt_length=16384 \
    data.max_response_length=2048 \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    env.env_name=rrg \
    env.seed=42 \
    env.max_steps=20 \
    env.rollout.n=4 \
    env.rrg.trajectory_data_path=$TRAJECTORY_DATA \
    env.rrg.reasoning_history_length=5 \
    trainer.total_epochs=10 \
    trainer.project_name=rrg_training \
    trainer.experiment_name=rrg_boot \
    trainer.logger='["console"]' \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=False
