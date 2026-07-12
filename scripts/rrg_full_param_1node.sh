#!/usr/bin/env bash
#
# RRG full-parameter fine-tune on a single 8-GPU node.
#
# Usage:
#   bash scripts/rrg_full_param_1node.sh
#
# Required environment variables:
#   POLICY_MODEL     Path to the policy checkpoint, e.g. /preset-models or
#                    /path/to/rrg-sft-v1
#
# Optional environment variables (override below defaults):
#   TRAIN_DATA       Path to the train parquet (default: train_aug.parquet).
#   VAL_DATA         Path to the val parquet (default: val_aug.parquet).
#   TRAIN_TASK_ROOT  Directory containing tasks.json / rrg-source.csv / params_cache.json
#                    for the train replay env (default: rrg-train-aug-combined).
#   VAL_TASK_ROOT    Same for val. By default we point at the augmented task root so
#                    the val env sees the same 142 trajectories whose rows are in val_aug.parquet;
#                    pointing at rrg-val/ won't work because that dir lacks rrg-source.csv.
#   SYSTEM_PROMPT    Path to the system prompt file.
#   PROJECT_NAME     W&B / console / swanlab project name.
#   EXPERIMENT_NAME  Run name (default: timestamped).
#   N_GPUS           GPUs per node (default 8).
#
# This script assumes:
#   - vllm >= 0.13.0 (vllm 0.11.0 has the multi-modal + logprobs crash, see
#     /root/.claude/projects/-root-verl-agent/memory/vllm-011-multimodal-logprobs-bug.md).
#   - Required vllm 0.13.0 compat patches already on the branch:
#       * verl/utils/vllm_utils.py:72           — LoRAModel import fallback
#       * verl/utils/vllm_utils.py:163-243      — LoRA handoff kwargs renamed
#       * agent_system/environments/env_package/rrg/self_judge_client.py:167
#                                              — compute_position_id_with_mask import
#   - Ray cluster running on the head node (this script does not start one).
#   - The parquets under $TRAIN_DATA / $VAL_DATA exist. Regenerate with:
#       /root/verl-agent/.venv/bin/python /root/verl-agent/scripts/prep_rrg_aug_parquet.py \
#           --task_root "$TRAIN_TASK_ROOT" --out_dir "$(dirname "$TRAIN_DATA")"

set -euo pipefail
# ---------------------------------------------------------------------------
# Inputs (env-overridable)
# ---------------------------------------------------------------------------

: "${POLICY_MODEL:?POLICY_MODEL must be set, e.g. /preset-models}"

# Two separate data paths drive RRG: (1) the verl parquet (data.train_files,
# data.val_files) gives batch-shape scheduling only; (2) env.rrg.{train,val}_task_root
# gives the trajectory replay content read each step by RRGEnvironmentManager. Both
# paths must point at the augmented set OR row count / on-disk task count mismatch will
# silently produce empty rollouts (env will load 0 episodes and surface a RuntimeError).

TRAIN_DATA="${TRAIN_DATA:-/root/verl-agent/data/rrg-aligned/train_aug_1280.parquet}"
VAL_DATA="${VAL_DATA:-/root/verl-agent/data/rrg-aligned/val_aug_144.parquet}"
TRAIN_TASK_ROOT="${TRAIN_TASK_ROOT:-/root/verl-agent/data/rrg-train-aug-combined}"
VAL_TASK_ROOT="${VAL_TASK_ROOT:-/root/verl-agent/data/rrg-train-aug-combined}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-/root/verl-agent/data/system_prompt.txt}"
PROJECT_NAME="${PROJECT_NAME:-verl_agent_rrg}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-rrg_fullparam_1node_$(date +%Y%m%d_%H%M%S)}"
N_GPUS="${N_GPUS:-8}"

# ---------------------------------------------------------------------------
# Production recipe (rrg-recipe-recommendation.md):
#
#   train_batch_size / ppo_mini_batch_size  = 16-32, scaled with world_size
#   ppo_micro_batch_size_per_gpu            = 2-4
#   lr                                      = 1e-6 to 5e-6
#   lr_warmup_steps_ratio                   = 0.03
#   rollout gpu_memory_utilization           = 0.6  (8 GPUs fit 0.6; was 0.4 on 1 GPU)
#   rollout max_num_batched_tokens           = 32768
#   rollout max_model_len                    = 16384 (long-horizon 30-50 step)
#   save_freq, test_freq                     = 50
#   total_epochs                             = 4
#
# RRG-specific knobs (per /root/.claude/projects/-root-verl-agent/memory/rrg-recipe-recommendation.md):
#   env.rollout.n                           = 4     # GiGPO macro-group size; 2 was smoke-only
#   env.rrg.num_episodes                    unset  # load all train replay tasks
#   env.rrg.val_num_episodes                144    # 8-aligned validation rows
#   env.rrg.concurrency                     16     # self-judge: bounded by vllm queue depth
#   env.max_steps                           50     # RRG ignores but matches paper "30-50 step" framing
#   data.val_batch_size                     144    # match 8-aligned val parquet
#
# Rollout fan-out math:
#   16 rollout slots × env.rollout.n=4 ⇒ 4 tasks per PPO step ⇒ 1280 / 4 = 320 PPO steps per epoch
#   × 4 epochs = ~1280 PPO steps. With test_freq=50, expect ~25 validations over the run.
#
# Validation math:
#   val_aug_144.parquet has 144 rows; val_batch_size=144 ⇒ 1 validation batch covers all tasks.
#   Each test_freq=50 step runs 1 rollout per validation task (no env.rollout.n for val).
#   val_batch_size is marked "deprecated" by verl but still wired through StateFulDataLoader;
#   setting it to len(val_dataset) is the cleanest "use every task every time" guarantee.

.venv/bin/python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    algorithm.gamma=0.0 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=mean_norm \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.train_batch_size=16 \
    data.val_batch_size=144 \
    data.max_prompt_length=4096 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.image_key=images \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="${POLICY_MODEL}" \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    reward_model.reward_manager=rrg \
    reward_model.launch_reward_fn_async=False \
    env.env_name=rrg \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=4 \
    env.resources_per_worker.num_cpus=0.1 \
    env.rrg.data_kind=rrg \
    env.rrg.train_task_root="${TRAIN_TASK_ROOT}" \
    env.rrg.val_task_root="${VAL_TASK_ROOT}" \
    env.rrg.val_num_episodes=144 \
    env.rrg.system_prompt_file="${SYSTEM_PROMPT}" \
    env.rrg.concurrency=16 \
    env.rrg.self_judge=True \
    env.rrg.answer_max_tokens=512 \
    env.rrg.policy_image_max_pixels=786432 \
    env.rrg.answer_step_credit=true \
    env.rrg.step_credit_w=1.0 \
    env.rrg.step_credit_mode=delta \
    env.rrg.step_credit_combine=add \
    trainer.critic_warmup=0 \
    trainer.logger='[console,swanlab]' \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=4 \
    trainer.val_before_train=True
