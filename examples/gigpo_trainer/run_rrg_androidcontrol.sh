set -x
# ---------------------------------------------------------------------------
# RRG reverse-reasoning RL — AndroidControl ANSWER-RECOVERY tasks, 8 GPUs.
# Two-channel GiGPO:
#   STEP  = action-recovery margin (env, 8B reader SEES the screenshot)
#   TRAJ  = generative action-SEQUENCE-recovery RECALL (reward_manager=rrg; blind 8B reader
#           reconstructs AndroidControl's own gold action sequence from the policy's reasonings
#           alone -> field-level recall in [0,1]).
# This is the E6-validated (rrg-reward-probe-e6-androidcontrol) trajectory reward: beats
# completion_abs on validity, real none-control margin (0.958 vs 0.0), no leakage.
#
# Differences from run_rrg_answer.sh (the AMEX-domain RRG script this is forked from):
#   - task roots point at the converted AndroidControl corpora (rrg-androidcontrol-rrpo-wiring)
#   - env.rrg.answer_prompt_path swaps the reader's answer-assembly framing to the blind
#     action-sequence-recovery prompt (prompts/answer_actionrecovery.txt) instead of RRG's
#     default info-retrieval one -- this is the config knob this session added.
#   - answer_max_tokens defaults bumped (AndroidControl episodes run up to ~39 steps; E6 found
#     the JSON reconstruction truncates below ~3072 at that length).
#   - policy init still AMEX-SFT-4B (patched_policy) -- cross-domain transfer, untested how well
#     it generalizes to AndroidControl specifically; no AndroidControl-specific SFT data yet.
#   - step-credit / repetition-penalty left OFF (both defaults already; step-credit was a NET
#     NEGATIVE on RRG's own domain per rrg-step-credit-eval, no prior it helps here either).
#
#   bash examples/gigpo_trainer/run_rrg_androidcontrol.sh
# Smoke (2 tiny steps): RRG_NUM_EPISODES=4 STEPS_PER_EPOCH=2 TOTAL_EPOCHS=1 bash ...
# Optional SwanLab: prepend  SWANLAB_MODE=local  (or  SWANLAB_API_KEY=...).
# ---------------------------------------------------------------------------
RRG_DATA_ROOT=${RRG_DATA_ROOT:-/data/liuguohong/workspace/ReverseReasoningGenerator}
POLICY_MODEL=${POLICY_MODEL:-/data/liuguohong/workspace/rrpo/patched_policy}
SYSTEM_PROMPT_FILE=${SYSTEM_PROMPT_FILE:-$RRG_DATA_ROOT/prompts/system_amex.txt}
ANSWER_PROMPT_PATH=${ANSWER_PROMPT_PATH:-$RRG_DATA_ROOT/prompts/answer_actionrecovery.txt}

num_cpus_per_env_worker=0.1
train_data_size=${TRAIN_DATA_SIZE:-16}   # distinct tasks sampled per batch (env_num); divisible by n_gpus=8
val_data_size=${VAL_DATA_SIZE:-16}       # val tasks per batch (G=1 for val)
group_size=${GROUP_SIZE:-8}              # G rollouts per task (env.rollout.n) -> GiGPO group size
mode="mean_norm"

# steps_per_epoch = train_rows / train_data_size. Initial scope (per rrg-androidcontrol-rrpo-
# wiring) is num_episodes=1000 of the full 10037-task train corpus -> ~62 steps/epoch covers
# that 1000-task pool ~once per epoch (env re-samples real episodes every batch; the parquet
# content is dummy, only its ROW COUNT sets the step budget).
steps_per_epoch=${STEPS_PER_EPOCH:-62}
train_rows=$(( steps_per_epoch * train_data_size ))
val_rows=$val_data_size
total_epochs=${TOTAL_EPOCHS:-3}
num_episodes=${RRG_NUM_EPISODES:-1000}   # initial scope; full converted pool is 10037
val_num_episodes=${RRG_VAL_NUM_EPISODES:-$val_data_size}

# --- full-answer-correctness levers (see memory rrg-completeness-levers) ---
# #1 train-reward reader token budget. Bumped from the RRG-domain 2048 default: AndroidControl
#    episodes run up to ~39 steps and the action-sequence JSON reconstruction gets
#    truncated/unparseable below ~3072 at that length (rrg-reward-probe-e6-androidcontrol).
ANSWER_MAX_TOKENS=${ANSWER_MAX_TOKENS:-3072}
# #2 optional stronger VAL reader (doubao via Ark). Leave null -> falls back to the 8B reader.
#    SECURITY: the key is read from env var RRG_VAL_READER_KEY at point of use and is NEVER
#    placed in the Hydra config (which verl prints to stdout AND uploads to swanlab). Launch with:
#      RRG_VAL_READER_KEY=$OPENAI_API_KEY VAL_READER_URL=https://ark.cn-beijing.volces.com/api/v3 \
#      VAL_READER_MODEL=doubao-seed-2-0-lite-260428 SWANLAB_MODE=local bash ...
#    If the TRAIN-side reader also needs a real key (e.g. it's an authenticated API too, not just
#    the unauthenticated 8B), set RRG_READER_KEY the same way -- also never enters the config, and
#    is used as val's fallback too when RRG_VAL_READER_KEY/VAL_READER_KEY aren't set.
# reward reader (step-margin + trajectory answer-recovery). Override RRG_READER_URL if the 8B moved.
READER_URL=${RRG_READER_URL:-http://117.74.66.190:10727/v1}
VAL_READER_URL=${VAL_READER_URL:-null}
VAL_READER_MODEL=${VAL_READER_MODEL:-null}
VAL_ANSWER_MAX_TOKENS=${VAL_ANSWER_MAX_TOKENS:-4096}
# Re-export under the name the code reads; keep it off the trace so `set -x` can't echo it.
{ set +x; } 2>/dev/null
export RRG_VAL_READER_KEY="${RRG_VAL_READER_KEY:-}"
# #3 GiGPO trajectory-reward shaping for completeness. Offline-best on RRG's own domain =
# threshold (rrg-completeness-levers); untested on AndroidControl -- default off (none) here.
TRAJ_REWARD_SHAPING=${TRAJ_REWARD_SHAPING:-none}   # none|square|correct_bonus|threshold
RECALL_THRESHOLD=${RECALL_THRESHOLD:-0.9}
THRESHOLD_BONUS=${THRESHOLD_BONUS:-0.5}
# per-step answer-field credit (GiGPO micro channel); OFF -- rrg-step-credit-eval found it a
# net regression on RRG's own domain (more repetition, worse recall), no prior it helps here.
ANSWER_STEP_CREDIT=${ANSWER_STEP_CREDIT:-false}
STEP_CREDIT_W=${STEP_CREDIT_W:-1.0}
STEP_CREDIT_MODE=${STEP_CREDIT_MODE:-first_appearance}  # first_appearance|delta
STEP_CREDIT_COMBINE=${STEP_CREDIT_COMBINE:-add}    # add|replace
# repetition penalty (GiGPO micro channel); off by default = plain-recall baseline.
REPETITION_PENALTY=${REPETITION_PENALTY:-false}    # true to enable
REPETITION_PENALTY_W=${REPETITION_PENALTY_W:-1.0}
REPETITION_LOOKBACK=${REPETITION_LOOKBACK:-15}
REPETITION_THRESHOLD=${REPETITION_THRESHOLD:-0.97}
# teacher-demo injection (EXPLORATION); off by default. TEACHER_DATA_PATH=<store.json> + TEACHER_SEED_K>0 enables.
TEACHER_DATA_PATH=${TEACHER_DATA_PATH:-null}
TEACHER_SEED_K=${TEACHER_SEED_K:-0}
TEACHER_ANNEAL_END_STEP=${TEACHER_ANNEAL_END_STEP:-null}
MAX_PREFIXES=${MAX_PREFIXES:-8}
MAX_STEPS=${MAX_STEPS:-42}   # AndroidControl episodes run up to ~39 steps; small safety margin

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
    data.max_prompt_length=32768 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.image_key=images \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$POLICY_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    reward_model.reward_manager=rrg \
    reward_model.launch_reward_fn_async=False \
    env.env_name=rrg \
    env.seed=0 \
    env.max_steps=$MAX_STEPS \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.rrg.data_kind=rrg \
    env.rrg.train_task_root=$RRG_DATA_ROOT/data/rrg-androidcontrol-train \
    env.rrg.val_task_root=$RRG_DATA_ROOT/data/rrg-androidcontrol-val \
    env.rrg.num_episodes=$num_episodes \
    env.rrg.val_num_episodes=$val_num_episodes \
    env.rrg.system_prompt_file=$SYSTEM_PROMPT_FILE \
    env.rrg.answer_prompt_path=$ANSWER_PROMPT_PATH \
    env.rrg.concurrency=96 \
    env.rrg.answer_max_tokens=$ANSWER_MAX_TOKENS \
    env.rrg.reader_url=$READER_URL \
    env.rrg.val_reader_url=$VAL_READER_URL \
    env.rrg.val_reader_model=$VAL_READER_MODEL \
    env.rrg.val_answer_max_tokens=$VAL_ANSWER_MAX_TOKENS \
    env.rrg.traj_reward_shaping=$TRAJ_REWARD_SHAPING \
    env.rrg.recall_threshold=$RECALL_THRESHOLD \
    env.rrg.threshold_bonus=$THRESHOLD_BONUS \
    env.rrg.answer_step_credit=$ANSWER_STEP_CREDIT \
    env.rrg.step_credit_w=$STEP_CREDIT_W \
    env.rrg.step_credit_mode=$STEP_CREDIT_MODE \
    env.rrg.step_credit_combine=$STEP_CREDIT_COMBINE \
    env.rrg.max_prefixes=$MAX_PREFIXES \
    env.rrg.repetition_penalty=$REPETITION_PENALTY \
    env.rrg.repetition_penalty_w=$REPETITION_PENALTY_W \
    env.rrg.repetition_lookback=$REPETITION_LOOKBACK \
    env.rrg.repetition_threshold=$REPETITION_THRESHOLD \
    env.rrg.teacher_data_path=$TEACHER_DATA_PATH \
    env.rrg.teacher_seed_k=$TEACHER_SEED_K \
    env.rrg.teacher_anneal_end_step=$TEACHER_ANNEAL_END_STEP \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='RRG-RL' \
    trainer.experiment_name=${EXPERIMENT_NAME:-gigpo-androidcontrol-answer-4b-g8} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ:-5} \
    trainer.test_freq=${TEST_FREQ:-25} \
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR:-rollout_dumps/$(date +%s)}" \
    trainer.rollout_data_max_dump=${ROLLOUT_DATA_MAX_DUMP:-32} \
    trainer.log_val_generations=${LOG_VAL_GENERATIONS:-16} \
    trainer.total_epochs=$total_epochs \
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-False} $@
