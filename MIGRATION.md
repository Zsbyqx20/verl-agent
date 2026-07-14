# RRG — Research Context & Migration Guide

Written 2026-07-09 to bootstrap work on a new machine. Covers what the project is,
where it stands, and the concrete steps to get RL training running again.

## 1. What this project is

**ReverseReasoningGenerator (RRG)**: train a GUI/note-taking agent to keep a working
"observation memory" during long-horizon information-retrieval tasks (e.g. "browse the
first 10 posts, find the one with the most likes, record its author and comment count").
The agent only sees goal + current screen + its own notes at each step — no full history —
so it must decide what's worth writing down and when.

**Two-stage pipeline:**
1. **Reverse-generation** (`src/extract_demo.py` + `prompts/system.txt`): given ground-truth
   action-only trajectories (screenshot + GT action, no reasoning), backfill per-step
   reasoning + observation-memory updates using a strong LLM, conditioned on the *known*
   next action. This produces SFT training data.
2. **RL** (`rrpo/`, a fork of `verl-agent`/GiGPO): fine-tune the note-taker with RL. Reward
   is **answer-recovery**, not an LLM judge — see design decisions below, this was
   extensively validated (memories `rrg-reward-probe-e1` through `e5`).

**Current phase**: an RL run is actively training (see §3). A parallel exploration is
underway to replace the frozen reward-model reader with the policy itself (self-judge,
validated 2026-07-09, not yet wired into the trainer).

There was also a considered pivot to long-video understanding (same idea, GUI→video) that
is **exploratory, not decided** — see `rrg-video-pivot` memory. Not covered further here
since it's not the active RL work.

## 2. Key design decisions (don't re-litigate these)

- **Observation list is the only persistent memory.** Reasoning is discarded between
  steps; only what's written to observations survives. Any running count/record must be
  in the notes.
- **Forward-only generation, no lookahead.** Salience is fully determined by the task goal.
- **Reward = terminal answerability, NOT an LLM judge.** Extensively tested (probes E1-E5):
  MCQ/completion-judge rewards saturate (near-ceiling, no gradient) on both GUI and
  video-style navigation data. The one signal that stays non-saturated is **generative
  answer-recovery**: a frozen reader reconstructs the gold-schema answer from the policy's
  reasonings alone (no screenshots) and gets scored against the gold JSON by field recall.
- **Two-channel GiGPO reward** (`rrpo/agent_system/environments/env_package/rrg/`):
  - STEP (micro): action-recovery margin — reader SEES the screenshot, K-way forced choice
    over GT action + distractors given goal+screenshot+reasoning.
  - TRAJECTORY (macro): answer-recovery recall — reader BLIND to screenshots, reconstructs
    the gold JSON from the full reasoning trace, scored by per-field recall in [0,1].
  - `gamma=0` so `step_rewards` = immediate per-step margin (no discounting needed, GiGPO
    handles both channels natively — no core GiGPO/trainer edits required).
- **Full-answer-correct ceiling is ~5-7%** even though field recall is much higher (~0.48
  greedy). Diagnosed (`rrg-full-correct-diagnosis`): ~half is a fixable *consolidation* gap
  (the right note exists in SOME rollout of a group, just not all in one coherent
  trajectory — best-of-G self-distillation / step-credit reward addresses this), ~half is
  a harder *note-content* gap (no rollout ever captures certain fields — needs
  teacher-demo exploration).

## 3. Current state of the running experiment (as of 2026-07-09)

- Experiment: `gigpo-answer-4b-aug1420-g4reg-6gpu`, running on GPUs 0-5.
- Checkpoint progress: saving every 5 steps, latest = `global_step_210` (18:45).
- Training pool: `data/rrg-train-aug-combined` (1420 tasks — 397 native + 1023 mined
  augmentation tasks; see `rrg-question-mining-probe` memory).
- Reader: frozen Qwen3-VL-8B-Instruct, served on `http://127.0.0.1:4000/v1` (confirmed
  live). This is reached via an SSH tunnel that **must bind IPv4 explicitly**
  (`-L 127.0.0.1:4000:localhost:4000`, not bare `-L 4000:...`) — binding to `::1` silently
  breaks the reward client while `curl` misleadingly still works.
- Config carries hard-won stability fixes (do not casually change these):
  `entropy_coeff=0`, `kl_loss_coef=0.02`, `max_response_length=768`, `enforce_eager=True`,
  `free_cache_engine=True`, `tensor_model_parallel_size=1` — an earlier run without these
  diverged (entropy/length runaway, policy degenerated into gibberish by step 50).
- Best fully-merged checkpoint on disk: `rrpo/merged/rrg-answer-4b-aug1420-g4reg-6gpu-step175`
  (step175, mid-training when merged — a later step will be better once this run finishes
  or is intentionally stopped).
- **Active exploration, not yet integrated**: `src/self_judge_probe.py` validated
  (n=320/40 tasks) that the policy itself can replace the frozen 8B reader as the
  *answer-assembler* (Spearman +0.879 vs oracle, not differentially gameable by leakage).
  This matters for scaling training to more machines without needing the reader served
  separately. NOT yet wired into `reward_client.py`/`reward_manager/rrg.py` — that's the
  next planned code change. A cheaper *self-logprob* variant was tested and REJECTED
  (uniquely exploitable by verbatim leakage, ~2x gap vs honest use) — don't implement it
  without a dedicated leakage veto.

## 4. Migrating to a new machine

### 4a. Code
`rrpo/` is a **separate git repo** (fork of `langfengQ/verl-agent`), symlinked in from
`/data/liuguohong/workspace/rrpo`. All RRG-specific code is committed on branch
**`rrg-native`** at `https://github.com/Zsbyqx20/verl-agent` (pushed 2026-07-09, commit
`e1230a3`). NOTE: this same fork's `master` branch has a *different*, unrelated
judge-based RRG implementation (v2/v3, abandoned direction) — make sure to check out
`rrg-native`, not `master`.

This repo (`ReverseReasoningGenerator`) is a separate git repo — bring it too (it holds
`src/`, `prompts/`, `data/` referenced below).

```bash
git clone https://github.com/Zsbyqx20/verl-agent.git rrpo
cd rrpo && git checkout rrg-native
```

### 4b. Python environment
`rrpo` has its own `.venv` (Python 3.12.10, uv-managed), separate from this project's.
Exact pinned packages are exported to `rrpo/requirements-freeze.txt` (227 packages).

```bash
cd rrpo
uv venv --python 3.12.10 .venv
uv pip install -r requirements-freeze.txt --python .venv/bin/python
```
Caveat: torch/vllm wheels are tied to a CUDA build. If the new machine's CUDA version
differs, some pins may need adjusting — check `torch`/`vllm` versions in the freeze file
against what's installable for the target CUDA before assuming a clean install.

This project's own `.venv` also needs setting up separately for `src/*.py` scripts
(reverse-generation, probes, eval) — no freeze file exported for it yet; check
`pyproject.toml`/imports if needed.

### 4c. Data — only what's needed for RL (see below for full breakdown)
Self-contained, no external dataset dependency required for RL itself:
- `data/rrg-train/` (2.3G) — 397 tasks + real screenshot PNGs per task.
- `data/rrg-val/` (1.5G) — 40 val tasks + `tasks.json` + `rrg-source-real.csv`.
- `data/rrg-train-aug-combined/` (3.6M) — the 1420-task augmented pool (the CURRENT
  run's training data). Its `tasks.json` references image paths inside `data/rrg-train/`,
  so both must travel together.
- `prompts/system_amex.txt` (in this repo) — the system prompt the run script points at.

**NOT needed for RL**: the three big symlinked raw datasets (`data/AMEX` 188G,
`data/GUIOdyssey` 176G, `data/AndroTMem-Bench` 22G) — these were only used upstream to
*build* `rrg-train`/`rrg-val`, which are already self-contained snapshots.

### 4d. Model weights
- **Base/SFT init**: `POLICY_MODEL` defaults to `rrpo/patched_policy`, which is a folder
  of *symlinks* pointing at
  `/data/liuguohong/workspace/ms-swift/output/rrg-amex-20260616/checkpoint-846-merged/`
  (8.3G, real weights — SFT'd Qwen3-VL-4B). Copy the **real target directory**, not the
  symlink folder (symlinks won't resolve on another machine). Then either recreate
  `patched_policy` pointing at the copied path, or just set `POLICY_MODEL` directly to it.
- **To resume the in-progress run** instead of starting fresh: copy the specific
  `global_step_N` checkpoint dir under
  `rrpo/checkpoints/RRG-RL/gigpo-answer-4b-aug1420-g4reg-6gpu/`. Caveat: FSDP checkpoints
  hardcode `world_size` in shard filenames — you can only resume directly on the **same
  GPU count** used to save it (6 here). To resume on a different GPU count, merge to HF
  first (`rrpo/scripts/model_merger.py merge --backend fsdp --local_dir <ckpt>/actor
  --target_dir <out>`) and restart fresh from those merged weights (optimizer state and
  step counter reset, but model weights carry over).
- Total checkpoint dir is 1.2T across ALL past runs — only copy the one run/step you
  actually need, not the whole tree.

### 4e. Services (not files — must be running on/reachable from the new machine)
- **Frozen 8B reader** (Qwen3-VL-8B-Instruct, served via vLLM): reward computation calls
  this over HTTP. Either re-serve it on the new machine, or route to wherever it's
  currently served (mind the IPv4-tunnel gotcha above if going over SSH).
- Ark API (doubao) if using it as an alternate/stronger val reader — needs
  `OPENAI_API_KEY`/`OPENAI_BASE_URL` from `.env`, transferred carefully (it's a secret,
  don't commit it).

### 4f. Launching training on the new machine
Reference script: `rrpo/examples/gigpo_trainer/run_rrg_answer.sh`. Example launch
(6-GPU, matching the currently-running config):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 RRG_DATA_ROOT=/path/to/ReverseReasoningGenerator \
RRG_READER_URL=http://127.0.0.1:4000/v1 \
STEPS_PER_EPOCH=118 TOTAL_EPOCHS=2 TRAIN_DATA_SIZE=12 VAL_DATA_SIZE=6 GROUP_SIZE=4 \
SAVE_FREQ=5 TEST_FREQ=25 VAL_BEFORE_TRAIN=False \
EXPERIMENT_NAME=<pick-a-new-name-or-match-existing-to-resume> SWANLAB_MODE=local \
POLICY_MODEL=<path-to-copied-SFT-checkpoint-or-merged-RL-checkpoint> \
bash examples/gigpo_trainer/run_rrg_answer.sh \
  env.rrg.train_task_root=$RRG_DATA_ROOT/data/rrg-train-aug-combined env.max_steps=40 \
  trainer.n_gpus_per_node=6 actor_rollout_ref.actor.ppo_mini_batch_size=12 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.kl_loss_coef=0.02 \
  data.max_response_length=768
```
Before leaving it unattended: check the printed Hydra config dump matches every override
above, and confirm `nvidia-smi` shows all target GPUs actually free (this training box is
historically shared — contention on ANY GPU has caused multiple OOM crashes in the past,
not just a fixed subset).
