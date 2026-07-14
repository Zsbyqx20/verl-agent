# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`verl-agent` is an extension of [veRL](https://github.com/volcengine/verl) for training **LLM agents via reinforcement learning**. Its core contribution is **GiGPO** (Group-in-Group Policy Optimization, NeurIPS 2025) — a step-independent multi-turn rollout mechanism that avoids concatenating full interaction histories, enabling RL training on long-horizon (30-50 step) agent tasks. Also includes **HGPO** (ICLR 2026), **GraphGPO** (ICML 2026), and standard algorithms (GRPO, PPO, DAPO, GSPO, RLOO, REINFORCE++).

## Build & Test Commands

```bash
# Install (Python 3.12 conda env)
pip install -e .

# Install with optional deps
pip install -e ".[vllm]"        # vLLM inference
pip install -e ".[sglang]"      # SGLang inference
pip install -e ".[test]"        # Test deps (pytest, pre-commit, py-spy)
pip install -e ".[gpu]"         # GPU deps (liger-kernel, flash-attn)

# Lint & format (ruff)
ruff check verl/ agent_system/ gigpo/ recipe/ tests/
ruff format verl/ agent_system/ gigpo/ recipe/ tests/

# Run a single test
python -m pytest tests/sanity/test_import.py -xvs
python -m pytest tests/trainer/ppo/test_metric_utils.py -xvs

# Run all CPU tests
python -m pytest tests/ -x --ignore=tests/ray_gpu --ignore=tests/utils/gpu_tests -m "not gpu"

# Run pre-commit
pre-commit run --all-files
```

## Architecture

### Source Tree Layout

```
verl/                   # Core RL framework (forked from veRL, kept in sync with upstream)
  trainer/
    main_ppo.py         # Primary entry point: python -m verl.trainer.main_ppo
    main_eval.py        # Evaluation entry point
    main_generation.py  # Generation-only entry point
    config/
      ppo_trainer.yaml  # Main Hydra config (ALL algorithm/env/training settings)
    ppo/
      ray_trainer.py    # RayPPOTrainer — distributed PPO training loop
      core_algos.py     # Advantage estimation (GAE, GRPO, RLOO, REINFORCE++, GiGPO)
      reward.py         # Reward computation
    fsdp_sft_trainer.py # SFT training
  workers/
    fsdp_workers.py     # FSDP Actor/Critic/RewardModel workers
    megatron_workers.py # Megatron backend workers
    rollout/            # Inference backends (vLLM, SGLang, HF, Naive)
    reward_manager/     # Reward model management
  single_controller/    # Ray-based distributed worker orchestration
  models/               # Model registry, weight loading, monkey-patches
  utils/                # FSDP utils, torch functional, dataset, checkpoint, etc.
  tools/                # Tool definitions (search, sandbox, GSM8K)
  protocol.py           # DataProto — the core data container passed between workers

agent_system/           # verl-agent custom additions on top of veRL
  environments/
    base.py             # EnvironmentManagerBase — gym-style env interface
    env_manager.py      # Per-environment managers (ALFWorld, WebShop, Search, Sokoban, Gym Cards, AppWorld, RRG)
    prompts/            # Per-environment prompt templates
    env_package/        # Bundled environment implementations + projection functions
  multi_turn_rollout/
    rollout_loop.py     # TrajectoryCollector — multi-turn agent rollout orchestration
    utils.py            # Image processing, batch adjustment utilities
  memory/
    memory.py           # SimpleMemory, SearchMemory (customizable per-step history management)
    base.py             # Abstract memory interface
  reward_manager/
    episode.py          # EpisodeRewardManager — default trajectory-reward from env info
    rrg.py              # RRGTrajectoryRewardManager — generative answer-recovery recall reward

gigpo/                  # GiGPO algorithm implementation
  core_gigpo.py         # Group-in-group advantage normalization, anchor grouping, similarity clustering

recipe/                 # Standalone algorithm recipes (each with its own trainer + main)
  hgpo/                 # HGPO (ICLR 2026) — hierarchical group policy optimization
  GraphGPO/             # GraphGPO (ICML 2026) — graph-based group policy optimization
  dapo/                 # DAPO recipe
  sppo/                 # SPPO recipe
  prime/                # PRIME recipe
  spin/                 # SPIN recipe
  r1/                   # Search-R1 data processing & evaluation

examples/               # Shell scripts to launch training runs
  gigpo_trainer/        # GiGPO training scripts per environment
  grpo_trainer/         # GRPO training scripts
  ppo_trainer/          # PPO training scripts
  dapo_trainer/         # DAPO training scripts
  gspo_trainer/         # GSPO training scripts
  rloo_trainer/         # RLOO training scripts
  data_preprocess/      # Data preparation scripts
  gigpo_dynamic_trainer/# Dynamic-sampling GiGPO variants

tests/                  # Organized by category: sanity, trainer, ray_gpu, models, utils, kernels, e2e
```

### Key Design Decisions

- **Hydra + OmegaConf** for all configuration. The primary config is `verl/trainer/config/ppo_trainer.yaml`. All algorithm, environment, and training settings live there. CLI overrides use dot notation (e.g., `algorithm.adv_estimator=gigpo env.env_name=alfworld/AlfredTWEnv`).

- **Ray** is the distributed backend. The entry point (`main_ppo.py`) initializes Ray, then spawns a `TaskRunner` remote actor. Workers are managed through `verl.single_controller.ray.RayWorkerGroup`.

- **DataProto** (`verl/protocol.py`) is the universal data container. It carries both tensor data (`.batch`) and non-tensor metadata (`.non_tensor_batch`). All communication between workers (actor → rollout → critic → reward) flows through DataProto.

- **Multi-turn rollouts** are handled by `TrajectoryCollector` (in `agent_system/multi_turn_rollout/`). It iteratively: builds per-step inputs from env observations → generates actions → steps environments → collects rewards. The key innovation is that each step's input is independently constructed (not concatenated history), enabling long-horizon training.

- **GiGPO** uses a two-level grouping: micro-groups within trajectories (step-level advantages) and macro-groups across trajectories (episode-level advantages). Core logic in `gigpo/core_gigpo.py` and `verl/trainer/ppo/core_algos.py`.

- **Environment interface**: `EnvironmentManagerBase` in `agent_system/environments/base.py`. Environments follow a gym-like `reset()`/`step()` API. `projection_f` maps text actions to environment actions. Per-environment managers in `agent_system/environments/env_manager.py` handle prompt building, memory, and reward computation.

- **Memory system**: `SimpleMemory` stores per-environment history; `SearchMemory` extends it with search-specific document tracking. The `fetch()` method supports configurable history length and summarization.

- **Recipe convention**: Each recipe in `recipe/` has its own `main_*.py` entry point and `*_ray_trainer.py` that subclasses `RayPPOTrainer`. They are launched directly (e.g., `python -m recipe.hgpo.main_hgpo`) rather than through `verl.trainer.main_ppo`.

### Inference Backends

Configured via `actor_rollout_ref.rollout.name`: `vllm` (default), `sglang`, `hf` (HuggingFace), or `naive`. vLLM is the primary backend; SGLang is required for multi-turn tool-interaction tasks (`rollout.multi_turn.enable=True`).

### Algorithm Selection

Set `algorithm.adv_estimator` in config: `gae` (PPO), `grpo`, `gigpo`, `rloo`, `reinforce_plus_plus`, `reinforce_plus_plus_baseline`, `remax`, `grpo_passk`.
