# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**verl-agent** is an extension of [veRL](https://github.com/volcengine/verl) for training LLM agents via reinforcement learning. The key contribution is a **step-independent multi-turn rollout mechanism** that avoids concatenating full interaction histories, enabling scalable training on long-horizon tasks (30-50 steps). The primary novel algorithm is **GiGPO** (Group-in-Group Policy Optimization), which performs two-level credit assignment (episode-level and step-level).

## Installation

```bash
conda create -n verl-agent python==3.12 -y
conda activate verl-agent
pip3 install vllm==0.11.0
pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .
```

Optional extras: `pip install -e .[test,gpu,vllm,sglang,mcore]`

## Common Commands

### Linting and Formatting
```bash
ruff check .        # lint
ruff format .       # format
```
Configured in `pyproject.toml`: line-length=300, rules E/F/UP/B/I/G.

### Running Tests
```bash
pip install -e .[test]

# CPU distributed tests
cd tests/ray_cpu && pytest -s -x --ignore=test_check_worker_alive.py .

# Protocol and trainer tests
pytest tests/test_protocol.py
pytest tests/trainer/ppo/
```

### Training
```bash
# Run a training recipe (example)
bash examples/gigpo_trainer/run_alfworld.sh

# Direct invocation with Hydra overrides
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_batch_size=16 \
    actor_rollout_ref.model.path=<model_path> \
    ...
```

### Data Preparation
```bash
python examples/data_preprocess/prepare.py  # outputs to ~/data/verl-agent/
```

### Documentation
```bash
cd docs && pip install -r requirements-docs.txt && make html
```

## Architecture

### Training Data Flow

```
Raw Task Data
  → Data Preparation (examples/data_preprocess/)
  → Environment Reset (agent_system/environments/)
  → Multi-turn Rollout (agent_system/multi_turn_rollout/rollout_loop.py)
      → Environment Step (gym-style step())
      → Memory Module (agent_system/memory/)
  → DataProto batching (verl/protocol.py)
  → RL Advantage Computation (gigpo/core_gigpo.py or verl/trainer/ppo/core_algos.py)
  → Training Update (verl/trainer/ppo/ray_trainer.py)
```

### Key Modules

**`verl/protocol.py` — DataProto**
The universal data exchange format between all modules. Contains:
- `batch`: TensorDict for GPU tensors
- `non_tensor_batch`: dict for numpy arrays / string trajectories
- `meta_info`: metadata dict
All workers communicate exclusively through DataProto.

**`verl/trainer/ppo/ray_trainer.py` — RayPPOTrainer**
Main training orchestrator. Manages the Ray worker group, calls rollout, computes advantages, and updates model weights. This is where algorithm-level logic integrates.

**`verl/trainer/main_ppo.py` — Entry Point**
Hydra-based entry point. All training config is driven by `verl/trainer/config/ppo_trainer.yaml` with command-line overrides.

**`agent_system/multi_turn_rollout/rollout_loop.py` — TrajectoryCollector**
The core of the step-independent rollout. Collects observations from environments, builds model inputs, and compiles multi-turn trajectories without concatenating full histories (the key innovation).

**`agent_system/environments/env_manager.py` — EnvironmentManagerBase**
Unified gym-like interface for all supported environments. Environment-specific logic lives in `agent_system/environments/env_package/{alfworld,webshop,search,sokoban,gym_cards,appworld}/`.

**`agent_system/memory/` — Memory Modules**
Pluggable history management. `BaseMemory` (base.py) defines the interface; `SimpleMemory` (memory.py) is the default. `FactBankMemory` (fact_bank.py) is a versioned observation bank used by RRG — each slot tracks add/update history per step. Custom memory modules control what context the agent sees at each step.

**`gigpo/core_gigpo.py` — GiGPO Algorithm**
Group-in-group advantage estimation. Key functions:
- `compute_step_discounted_returns()`: step-level credit assignment
- `to_hashable()`: groups trajectories by observation state similarity
- Called from `verl/trainer/ppo/core_algos.py`

**`rrg/` — RRG Algorithm**
Reverse Reasoning Generator algorithm. Key modules:
- `core_rrg.py`: `compute_rrg_advantage()` — combines citation and writing reward signals into advantages
- `output_parser.py`: `parse_rrg_output()` parses model outputs into citation indices, reasoning text, and fact bank writes; `build_span_masks()` produces token-level masks for cite/write spans
- `reward_calc_batch.py`: batch reward computation helpers

**`verl/single_controller/ray/` — Distributed Orchestration**
RayWorkerGroup manages the distributed worker lifecycle. Worker roles: ActorRolloutRef, Critic (optional), Reward Model (optional). Supports FSDP, FSDP2, and Megatron-LM training strategies.

### Supported Algorithms
`algorithm.adv_estimator` config key selects: `gigpo`, `grpo`, `gae` (PPO), `dapo`, `rloo`, `reinforce_plus_plus`, `rrg`

### Supported Environments
ALFWorld, WebShop, Search-R1 (open-domain QA), Sokoban, Gym Cards, AppWorld (experimental), RRG (Reverse Reasoning Generator — GUI trajectory replay)

### Supported Models
Qwen3, Qwen3-VL, Qwen2.5, Qwen2.5-VL, LLaMA3.2; LoRA fine-tuning supported

### Inference Backends
vLLM (default, `rollout.name=vllm`) and SGLang (`rollout.name=sglang`)

## Configuration System

All training is configured via Hydra. The base config is `verl/trainer/config/ppo_trainer.yaml`. Recipe-specific configs are in `recipe/*/`. Run scripts (e.g., `examples/gigpo_trainer/run_alfworld.sh`) override the base config via CLI.

Key config sections: `data`, `actor_rollout_ref`, `critic`, `algorithm`, `trainer`, `env`

RRG-specific config lives under `algorithm.rrg` (citation/fact reward weights, judge model, debug logging) and `env.rrg` (trajectory data path, reasoning history length). Setting `env.max_steps` to `null` or `0` enables unlimited steps (loop runs until all environments are done).

`algorithm.reward_manager: rrg` selects `RRGRewardManager`, which uses frozen LLM judges (J_cite, J_fact) called via OpenAI-compatible API to compute per-step citation precision/recall and fact quality rewards. Requires `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`.

## Repository Layout

```
verl/                      # Core training framework (PPO, workers, models, utils)
agent_system/              # Agent extensions: rollout, environments, memory, rewards
  environments/
    env_package/           # Per-environment implementations (includes rrg/)
    prompts/               # Environment-specific system prompts (includes rrg.py)
  multi_turn_rollout/      # Step-independent trajectory collection
  memory/                  # Pluggable history/context management (SimpleMemory, FactBankMemory)
  reward_manager/          # Reward managers (EpisodeRewardManager, RRGRewardManager)
gigpo/                     # GiGPO algorithm implementation
rrg/                       # RRG algorithm (core_rrg.py, output_parser.py, reward_calc_batch.py)
examples/                  # Training scripts organized by algorithm
  rrg_trainer/             # RRG run scripts (run_rrg.sh, train.sh)
  data_preprocess/         # Data prep scripts (prepare.py, prepare_rrg.py)
recipe/                    # End-to-end recipes (hgpo, dapo, r1, etc.)
tests/                     # Test suites (ray_cpu/, trainer/ppo/, test_protocol.py)
docs/                      # Sphinx documentation
```

## CI

GitHub Actions workflows run: Ray CPU tests, pre-commit checks, GPU utility tests, vLLM integration tests, and end-to-end AIME24 evaluation. See `.github/workflows/`.
