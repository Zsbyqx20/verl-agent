# Session Notes — RRG Full-Parameter Run Debugging

Date: 2026-07-11

## Current Work Context

- Active branch/work area is native RRG training on top of `verl-agent`/GiGPO.
- Main run script used in production: `/workspace/exp_scripts/rrg_full_param_1node.sh`.
- Repo-local script copy: `scripts/rrg_full_param_1node.sh`.
- Active training config uses:
  - `algorithm.adv_estimator=gigpo`
  - `env.env_name=rrg`
  - `reward_model.reward_manager=rrg`
  - `env.rrg.self_judge=True`
  - `env.rrg.answer_step_credit=true`
  - `data.max_prompt_length=4096`
  - `data.max_response_length=256`
  - `env.rrg.answer_max_tokens=512`

## Important Length Knobs

- `data.max_prompt_length` caps per-step policy input prompt tokens after chat/multimodal preprocessing.
- `data.max_response_length` caps the trainable policy's per-step rollout generation length; in logs this becomes rollout `max_tokens=256`.
- `env.rrg.answer_max_tokens` caps answer-recovery judge generation length, not policy step output length.
- For RRG these are per-step limits; full trajectory history is not concatenated in the normal GiGPO step-independent flow.

## Production Failure Root Cause

- Original failure with validation size `142` was:
  - `AssertionError: only support equal chunk. Got size of DataProto 142 and chunk 8.`
- Aligned parquet files fixed that specific whole-validation divisibility issue, but production later failed with:
  - `AssertionError: only support equal chunk. Got size of DataProto 15 and chunk 8.`
- This proves dataset-size alignment alone is insufficient.
- Root cause is `SelfJudgeClient` building internal reward/evaluation `DataProto` batches of arbitrary active/scorable sizes and calling `actor_rollout_wg.generate_sequences()` directly.
- Ray worker dispatch chunks by world size (`8`) and requires equal chunking unless the `DataProto` is padded.

## Required Code Fix

Patch `agent_system/environments/env_package/rrg/self_judge_client.py` so every internal self-judge generation call pads to `self.wg.world_size` and unpads after generation.

Local repo currently has this fix:

- Import at `agent_system/environments/env_package/rrg/self_judge_client.py`:
  - `from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto`
- Helper added:
  - `_generate_sequences(self, batch: DataProto) -> DataProto`
  - internally calls `pad_dataproto_to_divisor(batch, self.wg.world_size)`, then `self.wg.generate_sequences(...)`, then `unpad_dataproto(...)`.
- Direct calls replaced with `_generate_sequences(...)` in:
  - step-margin scoring
  - subtractive-control scoring
  - trajectory answer recovery
  - prefix answer recovery

Production should verify with:

```bash
rg -n "def _generate_sequences|self\.wg\.generate_sequences|pad_dataproto" \
  /root/verl-agent/agent_system/environments/env_package/rrg/self_judge_client.py
```

If production still shows `gen_batch = self.wg.generate_sequences(batch)` inside `score_traj_recovery`, it does not have the fix.

## 8-Aligned Parquet Files

Created local aligned copies:

- `data/rrg-aligned/train_aug_1280.parquet`: 1280 rows, duplicated first 2 rows from original 1278.
- `data/rrg-aligned/val_aug_144.parquet`: 144 rows, duplicated first 2 rows from original 142.

Original counts:

- `/root/data/verl-agent/rrg/train_aug.parquet`: 1278 rows, `1278 % 8 = 6`.
- `/root/data/verl-agent/rrg/val_aug.parquet`: 142 rows, `142 % 8 = 6`.

Repo-local `scripts/rrg_full_param_1node.sh` was updated to default to these aligned files and set:

- `data.val_batch_size=144`
- `env.rrg.val_num_episodes=144`

Important: production used `/workspace/exp_scripts/rrg_full_param_1node.sh`, so changes to the repo-local script must be copied there or passed via overrides/env vars.

## Warning Triage

The warning below is expected/noisy and not the cause of failures:

```text
transformers/image_processing_utils_fast.py:585: UserWarning: The given NumPy array is not writable...
image = torch.from_numpy(image).contiguous()
```

It means a read-only NumPy image array was converted to a Torch tensor. Since it is normally only read and immediately made contiguous, it should not affect training correctness unless image corruption or related crashes appear.

## Other Local Changes Noted

- `verl/utils/dataset/vision_utils.py` has a local fix for image dicts with `bytes`: open the `BytesIO` directly before `fetch_image`, because `fetch_image` expects string-like image fields.
- `scripts/rrg_full_param_1node.sh` is untracked/local-run infrastructure and was patched to use aligned parquets.

