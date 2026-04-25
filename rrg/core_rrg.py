"""RRG advantage estimator with span-masked, group-normalized advantages.

RRG v2: two span types (reasoning, fact). Three reward signals:
    - fact_rewards:  per-step scalar, already token-discounted.
    - reason_rewards: per-step scalar from rank judge, in [-1, 1].
    - final_rewards: per-trajectory (can_conclude binary) broadcast to all
      siblings of the trajectory before group normalization.

Tokens in the fact span receive w_fact * A_fact + w_final * A_final.
Tokens in the reasoning span receive w_reason * A_reason (no A_final).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


def _group_normalize(
    scores: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    remove_std: bool = True,
) -> torch.Tensor:
    """Group-normalize scalar scores across rollouts sharing the same ``index``.

    Mirrors ``gigpo/core_gigpo.py:episode_norm_reward``: mean-center by default
    (Dr.GRPO style); if ``remove_std=False``, also divide by std.
    """
    id2scores: dict[str, list[torch.Tensor]] = defaultdict(list)
    bsz = scores.shape[0]
    with torch.no_grad():
        for i in range(bsz):
            id2scores[index[i]].append(scores[i])

        id2mean: dict = {}
        id2std: dict = {}
        for idx, vals in id2scores.items():
            if len(vals) <= 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            else:
                stacked = torch.stack(vals)
                id2mean[idx] = stacked.mean()
                id2std[idx] = stacked.std()

        out = scores.clone()
        for i in range(bsz):
            mean = id2mean[index[i]]
            std = id2std[index[i]]
            if remove_std:
                out[i] = out[i] - mean
            else:
                out[i] = (out[i] - mean) / (std + epsilon)

    return out


def compute_rrg_advantage(
    token_level_rewards: torch.Tensor,   # (bs, response_length) — combined scalar, unused but kept for API parity
    response_mask: torch.Tensor,         # (bs, response_length)
    index: np.ndarray,                   # (bs,) — step-group uid (e.g. f"{uid}_{step_t}")
    traj_index: np.ndarray,              # (bs,) — sibling-rollout group uid for final reward normalization
    fact_rewards: np.ndarray,            # (bs,) — r_fact per step (already token-discounted)
    reason_rewards: np.ndarray,          # (bs,) — r_reason per step in [-1, 1]
    final_rewards: np.ndarray,           # (bs,) — r_final per trajectory (broadcast across all steps of that trajectory)
    fact_masks: torch.Tensor,            # (bs, response_length) — bool
    reason_masks: torch.Tensor,          # (bs, response_length) — bool
    w_fact: float = 1.0,
    w_reason: float = 1.0,
    w_final: float = 1.0,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute span-masked advantages for RRG v2.

    Returns ``(advantages, returns)`` both shape ``(bs, response_length)``.
    """
    device = token_level_rewards.device

    fact_t = torch.tensor(fact_rewards, dtype=torch.float32, device=device)
    reason_t = torch.tensor(reason_rewards, dtype=torch.float32, device=device)
    final_t = torch.tensor(final_rewards, dtype=torch.float32, device=device)

    a_fact = _group_normalize(fact_t, index, epsilon, remove_std=True)
    a_reason = _group_normalize(reason_t, index, epsilon, remove_std=True)
    a_final = _group_normalize(final_t, traj_index, epsilon, remove_std=True)

    fact_mask_f = fact_masks.float().to(device)
    reason_mask_f = reason_masks.float().to(device)

    advantages = (
        w_fact * a_fact.unsqueeze(-1) * fact_mask_f
        + w_reason * a_reason.unsqueeze(-1) * reason_mask_f
        + w_final * a_final.unsqueeze(-1) * fact_mask_f
    )
    advantages = advantages * response_mask

    return advantages, advantages
