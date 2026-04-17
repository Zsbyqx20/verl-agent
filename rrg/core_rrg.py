"""RRG advantage estimator with span-masked, group-normalized advantages.

The design parallels ``gigpo/core_gigpo.py``'s group normalization but adds
span-level masking so that citation tokens, writing tokens, and reasoning
tokens receive different advantage signals.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

# ------------------------------------------------------------------ #
# Group normalization (mirrors gigpo/core_gigpo.py:episode_norm_reward)
# ------------------------------------------------------------------ #


def _group_normalize(
    scores: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    remove_std: bool = True,
) -> torch.Tensor:
    """Group-normalize scalar scores across rollouts sharing the same ``index``.

    Args:
        scores: ``(bs,)`` scalar per sample.
        index: ``(bs,)`` group IDs (samples with the same ID form a group).
        traj_index: ``(bs,)`` trajectory UIDs (used for deduplication within a
            group when ``compute_mean_std_cross_steps`` is True).
        epsilon: numerical stability term.
        remove_std: if True, only subtract mean (Dr.GRPO style); if False,
            also divide by std (original GRPO style).

    Returns:
        Normalized scores ``(bs,)``.
    """
    id2scores: dict[str, list[torch.Tensor]] = defaultdict(list)
    seen_pairs: set[tuple] = set()

    bsz = scores.shape[0]
    with torch.no_grad():
        for i in range(bsz):
            pair = (index[i], traj_index[i])
            if pair in seen_pairs:
                continue
            id2scores[index[i]].append(scores[i])
            # Don't deduplicate by (index, traj) — include all steps

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


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #

def compute_rrg_advantage(
    token_level_rewards: torch.Tensor,   # (bs, response_length) — combined scalar
    response_mask: torch.Tensor,         # (bs, response_length)
    index: np.ndarray,                   # (bs,) — group uid for normalization
    traj_index: np.ndarray,              # (bs,) — trajectory uid
    cite_rewards: np.ndarray,            # (bs,) — R_cite per step
    write_rewards: np.ndarray,           # (bs,) — R_write per step
    cite_masks: torch.Tensor,            # (bs, response_length) — bool
    write_masks: torch.Tensor,           # (bs, response_length) — bool
    w_cite: float = 1.0,
    w_write: float = 1.0,
    w_final: float = 0.0,               # phase 2
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute span-masked advantages for RRG.

    1. Group-normalize cite and write rewards across rollouts of the same
       source trajectory.
    2. Broadcast to token level via span masks.
    3. Combine into final ``(bs, response_length)`` advantages.

    Returns:
        ``(advantages, returns)`` both of shape ``(bs, response_length)``.
    """
    device = token_level_rewards.device
    bsz, response_length = token_level_rewards.shape

    cite_rewards_t = torch.tensor(cite_rewards, dtype=torch.float32, device=device)
    write_rewards_t = torch.tensor(write_rewards, dtype=torch.float32, device=device)

    # Group normalize each signal
    a_cite = _group_normalize(cite_rewards_t, index, traj_index, epsilon, remove_std=True)
    a_write = _group_normalize(write_rewards_t, index, traj_index, epsilon, remove_std=True)

    # Broadcast to token level via span masks
    cite_masks_f = cite_masks.float().to(device)
    write_masks_f = write_masks.float().to(device)

    advantages = (
        w_cite * a_cite.unsqueeze(-1) * cite_masks_f
        + w_write * a_write.unsqueeze(-1) * write_masks_f
    )

    # Phase 2: add w_final * A_final (trajectory-level) to both cite and write spans
    # Currently w_final=0.0, so this is a no-op.

    # Apply response mask
    advantages = advantages * response_mask

    return advantages, advantages
