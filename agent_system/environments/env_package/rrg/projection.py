# Copyright 2025 the ReverseReasoningGenerator (RRG) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Action projection for the RRG replay env.

In reverse-reasoning RL the policy output IS the reasoning (and, for RRG obs-memory tasks,
the observation updates) — NOT an action. The action is REPLAYED from the recording, so the
"action" the env consumes is a placeholder. Projection therefore only does a well-formedness
check (non-empty generation). The stronger leakage/restatement VETO needs the GT coordinates
and is applied in RRGEnvironmentManager.step (which has the recorded frame); both are combined
into info['is_action_valid'] → invalid_action_penalty.
"""
from typing import List, Tuple


def rrg_projection(text_actions: List[str]) -> Tuple[List[int], List[int]]:
    actions, valids = [], []
    for t in text_actions:
        ok = bool(t and t.strip())
        actions.append(0)          # placeholder; the replay env ignores it
        valids.append(1 if ok else 0)
    return actions, valids
