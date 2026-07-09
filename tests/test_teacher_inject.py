#!/usr/bin/env python3
"""No-GPU unit tests for teacher-demo injection (exploration).

Covers the two risky new pieces:
  - TrajectoryCollector._splice_forced_responses  (tensor surgery on responses/input_ids/attn)
  - RRGEnvironmentManager._forced_responses        (first-k-of-group teacher slot designation + gating)

Run: rrpo/.venv/bin/python rrpo/tests/test_teacher_inject.py   (from workspace root)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rrpo/

import torch
from verl import DataProto
from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
from agent_system.environments.env_manager import RRGEnvironmentManager


class MockTok:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [3 + (ord(c) % 50) for c in text]  # ids in [3,52]; never pad(0)/eos(2)


class Stub:  # minimal `self` carrying only what the method reads
    pass


def test_splice():
    tok = MockTok()
    bs, prompt_len, resp_len = 3, 4, 6
    seq = prompt_len + resp_len
    # input_ids: prompt region = 7, response region = sentinel 99
    input_ids = torch.full((bs, seq), 99, dtype=torch.long)
    input_ids[:, :prompt_len] = 7
    responses = torch.full((bs, resp_len), 99, dtype=torch.long)
    attn = torch.ones((bs, seq), dtype=torch.long)
    batch = DataProto.from_single_dict({"input_ids": input_ids, "responses": responses,
                                        "attention_mask": attn})
    obs = {"forced_response": ["ab", None, "abcdefghij"]}  # slot1 untouched; slot2 over-length
    s = Stub(); s.tokenizer = tok
    n = TrajectoryCollector._splice_forced_responses(s, batch, obs)
    assert n == 2, n
    r = batch.batch["responses"]; ii = batch.batch["input_ids"]; am = batch.batch["attention_mask"]

    # slot 0: "ab" -> [enc(a),enc(b),eos,pad,pad,pad]; valid len 3
    exp0 = tok.encode("ab") + [tok.eos_token_id] + [tok.pad_token_id] * 3
    assert r[0].tolist() == exp0, r[0].tolist()
    assert ii[0, prompt_len:].tolist() == exp0
    assert am[0, :prompt_len].tolist() == [1, 1, 1, 1]              # prompt mask untouched
    assert am[0, prompt_len:].tolist() == [1, 1, 1, 0, 0, 0]        # 3 real tokens then pad
    assert ii[0, :prompt_len].tolist() == [7, 7, 7, 7]             # prompt ids untouched

    # slot 1: forced None -> entirely unchanged
    assert r[1].tolist() == [99] * resp_len
    assert ii[1, prompt_len:].tolist() == [99] * resp_len
    assert am[1].tolist() == [1] * seq

    # slot 2: over-length -> truncated to resp_len-1 tokens + eos (full, vlen=6)
    exp2 = tok.encode("abcdefghij")[:resp_len - 1] + [tok.eos_token_id]
    assert len(exp2) == resp_len
    assert r[2].tolist() == exp2, r[2].tolist()
    assert am[2, prompt_len:].tolist() == [1] * resp_len
    print("test_splice: OK")


def _forced(store, k, group_n, frames, train=True, anneal=None, reset_count=0):
    s = Stub()
    s.teacher_store = store; s.teacher_seed_k = k; s.group_n = group_n
    s.is_train_env = train; s.teacher_anneal_end_step = anneal; s._reset_count = reset_count
    return RRGEnvironmentManager._forced_responses(s, frames)


def test_forced_responses():
    store = {"A": ["a0", "a1", "a2"], "B": ["b0"]}
    # 4 slots, group_n=2 -> groups [0,1],[2,3]; teacher = first 1 of each group = slots 0 and 2
    frames = [{"task_id": "A", "step_idx": 1}, {"task_id": "A", "step_idx": 1},
              {"task_id": "B", "step_idx": 0}, {"task_id": "B", "step_idx": 0}]
    assert _forced(store, 1, 2, frames) == ["a1", None, "b0", None]

    # k=2 of group_n=2 -> both slots of each group are teacher
    assert _forced(store, 2, 2, frames) == ["a1", "a1", "b0", "b0"]

    # disabled paths -> whole-list None
    assert _forced(store, 0, 2, frames) is None                 # k=0
    assert _forced(store, 1, 2, frames, train=False) is None    # val env
    assert _forced({}, 1, 2, frames) is None                    # empty store
    assert _forced(store, 1, 2, frames, anneal=5, reset_count=5) is None  # anneal cutoff reached
    assert _forced(store, 1, 2, frames, anneal=5, reset_count=4) == ["a1", None, "b0", None]  # before cutoff

    # step out of range / task not in store -> None for that slot
    frames2 = [{"task_id": "A", "step_idx": 9}, {"task_id": "X", "step_idx": 0},
               {"task_id": "B", "step_idx": 0}, {"task_id": "B", "step_idx": 0}]
    assert _forced(store, 1, 2, frames2) == [None, None, "b0", None]  # slot0 step OOR, slot1 unknown task
    print("test_forced_responses: OK")


if __name__ == "__main__":
    test_splice()
    test_forced_responses()
    print("ALL TEACHER-INJECT UNIT TESTS PASSED")
