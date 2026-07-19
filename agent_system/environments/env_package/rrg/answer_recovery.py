# Copyright 2025 the ReverseReasoningGenerator (RRG) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Pure-python gold-answer loading, schema lookup, and answer-recovery scoring for RRG.

Self-contained port of the validated helpers in the main project
(src/extract_demo.py + src/rrg_reward_probe.py) so the rrpo training stack does NOT
import from the main repo's src/. Everything here is HTTP/PIL/stdlib-only (no torch),
safe to live in rrpo/.venv beside the trainer.

Used by:
  - envs.py     : RRG episode loader (load_rrg_tasks) -> per-task goal + gold + schema.
  - reward_client.py : generative answer-recovery reward (assemble + score_recovery).
  - reward_manager/rrg.py : task_id -> {gold, schema} table + final-answer correctness.

The trajectory reward is generative answer-recovery RECALL (E1/E4-validated, non-saturated):
a blind reader assembles the gold-schema JSON from goal + the policy's reasonings only, and
score_recovery returns field-level recall in [0,1]. compare_answer gives the binary
correctness used for the val success_evaluator.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

# The reader's answer-assembly system prompt (copied from prompts/answer.txt).
ANSWER_PROMPT = (Path(__file__).parent / "answer_prompt.txt").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# JSON extraction + schema validation
# --------------------------------------------------------------------------- #
def extract_json(raw: str) -> tuple[dict | None, str]:
    """Pull the outermost JSON object from a model reply. Returns (obj_or_None, err)."""
    txt = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw.strip()).strip()
    start, end = txt.find("{"), txt.rfind("}")
    if start == -1 or end <= start:
        return None, "no JSON object found in reply"
    try:
        return json.loads(txt[start:end + 1]), ""
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"


def validate_answer(answer: dict, schema: dict) -> list[str]:
    """Return a list of schema-validation error messages ([] == valid)."""
    import jsonschema
    validator = jsonschema.Draft7Validator(schema)
    errs = []
    for e in sorted(validator.iter_errors(answer), key=lambda x: list(x.path)):
        loc = "/".join(str(p) for p in e.path) or "<root>"
        errs.append(f"{loc}: {e.message}")
    return errs


# --------------------------------------------------------------------------- #
# Gold-answer loading from the source CSV + schema cache
# --------------------------------------------------------------------------- #
# _csv_row used to re-open and linearly re-scan the whole CSV on every call, and
# load_gold called it a second time internally -> O(n^2) file I/O + comparisons
# across a task-loop of n tasks (fine at AMEX's ~400 tasks, painfully slow once
# AndroidControl-scale task-roots with 10000+ tasks reuse this same loader).
# Fix: parse each CSV file into an (App, ID) -> row index once, cache it by path.
_CSV_INDEX_CACHE: dict[str, dict[tuple[str, str], dict]] = {}


def _csv_index(csv_path: Path) -> dict[tuple[str, str], dict]:
    key = str(csv_path)
    idx = _CSV_INDEX_CACHE.get(key)
    if idx is None:
        import csv as _csv
        idx = {}
        with open(csv_path, encoding="utf-8-sig") as f:
            for r in _csv.DictReader(f):
                idx[(r.get("App"), str(r.get("ID")))] = r
        _CSV_INDEX_CACHE[key] = idx
    return idx


def _csv_row(csv_path: Path, env: str, task_id) -> dict | None:
    """Find the source-CSV row for {env}:{task_id} (utf-8-sig strips the BOM on 'App').

    O(1) after the first call per csv_path (cached index); see _csv_index."""
    if not csv_path or not csv_path.exists():
        return None
    return _csv_index(csv_path).get((str(env), str(task_id)))


def load_gold(csv_path: Path, env: str, task_id, row: dict | None = None) -> dict | None:
    """Parse the gold answer from the source CSV's 'Validate 备注' column.

    Pass `row` (already fetched via _csv_row, e.g. for resolve_goal_and_lang) to skip
    the redundant second lookup."""
    import json5
    r = row if row is not None else _csv_row(csv_path, env, task_id)
    if not r:
        return None
    raw = (r.get("Validate 备注") or "").strip()
    if not raw:
        return None
    try:
        gold = json5.loads(raw)  # tolerant: single quotes / unquoted keys
    except Exception:  # noqa: BLE001
        return None
    if isinstance(gold, dict):
        gold.pop("taskId", None)  # annotation metadata, not a schema field
    return gold


def resolve_goal_and_lang(row: dict | None, existing_goal: str) -> tuple[str, str]:
    """Pick the goal text and language. LANG column is authoritative; for train
    (empty tasks.json goal) it selects the EN Task vs Task column. Falls back to
    detecting CJK in the goal when no CSV row is available."""
    lang = None
    if row:
        lv = (row.get("LANG") or "").strip().upper()
        lang = "en" if lv.startswith("EN") else ("zh" if lv else None)
    goal = (existing_goal or "").strip()
    if not goal and row:  # train: tasks.json leaves goal empty -> choose CSV column
        if lang == "en":
            goal = (row.get("EN Task") or row.get("Task") or "").strip()
        else:
            goal = (row.get("Task") or "").strip()
    if lang is None:  # no LANG recorded -> infer from the goal text
        lang = "zh" if any("一" <= c <= "鿿" for c in goal) else "en"
    return goal, lang


def auto_gold_csv(task_root: Path) -> Path | None:
    for name in ("rrg-source.csv", "rrg-source-real.csv"):
        if (task_root / name).exists():
            return task_root / name
    return None


# --------------------------------------------------------------------------- #
# Tolerant value/structure comparison (binary correctness)
# --------------------------------------------------------------------------- #
def coerce_number(x) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if not isinstance(x, str):
        return None
    s = x.strip().replace(",", "").replace("¥", "").replace("￥", "").replace("$", "")
    mult = 1.0
    for suf, m in (("万", 1e4), ("亿", 1e8), ("w", 1e4), ("k", 1e3), ("K", 1e3)):
        if s.endswith(suf):
            s, mult = s[: -len(suf)], m
            break
    s = s.rstrip("%元件条次个人km米").strip()
    try:
        return float(s) * mult
    except ValueError:
        return None


def _values_match(a, g) -> bool:
    na, ng = coerce_number(a), coerce_number(g)
    if na is not None and ng is not None:
        return abs(na - ng) <= max(1e-6, abs(ng) * 0.005)  # 0.5% tol for floats
    sa = re.sub(r"\s+", "", str(a)).casefold()
    sg = re.sub(r"\s+", "", str(g)).casefold()
    if sg.endswith("...") or sg.endswith("…"):  # rare truncated gold value
        pref = sg.rstrip(".… ")
        return bool(pref) and (sa.startswith(pref) or pref in sa)
    return sa == sg


def compare_answer(ans, gold, path: str = "") -> list[str]:
    """Tolerant structural compare -> list of mismatch messages ([] == correct).

    Numbers are coerced ('1.9万'->19000); arrays match order-insensitively; strings
    normalize whitespace/case. Only gold keys are required (extra answer keys ignored).
    """
    if isinstance(gold, dict):
        if not isinstance(ans, dict):
            return [f"{path or '<root>'}: expected object, got {type(ans).__name__}"]
        diffs = []
        for k, gv in gold.items():
            if k not in ans:
                diffs.append(f"{path}/{k}: missing")
            else:
                diffs += compare_answer(ans[k], gv, f"{path}/{k}")
        return diffs
    if isinstance(gold, list):
        if not isinstance(ans, list):
            return [f"{path}: expected array, got {type(ans).__name__}"]
        diffs, used = [], set()
        for i, gi in enumerate(gold):
            match = next((j for j in range(len(ans))
                          if j not in used and not compare_answer(ans[j], gi, "<m>")), None)
            if match is None:
                diffs.append(f"{path}[{i}]: no matching item for {gi!r}")
            else:
                used.add(match)
        if len(ans) - len(used) > 0:
            diffs.append(f"{path or '<root>'}: {len(ans) - len(used)} unexpected extra item(s)")
        return diffs
    return [] if _values_match(ans, gold) else [f"{path or '<root>'}: got {ans!r}, expected {gold!r}"]


# --------------------------------------------------------------------------- #
# Graded field-level recall (the continuous reward target)
# --------------------------------------------------------------------------- #
def score_recovery(ans, gold) -> tuple[int, int]:
    """Graded field-level recall: (matched_leaves, total_leaves) of the gold structure.

    Continuous counterpart of compare_answer's binary verdict -- gives the reward dynamic
    range (partial recovery). Arrays are matched greedily by best leaf-overlap; missing
    keys/items count their gold leaves as total-but-unmatched.
    """
    if isinstance(gold, dict):
        m = tot = 0
        if not isinstance(ans, dict):
            ans = {}
        for k, gv in gold.items():
            mm, tt = score_recovery(ans.get(k, None) if k in ans else None, gv)
            m += mm
            tot += tt
        return m, tot
    if isinstance(gold, list):
        m = tot = 0
        pool = list(range(len(ans))) if isinstance(ans, list) else []
        for gi in gold:
            best = None
            best_m = -1
            best_t = 0
            for j in pool:
                mm, tt = score_recovery(ans[j], gi)
                if mm > best_m:
                    best, best_m, best_t = j, mm, tt
            gm, gtot = score_recovery(None, gi)  # gold-leaf count for this item
            if best is not None:
                pool.remove(best)
                m += best_m
                tot += max(best_t, gtot)
            else:
                tot += gtot
        return m, tot
    # scalar leaf
    if ans is None:
        return 0, 1
    return (1 if _values_match(ans, gold) else 0), 1


def recall(ans, gold) -> float:
    m, tot = score_recovery(ans, gold)
    return (m / tot) if tot else 0.0


def matched_field_paths(ans, gold, path: str = "") -> set:
    """Set of gold leaf-paths that MATCH in `ans` -- the companion to score_recovery, which
    returns only counts. Same recursion and greedy list pairing, so
    `len(matched_field_paths(a, g)) == score_recovery(a, g)[0]` and
    `matched_field_paths(g, g)` is the full gold leaf-path universe (size == total_leaves).
    Used for per-step first-appearance credit (which gold field became recoverable WHEN)."""
    out: set = set()
    if isinstance(gold, dict):
        if not isinstance(ans, dict):
            ans = {}
        for k, gv in gold.items():
            sub = ans.get(k, None) if k in ans else None
            out |= matched_field_paths(sub, gv, f"{path}/{k}")
        return out
    if isinstance(gold, list):
        pool = list(range(len(ans))) if isinstance(ans, list) else []
        for idx, gi in enumerate(gold):
            best, best_m = None, -1
            for j in pool:
                mm, _ = score_recovery(ans[j], gi)  # same selection rule as score_recovery
                if mm > best_m:
                    best, best_m = j, mm
            if best is not None:
                pool.remove(best)
                out |= matched_field_paths(ans[best], gi, f"{path}[{idx}]")
        return out
    # scalar leaf
    if ans is None:
        return out
    if _values_match(ans, gold):
        out.add(path or "<root>")
    return out
