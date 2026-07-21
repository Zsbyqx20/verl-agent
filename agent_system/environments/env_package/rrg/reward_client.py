# Copyright 2025 the ReverseReasoningGenerator (RRG) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""8B reader reward client for the RRG replay env (GiGPO two-channel reward).

HTTP-only (openai + PIL, no torch), so it lives safely in rrpo/.venv next to the
training stack and talks to the frozen reader served separately (Qwen3-VL-8B-Instruct
@ 117.74.66.190:10727). Two reward channels, both ported from the validated probes in
the main project (src/action_recovery_eval.py, src/traj_judge_probe.py):

  STEP  (action-recovery margin): reader SEES the screenshot; K-way forced choice over the
        GT action + hard type-aware distractors given goal+screenshot+reasoning. Score =
        renormalized P(correct letter) in [0,1] (optionally minus a content-free control).
        Decision (rrg-sft-quality): 8B base reader, fluency prior ~0, gold/gen >> none/random.

  TRAJ  (completion judge): reader BLIND to state; P(yes=done) given goal + the trajectory's
        reasonings. (Toy: absolute P(yes); production refinement = contrastive vs truncated.)

Both are exposed as SYNC batch entry points (score_step_margins / score_traj_pyes) so the
in-process env.step() can call them once per turn over the whole env vector.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import random
import re
from typing import Any, Dict, List, Optional

from PIL import Image

# --------------------------------------------------------------------------- #
# Prompts (kept identical to the validated probes)
# --------------------------------------------------------------------------- #
STEP_SYSTEM = (
    "You are a GUI agent operating an Android phone. Given the user's goal, the "
    "current screenshot, and any notes taken so far, you judge the next action to take."
)
JUDGE_SYSTEM = (
    "You are evaluating whether a phone GUI agent has FULLY completed its task. "
    "You are given the task goal and the agent's step-by-step reasoning notes. "
    "You cannot see the screen; judge only from the notes. Be strict: only say the "
    "task is complete if the notes provide concrete evidence that every part of the "
    "goal has been accomplished."
)

CONTROL_CLAUSES = [
    "Let me look at the current screen carefully.",
    "I should think about what to do next to make progress on the task.",
    "It is important to choose an appropriate action here.",
    "I will consider the available options on this screen.",
    "Taking the right step now will help complete the task.",
    "Let me proceed in a sensible and careful manner.",
]
PRESS_KEYS = ["BACK", "HOME", "ENTER", "APP_SWITCH", "MENU"]


# --------------------------------------------------------------------------- #
# Action canonicalization + hard distractors (ported from action_recovery_eval.py)
# --------------------------------------------------------------------------- #
# AMEX swipes carry two raw points (coordinate/coordinate2) with no direction field;
# AndroidControl swipes (from androidcontrol_to_rrg_format.py's _androidcontrol_action)
# already carry "direction" directly and have NO coordinate/coordinate2 at all -- calling
# swipe_dir() on one raises KeyError. action_str() below checks for "direction" first so
# both schemas work.
def swipe_dir(a: dict) -> str:
    (x1, y1), (x2, y2) = a["coordinate"], a["coordinate2"]
    dx, dy = x2 - x1, y2 - y1
    if abs(dy) >= abs(dx):
        return "up" if dy < 0 else "down"
    return "left" if dx < 0 else "right"


ANDROIDCONTROL_BUTTONS = ["back", "home"]


def action_str(a: dict) -> str:
    v = a.get("action")
    if v == "click":
        x, y = a["coordinate"]
        return f"click({x}, {y})"
    if v == "long_press":  # AndroidControl: same coordinate schema as click
        x, y = a["coordinate"]
        return f"long_press({x}, {y})"
    if v == "type":
        return f'type("{a.get("text", "")}")'
    if v == "swipe":
        direction = a["direction"] if "direction" in a else swipe_dir(a)
        return f"swipe({direction})"
    if v == "terminate":
        return f"terminate({a.get('status', 'success')})"
    if v == "press_key":
        return f"press_key({a.get('key', '')})"
    if v == "open_app":  # AndroidControl
        return f'open_app("{a.get("app", "")}")'
    if v == "wait":  # AndroidControl
        return "wait()"
    if v == "system_button":  # AndroidControl
        return f"system_button({a.get('button', '')})"
    return json.dumps(a, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Open-generation step reward (action-recovery SSR margin).
#
# Instead of a K-way MC over [gold + distractors], the reader GENERATES the next action
# (same DSL as action_str) given goal + screenshot + reasoning; we score that generated
# action against gold with an SSR-style match (exact action-type + params; smooth distance
# decay for click/long_press coordinates; soft text score for type). The reward passed to
# GiGPO is the MARGIN score(reasoning) - score(control): what the reasoning adds ON TOP OF
# the screenshot the reader already sees (== what it would add for the sighted downstream
# agent). This drops the fragile hand-tuned distractor machinery entirely.
#
# Coordinate handling: gold coords in tasks.json are ALREADY in 0-1000 NORMALIZED space
# (empirically x,y p99 ~944/947, max ~1000; the SSR eval treats them as 0-1000 and scales
# BOTH gold and prediction up by W/H before its pixel-threshold compare). The reader is
# instructed to emit 0-1000 coords too, so gold and prediction share one space -- compute a
# single isotropic Euclidean distance directly, NO per-image W/H normalization. tau=90 ->
# reward ~exp(-1)=0.37 at 90 normalized units; the eval's 140px threshold is ~130 units on X
# and ~58 on Y for a 1080x2400 screen, so tau=90 straddles the eval boundary isotropically.
# --------------------------------------------------------------------------- #
STEP_GEN_SYSTEM = (
    "You are a GUI agent operating an Android phone. Given the current screenshot and the "
    "agent's reasoning for this step, output the single next action to "
    "take. Use exactly one of these forms and nothing else:\n"
    "  open_app(\"AppName\")\n  click(x, y)\n  long_press(x, y)\n  type(\"text\")\n"
    "  swipe(direction)   # direction in up/down/left/right\n"
    "  system_button(button)   # button in back/home\n  wait()\n"
    "Coordinates x, y are integers normalized to 0-1000 on both axes (0,0 = top-left)."
)
COORD_TAU = 90.0  # 0-1000-space distance scale for the click/long_press decay

_GEN_ACT_RE = re.compile(
    r'(open_app|click|long_press|type|swipe|system_button|wait)\s*\(([^)]*)\)', re.IGNORECASE)


def parse_gen_action(raw: str) -> Optional[dict]:
    """Parse the reader's generated action DSL back into an action dict (last match wins,
    so trailing chatter before the call still parses). Returns None if unparseable."""
    if not raw:
        return None
    matches = list(_GEN_ACT_RE.finditer(raw))
    if not matches:
        return None
    m = matches[-1]
    verb = m.group(1).lower()
    arg = m.group(2).strip()
    try:
        if verb in ("click", "long_press"):
            parts = [p.strip() for p in arg.split(",")]
            if len(parts) != 2:
                return None
            return {"action": verb, "coordinate": [float(parts[0]), float(parts[1])]}
        if verb == "type":
            return {"action": "type", "text": arg.strip().strip('"\'')}
        if verb == "swipe":
            return {"action": "swipe", "direction": arg.strip().strip('"\'').lower()}
        if verb == "system_button":
            return {"action": "system_button", "button": arg.strip().strip('"\'').lower()}
        if verb == "open_app":
            return {"action": "open_app", "app": arg.strip().strip('"\'')}
        if verb == "wait":
            return {"action": "wait"}
    except (ValueError, IndexError):
        return None
    return None


def _text_score(pred: str, gold: str) -> float:
    """Soft text match for type() content: token-level F1 (case/space-insensitive), so
    right-entity-wrong-surface-form ('Miami' vs 'miami florida') gets partial credit."""
    pt = (pred or "").lower().split()
    gt = (gold or "").lower().split()
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    from collections import Counter
    cp, cg = Counter(pt), Counter(gt)
    overlap = sum((cp & cg).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(pt), overlap / len(gt)
    return 2 * prec * rec / (prec + rec)


def score_gen_action(pred: Optional[dict], gold: dict, img_wh: tuple) -> float:
    """SSR-style score in [0,1] of a generated action vs gold. Action-type must match
    (else 0). Then: click/long_press -> exp(-d/tau) on the single isotropic distance in
    0-1000 normalized space; type -> token-F1 on text; open_app/swipe/system_button/wait
    -> exact categorical (1/0)."""
    if pred is None:
        return 0.0
    gv, pv = gold.get("action"), pred.get("action")
    if pv != gv:
        return 0.0
    if gv in ("click", "long_press"):
        gx, gy = gold["coordinate"]   # already 0-1000 (see module note above)
        px, py = pred["coordinate"]   # reader emits 0-1000
        d = math.hypot(px - gx, py - gy)
        return math.exp(-d / COORD_TAU)
    if gv == "type":
        return _text_score(pred.get("text", ""), gold.get("text", ""))
    if gv == "swipe":
        return 1.0 if pred.get("direction") == gold.get("direction") else 0.0
    if gv == "system_button":
        return 1.0 if pred.get("button") == gold.get("button") else 0.0
    if gv == "open_app":
        return 1.0 if (pred.get("app", "").lower() == gold.get("app", "").lower()) else 0.0
    if gv == "wait":
        return 1.0
    return 0.0


def _build_gen_text(reasoning: Optional[str]) -> str:
    """NOTE: the task GOAL is deliberately WITHHELD. With the goal present, a sighted reader
    infers the task-relevant action from goal+screenshot and ignores the reasoning -> the
    reasoning earns ~0 margin on click/open_app/type (screenshot-dominated). Withholding the
    goal makes the reasoning the SOLE carrier of intent: reasoning that names the target/app/
    text earns large margin vs content-free control, while a screenshot-obvious single-button
    step correctly still earns ~0 (no reasoning contribution). Validated on a mixed-action
    sample: mean margin ~0 (with goal) -> ~0.57 (no goal), open_app/click/type all opened up."""
    rblock = f"Agent's reasoning for this step:\n{reasoning}\n\n" if reasoning else ""
    return (f"{rblock}Output the single next action now.")


# Empirically observed action-type confusion pairs: which OTHER action type a trained model
# is most likely to mistakenly predict for a given gold type, from the AndroidControl SSR
# eval's confusion matrix (base Qwen3-VL-4B, AMEX-SFT-4B-backfilled agent, and RL-step40-
# backfilled agent, aggregated over the same held-out 1255-step test set). click dominates as
# the wrong guess for nearly everything -- it's the majority action by volume (~65% of all
# steps), so it's the "safe default" whenever a model is unsure: wait->click 49.5%,
# system_button->click 40.8%, open_app->click 23.4%, swipe->click 21.5%, type->click 16.5%.
# click itself is most often mistaken for swipe (9.6%) or wait (5.4%). The plain uniform-random
# cross-verb pool below doesn't deliberately include these -- this ordering does, so the
# distractor set actually contains the confusions a model is likely to make, not just any
# other action.
CONFUSION_PRIORITY: Dict[str, List[str]] = {
    "click": ["swipe", "wait", "system_button", "open_app"],
    "long_press": ["click", "swipe"],
    "swipe": ["click", "system_button", "wait"],
    "type": ["click", "wait"],
    "open_app": ["click", "system_button", "swipe"],
    "wait": ["click", "swipe", "type", "open_app"],
    "system_button": ["click", "open_app", "swipe"],
}


def _prioritized_cross(gt: str, v: str, pool: List[str], rng: random.Random) -> List[str]:
    """Cross-verb distractor pool ordered by CONFUSION_PRIORITY (most-confused-for action
    types first) instead of uniform-random. Falls back to plain shuffle for unlisted/legacy
    (AMEX-only) types, same behavior as before this existed."""
    by_verb: Dict[str, List[str]] = {}
    for c in pool:
        if c == gt:
            continue
        by_verb.setdefault(c.split("(", 1)[0], []).append(c)
    ordered: List[str] = []
    seen_verbs = set()
    for verb in CONFUSION_PRIORITY.get(v, []):
        items = by_verb.get(verb, [])
        rng.shuffle(items)
        ordered += items
        seen_verbs.add(verb)
    remaining = [c for verb, items in by_verb.items() if verb not in seen_verbs for c in items]
    rng.shuffle(remaining)
    ordered += remaining
    return ordered


def hard_distractors(a: dict, pool: List[str], rng: random.Random, k: int) -> List[str]:
    gt = action_str(a)
    v = a.get("action")
    cands: List[str] = []
    if v in ("click", "long_press"):
        x, y = a["coordinate"]
        # 2 "hard" near-misses placed just outside the ~140px SSR match tolerance (rather than
        # only the far ones below, which let vague "top-right corner"-style reasoning pass the
        # MC test without naming the actual target -- see rrg-androidcontrol click-regression
        # finding). Offsets are scaled per axis (not a single symmetric value): AndroidControl
        # screens are tall portraits (~1080x2400 dominant), so the same normalized-space delta
        # is ~2.2x more actual pixels on Y than X; 140 on X and 65 on Y both land at ~150-160px
        # real distance on that dominant profile. This is a fixed approximation (hard_distractors
        # has no access to the actual per-image width/height), not exact for every screen.
        # Listed BEFORE the far ones so they always survive the k truncation: at the default
        # num_distractors=4 the selected set is [near1, near2, far1, far2] -- a real mix, not
        # all-coarse and not all-fine (see the design discussion this implements). A higher
        # num_distractors pulls in more of the far set too.
        near_offsets = [(140, 0), (0, -65)]
        far_offsets = [(300, 0), (-300, 0), (0, 400), (0, -400), (380, 380), (-380, -380)]
        for dx, dy in near_offsets + far_offsets:
            nx, ny = min(990, max(10, x + dx)), min(990, max(10, y + dy))
            cands.append(f"{v}({nx}, {ny})")
    elif v == "swipe":
        cands += [f"swipe({d})" for d in ("up", "down", "left", "right")]
    elif v == "press_key":
        cands += [f"press_key({key})" for key in PRESS_KEYS]
    elif v == "terminate":
        cands += ["terminate(failure)", "terminate(success)"]
    elif v == "type":
        cands += ['type("search")', 'type("home")', 'type("settings")', 'type("login")']
    elif v == "open_app":  # AndroidControl: reuse other apps' real names from the global pool
        cands += [c for c in pool if c.startswith("open_app(")]
    elif v == "system_button":  # AndroidControl
        cands += [f"system_button({b})" for b in ANDROIDCONTROL_BUTTONS]
    # "wait" (AndroidControl) has no parameters -> no same-type near-miss is possible;
    # falls through to the cross-verb pool below like AMEX's other unhandled cases.
    cands += _prioritized_cross(gt, v, pool, rng)
    seen, out = {gt}, []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out[:k]


def make_control(reasoning: str, rng: random.Random) -> str:
    target = max(1, len(reasoning.split()))
    out, n = [], 0
    while n < target:
        c = rng.choice(CONTROL_CLAUSES)
        out.append(c)
        n += len(c.split())
    return " ".join(" ".join(out).split()[:target])


def _build_mc_text(goal: str, reasoning: Optional[str], options: List[str]) -> str:
    letters = [chr(ord("A") + i) for i in range(len(options))]
    opt_lines = "\n".join(f"{l}. {o}" for l, o in zip(letters, options))
    rblock = f"Reasoning for the current step:\n{reasoning}\n\n" if reasoning else ""
    return (
        f"Goal: {goal}\n\nNotes so far:\n(none)\n\n{rblock}"
        f"Candidate next actions:\n{opt_lines}\n\n"
        "Which single candidate is the correct next action? Answer with only its letter."
    )


def _render_trace(reasonings: List[str]) -> str:
    return "\n".join(f"{i}. {r}" for i, r in enumerate(reasonings, 1))


def data_url(path_or_img, max_long: int) -> str:
    im = Image.open(path_or_img).convert("RGB") if isinstance(path_or_img, str) else path_or_img.convert("RGB")
    w, h = im.size
    if max(w, h) > max_long:
        s = max_long / max(w, h)
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------- #
# Reward client
# --------------------------------------------------------------------------- #
class RRGRewardClient:
    def __init__(self, base_url: str, model_name: str, max_image_long: int = 768,
                 num_distractors: int = 4, concurrency: int = 64, seed: int = 0,
                 subtract_control: bool = False, api_key: str = "sk-dummy",
                 answer_prompt_path: Optional[str] = None,
                 step_reward_mode: str = "mc", gen_max_tokens: int = 64,
                 gen_n: int = 8, gen_temperature: float = 0.8):
        self.base_url = base_url
        self.model = model_name
        self.max_image_long = max_image_long
        self.num_distractors = num_distractors
        self.concurrency = concurrency
        self.subtract_control = subtract_control
        self.seed = seed
        self.api_key = api_key
        # Step-reward mode:
        #   "mc"  (default, byte-identical prior behavior): K-way forced-choice over
        #         gold+distractors, score = renormalized P(gold letter). Optional
        #         subtract_control margin.
        #   "gen": reader OPEN-GENERATES the next action from screenshot+reasoning with the GOAL
        #         WITHHELD (see _build_gen_text -- withholding the goal is what makes the reasoning
        #         the sole carrier of intent, so reasoning that names the target earns signal).
        #         Score = SSR-style match vs gold (score_gen_action). Per rollout we draw gen_n
        #         samples at gen_temperature and use the MEAN SSR (expectation over the reader's
        #         action distribution given the reasoning -> smooth/continuous even for categorical
        #         actions; vLLM shares the image prefill across the n samples so it's ~15% over a
        #         single greedy call, not n x). NO explicit content-free control call: GiGPO's
        #         step_norm_reward subtracts the anchor-group mean (all G rollouts at a step share
        #         the same screenshot+gold), which removes the screenshot-obviousness baseline that
        #         control was there to subtract -- so control is redundant on the micro channel.
        #         *** This redundancy holds ONLY on the step/micro channel with step_advantage_w>0;
        #         if the micro channel is off, raw mean-SSR leaks the screenshot baseline. The
        #         env/trainer should guard on that (warn if gen mode + step_advantage_w==0). ***
        self.step_reward_mode = step_reward_mode
        self.gen_max_tokens = gen_max_tokens
        self.gen_n = max(1, int(gen_n))
        self.gen_temperature = float(gen_temperature)
        # Answer-assembly system prompt override (e.g. AndroidControl's blind
        # action-sequence-recovery framing instead of RRG's default info-retrieval one).
        # None (default) -> _assemble_answer falls back to answer_recovery.ANSWER_PROMPT,
        # byte-identical to the pre-existing hardcoded behavior.
        if answer_prompt_path:
            from pathlib import Path
            self.answer_prompt = Path(answer_prompt_path).read_text(encoding="utf-8")
        else:
            self.answer_prompt = None

    def _client(self):
        from openai import AsyncOpenAI
        import httpx
        # trust_env=False => ignore ambient http_proxy/all_proxy (the training box exports a
        # SOCKS proxy); both the local 8B reader and the Ark/doubao endpoint are directly
        # reachable, and routing reward calls through the proxy would fail (no socksio).
        # timeout bumped to 240s: a long answer assembly (answer_max_tokens up to ~4k) on the
        # 8B reader can otherwise hit the proxy's request timeout.
        return AsyncOpenAI(base_url=self.base_url, api_key=self.api_key,
                           http_client=httpx.AsyncClient(trust_env=False, timeout=240.0))

    @staticmethod
    async def _retry(coro_factory, *, attempts: int = 4):
        """Await coro_factory() with retries on transient reader errors (transport drops,
        408/409/425/429, 5xx). A blip must not silently zero a reward. Re-raises the last
        error after `attempts` tries so the caller's return_exceptions handling still applies."""
        import asyncio as _asyncio
        from openai import APIConnectionError, APITimeoutError, RateLimitError, InternalServerError, APIStatusError
        last = None
        for i in range(attempts):
            try:
                return await coro_factory()
            except (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError) as e:
                last = e
            except APIStatusError as e:
                if getattr(e, "status_code", 0) in (408, 409, 425, 429) or getattr(e, "status_code", 0) >= 500:
                    last = e
                else:
                    raise
            await _asyncio.sleep(1.5 * (i + 1))
        raise last

    async def _p_correct(self, client, sem, durl: str, text: str, n_opts: int, gold_idx: int) -> float:
        """Renormalized P(correct letter) over the candidate set in [0,1]."""
        letters = [chr(ord("A") + i) for i in range(n_opts)]
        async with sem:
            r = await self._retry(lambda: client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": STEP_SYSTEM},
                          {"role": "user", "content": [
                              {"type": "image_url", "image_url": {"url": durl}},
                              {"type": "text", "text": text}]}],
                max_tokens=1, temperature=0, logprobs=True, top_logprobs=20))
        top = r.choices[0].logprobs.content[0].top_logprobs
        floor = min(t.logprob for t in top) - 5.0
        vals = []
        for L in letters:
            cand = [t.logprob for t in top if t.token.strip() == L]
            vals.append(max(cand) if cand else floor)
        m = max(vals)
        denom = math.log(sum(math.exp(v - m) for v in vals)) + m
        return math.exp(vals[gold_idx] - denom)

    async def _p_yes(self, client, sem, goal: str, reasonings: List[str]) -> float:
        text = (
            f"Task goal: {goal}\n\n"
            f"The agent's step-by-step reasoning notes:\n{_render_trace(reasonings)}\n\n"
            "Based only on these notes, has the agent FULLY completed the task? "
            "Answer with a single word: Yes or No.")
        async with sem:
            r = await self._retry(lambda: client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": JUDGE_SYSTEM},
                          {"role": "user", "content": text}],
                max_tokens=1, temperature=0, logprobs=True, top_logprobs=20))
        top = r.choices[0].logprobs.content[0].top_logprobs
        floor = min(t.logprob for t in top) - 5.0

        def lp(word):
            c = [t.logprob for t in top if t.token.strip().lower() == word]
            return max(c) if c else floor
        ly, ln = lp("yes"), lp("no")
        m = max(ly, ln)
        denom = math.log(math.exp(ly - m) + math.exp(ln - m)) + m
        return math.exp(ly - denom)

    # ----- sync batch entry points (called from the in-process env / reward mgr) ----- #
    def score_step_margins(self, items: List[Dict[str, Any]], action_pool: List[str]) -> List[float]:
        """items: [{goal, image, action(dict), reasoning}]; returns per-item margin in [-1,1].
        margin = P(correct|reasoning) [- P(correct|control) if subtract_control]."""
        return asyncio.run(self._score_step_margins(items, action_pool))

    async def _gen_score(self, client, sem, durl, gold, img_wh, reasoning) -> float:
        """Reader open-generates the next action from screenshot+reasoning (GOAL WITHHELD --
        see _build_gen_text). Draws gen_n samples in ONE call (vLLM shares the image prefill,
        so ~15% over a single greedy call, not n x) and returns the MEAN SSR-style match vs
        gold in [0,1] -- an estimate of E[SSR] over the reader's action distribution given the
        reasoning, so categorical actions become continuous and clicks reward tight clustering.
        gen_n=1 with gen_temperature=0 recovers greedy single-sample scoring."""
        text = _build_gen_text(reasoning)
        n = self.gen_n
        async with sem:
            r = await self._retry(lambda: client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": STEP_GEN_SYSTEM},
                          {"role": "user", "content": [
                              {"type": "image_url", "image_url": {"url": durl}},
                              {"type": "text", "text": text}]}],
                max_tokens=self.gen_max_tokens,
                temperature=(self.gen_temperature if n > 1 else 0.0), n=n))
        scores = [score_gen_action(parse_gen_action(ch.message.content or ""), gold, img_wh)
                  for ch in r.choices]
        return sum(scores) / len(scores) if scores else 0.0

    async def _score_step_margins(self, items, action_pool):
        client, sem = self._client(), asyncio.Semaphore(self.concurrency)
        rng = random.Random(self.seed)

        if self.step_reward_mode == "gen":
            # Open-generation mean-SSR, ONE call per rollout (gen_n samples averaged inside
            # _gen_score). No explicit control call: GiGPO's step_norm_reward subtracts the
            # anchor-group mean, which removes the shared screenshot-obviousness baseline that
            # a control would subtract (see __init__ note + the step_advantage_w>0 guard).
            jobs = []
            for it in items:
                im = Image.open(it["image"]).convert("RGB") if isinstance(it["image"], str) \
                    else it["image"].convert("RGB")
                durl = data_url(im, self.max_image_long)
                jobs.append(self._gen_score(client, sem, durl, it["action"], im.size,
                                            it.get("reasoning") or None))
            res = await asyncio.gather(*jobs, return_exceptions=True)
            return [0.0 if isinstance(v, Exception) else float(v) for v in res]

        # default: MC forced-choice (unchanged)
        jobs, meta = [], []
        for idx, it in enumerate(items):
            a = it["action"]
            distract = hard_distractors(a, action_pool, rng, self.num_distractors)
            options = distract + [action_str(a)]
            rng.shuffle(options)
            gold_idx = options.index(action_str(a))
            durl = data_url(it["image"], self.max_image_long)
            text = _build_mc_text(it["goal"], it.get("reasoning") or None, options)
            jobs.append(self._p_correct(client, sem, durl, text, len(options), gold_idx))
            meta.append((idx, "gen"))
            if self.subtract_control:
                ctext = _build_mc_text(it["goal"], make_control(it.get("reasoning") or "", rng), options)
                jobs.append(self._p_correct(client, sem, durl, ctext, len(options), gold_idx))
                meta.append((idx, "ctrl"))
        res = await asyncio.gather(*jobs, return_exceptions=True)
        gen = [0.0] * len(items)
        ctrl = [0.0] * len(items)
        for (idx, kind), v in zip(meta, res):
            val = 0.0 if isinstance(v, Exception) else float(v)
            if kind == "gen":
                gen[idx] = val
            else:
                ctrl[idx] = val
        return [gen[i] - (ctrl[i] if self.subtract_control else 0.0) for i in range(len(items))]

    def score_traj_pyes(self, items: List[Dict[str, Any]]) -> List[float]:
        """items: [{goal, reasonings:[str]}]; returns per-item P(yes=done) in [0,1]."""
        return asyncio.run(self._score_traj_pyes(items))

    async def _score_traj_pyes(self, items):
        client, sem = self._client(), asyncio.Semaphore(self.concurrency)
        jobs = [self._p_yes(client, sem, it["goal"], it["reasonings"]) for it in items]
        res = await asyncio.gather(*jobs, return_exceptions=True)
        return [0.0 if isinstance(v, Exception) else float(v) for v in res]

    # ----- generative answer-recovery (the RRG trajectory reward) ----- #
    async def _assemble_answer(self, client, sem, goal: str, reasonings: List[str],
                               schema: dict, max_tokens: int):
        """Blind reader assembles the gold-schema JSON answer from goal + reasonings ONLY
        (no screenshots). This is the answerability test: are the policy's notes sufficient
        to reconstruct the answer? Returns (answer_obj_or_None, parse_err)."""
        from agent_system.environments.env_package.rrg import answer_recovery as A
        system_prompt = self.answer_prompt if self.answer_prompt is not None else A.ANSWER_PROMPT
        user = (f"# Task goal\n{goal}\n\n"
                f"# Agent's step-by-step reasoning (its only memory of the trajectory)\n"
                f"{_render_trace(reasonings)}\n\n"
                f"# Required output JSON Schema\n{json.dumps(schema, ensure_ascii=False, indent=2)}")
        async with sem:
            r = await self._retry(lambda: client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user}],
                max_tokens=max_tokens, temperature=0.0))
        raw = r.choices[0].message.content or ""
        return A.extract_json(raw)

    def score_traj_recovery(self, items: List[Dict[str, Any]],
                            max_tokens: int = 1024) -> List[Dict[str, Any]]:
        """items: [{goal, reasonings:[str], gold, schema}]. For each trajectory the blind
        reader assembles the gold-schema answer from the reasonings, then we score field-level
        recall in [0,1] against gold. Returns per-item dicts:
            {recall: float, correct: bool, answer: obj|None, parse_error: str}
        recall is the GiGPO macro (trajectory) reward; correct (compare_answer empty) is the
        val success signal. On any error recall=0.0 (a failed assemble == no recovery)."""
        return asyncio.run(self._score_traj_recovery(items, max_tokens))

    async def _score_traj_recovery(self, items, max_tokens):
        from agent_system.environments.env_package.rrg import answer_recovery as A
        client, sem = self._client(), asyncio.Semaphore(self.concurrency)
        jobs = [self._assemble_answer(client, sem, it["goal"], it["reasonings"],
                                      it["schema"], max_tokens) for it in items]
        res = await asyncio.gather(*jobs, return_exceptions=True)
        out = []
        for it, r in zip(items, res):
            if isinstance(r, Exception):
                out.append({"recall": 0.0, "correct": False, "answer": None,
                            "parse_error": f"{type(r).__name__}: {r}"})
                continue
            answer, perr = r
            if answer is None:
                out.append({"recall": 0.0, "correct": False, "answer": None,
                            "parse_error": perr})
                continue
            rec = A.recall(answer, it["gold"])
            diffs = A.compare_answer(answer, it["gold"])
            out.append({"recall": float(rec), "correct": (not diffs),
                        "answer": answer, "parse_error": ""})
        return out

    # ----- per-step answer-recovery (prefix-wise; the GiGPO MICRO answer credit) ----- #
    def score_prefix_recovery(self, items: List[Dict[str, Any]],
                              max_tokens: int = 1024,
                              max_prefixes: int = 8) -> List[Dict[str, Any]]:
        """For each trajectory, assemble the answer from NOTE PREFIXES notes[0:k] to find WHEN
        each gold field first becomes recoverable. items: [{goal, reasonings:[str], gold, schema}].
        Returns per-item dicts:
            {recalls: [r_0..r_{N-1}], matched_paths: [set_0..set_{N-1}],
             recall: float, correct: bool, answer: obj|None, parse_error: str, n_reader_calls: int}
        recall/answer/correct come from the FULL (final) prefix and equal score_traj_recovery's
        result, so the caller can reuse this as the MACRO reward with no extra assemble. The
        number of reader calls per trajectory is capped at max_prefixes (subsample evenly, always
        including prefix 1 and the final prefix N); un-sampled steps are forward-filled."""
        return asyncio.run(self._score_prefix_recovery(items, max_tokens, max_prefixes))

    @staticmethod
    def _prefix_lengths(n: int, max_prefixes: int) -> list:
        """Prefix lengths in [1, n] to actually score: all of them if n<=max_prefixes, else an
        evenly-spaced subset that always includes 1 and n."""
        if n <= 0:
            return []
        if n <= max_prefixes or max_prefixes <= 1:
            return list(range(1, n + 1)) if max_prefixes > 1 else [n]
        step = (n - 1) / (max_prefixes - 1)
        return sorted({int(round(1 + i * step)) for i in range(max_prefixes)})

    async def _score_prefix_recovery(self, items, max_tokens, max_prefixes):
        from agent_system.environments.env_package.rrg import answer_recovery as A
        client, sem = self._client(), asyncio.Semaphore(self.concurrency)
        plans = []  # (item_idx, prefix_len)
        for ii, it in enumerate(items):
            for L in self._prefix_lengths(len(it["reasonings"]), max_prefixes):
                plans.append((ii, L))
        jobs = [self._assemble_answer(client, sem, items[ii]["goal"],
                                      items[ii]["reasonings"][:L], items[ii]["schema"], max_tokens)
                for (ii, L) in plans]
        res = await asyncio.gather(*jobs, return_exceptions=True)
        per = [dict() for _ in items]  # item_idx -> {prefix_len: (recall|None, matched_set|None, answer, perr)}
        for (ii, L), r in zip(plans, res):
            gold = items[ii]["gold"]
            if isinstance(r, Exception):
                per[ii][L] = (None, None, None, f"{type(r).__name__}: {r}")
                continue
            answer, perr = r
            if answer is None:
                per[ii][L] = (0.0, set(), None, perr)
                continue
            per[ii][L] = (float(A.recall(answer, gold)), A.matched_field_paths(answer, gold), answer, "")
        out = []
        for ii, it in enumerate(items):
            n = len(it["reasonings"])
            gold = it["gold"]
            sampled = per[ii]
            recalls, mpaths = [], []
            last_rec, last_mp = 0.0, set()
            for L in range(1, n + 1):  # forward-fill every step from the most recent sampled prefix
                if L in sampled:
                    rec, mp, _ans, _perr = sampled[L]
                    if rec is not None:  # an exception keeps the previous value
                        last_rec, last_mp = rec, mp
                recalls.append(last_rec)
                mpaths.append(set(last_mp))
            final = sampled.get(n)
            if final and final[0] is not None and final[2] is not None:
                f_rec, _f_mp, f_ans, f_perr = final
                correct = not A.compare_answer(f_ans, gold)
                answer_obj, parse_error = f_ans, f_perr
            else:
                f_rec = recalls[-1] if recalls else 0.0
                correct, answer_obj = False, None
                parse_error = (final[3] if final else "no reasonings")
            out.append({"recalls": recalls, "matched_paths": mpaths,
                        "recall": float(recalls[-1] if recalls else 0.0),
                        "correct": bool(correct), "answer": answer_obj,
                        "parse_error": parse_error, "n_reader_calls": len(sampled)})
        return out
