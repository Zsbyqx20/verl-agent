# Copyright 2025 the ReverseReasoningGenerator (RRG) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Self-judge reward client: uses the policy's own vLLM engine instead of an external reader.

Same public API as RRGRewardClient (score_step_margins, score_traj_recovery,
score_prefix_recovery) but routes inference through actor_rollout_wg (RayWorkerGroup
wrapping the policy's vLLM engine). This eliminates the external 8B reader dependency.

The policy evaluates its own outputs: stale weights are fine -- this is self-assessment,
and the Spearman correlation vs oracle is +0.879 (validated by self_judge_probe.py).

Two inference modes:
  - STEP margin: generate_with_logprobs (needs full per-token logprob distributions for
    MC forced-choice renormalization).
  - TRAJECTORY recovery: standard generate_sequences (text-only JSON assembly).
"""
from __future__ import annotations

import asyncio
import io
import json
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils import torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

from .reward_client import (
    JUDGE_SYSTEM, STEP_SYSTEM, CONTROL_CLAUSES, PRESS_KEYS,
    action_str, hard_distractors, make_control, _render_trace,
)
from . import answer_recovery as A


@dataclass
class _MCPending:
    """Handle returned by SelfJudgeClient.submit_score_step_margins. Joins the
    background MC-scoring thread and decodes per-item margins on .get()."""
    thread: Any                 # threading.Thread running the sync MC RPC
    result_box: Dict[str, Any]  # {'gen_batch': DataProto, 'dt': float, 'error'?: Exception}
    gold_letters: List[str]
    option_letters: List[str]
    decoder: Any                # tokenizer with .decode([tid]) for letter lookup

    def get(self) -> List[float]:
        """Block on the background thread and return per-item margins."""
        self.thread.join()
        if "error" in self.result_box:
            raise self.result_box["error"]
        gen_batch = self.result_box["gen_batch"]
        if "dt" in self.result_box:
            print(
                f"[rrg-self-judge] step_margins (async) n={len(self.gold_letters)} "
                f"dt={self.result_box['dt']:.3f}s",
                flush=True,
            )
        return self._decode(gen_batch)

    def _decode(self, gen_batch) -> List[float]:
        full_lps = gen_batch.non_tensor_batch.get("full_logprobs")
        if full_lps is None:
            return [0.0] * len(self.gold_letters)
        margins = []
        for i, gold_letter in enumerate(self.gold_letters):
            lp_dicts = full_lps[i]
            if not lp_dicts:
                margins.append(0.0)
                continue
            pos0 = lp_dicts[0]
            gold_lp = self._logprob_for_letter(pos0, gold_letter)
            lps = [self._logprob_for_letter(pos0, L) for L in self.option_letters]
            m = max(lps)
            denom = math.log(sum(math.exp(v - m) for v in lps)) + m
            margins.append(float(max(0.0, min(1.0, math.exp(gold_lp - denom)))))
        return margins

    def _logprob_for_letter(self, logprob_dict, letter: str) -> float:
        for tid, lp in logprob_dict.items():
            tok = self.decoder.decode([tid]).strip()
            if tok == letter:
                return lp
        max_lp = max(logprob_dict.values()) if logprob_dict else 0.0
        return max_lp - 10.0


# --------------------------------------------------------------------------- #
# SelfJudgeClient — same external API as RRGRewardClient
# --------------------------------------------------------------------------- #
class SelfJudgeClient:
    """Policy-as-judge client using the local vLLM engine (via actor_rollout_wg).

    Constructed lazily by RRGEnvironmentManager / RRGTrajectoryRewardManager when
    set_self_judge_wg() is called and env.rrg.self_judge=True.
    """

    def __init__(self, tokenizer, processor, actor_rollout_wg, config):
        self.tokenizer = tokenizer
        self.processor = processor
        self.wg = actor_rollout_wg
        self.max_image_long = config.get("max_image_long", 768)
        self.num_distractors = config.get("num_distractors", 4)
        self.subtract_control = config.get("subtract_control", False)
        self.seed = config.get("seed", 0)
        self.max_prompt_length = config.get("max_prompt_length", 32768)
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        # Answer-assembly system prompt override (see RRGRewardClient) -- same knob, same
        # default fallback, so self-judge and the HTTP reader stay in sync.
        answer_prompt_path = config.get("answer_prompt_path")
        if answer_prompt_path:
            from pathlib import Path
            self.answer_prompt = Path(answer_prompt_path).read_text(encoding="utf-8")
        else:
            self.answer_prompt = A.ANSWER_PROMPT
        # Detect processor class for position_id computation
        pc_name = processor.__class__.__name__ if processor is not None else ""
        if "Qwen3VL" in pc_name:
            from verl.models.transformers.qwen3_vl import get_rope_index
            self._get_rope_index = get_rope_index
        elif "Qwen2VL" in pc_name:
            from verl.models.transformers.qwen2_vl import get_rope_index
            self._get_rope_index = get_rope_index
        else:
            self._get_rope_index = None
        # LRU cache for the image→(image_grid_thw, expanded template string) expansion.
        # Keyed on the image bytes hash so identical screenshots reuse the processor output
        # across calls (and across training iterations). Bounded to avoid unbounded growth
        # on a long-running train loop. Survives across rollouts within this client.
        self._vision_cache: Dict[Tuple[int, int], Tuple[Any, str]] = {}
        self._vision_cache_max = 256

    def _generate_sequences(self, batch: DataProto) -> DataProto:
        """Run rollout generation with padding for non-divisible reward batches."""
        batch_padded, pad_size = pad_dataproto_to_divisor(batch, self.wg.world_size)
        gen_padded = self.wg.generate_sequences(batch_padded)
        return unpad_dataproto(gen_padded, pad_size=pad_size)

    def _generate_with_logprobs(self, batch: DataProto) -> DataProto:
        """Run rollout generation with full logprobs for non-divisible reward batches."""
        batch_padded, pad_size = pad_dataproto_to_divisor(batch, self.wg.world_size)
        gen_padded = self.wg.generate_with_logprobs(batch_padded)
        return unpad_dataproto(gen_padded, pad_size=pad_size)

    # ----- helpers ----- #
    @staticmethod
    def _data_url(img_path_or_pil, max_long: int) -> str:
        im = Image.open(img_path_or_pil).convert("RGB") if isinstance(img_path_or_pil, str) else img_path_or_pil.convert("RGB")
        w, h = im.size
        if max(w, h) > max_long:
            s = max_long / max(w, h)
            im = im.resize((max(1, round(w * s)), max(1, round(h * s))))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return "data:image/png;base64," + __import__("base64").b64encode(buf.getvalue()).decode()

    def _vision_expand_cached(self, img: Image.Image) -> Tuple[str, Any]:
        """Return (vision_block_str, image_grid_thw) for an image. Caches on bytes hash so
        repeated screenshots (e.g., same `task_id:step_idx` across training iterations, or
        identical images within a batch) skip processor.image_processor and the placeholder
        expansion loop. image_grid_thw is the per-image tensor returned by the processor; the
        vision_block_str is `<|vision_start|><image_token>×N<|vision_end|>` ready to splice
        into a chat-templated prompt in place of '<image>'.
        """
        key = (hash(img.tobytes()), img.size[0] * img.size[1])
        cached = self._vision_cache.get(key)
        if cached is not None:
            return cached
        row_mm_data = {'image': [np.array(img)]}
        image_inputs = self.processor.image_processor(row_mm_data['image'], return_tensors='pt')
        image_grid_thw = image_inputs['image_grid_thw']
        merge_length = self.processor.image_processor.merge_size ** 2
        n_placeholders = int(image_grid_thw[0].prod().item() // merge_length)
        vision_block = (
            '<|vision_start|>' + self.processor.image_token * n_placeholders + '<|vision_end|>')
        result = (vision_block, image_grid_thw)
        if len(self._vision_cache) >= self._vision_cache_max:
            # Cheap LRU-ish eviction: drop the oldest insertion (FIFO). Dict preserves
            # insertion order in CPython 3.7+.
            self._vision_cache.pop(next(iter(self._vision_cache)))
        self._vision_cache[key] = result
        return result

    def _tokenize_batch(self, messages_list: List[List[Dict]], images_list: List = None) -> DataProto:
        """Tokenize a list of chat message lists into a DataProto for vLLM generation.

        Mirrors the exact multi-modal tokenization pattern from TrajectoryCollector.preprocess_batch
        in rollout_loop.py. Image-bearing prompts use a literal '<image>' marker in the text,
        replaced with vision tokens after chat-template application.
        """
        bs = len(messages_list)
        all_input_ids = []
        all_attention_mask = []
        all_position_ids = []
        multi_modal_data = []
        has_images = images_list is not None and any(im is not None for im in images_list)

        for i in range(bs):
            messages = messages_list[i]
            # For image prompts: extract the text part only (strip image content),
            # insert a literal '<image>' marker, apply chat template, then replace.
            if has_images and images_list[i] is not None:
                img = images_list[i]
                if isinstance(img, str):
                    img = Image.open(img).convert("RGB")
                # Extract text-only user content (strip image entries)
                user_msgs = [m for m in messages if m["role"] == "user"]
                text_parts = []
                for um in user_msgs:
                    content = um.get("content", "")
                    if isinstance(content, list):
                        # Multi-modal content: keep only text parts
                        text_parts.extend(
                            part["text"] for part in content if part.get("type") == "text")
                    else:
                        text_parts.append(str(content))
                user_text = "\n".join(text_parts)

                # Build a simple chat message with <image> marker prepended
                chat = [{"role": "user", "content": f"<image>\n{user_text}"}]
                prompt_with_template = self.tokenizer.apply_chat_template(
                    chat, add_generation_prompt=True, tokenize=False)

                # Vision expansion (cached): see _vision_expand_cached docstring.
                vision_block, image_grid_thw = self._vision_expand_cached(img)
                prompt_with_template = prompt_with_template.replace('<image>', vision_block, 1)
                row_mm_data = {'image': [np.array(img)]}
                multi_modal_data.append(row_mm_data)
            else:
                # Text-only: apply chat template normally
                prompt_with_template = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False)
                multi_modal_data.append(None)
                image_grid_thw = None

            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_with_template, tokenizer=self.tokenizer,
                max_length=self.max_prompt_length, pad_token_id=self.pad_token_id,
                left_pad=True, truncation='error')

            # Compute position IDs
            if has_images and image_grid_thw is not None and self._get_rope_index is not None:
                vision_position_ids = self._get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask[0],
                )  # (3, seq_length)
                valid_mask = attention_mask[0].bool()
                text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
                text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
                position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
                position_ids = position_ids[0]  # -> (4, seq_length)
            else:
                position_ids = compute_position_id_with_mask(attention_mask)

            all_input_ids.append(input_ids[0])
            all_attention_mask.append(attention_mask[0])
            all_position_ids.append(position_ids[0] if position_ids.dim() == 2 else position_ids)

        # Pad to same length
        max_len = max(ids.size(0) for ids in all_input_ids)
        pad_id = self.pad_token_id

        def pad_tensor(t, max_l, pad_val=0):
            if t.size(0) < max_l:
                return torch.nn.functional.pad(t, (0, max_l - t.size(0)), value=pad_val)
            return t

        input_ids_batch = torch.stack([pad_tensor(ids, max_len, pad_id) for ids in all_input_ids])
        attn_batch = torch.stack([pad_tensor(a, max_len, 0) for a in all_attention_mask])

        pos_dim = all_position_ids[0].dim()
        if pos_dim == 2:
            pos_batch = torch.stack([pad_tensor(p, max_len, 0) for p in all_position_ids])
        else:
            max_pos_len = max(p.size(-1) for p in all_position_ids)
            pos_padded = []
            for p in all_position_ids:
                if p.size(-1) < max_pos_len:
                    p = torch.nn.functional.pad(p, (0, max_pos_len - p.size(-1)), value=0)
                pos_padded.append(p)
            pos_batch = torch.stack(pos_padded)

        batch = {
            "input_ids": input_ids_batch,
            "attention_mask": attn_batch,
            "position_ids": pos_batch,
        }
        non_tensor = {}
        if any(mmd is not None for mmd in multi_modal_data):
            non_tensor["multi_modal_data"] = np.array(multi_modal_data, dtype=object)

        return DataProto.from_dict(
            tensors=batch,
            non_tensors=non_tensor,
            meta_info={"eos_token_id": self.tokenizer.eos_token_id},
        )

    def _logprob_for_letter(self, logprob_dict: Dict[int, float], letter: str) -> float:
        """Return the logprob of a single-letter token from a position's top-K dict."""
        # Try exact match on token text
        for tid, lp in logprob_dict.items():
            tok = self.tokenizer.decode([tid]).strip()
            if tok == letter:
                return lp
        # Fallback: use a very low logprob
        max_lp = max(logprob_dict.values()) if logprob_dict else 0.0
        return max_lp - 10.0  # ~0.000045 probability

    # ----- public API (mirrors RRGRewardClient) ----- #
    def _prepare_step_margin_batch(self, items, action_pool):
        """Shared prompt-prep for sync and async step-margin calls. Returns
        (batch, gold_letters, option_letters, items_for_redecode).
        The batch is ready to feed _generate_with_logprobs / submit_score_step_margins."""
        rng = random.Random(self.seed)
        n_opts = self.num_distractors + 1
        option_letters = [chr(ord("A") + j) for j in range(n_opts)]

        prompts_messages = []
        images = []
        gold_letters = []

        for idx, it in enumerate(items):
            a = it["action"]
            distract = hard_distractors(a, action_pool, rng, self.num_distractors)
            options = distract + [action_str(a)]
            rng.shuffle(options)
            gold_idx = options.index(action_str(a))
            gold_letter = chr(ord("A") + gold_idx)

            mc_text = self._build_mc_text(it["goal"], it.get("reasoning") or None, options)

            user_content = []
            img_path = it["image"]
            if isinstance(img_path, str):
                user_content.append({"type": "image", "image": self._data_url(img_path, self.max_image_long)})
                images.append(img_path)
            else:
                images.append(None)
            user_content.append({"type": "text", "text": mc_text})

            messages = [
                {"role": "system", "content": STEP_SYSTEM},
                {"role": "user", "content": user_content},
            ]
            prompts_messages.append(messages)
            gold_letters.append(gold_letter)

        batch = self._tokenize_batch(prompts_messages, images)
        for k, v in ("max_tokens", 1), ("logprobs", 20), ("temperature", 0), ("n", 1), ("do_sample", False):
            batch.meta_info[k] = v
        return batch, gold_letters, option_letters, items, rng

    def _decode_step_margins(self, gen_batch, gold_letters, option_letters):
        """Decode per-item margins from a generate_with_logprobs result. Shared by sync
        and async paths."""
        full_lps = gen_batch.non_tensor_batch.get("full_logprobs")
        if full_lps is None:
            return [0.0] * len(gold_letters)
        margins = []
        for i, gold_letter in enumerate(gold_letters):
            lp_dicts = full_lps[i]
            if not lp_dicts:
                margins.append(0.0)
                continue
            pos0 = lp_dicts[0]
            gold_lp = self._logprob_for_letter(pos0, gold_letter)
            lps = [self._logprob_for_letter(pos0, L) for L in option_letters]
            m = max(lps)
            denom = math.log(sum(math.exp(v - m) for v in lps)) + m
            margins.append(float(max(0.0, min(1.0, math.exp(gold_lp - denom)))))
        return margins

    def score_step_margins(self, items: List[Dict[str, Any]], action_pool: List[str]) -> List[float]:
        """items: [{goal, image, action(dict), reasoning}]; returns per-item margin in [-1,1].

        Sync path: uses the policy's own vLLM engine via generate_with_logprobs to compute
        renormalized P(correct|screenshot+reasoning) over K-way MC forced choice. Blocks
        until the GPU run completes. Honors self.subtract_control (off by default) by
        also scoring a content-free reasoning and subtracting its probability."""
        batch, gold_letters, option_letters, _, rng = self._prepare_step_margin_batch(items, action_pool)
        t0 = time.perf_counter()
        gen_batch = self._generate_with_logprobs(batch)
        print(
            f"[rrg-self-judge] step_margins n={len(items)} "
            f"prompt_len={batch.batch['input_ids'].shape[-1]} dt={time.perf_counter() - t0:.3f}s",
            flush=True,
        )
        margins = self._decode_step_margins(gen_batch, gold_letters, option_letters)

        if self.subtract_control:
            # Optional: subtract P(correct|control) where control is content-free reasoning.
            # Off by default. Implements the same MC pattern with a fresh batch.
            ctrl_margins = self._score_step_margins_control(items, action_pool, rng)
            margins = [m - c for m, c in zip(margins, ctrl_margins)]
            margins = [max(-1.0, min(1.0, m)) for m in margins]

        return margins

    def _score_step_margins_control(self, items, action_pool, rng):
        """P(correct | content-free reasoning) — used by subtract_control to remove the
        baseline reward the model gives without seeing the actual reasoning text."""
        n_opts = self.num_distractors + 1
        option_letters = [chr(ord("A") + j) for j in range(n_opts)]
        prompts_messages = []
        images = []
        gold_letters = []
        for it in items:
            a = it["action"]
            it_options = hard_distractors(a, action_pool, rng, self.num_distractors)
            it_options.append(action_str(a))
            ctrl_text = self._build_mc_text(
                it["goal"], make_control(it.get("reasoning") or "", rng),
                it_options)
            ctrl_content = []
            img_path = it["image"]
            if isinstance(img_path, str):
                ctrl_content.append({"type": "image", "image": self._data_url(img_path, self.max_image_long)})
                images.append(img_path)
            else:
                images.append(None)
            ctrl_content.append({"type": "text", "text": ctrl_text})
            prompts_messages.append([
                {"role": "system", "content": STEP_SYSTEM},
                {"role": "user", "content": ctrl_content},
            ])
            gold_idx = it_options.index(action_str(a))
            gold_letters.append(chr(ord("A") + gold_idx))
        batch = self._tokenize_batch(prompts_messages, images)
        for k, v in ("max_tokens", 1), ("logprobs", 20), ("temperature", 0), ("n", 1), ("do_sample", False):
            batch.meta_info[k] = v
        gen_batch = self._generate_with_logprobs(batch)
        return self._decode_step_margins(gen_batch, gold_letters, option_letters)

    def submit_score_step_margins(self, items, action_pool) -> "_MCPending":
        """Async path: fire the MC scoring RPC in a background thread, return immediately
        with an _MCPending. The thread calls the sync wg.generate_with_logprobs(batch)
        (which goes through the @register dispatch + ray.get on the Ray actor task).
        Because the Ray actor task queue serializes anyway, we don't save GPU work —
        we save DRIVER-SIDE blocking time: the rollout loop can run the next iteration's
        actor_rollout_wg.generate_sequences in parallel with the MC scoring thread.
        This is the simplest async that works with the existing @register dispatch."""
        import threading
        batch, gold_letters, option_letters, _, _ = self._prepare_step_margin_batch(items, action_pool)
        batch_padded, pad_size = pad_dataproto_to_divisor(batch, self.wg.world_size)
        result_box: Dict[str, Any] = {}

        def _run():
            try:
                t0 = time.perf_counter()
                gen_batch = self.wg.generate_with_logprobs(batch_padded)
                # The sync path's _generate_with_logprobs wrapper added pad/unpad and
                # the registered dispatch; for the threaded path we do it inline.
                if pad_size:
                    gen_batch = unpad_dataproto(gen_batch, pad_size=pad_size)
                result_box["gen_batch"] = gen_batch
                result_box["dt"] = time.perf_counter() - t0
            except Exception as e:
                result_box["error"] = e

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return _MCPending(
            thread=thread,
            result_box=result_box,
            gold_letters=gold_letters,
            option_letters=option_letters,
            decoder=self.tokenizer,
        )

    @staticmethod
    def _build_mc_text(goal: str, reasoning: Optional[str], options: List[str]) -> str:
        letters = [chr(ord("A") + i) for i in range(len(options))]
        opt_lines = "\n".join(f"{l}. {o}" for l, o in zip(letters, options))
        rblock = f"Reasoning for the current step:\n{reasoning}\n\n" if reasoning else ""
        return (
            f"Goal: {goal}\n\nNotes so far:\n(none)\n\n{rblock}"
            f"Candidate next actions:\n{opt_lines}\n\n"
            "Which single candidate is the correct next action? Answer with only its letter."
        )

    # ----- trajectory recovery (text-only, standard generate) ----- #
    def score_traj_recovery(self, items: List[Dict[str, Any]],
                            max_tokens: int = 1024) -> List[Dict[str, Any]]:
        """items: [{goal, reasonings:[str], gold, schema}]. Blind text-only answer assembly.

        Uses the policy's vLLM engine (standard generate_sequences) to assemble the
        gold-schema JSON answer, then scores field-level recall against gold.
        """
        prompts_messages = []
        for it in items:
            user_text = (
                f"# Task goal\n{it['goal']}\n\n"
                f"# Agent's step-by-step reasoning (its only memory of the trajectory)\n"
                f"{_render_trace(it['reasonings'])}\n\n"
                f"# Required output JSON Schema\n{json.dumps(it['schema'], ensure_ascii=False, indent=2)}"
            )
            prompts_messages.append([
                {"role": "system", "content": self.answer_prompt},
                {"role": "user", "content": user_text},
            ])

        batch = self._tokenize_batch(prompts_messages, images_list=[None] * len(items))
        for k, v in ("max_tokens", max_tokens), ("temperature", 0), ("n", 1), ("do_sample", False):
            batch.meta_info[k] = v
        gen_batch = self._generate_sequences(batch)

        responses = gen_batch.batch["responses"]  # (bs, max_resp_len)
        out = []
        for i, it in enumerate(items):
            resp_ids = responses[i]
            # Strip padding
            valid_mask = resp_ids != self.pad_token_id
            if valid_mask.any():
                resp_ids = resp_ids[valid_mask]
            raw = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
            answer, perr = A.extract_json(raw)
            if answer is None:
                out.append({"recall": 0.0, "correct": False, "answer": None,
                           "parse_error": perr})
                continue
            rec = A.recall(answer, it["gold"])
            diffs = A.compare_answer(answer, it["gold"])
            out.append({"recall": float(rec), "correct": (not diffs),
                        "answer": answer, "parse_error": ""})
        return out

    def score_prefix_recovery(self, items: List[Dict[str, Any]],
                              max_tokens: int = 1024,
                              max_prefixes: int = 8) -> List[Dict[str, Any]]:
        """Per-prefix answer recovery (for step-credit). Reuses same logic pattern as
        RRGRewardClient.score_prefix_recovery but routes through local vLLM."""
        from .reward_client import RRGRewardClient
        # Plan which prefixes to score
        plans = []  # (item_idx, prefix_len)
        for ii, it in enumerate(items):
            lengths = RRGRewardClient._prefix_lengths(len(it["reasonings"]), max_prefixes)
            for L in lengths:
                plans.append((ii, L))

        # Build prompts for all prefix queries
        prompts_messages = []
        for ii, L in plans:
            it = items[ii]
            user_text = (
                f"# Task goal\n{it['goal']}\n\n"
                f"# Agent's step-by-step reasoning (its only memory of the trajectory)\n"
                f"{_render_trace(it['reasonings'][:L])}\n\n"
                f"# Required output JSON Schema\n{json.dumps(it['schema'], ensure_ascii=False, indent=2)}"
            )
            prompts_messages.append([
                {"role": "system", "content": self.answer_prompt},
                {"role": "user", "content": user_text},
            ])

        batch = self._tokenize_batch(prompts_messages, images_list=[None] * len(plans))
        for k, v in ("max_tokens", max_tokens), ("temperature", 0), ("n", 1), ("do_sample", False):
            batch.meta_info[k] = v
        gen_batch = self._generate_sequences(batch)

        responses = gen_batch.batch["responses"]
        # Parse results per prefix query
        per = [dict() for _ in items]  # item_idx -> {prefix_len: (recall, matched_set, answer, perr)}
        for p_idx, (ii, L) in enumerate(plans):
            it = items[ii]
            resp_ids = responses[p_idx]
            valid_mask = resp_ids != self.pad_token_id
            if valid_mask.any():
                resp_ids = resp_ids[valid_mask]
            raw = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
            answer, perr = A.extract_json(raw)
            if answer is None:
                per[ii][L] = (0.0, set(), None, perr)
                continue
            per[ii][L] = (float(A.recall(answer, it["gold"])),
                         A.matched_field_paths(answer, it["gold"]), answer, "")

        # Forward-fill for unsampled prefixes
        out = []
        for ii, it in enumerate(items):
            n = len(it["reasonings"])
            sampled = per[ii]
            recalls, mpaths = [], []
            last_rec, last_mp = 0.0, set()
            for L in range(1, n + 1):
                if L in sampled:
                    rec, mp, _ans, _perr = sampled[L]
                    if rec is not None:
                        last_rec, last_mp = rec, mp
                recalls.append(last_rec)
                mpaths.append(set(last_mp))
            final = sampled.get(n)
            if final and final[0] is not None and final[2] is not None:
                f_rec, _f_mp, f_ans, f_perr = final
                correct = not A.compare_answer(f_ans, it["gold"])
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
