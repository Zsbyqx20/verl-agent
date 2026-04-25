from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Prompt-injection sanitization
# ---------------------------------------------------------------------------
# Tokens that would be reinterpreted as vision placeholders if a model-authored
# string is concatenated back into a multimodal prompt. The rollout's
# ``preprocess_single_sample`` counts ``<image>`` occurrences, and the Qwen-VL
# substitutions consume ``<|vision_start|>``/``<|image_pad|>``/``<|vision_end|>``.
# A model that learns to emit any of these literally will desync image_grid_thw
# from the prompt's vision placeholder count and crash the rollout.
_SANITIZE_PATTERNS = (
    "<image>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|placeholder|>",
)


def sanitize_for_prompt(text: str) -> str:
    """Strip tokens that would be reinterpreted as image markers when re-injected into a multimodal prompt."""
    if not text:
        return text
    for pat in _SANITIZE_PATTERNS:
        if pat in text:
            text = text.replace(pat, "")
    return text


@dataclass
class RRGParseResult:
    """Parsed model output for one RRG generation.

    RRG v2 output has two sections only: action reasoning and observation
    writing (add-only). Citation is removed.
    """
    reasoning_text: str = ""
    writing_text: str = ""
    writing_updates: list[dict[str, Any]] = field(default_factory=list)
    # Character offsets in the full output string (start, end)
    reasoning_span: tuple[int, int] = (0, 0)
    writing_span: tuple[int, int] = (0, 0)


# ---------------------------------------------------------------------------
# Section header patterns (legacy plain-text format)
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r"\[Action Reasoning\]|\[Fact Writing\]|\[Observation Writing\]",
    re.IGNORECASE,
)

_ADD_RE = re.compile(r"ADD\s+(\d+)\s*:\s*(.+)", re.IGNORECASE)


def _find_json_object_span(text: str) -> tuple[int, int] | None:
    """Return the first balanced top-level JSON object span in text."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return start, i + 1
        start = text.find("{", start + 1)
    return None


def _find_json_field_value_span(text: str, field: str, object_start: int, object_end: int) -> tuple[int, int]:
    """Best-effort character span for a top-level JSON field value."""
    field_pattern = re.compile(rf'"{re.escape(field)}"\s*:', re.IGNORECASE)
    match = field_pattern.search(text, object_start, object_end)
    if not match:
        return (0, 0)

    value_start = match.end()
    while value_start < object_end and text[value_start].isspace():
        value_start += 1
    if value_start >= object_end:
        return (0, 0)

    opener = text[value_start]
    if opener == '"':
        in_escape = False
        for i in range(value_start + 1, object_end):
            ch = text[i]
            if in_escape:
                in_escape = False
            elif ch == "\\":
                in_escape = True
            elif ch == '"':
                return value_start, i + 1
        return value_start, object_end

    if opener in "[{":
        closer = "]" if opener == "[" else "}"
        depth = 0
        in_string = False
        escape = False
        for i in range(value_start, object_end):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return value_start, i + 1
        return value_start, object_end

    value_end = value_start
    while value_end < object_end and text[value_end] not in ",}":
        value_end += 1
    return value_start, value_end


def _normalize_update(update: Any) -> dict[str, Any] | None:
    """Coerce a writing entry into an add-only dict. UPDATE entries are rejected.

    Primary schema: ``observation_update`` is a list[str] — each string is one
    atomic fact and the index is auto-assigned. Dict entries are still accepted
    for backward compatibility but must have ``action == "add"``.
    """
    if isinstance(update, str):
        stripped = update.strip()
        if not stripped:
            return None
        m_add = _ADD_RE.match(stripped)
        if m_add:
            return {
                "action": "add",
                "observation_index": int(m_add.group(1)),
                "observation": m_add.group(2).strip(),
            }
        # Bare content → treat as add with auto-assigned index
        return {"action": "add", "observation_index": -1, "observation": stripped}

    if not isinstance(update, dict):
        return None

    action = str(update.get("action", "add")).lower()
    if action != "add":
        return None  # RRG v2 disallows update

    raw_index = update.get("observation_index", update.get("index", -1))
    try:
        observation_index = int(raw_index)
    except (TypeError, ValueError):
        observation_index = -1

    observation = update.get("observation", update.get("content", update.get("text", "")))
    if observation is None:
        observation = ""
    observation = str(observation).strip()
    if not observation:
        return None

    return {
        "action": "add",
        "observation_index": observation_index,
        "observation": observation,
    }


def _parse_json_output(text: str) -> RRGParseResult | None:
    object_span = _find_json_object_span(text)
    if object_span is None:
        return None

    object_start, object_end = object_span
    try:
        payload = json.loads(text[object_start:object_end])
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    result = RRGParseResult()
    result.reasoning_span = _find_json_field_value_span(text, "action_reasoning", object_start, object_end)
    result.writing_span = _find_json_field_value_span(text, "observation_update", object_start, object_end)

    reasoning_value = payload.get("action_reasoning", payload.get("reasoning", ""))
    writing_value = payload.get("observation_update", payload.get("fact_writing", []))

    result.reasoning_text = "" if reasoning_value is None else str(reasoning_value).strip()
    result.writing_text = json.dumps(writing_value, ensure_ascii=False)

    if isinstance(writing_value, list):
        for item in writing_value:
            normalized = _normalize_update(item)
            if normalized is not None:
                result.writing_updates.append(normalized)
    else:
        normalized = _normalize_update(writing_value)
        if normalized is not None:
            result.writing_updates.append(normalized)

    return result


def parse_rrg_output(text: str) -> RRGParseResult:
    """Parse a model generation into reasoning and writing spans.

    Supports the current JSON schema and the legacy section format.
    Tolerates missing sections — any absent section gets empty defaults.
    Citations are no longer parsed. UPDATE writing ops are rejected.
    """
    json_result = _parse_json_output(text)
    if json_result is not None:
        return json_result

    result = RRGParseResult()

    # Find all section header positions
    headers: list[tuple[str, int, int]] = []
    for m in _SECTION_RE.finditer(text):
        headers.append((m.group().lower(), m.start(), m.end()))

    if not headers:
        # No structured output — treat entire text as reasoning
        result.reasoning_text = text.strip()
        result.reasoning_span = (0, len(text))
        return result

    # Build section slices
    sections: dict[str, tuple[int, int, str]] = {}  # key -> (body_start, body_end, body_text)
    for i, (name, _hdr_start, hdr_end) in enumerate(headers):
        body_start = hdr_end
        body_end = headers[i + 1][1] if i + 1 < len(headers) else len(text)
        body = text[body_start:body_end].strip()

        if "reasoning" in name:
            sections["reasoning"] = (_hdr_start, body_end, body)
        elif "writing" in name:
            sections["writing"] = (_hdr_start, body_end, body)

    # Reasoning
    if "reasoning" in sections:
        span_start, span_end, body = sections["reasoning"]
        result.reasoning_text = body
        result.reasoning_span = (span_start, span_end)

    # Writing (add-only)
    if "writing" in sections:
        span_start, span_end, body = sections["writing"]
        result.writing_text = body
        result.writing_span = (span_start, span_end)
        for line in body.splitlines():
            line = line.strip().lstrip("-•* ")
            m_add = _ADD_RE.match(line)
            if m_add:
                result.writing_updates.append({
                    "action": "add",
                    "observation_index": int(m_add.group(1)),
                    "observation": m_add.group(2).strip(),
                })
            elif line:
                # Bare line — treat as add with auto-assigned index
                result.writing_updates.append({
                    "action": "add",
                    "observation_index": -1,
                    "observation": line,
                })

    return result


# ---------------------------------------------------------------------------
# Token-level span masks
# ---------------------------------------------------------------------------

def _char_to_token_mask(
    text: str,
    char_start: int,
    char_end: int,
    token_offsets: list[tuple[int, int]],
) -> list[bool]:
    """Return a boolean list (one per token) marking tokens overlapping [char_start, char_end)."""
    mask = []
    for tok_start, tok_end in token_offsets:
        overlap = tok_start < char_end and tok_end > char_start
        mask.append(overlap)
    return mask


def build_span_masks(
    response_text: str,
    response_ids: torch.Tensor,
    tokenizer,
) -> dict[str, torch.Tensor]:
    """Build boolean masks for reasoning and fact (writing) token spans.

    Args:
        response_text: Decoded response string.
        response_ids: Token IDs of the response, shape ``(response_length,)``.
        tokenizer: HuggingFace tokenizer with ``encode`` method.

    Returns:
        Dict with keys ``reasoning_mask`` and ``fact_mask``, each a bool
        tensor of shape ``(response_length,)``.
    """
    parse_result = parse_rrg_output(response_text)
    response_length = response_ids.shape[0]

    # Compute character offsets per token by re-encoding the response text
    encoding = tokenizer(response_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding.get("offset_mapping", None)

    if offsets is None or len(offsets) == 0:
        # Fallback: all tokens treated as reasoning, no fact tokens
        return {
            "reasoning_mask": torch.ones(response_length, dtype=torch.bool),
            "fact_mask": torch.zeros(response_length, dtype=torch.bool),
        }

    # Align offsets to response_length (may differ due to special tokens or truncation)
    if len(offsets) > response_length:
        offsets = offsets[:response_length]
    elif len(offsets) < response_length:
        offsets = offsets + [(0, 0)] * (response_length - len(offsets))

    reasoning_mask = _char_to_token_mask(response_text, *parse_result.reasoning_span, offsets)
    fact_mask = _char_to_token_mask(response_text, *parse_result.writing_span, offsets)

    def to_tensor(mask: list[bool]) -> torch.Tensor:
        t = torch.tensor(mask, dtype=torch.bool)
        if len(t) > response_length:
            t = t[:response_length]
        elif len(t) < response_length:
            t = torch.cat([t, torch.zeros(response_length - len(t), dtype=torch.bool)])
        return t

    return {
        "reasoning_mask": to_tensor(reasoning_mask),
        "fact_mask": to_tensor(fact_mask),
    }
