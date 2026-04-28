from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import BaseMemory


@dataclass
class FactVersion:
    """One version of a fact in the bank."""
    step_written: int
    content: str
    action: str  # "add" or "update"


@dataclass
class FactSlot:
    """A single observation slot with version history."""
    versions: list[FactVersion] = field(default_factory=list)

    @property
    def current(self) -> str:
        return self.versions[-1].content if self.versions else ""

    @property
    def step_written(self) -> int:
        return self.versions[0].step_written if self.versions else -1

    @property
    def latest_step_written(self) -> int:
        return self.versions[-1].step_written if self.versions else -1


class FactBankMemory(BaseMemory):
    """Versioned per-environment observation bank for RRG.

    Each environment maintains a list of observation slots. Each slot can be
    written (add) or updated with a new version. The bank also records
    reasoning history per environment for context.
    """

    def __init__(self):
        self._banks: list[list[FactSlot]] | None = None
        self._reasoning_history: list[list[str]] | None = None
        self.batch_size: int = 0

    # -- BaseMemory interface --

    def __len__(self) -> int:
        return len(self._banks) if self._banks else 0

    def __getitem__(self, idx: int) -> list[FactSlot]:
        assert self._banks is not None
        return self._banks[idx]

    def reset(self, batch_size: int) -> None:
        if self._banks is not None:
            self._banks.clear()
        self._banks = [[] for _ in range(batch_size)]
        self._reasoning_history = [[] for _ in range(batch_size)]
        self.batch_size = batch_size

    def store(self, record: dict[str, list[Any]]) -> None:
        """Store parsed model output for one step across all envs.

        Expected record keys:
            updates: List[List[dict]]  — per-env list of fact update dicts
            reasoning: List[str]       — per-env reasoning text
            step: List[int]            — per-env current step index
        """
        updates_batch = record.get("updates", [[] for _ in range(self.batch_size)])
        reasoning_batch = record.get("reasoning", [""] * self.batch_size)
        step_batch = record.get("step", [0] * self.batch_size)

        for env_idx in range(self.batch_size):
            self.apply_updates(env_idx, updates_batch[env_idx], step_batch[env_idx])
            if self._reasoning_history is not None:
                self._reasoning_history[env_idx].append(reasoning_batch[env_idx])

    def fetch(self, history_length: int = -1, **kwargs) -> tuple[list[str], list[int]]:
        """Return formatted observation bank text for each environment."""
        bank_texts = []
        valid_lengths = []
        for env_idx in range(self.batch_size):
            text = self.get_bank_formatted(env_idx)
            bank_texts.append(text)
            valid_lengths.append(len(self._banks[env_idx]) if self._banks else 0)
        return bank_texts, valid_lengths

    # -- RRG-specific methods --

    def apply_updates(
        self, env_idx: int, updates: list[dict[str, Any]], step: int
    ) -> None:
        """Apply a list of add operations to one environment's bank.

        RRG v2 is add-only: UPDATE operations are silently dropped to guard
        against upstream regressions; the parser is the primary gate.
        """
        assert self._banks is not None
        bank = self._banks[env_idx]
        for upd in updates:
            action = upd.get("action", "add")
            if action != "add":
                continue  # RRG v2: add-only
            obs_index = upd.get("observation_index", -1)
            content = upd.get("observation", "")
            if not content:
                continue

            version = FactVersion(step_written=step, content=content, action="add")

            # Auto-assign index if missing or out of range; always append to the
            # end so the bank grows monotonically.
            if obs_index < 0 or obs_index > len(bank):
                obs_index = len(bank)
            if obs_index == len(bank):
                bank.append(FactSlot(versions=[version]))
            else:
                # Add-to-existing-index: ignore to preserve add-only semantics.
                continue

    def get_bank(self, env_idx: int) -> list[str]:
        """Return current fact contents as a plain list of strings."""
        assert self._banks is not None
        return [slot.current for slot in self._banks[env_idx]]

    def get_bank_formatted(self, env_idx: int) -> str:
        """Return the observation bank as numbered text for prompt injection."""
        assert self._banks is not None
        bank = self._banks[env_idx]
        if not bank:
            return "empty"
        lines = []
        for i, slot in enumerate(bank):
            lines.append(f"[step {slot.step_written + 1}] {i}: {slot.current}")
        return "\n".join(lines)

    def get_reasoning_history(self, env_idx: int, max_steps: int = -1) -> str:
        """Return prior reasoning texts as formatted history."""
        assert self._reasoning_history is not None
        history = self._reasoning_history[env_idx]
        if max_steps > 0:
            history = history[-max_steps:]
        if not history:
            return "(none)"
        offset = max(0, len(self._reasoning_history[env_idx]) - len(history))
        lines = []
        for i, text in enumerate(history):
            lines.append(f"Step {offset + i + 1}: {text}")
        return "\n".join(lines)

    def get_version_history(self, env_idx: int) -> list[list[dict[str, Any]]]:
        """Return version history per slot for reward computation."""
        assert self._banks is not None
        result = []
        for slot in self._banks[env_idx]:
            versions = []
            for v in slot.versions:
                versions.append({
                    "step_written": v.step_written,
                    "content": v.content,
                    "action": v.action,
                })
            result.append(versions)
        return result
