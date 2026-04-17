from __future__ import annotations


def rrg_projection(text_actions: list[str]) -> tuple[list[str], list[bool]]:
    """Identity projection — model output is reasoning, not an env action.

    Returns ``(actions, valids)`` where actions are the original texts and all
    are marked valid.
    """
    return text_actions, [True] * len(text_actions)
