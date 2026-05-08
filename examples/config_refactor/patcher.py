"""Deterministic patcher — schema-violation-specific fixes.

For demo purposes we hard-code the recovery strategies for the three
canonical defect classes the schema critic emits in this example
(``required`` / ``type`` / ``enum``). In production this is exactly
where the LLM patcher (see ``examples/04_with_llm_patcher.py``) goes.
"""
from __future__ import annotations

import re
from typing import Any


_VERSION_RE = re.compile(r"^\d+(\.\d+)*$")


def _set_at(state: dict, pointer: str, value) -> bool:
    """Walk ``pointer`` (RFC 6901) and set the leaf to ``value``."""
    if not pointer.startswith("/"):
        return False
    parts = [p.replace("~1", "/").replace("~0", "~") for p in pointer.lstrip("/").split("/")]
    node = state
    for p in parts[:-1]:
        if isinstance(node, list):
            node = node[int(p)]
        else:
            node = node.get(p)
        if node is None:
            return False
    last = parts[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value
    return True


def fix_one(state: dict, pointer: str, description: str) -> tuple[bool, str]:
    """Return ``(fixed, reason)``. Heuristics:

      - If ``pointer`` ends in ``port`` and current value is a digit
        string, parse it to int.
      - If ``pointer`` ends in ``env`` and the value is one of the
        common synonyms ('production', 'staging-x'), normalize.
      - If ``pointer`` ends in ``version`` and the value matches a
        loose semver, pad to X.Y.Z with zeros.
    """
    if pointer.endswith("/port"):
        # Get current value via the description (or walk).
        # Simple walk:
        parts = pointer.lstrip("/").split("/")
        node = state
        for p in parts:
            node_parent = node
            node = node[p] if not isinstance(node, list) else node[int(p)]
        if isinstance(node, str) and node.isdigit():
            ok = _set_at(state, pointer, int(node))
            return ok, f"parsed port string {node!r} → int"

    if pointer.endswith("/env"):
        parts = pointer.lstrip("/").split("/")
        node = state
        for p in parts:
            node = node[p] if not isinstance(node, list) else node[int(p)]
        if node == "production":
            return _set_at(state, pointer, "prod"), "normalized 'production' → 'prod'"
        return False, "unrecognized env value"

    if pointer.endswith("/version") or pointer == "/version":
        parts = pointer.lstrip("/").split("/") if pointer != "/version" else ["version"]
        node = state
        for p in parts:
            node = node[p] if not isinstance(node, list) else node[int(p)]
        if isinstance(node, str) and _VERSION_RE.match(node):
            padded = node + ".0" * (3 - (node.count(".") + 1))
            return _set_at(state, pointer, padded), f"padded version {node!r} → {padded!r}"
        return False, "version not coercible"

    return False, "no rule for this pointer"


def make_patcher():
    counter = [0]

    def _apply(state, flagged_paths, feedback_by_path, model=None):
        traces = []
        for path in flagged_paths:
            counter[0] += 1
            ok, reason = fix_one(state, path, feedback_by_path.get(path, ""))
            traces.append(type("T", (), {
                "id": f"cfg-{counter[0]}",
                "requirement_id": path,
                "addressed": ok,
                "reason": reason,
            })())
        return traces

    return _apply
