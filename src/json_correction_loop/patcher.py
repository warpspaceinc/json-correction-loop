"""Surgical patcher — single entry point for applying critic-derived patch
requirements to a graph dict via LLM tool calling.

Design contract:
- Critics determine **what** to fix (defects + intent + target pointer).
- This module determines **how** to apply fixes (LLM emits RFC 6902 ops via
  tool calls, we validate and apply against the graph dict).

Why a single funnel: v1's pre-existing per-level surgical/full revise paths
each re-emitted whole pydantic objects, drifting state and burning tokens.
By routing every revise through one tool-calling LLM session per requirement,
patches stay localized to the targeted JSON pointer.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from json_correction_loop._config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_PATCHER_MAX_STEPS,
    DEFAULT_PATCHER_READ_ONLY_BAIL,
)
from json_correction_loop._observability import log_verbose
from json_correction_loop.llm import TransientLLMError

logger = logging.getLogger(__name__)


# ── Public types ────────────────────────────────────────────────────────────


@dataclass
class PatchRequest:
    """One thing a critic wants fixed.

    `target_pointer` is the canonical location of the defect (RFC 6901). The
    patcher exposes it to the LLM and constrains patches to land at-or-under
    that pointer. `intent` is human-readable critic guidance — it goes into
    the LLM prompt verbatim. `context_pointers` are read-only areas the
    patcher pre-fetches so the LLM has decision context without burning a
    `query` round-trip.
    """
    requirement_id: str
    target_pointer: str
    intent: str
    context_pointers: list[str] = field(default_factory=list)
    constraints: dict[str, Any] | None = None


@dataclass
class ToolCallRecord:
    """One tool invocation captured during a SurgicalPatcher requirement loop.

    `result_summary` is a short string (≤ 400 chars) — the inspector
    renders this verbatim, so we trim aggressively to keep manifests small.
    """
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    result_summary: str = ""


@dataclass
class CriticErrorRecord:
    """The patcher's diagnosis that the requirement itself was wrong.
    Mirrors :class:`a domain CriticError` (this dataclass
    stays inside the patcher module so the agent doesn't have to
    import the pydantic model at run time)."""
    kind: str                                  # "scope_mismatch" | "unfulfillable_intent"
    summary: str
    attempted_out_of_scope_paths: list[str] = field(default_factory=list)


@dataclass
class SubAgentTrace:
    """One sub-agent invocation's verdict + evidence trail. Used by
    request_validator (pre-loop), path_finder (per-op), and
    patch_evaluator (post-op). Stored on PatchResult for the inspector
    so reviewers can see why the patcher accepted/rewrote/refused."""
    kind: str                              # "request_validator" | "path_finder" | "patch_evaluator"
    verdict: str                            # subagent-specific enum
    confidence: str                         # "high" | "medium" | "low"
    rationale: str = ""
    op_index: int | None = None             # for path_finder/patch_evaluator: which op this judged
    proposed_path: str | None = None        # path_finder only
    final_path: str | None = None           # path_finder only — corrected pointer (== proposed if confirmed)
    calls: list[dict[str, Any]] = field(default_factory=list)  # tool-call trail


@dataclass
class PatchResult:
    requirement_id: str
    addressed: bool
    target_pointer: str = ""
    intent: str = ""
    applied_ops: list[dict[str, Any]] = field(default_factory=list)
    reason: str | None = None        # set when addressed=False
    llm_calls: int = 0
    calls: list[ToolCallRecord] = field(default_factory=list)
    # Set when the patcher infers the critic itself made a mistake
    # (target_pointer too narrow, intent points at non-existent thing).
    # Independent of `addressed` — best-effort partial fixes can carry
    # this AND addressed=True.
    critic_error: CriticErrorRecord | None = None
    # Set when P3+ expanded the scope mid-loop. `target_pointer` above
    # already reflects the widened pointer that was actually used by
    # the patcher; this captures the original critic-supplied pointer
    # for review.
    auto_widened_from: str | None = None
    # Sub-agent invocations (request_validator, path_finder, patch_evaluator)
    # captured for audit. Empty when the corresponding sub-agent is
    # disabled or didn't run for this requirement.
    subagent_traces: list[SubAgentTrace] = field(default_factory=list)


# ── RFC 6901 / 6902 minimal helpers ─────────────────────────────────────────
#
# We deliberately avoid the `jsonpatch` package — these helpers cover the
# add/replace/remove ops we use, and keeping them inline means we control
# exactly what's accepted (e.g. we reject `move`/`copy` for now since no
# critic needs them).


def _split_pointer(pointer: str) -> list[str]:
    if pointer == "":
        return []
    if not pointer.startswith("/"):
        raise ValueError(f"pointer must start with '/' or be empty: {pointer!r}")
    parts = pointer.split("/")[1:]
    # RFC 6901 unescaping: ~1 → /, ~0 → ~
    return [p.replace("~1", "/").replace("~0", "~") for p in parts]


def _common_parent_pointer(paths: list[str]) -> str:
    """Deepest common-prefix JSON pointer across `paths`.

    Returns the empty string when there is no shared prefix beyond root —
    callers treat that as "no useful widening available" and skip the
    auto-widen. Used by the patcher's scope auto-widen (P3+) when the LLM
    repeatedly tries siblings of `target_pointer`: if the rejected siblings
    share a non-trivial parent with the original target, we expand scope
    once instead of bailing.
    """
    if not paths:
        return ""
    seg_lists = [_split_pointer(p) for p in paths if p]
    if not seg_lists:
        return ""
    common: list[str] = []
    for items in zip(*seg_lists):
        first = items[0]
        if all(x == first for x in items):
            common.append(first)
        else:
            break
    if not common:
        return ""
    # Re-escape segments so we round-trip through _split_pointer cleanly.
    escaped = [s.replace("~", "~0").replace("/", "~1") for s in common]
    return "/" + "/".join(escaped)


def _resolve(graph: Any, pointer: str) -> Any:
    """Return the value at `pointer`. Raises KeyError/IndexError on miss.

    Slash-in-key fallback: ``world.spaces`` uses slash-keyed paths
    (e.g. ``"펜션/2층_객실_A"``) where the slash is part of the key,
    NOT a JSON Pointer separator. RFC 6901 says authors must encode
    these as ``~1`` — but LLMs reliably emit the raw slash. When a
    token doesn't match an existing dict key, we greedily recombine
    later tokens with ``/`` until we find a match (or run out). Safe
    because verbatim matches are tried first.
    """
    tokens = _split_pointer(pointer)
    node = graph
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if isinstance(node, list):
            node = node[int(tok)]
            i += 1
            continue
        if not isinstance(node, dict):
            raise KeyError(f"cannot descend into {type(node).__name__} at {tok!r}")
        # Verbatim key match first.
        if tok in node:
            node = node[tok]
            i += 1
            continue
        # Slash-in-key fallback: greedily try joining with later tokens.
        matched = False
        for j in range(i + 1, len(tokens) + 1):
            candidate = "/".join(tokens[i:j])
            if candidate in node:
                node = node[candidate]
                i = j
                matched = True
                break
        if not matched:
            raise KeyError(tok)
    return node


def _apply_op(graph: dict, op: dict) -> None:
    """Apply one op IN PLACE. Supports RFC 6902 add/replace/remove plus a
    custom `merge` op that shallow-merges a partial dict into the dict at
    `path` (preserves keys not present in `value`).

    `merge` is the safer way to update a dict because it forbids the LLM's
    common failure mode of `replace`-ing a whole nested object and dropping
    required sub-fields it didn't bother to re-emit.

    Mutates `graph`. Raises on any error (caller catches and records as
    a failed requirement). Whole-document add/replace (path="") not
    supported — patcher contract requires at least one segment.
    """
    op_name = op.get("op")
    path = op.get("path", "")
    if op_name not in ("add", "replace", "remove", "merge", "rename_key"):
        # Common LLM confusion: emitting a top-level TOOL name (e.g.
        # ``list_insert_at``) as ``op`` inside ``patch.ops``. Reject with
        # a hint that points at the right shape — auto-translation now
        # happens earlier in ``_translate_tool_op_to_rfc``, so reaching
        # here means an unrecognised name.
        raise ValueError(
            f"unsupported op: {op_name!r}; expected one of add/replace/remove/merge/rename_key. "
            f"If you meant a narrow tool like list_insert_at/list_append/"
            f"list_replace_at/set_field/rename_key, call it directly as a separate tool — "
            f"don't put its name inside patch.ops[].op."
        )

    # rename_key is structurally distinct: it doesn't read/write a value,
    # only swaps the dict key the value lives under. ``path`` resolves to
    # the *container dict* (not the renamed entry). Insertion order is
    # preserved by rebuilding the dict in place — important for keys like
    # ``spaces`` where downstream rendering iterates in declaration order.
    if op_name == "rename_key":
        old_key = op.get("old_key")
        new_key = op.get("new_key")
        if old_key is None or new_key is None:
            raise ValueError(
                "rename_key requires 'old_key' and 'new_key' alongside 'path' "
                "(JSON Pointer to the containing dict)"
            )
        if not isinstance(old_key, str) or not isinstance(new_key, str):
            raise ValueError(
                f"rename_key 'old_key' and 'new_key' must be strings, got "
                f"{type(old_key).__name__}/{type(new_key).__name__}"
            )
        if old_key == new_key:
            return  # no-op, treat as success
        container = _resolve(graph, path) if path else graph
        if not isinstance(container, dict):
            raise ValueError(
                f"rename_key target {path!r} is not a dict ({type(container).__name__}); "
                f"only dict keys can be renamed"
            )
        if old_key not in container:
            raise KeyError(f"rename_key: {old_key!r} not in dict at {path!r}")
        if new_key in container:
            raise ValueError(
                f"rename_key: {new_key!r} already exists in dict at {path!r}; "
                f"refusing to overwrite the existing entry"
            )
        # Rebuild in place to preserve insertion order — Python dicts are
        # ordered since 3.7 and downstream renderers (e.g. world spaces
        # tree walk) rely on declaration order.
        new_items = [
            (new_key if k == old_key else k, v) for k, v in container.items()
        ]
        container.clear()
        container.update(new_items)
        return

    parts = _split_pointer(path)
    if not parts:
        raise ValueError("whole-document patches not allowed")

    parent = graph
    for tok in parts[:-1]:
        if isinstance(parent, list):
            parent = parent[int(tok)]
        elif isinstance(parent, dict):
            parent = parent[tok]
        else:
            raise KeyError(f"cannot descend into {type(parent).__name__} at {tok!r}")
    last = parts[-1]

    if op_name == "remove":
        if isinstance(parent, list):
            del parent[int(last)]
        else:
            del parent[last]
        return

    if "value" not in op:
        raise ValueError(f"{op_name} op missing 'value'")
    value = op["value"]

    if op_name == "merge":
        # Resolve the target dict (one more step than replace/add — we merge
        # INTO the dict at `path`, not the dict's parent).
        if isinstance(parent, list):
            target = parent[int(last)]
        else:
            if last not in parent:
                raise KeyError(f"merge target missing: {path}")
            target = parent[last]
        if not isinstance(target, dict):
            raise ValueError(f"merge target is not a dict: {path} ({type(target).__name__})")
        if not isinstance(value, dict):
            raise ValueError(f"merge value must be a dict, got {type(value).__name__}")
        # Reject merging list-typed values. merge is shallow, so writing
        # ``{"foo": [...]}`` wholesale-replaces ``foo``'s existing list with
        # the new one — and LLMs reliably drop required fields when
        # re-emitting list items inline. Force list edits through narrow
        # tools (list_append/list_replace_at/list_insert_at) or RFC 6902
        # add/replace ops on the indexed path, both of which require the LLM
        # to address one item at a time with its full required shape.
        bad = [k for k, v in value.items() if isinstance(v, list)]
        if bad:
            raise ValueError(
                f"merge cannot set list-typed values for keys {bad} at {path}; "
                f"use list_append / list_replace_at / list_insert_at, or use "
                f"add/replace with the full item shape (every required field)."
            )
        target.update(value)
        return

    if op_name == "replace":
        # Reject wholesale replace of a list-of-dicts. Each item in such a
        # list typically carries required fields (e.g. dramatic_breakdown
        # items require master_event_id), and LLMs reliably drop or rename
        # those fields when re-emitting items inline. Scalar lists (list of
        # strings/numbers/bools) stay permitted — they have no per-item
        # contract to drop. For list-of-dicts edits, use list_replace_at /
        # list_append / list_insert_at one item at a time, or replace the
        # specific indexed path with a fully-shaped item.
        if isinstance(value, list) and any(isinstance(v, dict) for v in value):
            raise ValueError(
                f"replace cannot wholesale-overwrite a list-of-dicts at {path}; "
                f"use list_replace_at / list_append / list_insert_at item by item, "
                f"or replace the specific indexed path with a full item shape."
            )
        if isinstance(parent, list):
            parent[int(last)] = value
        else:
            if last not in parent:
                raise KeyError(f"replace target missing: {path}")
            parent[last] = value
        return

    # add
    if isinstance(parent, list):
        if last == "-":
            parent.append(value)
        else:
            parent.insert(int(last), value)
    else:
        parent[last] = value


# ── Tool-name-as-op auto-translation ────────────────────────────────────────


_NARROW_TOOL_NAMES = frozenset({
    "list_insert_at", "list_append", "list_replace_at", "set_field",
})

# Tools that can't be auto-translated into a single RFC op (they need
# graph access — scanning a list to find the matching index by identity
# field, or no equivalent at all). Recognising them in the translator
# lets us reject with a helpful error pointing at the right shape,
# instead of "unsupported op: X" which the LLM may keep hammering.
_NON_TRANSLATABLE_TOOL_NAMES = frozenset({
    "list_replace_where", "list_remove_where", "list_set_where",
    "list_str_remove", "list_move_at",
    "patch", "query", "get_schema", "diff", "undo",
})


def _translate_tool_op_to_rfc(op: dict) -> tuple[dict, str | None, str | None]:
    """Detect ``patch.ops[]`` entries that mistakenly use a top-level TOOL
    name as ``op`` and translate them into the equivalent RFC 6902 op.

    Returns ``(translated_op, note, hard_error)``:
      - ``translated_op`` — the RFC-shaped op to apply (or the input op
        unchanged when no translation matched).
      - ``note`` — one-line explanation surfaced in the tool's
        result_summary so the LLM learns the right shape on its next
        turn (set when translation succeeded).
      - ``hard_error`` — error string when the op references a tool
        that *can't* be auto-translated (e.g. ``list_set_where``
        needs graph access; ``patch`` is the wrapper itself). The
        caller skips applying the op and surfaces this string back
        to the LLM verbatim — much more actionable than the generic
        ``unsupported op: X`` rejection from ``_apply_op``.

    Why we translate instead of reject: the prior failure trace
    (academy-bottom-mage tr-4ab88997) showed the LLM looping on the
    same ``unsupported op`` rejection until ``max_steps``. A near-miss
    deserves to land — combined with the note, the LLM both progresses
    AND learns the correct shape.

    Translation table:
      list_append      → add at <pointer>/-
      list_insert_at   → add at <pointer>/<index> (or /- if no index)
      list_replace_at  → replace at <pointer>/<index>
      set_field        → replace at <pointer> (or add if path missing —
                         deferred to caller, since it needs graph access)

    Args may name the value as ``item`` (narrow-tool convention) or
    ``value`` (RFC convention) — accept either.
    """
    op_name = op.get("op")
    # Non-translatable tools used as op — reject with a clear hint at
    # the right shape. These can't become a single RFC op (graph access
    # required, or no equivalent at all), but recognising them stops
    # the LLM from looping on the generic "unsupported op" error.
    if op_name in _NON_TRANSLATABLE_TOOL_NAMES:
        if op_name == "patch":
            err = (
                "'patch' is the WRAPPER tool, not a value for op. "
                "Inside patch.ops use op=add/replace/remove/merge. "
                "If you wanted to clear a list, use op=replace with value=[]."
            )
        elif op_name in ("list_replace_where", "list_remove_where", "list_set_where"):
            err = (
                f"{op_name!r} is a separate TOOL (not an RFC op) — call it "
                f"directly as its own tool call, not inside patch.ops. "
                f"It needs graph access to resolve key→index at call time."
            )
        else:  # query / get_schema / diff / undo
            err = (
                f"{op_name!r} is a read/meta TOOL — call it directly, "
                f"not inside patch.ops."
            )
        return op, None, err

    if op_name not in _NARROW_TOOL_NAMES:
        return op, None, None
    pointer = (op.get("pointer") or op.get("path") or "").rstrip("/")
    item = op.get("item") if "item" in op else op.get("value")

    if op_name == "list_append":
        translated = {"op": "add", "path": f"{pointer}/-", "value": item}
    elif op_name == "list_insert_at":
        idx = op.get("index")
        if idx is None:
            translated = {"op": "add", "path": f"{pointer}/-", "value": item}
        else:
            translated = {"op": "add", "path": f"{pointer}/{int(idx)}", "value": item}
    elif op_name == "list_replace_at":
        idx = op.get("index")
        if idx is None:
            return op, None, None  # invalid; let _apply_op raise the unsupported-op error
        translated = {"op": "replace", "path": f"{pointer}/{int(idx)}", "value": item}
    elif op_name == "set_field":
        translated = {"op": "replace", "path": pointer, "value": item}
    else:
        return op, None, None

    note = (
        f"auto-translated patch.ops entry from tool name {op_name!r} → "
        f"{translated['op']} at {translated['path']}; next time, call "
        f"{op_name!r} as a separate tool (not inside patch.ops)."
    )
    return translated, note, None


# ── JSON Schema walker (for get_schema tool) ────────────────────────────────


def _hoist_defs(root_schema: dict) -> dict:
    """Return ``root_schema`` with every nested ``$defs`` lifted to the top.

    Pydantic's ``model_json_schema()`` puts ``$defs`` at the model's own
    root. When adapters wrap multiple model schemas under
    ``{"properties": {"plot": Plot.model_json_schema(), ...}}``, those
    ``$defs`` end up nested (e.g. ``properties.plot.$defs.CharacterArc``)
    and ``$ref: "#/$defs/CharacterArc"`` no longer resolves from the wrapper
    root. We walk the tree, pop every nested ``$defs``, and merge them
    into the wrapper's top-level ``$defs``.
    """
    out = dict(root_schema)
    merged: dict = dict(out.get("$defs") or {})

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            inner = node.pop("$defs", None)
            if isinstance(inner, dict):
                for k, v in inner.items():
                    merged.setdefault(k, v)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(out)
    if merged:
        out["$defs"] = merged
    return out


def _resolve_ref(schema: dict, ref: str) -> dict:
    """Resolve a JSON Schema `$ref` like `#/$defs/CharacterArc`."""
    if not ref.startswith("#/"):
        raise ValueError(f"unsupported $ref: {ref!r} (only local refs)")
    node: Any = schema
    for tok in ref[2:].split("/"):
        if isinstance(node, dict):
            node = node[tok]
        else:
            raise ValueError(f"can't follow $ref through {type(node).__name__} at {tok!r}")
    return node


def _resolve_schema_at(root_schema: dict, pointer: str) -> dict:
    """Walk a JSON Pointer through a JSON Schema tree and return the schema
    node at that location. Resolves `$ref` along the way. The pointer
    addresses VALUE space (e.g. `/sequences/2/title`); we navigate the
    matching SCHEMA space (`properties.sequences.items.properties.title`).

    Tolerant of failed descent: if a token doesn't match any property,
    we return the schema for the deepest level we did reach, with no
    error. This matters for slash-keyed dicts (e.g. ``world.spaces``,
    where keys like ``"펜션/2층_복도"`` look like multi-segment paths to
    the LLM and trigger `get_schema(/spaces/펜션/2층_복도)`). Returning
    the WorldSpace schema (the dict's value type) is the most useful
    answer in that case, instead of erroring.
    """
    node: dict = root_schema
    if "$ref" in node:
        node = _resolve_ref(root_schema, node["$ref"])
    for tok in _split_pointer(pointer):
        # Resolve any $ref at the current level before descending.
        if "$ref" in node:
            node = _resolve_ref(root_schema, node["$ref"])
        # Tolerate `oneOf`/`anyOf`: if multiple alternatives, prefer the one
        # that's an object/array. Best-effort — sufficient for v1 models.
        if "oneOf" in node or "anyOf" in node:
            alts = node.get("oneOf") or node.get("anyOf") or []
            picked = next(
                (a for a in alts
                 if isinstance(a, dict) and a.get("type") in ("object", "array")),
                alts[0] if alts else node,
            )
            node = picked

        node_type = node.get("type")
        if node_type == "array" or "items" in node:
            # Array item — index ignored, all items share the same schema.
            node = node.get("items") or {}
        elif node_type == "object" or "properties" in node:
            props = node.get("properties") or {}
            if tok in props:
                node = props[tok]
            elif "additionalProperties" in node and isinstance(node["additionalProperties"], dict):
                # dict[K, V] — every key shares the value schema.
                node = node["additionalProperties"]
            else:
                # Tolerant fallback: the LLM is asking deeper than the
                # schema can describe (often because the dict key has
                # internal slashes, e.g. ``"펜션/2층_복도"``). Return the
                # schema we have rather than raising.
                return node
        else:
            # Reached a leaf type (string/number/bool); stop descending.
            return node
        # Resolve $ref the descent-result might be itself.
        if isinstance(node, dict) and "$ref" in node:
            node = _resolve_ref(root_schema, node["$ref"])
    return node


def _trim_schema_for_llm(schema: dict, max_depth: int = 3) -> dict:
    """Shrink a schema dict for LLM display: drop $defs noise, cap nesting
    so a giant schema doesn't blow the context. Preserves the fields the
    LLM cares about: `type`, `required`, `properties` (1 level deep, with
    each property's `type`/`description` only), `items` (1 level)."""
    out: dict = {}
    if "type" in schema:
        out["type"] = schema["type"]
    if "description" in schema:
        out["description"] = schema["description"]
    if "required" in schema:
        out["required"] = list(schema["required"])
    if "enum" in schema:
        out["enum"] = list(schema["enum"])
    if max_depth <= 0:
        return out
    if "properties" in schema:
        out["properties"] = {}
        for k, v in (schema["properties"] or {}).items():
            if isinstance(v, dict):
                out["properties"][k] = _trim_schema_for_llm(v, max_depth - 1)
            else:
                out["properties"][k] = v
    if "items" in schema and isinstance(schema["items"], dict):
        out["items"] = _trim_schema_for_llm(schema["items"], max_depth - 1)
    if "additionalProperties" in schema and isinstance(schema["additionalProperties"], dict):
        out["additionalProperties"] = _trim_schema_for_llm(
            schema["additionalProperties"], max_depth - 1
        )
    return out


# ── Tool schemas ────────────────────────────────────────────────────────────


# Default exposure state for LLM-visible tools.
#   - Tools not listed here are exposed by default (writes, diff, undo, etc.)
#   - Read-language tools (jq/python/javascript) defaults reflect the
#     P58 benchmark outcome: jqBench predicts code > jq, and a 5-case
#     4-mode test (in this repo) showed javascript-only fastest at
#     97.8s avg vs 110s python-only vs 128s jq-only.
#   - Legacy read trio (query/find_paths/find_values) stays off — A/B
#     showed they fired in <3% of calls when jq was available.
_DEFAULT_TOOL_STATE: dict[str, bool] = {
    "query": False,
    "find_paths": False,
    "find_values": False,
    "find_index": True,
    "get_schema": True,
    "jq": False,
    "python": False,
    "javascript": True,
}


def _tool_enabled(name: str) -> bool:
    """Per-tool exposure check.

    Priority (high → low):
      1. ``JCL_TOOL_<UPPERCASE_NAME>`` env: ``1/0``, ``true/false``,
         ``on/off``, ``yes/no``.
      2. ``JCL_PATCHER_DISABLE_TOOLS`` env: comma-separated tool names
         to force-disable (legacy bulk knob, kept for back-compat).
      3. ``_DEFAULT_TOOL_STATE`` dict — falls back to ``True`` for tools
         not explicitly listed.
    """
    per_tool = os.environ.get(f"JCL_TOOL_{name.upper()}")
    if per_tool is not None:
        return per_tool.strip().lower() in ("1", "true", "yes", "on")
    bulk = os.environ.get("JCL_PATCHER_DISABLE_TOOLS")
    if bulk is not None:
        disabled = {n.strip() for n in bulk.split(",") if n.strip()}
        if name in disabled:
            return False
    return _DEFAULT_TOOL_STATE.get(name, True)


def _active_tool_schemas() -> list[dict]:
    """Return the LLM-visible subset of ``_TOOL_SCHEMAS`` per
    ``_tool_enabled``. Dispatch handlers stay registered regardless so
    sub-agents (which carry their own schemas) and cascade ops still
    work."""
    return [t for t in _TOOL_SCHEMAS if _tool_enabled(t["function"]["name"])]


_TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "query",
            "description": (
                "Read a slice of the graph at a JSON Pointer (RFC 6901). "
                "Use to inspect context before patching. Returns the value as JSON."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pointer": {
                        "type": "string",
                        "description": "JSON Pointer, e.g. /plot/character_arcs/김재혁",
                    },
                },
                "required": ["pointer"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_schema",
            "description": (
                "Return the JSON Schema (type, required, properties) of the value at "
                "a JSON Pointer. Call this BEFORE patching any object you don't fully "
                "control — it tells you which fields are required so your patch ops "
                "don't drop them. Returns {} when no schema is available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pointer": {
                        "type": "string",
                        "description": "JSON Pointer to the value whose schema you want.",
                    },
                },
                "required": ["pointer"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patch",
            "description": (
                "Apply ops to the graph. Each op is "
                "{'op':'add|replace|remove|merge','path':'/...','value':...}. "
                "PREFER `merge` when updating fields of an existing dict — it preserves "
                "untouched keys. Use `replace` only on scalar leaves or when intentionally "
                "swapping a whole sub-tree. For common single-purpose edits, prefer the "
                "narrower tools `set_field` / `list_append` / `list_replace_at` / "
                "`list_insert_at`. Submit empty ops to declare the requirement fully "
                "addressed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "description": "One-line natural-language summary of this patch.",
                    },
                    "ops": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "op": {
                                    "type": "string",
                                    "enum": ["add", "replace", "remove", "merge"],
                                },
                                "path": {"type": "string"},
                                "value": {},
                            },
                            "required": ["op", "path"],
                        },
                    },
                },
                "required": ["intent", "ops"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_field",
            "description": (
                "Set ONE field at `pointer` to `value`. Creates the field if "
                "missing, otherwise overwrites. Use this for single scalar-leaf "
                "edits (e.g. `set_field('/plot/theme', 'new theme')`). Safer than "
                "raw `patch` because the intent is unambiguous."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string"},
                    "pointer": {
                        "type": "string",
                        "description": "JSON Pointer to the field to set.",
                    },
                    "value": {
                        "description": "New value (any JSON-serializable type).",
                    },
                },
                "required": ["intent", "pointer", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_append",
            "description": (
                "Append `item` to the end of the list at `pointer`. Equivalent "
                "to `add` at `<pointer>/-`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string"},
                    "pointer": {"type": "string", "description": "JSON Pointer to the list."},
                    "item": {"description": "Item to append."},
                },
                "required": ["intent", "pointer", "item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_replace_at",
            "description": (
                "Replace the element at `index` in the list at `pointer` with "
                "`item`. The index must already exist."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string"},
                    "pointer": {"type": "string", "description": "JSON Pointer to the list."},
                    "index": {"type": "integer", "description": "0-based index to replace."},
                    "item": {"description": "Replacement value."},
                },
                "required": ["intent", "pointer", "index", "item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_insert_at",
            "description": (
                "Insert `item` at `index` in the list at `pointer`. Existing "
                "elements at and after `index` shift right by 1."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string"},
                    "pointer": {"type": "string", "description": "JSON Pointer to the list."},
                    "index": {"type": "integer", "description": "0-based insertion position."},
                    "item": {"description": "Item to insert."},
                },
                "required": ["intent", "pointer", "index", "item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_move_at",
            "description": (
                "Move the element at `from_index` to `to_index` in the list "
                "at `pointer`. Use when the list's order matters (scene "
                "playback order, sequence tension curve) and the critic asks "
                "to reorder rather than replace content. ``to_index`` is the "
                "FINAL position the moved item will occupy in the result "
                "list; e.g. moving index 0 to index 3 in [a,b,c,d,e] yields "
                "[b,c,d,a,e]. Equal indices are a no-op."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string"},
                    "pointer": {"type": "string", "description": "JSON Pointer to the list."},
                    "from_index": {"type": "integer", "description": "Current 0-based index of the item to move."},
                    "to_index": {"type": "integer", "description": "Final 0-based index where the moved item lands in the result list."},
                },
                "required": ["intent", "pointer", "from_index", "to_index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_replace_where",
            "description": (
                "Replace a list item identified by an identity field (label / id / "
                "name). Safer than `list_replace_at` when the list has stable "
                "identity keys, because it avoids the index-shift trap after "
                "siblings get added or removed. By default errors on zero or "
                "multiple matches; pass ``match_policy`` to relax this."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string"},
                    "pointer": {"type": "string", "description": "JSON Pointer to the list."},
                    "key_field": {
                        "type": "string",
                        "description": "Identity field on each item (e.g. 'label', 'id', 'name').",
                    },
                    "key_value": {
                        "description": "Value of `key_field` to look up — exact equality match.",
                    },
                    "item": {"description": "Replacement value (full item shape)."},
                    "match_policy": {
                        "type": "string",
                        "enum": ["single", "first", "last", "all"],
                        "description": (
                            "single (default): error on 0 or >1 matches. "
                            "first / last: pick the matching item at that position. "
                            "all: replace EVERY matching item with the same value."
                        ),
                    },
                },
                "required": ["intent", "pointer", "key_field", "key_value", "item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_remove_where",
            "description": (
                "Remove a list item identified by an identity field. Same "
                "advantage as `list_replace_where`: identity-keyed instead of "
                "index-keyed, so subsequent ops are not corrupted by an index "
                "shift. By default errors on zero or multiple matches; pass "
                "``match_policy`` to relax this. ``match_policy='all'`` is "
                "the right tool for removing duplicate entries that share a "
                "label / id."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string"},
                    "pointer": {"type": "string", "description": "JSON Pointer to the list."},
                    "key_field": {"type": "string"},
                    "key_value": {},
                    "match_policy": {
                        "type": "string",
                        "enum": ["single", "first", "last", "all"],
                        "description": (
                            "single (default): error on 0 or >1 matches. "
                            "first / last: pick the matching item at that position. "
                            "all: remove EVERY matching item — use when "
                            "deduplicating a list with repeated identity keys."
                        ),
                    },
                },
                "required": ["intent", "pointer", "key_field", "key_value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_set_where",
            "description": (
                "Set ONE field of a list item identified by an identity field. "
                "Safer than `list_replace_where` when only a single sub-field "
                "needs updating — avoids re-emitting the whole item shape and "
                "the risk of dropping required sub-fields. By default errors "
                "on zero or multiple matches; pass ``match_policy`` to relax."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string"},
                    "pointer": {"type": "string", "description": "JSON Pointer to the list."},
                    "key_field": {"type": "string"},
                    "key_value": {},
                    "field": {
                        "type": "string",
                        "description": "Field on the matched item to set.",
                    },
                    "value": {"description": "New value for that field."},
                    "match_policy": {
                        "type": "string",
                        "enum": ["single", "first", "last", "all"],
                        "description": (
                            "single (default): error on 0 or >1 matches. "
                            "first / last: pick the matching item at that position. "
                            "all: set the field on EVERY matching item."
                        ),
                    },
                },
                "required": ["intent", "pointer", "key_field", "key_value", "field", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_str_remove",
            "description": (
                "Remove all occurrences of a specific string `value` from a "
                "list of strings at `pointer`. Use this when an item id was "
                "removed elsewhere and now appears as a stale reference inside "
                "another item's id-list (e.g. `prerequisite_event_ids`, "
                "`assigned_event_ids`). Without this tool the alternative is "
                "set_field with the whole list re-emitted, which is brittle "
                "(LLM may drop unrelated entries). No-op when value isn't "
                "present (returns ok=true with removed=0)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string"},
                    "pointer": {
                        "type": "string",
                        "description": "JSON Pointer to the list[str].",
                    },
                    "value": {
                        "type": "string",
                        "description": "Exact string to remove from the list.",
                    },
                },
                "required": ["intent", "pointer", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_paths",
            "description": (
                "Search the graph's KEY space for `keyword` (substring, "
                "case-sensitive). Returns up to `max_results` JSON Pointers "
                "where any path segment (dict key) contains `keyword`. Use "
                "this when you don't know the exact JSON Pointer for a thing "
                "you can name — especially for slash-keyed dicts like "
                "world.spaces (e.g. `find_paths(\"로비\")` → "
                "`['/spaces/펜션/로비']`). For finding by VALUE (e.g. \"which "
                "extension mentions '강현우'?\"), use `find_values` instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Substring to look for in dict keys.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Cap on results returned. Defaults to 30.",
                        "minimum": 1,
                    },
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_values",
            "description": (
                "Search the graph's VALUE space for `keyword` (substring, "
                "case-sensitive, in string-typed leaf values only). Returns "
                "up to `max_results` JSON Pointers whose leaf value contains "
                "`keyword`. Use this to locate where a specific name / id / "
                "phrase appears in the data — e.g. `find_values(\"강현우\")` "
                "returns every path whose string value mentions that name. "
                "For finding by KEY (e.g. dict keys / labels), use "
                "`find_paths` instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Substring to look for in leaf string values.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Cap on results returned. Defaults to 30.",
                        "minimum": 1,
                    },
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_index",
            "description": (
                "Look up the integer index of a list-of-dicts item by an "
                "identity field, in one step. Use this immediately before "
                "an index-keyed op (`list_insert_at`, `list_replace_at`, "
                "`list_remove_at`) when you already know the item id but "
                "not the position. Returns ``{\"index\": N}`` for a single "
                "match; ``{\"indices\": [N, M, ...]}`` for multi; "
                "``{\"error\": ..., \"available_values\": [...]}`` when no "
                "item matches (so you can spot a typo). Identity-keyed "
                "edits (where the index doesn't matter) should use "
                "`list_replace_where` / `list_set_where` / "
                "`list_remove_where` directly — those resolve the index "
                "internally."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pointer": {
                        "type": "string",
                        "description": "JSON Pointer to the list of dicts.",
                    },
                    "key_field": {
                        "type": "string",
                        "description": "Identity field on each item (e.g. 'id', 'label', 'name').",
                    },
                    "key_value": {
                        "description": "Value of `key_field` to look up — exact equality match.",
                    },
                },
                "required": ["pointer", "key_field", "key_value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "jq",
            "description": (
                "Run a `jq` expression against the graph (or a subtree) "
                "to extract / project / filter in one step. Replaces "
                "long chains of `query` + `find_values` + per-item `query` "
                "when you need bulk info from many items. Returns the jq "
                "output (multiple values are wrapped in a JSON array). "
                "Common patterns:\n"
                "  • `.key_events | map({id, summary: .summary[:60]})` — "
                "id+brief summary of every item\n"
                "  • `.key_events | map(select(.act_number == 2))` — "
                "filter by field\n"
                "  • `.key_events | to_entries | map({i: .key, id: .value.id})` — "
                "index→id map\n"
                "  • `[paths(strings) as $p | select(getpath($p) | "
                "tostring | startswith(\"kev-legacy\")) | $p]` — every "
                "path to a stale id (returns list of pointer-segment lists)\n"
                "  • `.key_events[] | select(.id == \"kev-XYZ\")` — find by id\n"
                "Use `pointer` to scope the input (default: whole graph). "
                "Read-only — does not modify state. Errors (syntax / "
                "runtime) come back with the offending expression so you "
                "can correct on the next call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "jq expression to evaluate.",
                    },
                    "pointer": {
                        "type": "string",
                        "description": (
                            "Optional JSON Pointer to scope the input "
                            "(empty = whole graph). The expression sees "
                            "the value at this pointer as ``.``"
                        ),
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python",
            "description": (
                "Execute a Python expression / snippet against the graph "
                "(or a subtree). The graph is bound to ``data`` and "
                "``json`` is available; the value of the last expression "
                "(for one-liners) or a ``result`` variable (for "
                "multi-statement code) is returned, JSON-serialized.\n"
                "Per `jqBench` (Parnin et al. ICLR 2026), code-language "
                "tools beat jq on harder JSON read/edit tasks because "
                "the LLM can express conditionals, loops, and comprehensions "
                "natively. Prefer python for: nested filters, paired/joined "
                "lookups, indexed mutations, anything that takes >1 jq "
                "pipeline. Use jq for simple projections / shape inspection.\n"
                "Examples:\n"
                "  • ``[(i, e['id']) for i, e in enumerate(data) if 'kev-legacy' in e.get('prerequisite_event_ids', [''])[0]]``\n"
                "  • ``{e['id']: [m['entry_basis'] for m in e['character_motivations']] for e in data}``\n"
                "  • Multi-statement: ``out=[]\\nfor i,e in enumerate(data):\\n    if e['act_number']==2: out.append(e['id'])\\nresult=out``\n"
                "Read-only — modifications to ``data`` do NOT persist; "
                "use ``patch`` / ``set_field`` / ``list_*`` ops to commit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python expression or multi-statement code (with ``result=...`` at the end).",
                    },
                    "pointer": {
                        "type": "string",
                        "description": (
                            "Optional JSON Pointer to scope ``data`` "
                            "(empty = whole graph)."
                        ),
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "javascript",
            "description": (
                "Execute a JavaScript expression / snippet against the "
                "graph (or a subtree) via embedded V8. The graph is "
                "bound to the global ``data`` and the expression's value "
                "(or a ``result`` variable for multi-statement code) is "
                "returned, JSON-serialized.\n"
                "Same use cases as ``python``, different syntax — pick "
                "whichever expresses the transform most naturally. "
                "Examples:\n"
                "  • ``data.map(e => ({id: e.id, summary: e.summary.slice(0, 60)}))``\n"
                "  • ``data.filter(e => e.act_number === 2).map(e => e.id)``\n"
                "  • Multi-stmt: ``const out = {}; data.forEach(e => { out[e.id] = e.character_motivations.map(m => m.entry_basis); }); result = out;``\n"
                "Read-only — modifications to ``data`` do NOT persist; "
                "use ``patch`` / ``set_field`` / ``list_*`` ops to commit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "JavaScript expression or multi-statement code (with ``result = ...`` at the end).",
                    },
                    "pointer": {
                        "type": "string",
                        "description": (
                            "Optional JSON Pointer to scope ``data`` "
                            "(empty = whole graph)."
                        ),
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "diff",
            "description": (
                "Show what you have changed in this requirement loop so far. "
                "Returns the before/after value at every path you have patched "
                "(deduplicated, in op order). Use this to audit your work — "
                "especially before declaring done — when you are unsure what "
                "actually landed. If no ops have been applied yet, returns "
                "{\"changed\": false, \"note\": \"no ops applied yet\"}."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "undo",
            "description": (
                "Roll back the most recent `count` ops you applied in this "
                "requirement loop (default 1, or `all=true` to undo every op). "
                "Use when a previous op was wrong — undo, then re-emit the "
                "corrected op. Has no effect when nothing has been applied yet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {
                        "type": "integer",
                        "description": "How many trailing ops to undo. Defaults to 1.",
                        "minimum": 1,
                    },
                    "all": {
                        "type": "boolean",
                        "description": "If true, undo every op applied in this loop.",
                    },
                },
            },
        },
    },
]


# ── LLM call retry guard ────────────────────────────────────────────────────


def _chat_with_retry(client, *, max_attempts: int = 3, **kwargs):
    """Retry guard around ``client.chat_complete()``.

    Phase 2: ``client`` is now an ``LLMClient`` Protocol (``chat_complete``
    returning ``ChatResponse``). Provider knobs (``reasoning_effort``)
    that used to live in ``extra_body`` ride in ``extra=`` instead.
    """
    effort = os.environ.get("JCL_REASONING_EFFORT", "none").strip() or "none"
    kwargs.setdefault("extra", {})
    # Some backends (FriendliAI serverless) reject ``"none"`` enum
    # with HTTP 422. Omit the key entirely when effort is none.
    if effort and effort != "none":
        kwargs["extra"].setdefault("reasoning_effort", effort)
    last: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return client.chat_complete(**kwargs)
        except TransientLLMError as exc:
            last = exc
            if attempt + 1 == max_attempts:
                raise
            wait = 5 * (attempt + 1)
            logger.warning(
                "patcher transient error (attempt %d/%d), sleeping %ds: %s",
                attempt + 1, max_attempts, wait, exc,
            )
            time.sleep(wait)
    raise last  # type: ignore[misc]


# ── Patcher ─────────────────────────────────────────────────────────────────


_SYSTEM_PROMPT = """\
You are a surgical graph patcher. A critic has identified a defect in the graph
and described what needs to change. Your job: emit patch ops via the `patch`
tool that resolve the defect — nothing more, nothing less.

# Surgical principle (most important)
Make the SMALLEST POSSIBLE edit that resolves the critic's intent. Surgical means:
- Touch only the field(s) the defect is about. Sibling fields stay untouched.
- Reuse existing values verbatim — never re-emit a field with a paraphrased
  copy of its current value. If you find yourself writing out a field that the
  critic didn't flag, you are doing too much.
- Number of changes = number of defects the critic raised. One defect → one op
  (or one `merge` with one or two keys). Two defects → two ops. Never bundle
  unrelated improvements.
- When in doubt between a wide change and a narrow one, pick narrow. The
  critic loop will run again if more is needed.

# Tools
- `query(pointer)` — read a slice of the graph by JSON Pointer. Large
  values are auto-summarised structurally (key list + types) — drill in
  with a more specific pointer to see actual content. This keeps tool
  results small.
- `find_paths(keyword)` — search dict KEYS for `keyword`. Returns JSON
  Pointers whose any segment contains it. Use when you don't know the
  full path (especially for slash-keyed dicts like `world.spaces`).
- `find_values(keyword)` — search leaf STRING VALUES for `keyword`.
  Returns paths whose value mentions it (with a 120-char preview).
  Use to locate where a name / id / phrase appears in the data.
- `get_schema(pointer)` — return the JSON Schema (type, required, properties)
  of the value at `pointer`. **Call this BEFORE you patch any object you
  didn't author yourself.** It tells you which sub-fields are required so
  your patch doesn't drop them.
- `set_field(intent, pointer, value)` — set ONE scalar/leaf field.
- `list_append(intent, pointer, item)` — append to list at `pointer`.
- `list_replace_at(intent, pointer, index, item)` — replace one list element by index.
- `list_insert_at(intent, pointer, index, item)` — insert one list element by index.
- `list_replace_where(intent, pointer, key_field, key_value, item)` — replace
  the list item whose `key_field` equals `key_value`. **Prefer this over
  `list_replace_at` whenever items have an identity field (`label` / `id` /
  `name`).** Index-based ops are unsafe across multi-step edits because
  removing or inserting a sibling silently shifts every later index — this
  tool finds the right item at call time, regardless of position.
- `list_remove_where(intent, pointer, key_field, key_value)` — remove the
  matching item. Same advantage as `list_replace_where`.
- `list_set_where(intent, pointer, key_field, key_value, field, value)` —
  set ONE field on the matching item. Use when only a single sub-field of an
  identity-keyed item changes — avoids re-emitting the whole item shape.
- `patch(intent, ops)` — multi-op general path; use when the narrow tools
  above don't fit (e.g. multiple coordinated edits in one shot, `merge`,
  `remove`). Use `ops: []` to declare done.
- `diff()` — audit what you've changed so far in this loop (before/after
  per path). Useful before declaring done if you're unsure what landed,
  or after a `patch` whose result you want to verify. Returns
  `{changed: false}` when nothing has landed yet.
- `undo(count=N | all=true)` — roll back the most recent N ops you
  applied (default 1). Use when you realize a previous op was wrong:
  undo, then re-emit the corrected op. Has no effect when nothing has
  been applied yet.

# Tool selection
- One scalar field changing → `set_field`
- One list grows by one → `list_append`
- One list element gets a new value, list has identity keys (label/id/name) →
  `list_replace_where` (or `list_set_where` for one sub-field of that item)
- One list element gets a new value, list is positional only → `list_replace_at`
- One list element should be removed, list has identity keys → `list_remove_where`
- Several fields of one dict change at once → `patch` with a `merge` op
- Anything else (or a mix) → `patch` with an explicit ops list

# Index-shift hazard (important)
After a `remove` or `add` (insert), every later element's index changes.
NEVER chain index-based ops on the same list across multiple turns —
re-`query` the list between turns, or use the `where` family (which finds
items by identity at call time and is immune to index drift).

# Op semantics
- `replace` — overwrite the value at `path` wholesale. Safe for scalar leaves
  (strings/numbers/bools) and for scalar lists (list of strings/numbers).
  DANGEROUS on objects and on lists of objects: if you forget a required
  sub-field, or rename one (e.g. ``master_event_id`` → ``event_id``), the
  patcher REJECTS the op. For lists of objects, edit one item at a time
  with `list_replace_at` / `list_append` / `list_insert_at`, which require
  you to emit the item's full required shape per call.
- `merge` — shallow-merge a partial dict INTO the dict at `path`. Existing
  keys not present in your value stay. **Strongly prefer `merge` over
  `replace` when updating fields of an existing dict.**
  - **`merge` MUST NOT carry list-typed values.** A merge value like
    `{"items": [...]}` would wholesale-replace the existing list — and you
    will reliably drop required fields when you re-emit list items inline.
    For list edits use `list_append` / `list_replace_at` / `list_insert_at`
    (one item at a time, full required shape), or use `add`/`replace` ops
    on the indexed path with the complete item shape. The patcher will
    REJECT a merge that contains list values.
- `add` — insert a new key/element. Use for keys that don't yet exist.
- `remove` — delete the value at `path`. Use rarely.

# Required protocol
1. If `target_pointer` points to an object (not a scalar), call `get_schema`
   on it (or on the specific sub-field you intend to change) to learn the
   required-fields contract.
2. If you only need to change one or two sub-fields of an object, use
   `merge` with just those sub-fields — DO NOT replace the whole object.
3. After your patch lands, call `diff()` to AUDIT what actually changed.
   Confirm the before/after pair shows the intent was resolved (right field
   touched, value sensible, no accidental no-op or wrong target). If the
   diff reveals a miss, fix it with another patch — don't paper over it.
4. Once the diff confirms the intent is addressed, call `patch(ops=[])` to
   converge. The runtime REFUSES `patch(ops=[])` until at least one diff()
   call follows the latest successful patch — declaring done without
   auditing is a common failure mode and is now blocked.

# Discipline
- Patch ONLY at-or-under the requirement's `target_pointer`. Do not touch
  unrelated branches. If the fix needs a sibling change, the critic should have
  emitted a separate requirement.
- Stay within scope. The critic's intent is the contract.
- If a `patch` call returns rejection errors (path not found, validation
  failure, etc.), inspect with `query`/`get_schema` and retry — don't repeat
  the same op blindly.
"""


_USER_TMPL = """\
# Patch requirement
- requirement_id: {rid}
- target_pointer: {target}
- intent: {intent}
{subagent_hints_block}{template_seed_block}
# Pre-fetched context (read-only)
{context_block}

# Target current value
{target_value}

Apply ops that resolve the intent at `target_pointer`. After each patch
lands, call `diff()` to audit the change before declaring done. End with
`patch(ops=[])` once the diff confirms the intent is resolved.
"""


def _typecheck_json_schema(node: dict, value: Any) -> str | None:
    """P20: cheap JSON-Schema ``type`` enforcement.

    Used by ``SurgicalPatcher._validate_op_value_against_schema`` to
    catch obvious mismatches the LLM emits (e.g. ``-1`` for a string
    field, ``"42"`` for an integer field). Returns an error message
    when the value's runtime type doesn't satisfy ``node["type"]``,
    else ``None``.

    Behaviour:
    - Accepts ``None`` only when the schema lists ``"null"`` (or has
      ``nullable: true``, or appears in an ``anyOf`` with a null branch).
    - Treats ``bool`` as distinct from ``integer``/``number`` (Python
      ``bool`` is an int but JSON Schema treats them separately).
    - Skips when ``type`` is missing or unrecognised — we'd rather
      under-enforce than reject valid ops.
    """
    raw_type = node.get("type")
    if raw_type is None:
        return None
    types: tuple[str, ...]
    if isinstance(raw_type, str):
        types = (raw_type,)
    elif isinstance(raw_type, list):
        types = tuple(t for t in raw_type if isinstance(t, str))
    else:
        return None

    def _matches(t: str, v: Any) -> bool:
        if t == "null":
            return v is None
        if t == "boolean":
            return isinstance(v, bool)
        if t == "integer":
            return isinstance(v, int) and not isinstance(v, bool)
        if t == "number":
            return isinstance(v, (int, float)) and not isinstance(v, bool)
        if t == "string":
            return isinstance(v, str)
        if t == "array":
            return isinstance(v, list)
        if t == "object":
            return isinstance(v, dict)
        return True  # unknown type label — don't reject

    if any(_matches(t, value) for t in types):
        return None
    expected = "|".join(types)
    actual = type(value).__name__
    return f"value type {actual!r} does not match schema type {expected!r}"


def _extract_subagent_hints(traces: list["SubAgentTrace"]) -> str:
    """P17: distill validator/finder investigation into a compact prompt
    block for the main patcher LLM.

    Production traces showed the validator reliably finds the precise
    sub-path during its investigation (e.g. it queries
    ``/extensions/1/items/2`` to verify a critic claim, confirming
    that's where the relevant item lives), but the main patcher LLM
    only saw the coarse ``target_pointer`` (``/extensions/1``) and
    couldn't decide where to land. Without this propagation, the same
    intent that the validator just successfully located would die in
    the main loop with "abandoned: 3 consecutive steps no op landed".

    We surface the *paths the validator/finder touched* (with a
    success/failure indicator from the result summary), not free-form
    rationale — keeps the hint factual and bounded.
    """
    if not traces:
        return ""
    hints: list[str] = []
    for tr in traces:
        if tr.kind != "request_validator":
            continue
        for c in tr.calls:
            tool = c.get("tool") if isinstance(c, dict) else None
            args = (c.get("args") if isinstance(c, dict) else {}) or {}
            result = (c.get("result_summary") if isinstance(c, dict) else "") or ""
            is_error = result.startswith('{"error"')
            if tool == "query":
                ptr = args.get("pointer")
                if ptr and not is_error:
                    hints.append(f"  - validator confirmed `{ptr}` resolves (used during claim check)")
            elif tool == "find_paths":
                kw = args.get("keyword")
                if kw and '"matches": []' not in result:
                    summary = result.split("\n")[0][:120]
                    hints.append(f"  - validator searched key='{kw}' → {summary}")
            elif tool == "find_values":
                kw = args.get("keyword")
                if kw and '"matches": []' not in result:
                    summary = result.split("\n")[0][:120]
                    hints.append(f"  - validator searched value~'{kw}' → {summary}")
        if tr.rationale:
            hints.append(f"  - validator rationale: {tr.rationale[:200]}")
    # De-dup while preserving order; cap to keep the prompt compact.
    seen: set[str] = set()
    uniq: list[str] = []
    for h in hints:
        if h not in seen:
            seen.add(h)
            uniq.append(h)
        if len(uniq) >= 8:
            break
    if not uniq:
        return ""
    return (
        "\n# Sub-agent investigation (already performed — use these as path hints)\n"
        + "\n".join(uniq) + "\n"
    )


def _truncate_for_prompt(value: Any, max_chars: int = 6000) -> str:
    rendered = json.dumps(value, ensure_ascii=False, indent=2)
    if len(rendered) <= max_chars:
        return rendered
    return rendered[:max_chars] + f"\n… (truncated, full size {len(rendered)} chars)"


# ── Path / value search helpers (P5) ────────────────────────────────────────


def _encode_pointer_token(tok: str) -> str:
    """RFC 6901 token escaping: ~ → ~0, then / → ~1. Order matters."""
    return tok.replace("~", "~0").replace("/", "~1")


def _walk_paths(
    graph: Any,
    *,
    on_node: Callable[[list[str], Any], bool] | None = None,
    on_leaf: Callable[[list[str], Any], bool] | None = None,
) -> None:
    """DFS over ``graph``. For every dict key descent and list index
    descent, call ``on_node(parts, value)`` (return True to stop). For
    every scalar leaf, call ``on_leaf(parts, value)`` (return True to
    stop). ``parts`` is the JSON Pointer segment list (no leading ``/``).
    """
    stop = False

    def walk(node: Any, parts: list[str]) -> None:
        nonlocal stop
        if stop:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                child_parts = parts + [str(k)]
                if on_node and on_node(child_parts, v):
                    stop = True
                    return
                walk(v, child_parts)
                if stop:
                    return
        elif isinstance(node, list):
            for i, v in enumerate(node):
                child_parts = parts + [str(i)]
                if on_node and on_node(child_parts, v):
                    stop = True
                    return
                walk(v, child_parts)
                if stop:
                    return
        else:
            # Scalar leaf — string, number, bool, None.
            if on_leaf and on_leaf(parts, node):
                stop = True
                return

    walk(graph, [])


def _parts_to_pointer(parts: list[str]) -> str:
    """Build a JSON Pointer from segment list, encoding ``~``/``/`` per
    RFC 6901. Empty parts → ``""``."""
    return "".join("/" + _encode_pointer_token(p) for p in parts)


def _find_paths_by_key(graph: Any, keyword: str, max_results: int = 30) -> list[str]:
    """Return JSON Pointers whose any segment (dict key only — list
    indices are skipped because they aren't human-meaningful) contains
    ``keyword``. Capped at ``max_results``."""
    out: list[str] = []
    if not keyword:
        return out

    def visit(parts: list[str], _value: Any) -> bool:
        # Skip list-index segments; they aren't useful as a search target.
        last = parts[-1]
        if last.isdigit():
            return False
        if keyword in last:
            out.append(_parts_to_pointer(parts))
            if len(out) >= max_results:
                return True
        return False

    _walk_paths(graph, on_node=visit)
    return out


def _find_values(graph: Any, keyword: str, max_results: int = 30) -> list[dict]:
    """Return ``[{path, value_preview}]`` for every leaf-string value
    containing ``keyword``. Capped at ``max_results``. ``value_preview``
    is the matched string trimmed to 120 chars."""
    out: list[dict] = []
    if not keyword:
        return out

    def visit_leaf(parts: list[str], value: Any) -> bool:
        if not isinstance(value, str):
            return False
        if keyword in value:
            preview = value if len(value) <= 120 else value[:120] + "…"
            out.append({"path": _parts_to_pointer(parts), "value_preview": preview})
            if len(out) >= max_results:
                return True
        return False

    _walk_paths(graph, on_leaf=visit_leaf)
    return out


# ── Query auto-summary (P6) ─────────────────────────────────────────────────


def _js_to_python(value: Any) -> Any:
    """Recursively convert py-mini-racer's lazy JS proxy objects
    (``JSArrayImpl``, ``JSMappedObjectImpl``, ...) into plain Python
    lists / dicts so the result is JSON-serializable.

    Primitives (str/int/float/bool/None) come back as native Python
    already and pass through. ``JSUndefined`` becomes ``None``
    (Python has no separate ``undefined``), and other JS-only proxy
    types (functions, symbols, promises) collapse to their ``repr``
    so json.dumps doesn't crash — losing structure but keeping the
    result inspectable.
    """
    try:
        from py_mini_racer._objects import (
            JSArrayImpl,
            JSMappedObjectImpl,
            JSUndefinedType,
        )
    except ImportError:
        return value
    if isinstance(value, JSUndefinedType):
        return None
    if isinstance(value, JSArrayImpl):
        return [_js_to_python(x) for x in value]
    if isinstance(value, JSMappedObjectImpl):
        return {k: _js_to_python(value[k]) for k in value.keys()}
    # Anything else that came from V8 but isn't a primitive — e.g.
    # JSFunctionImpl, JSSymbolImpl, JSPromiseImpl. json.dumps can't
    # serialize them; render the repr so the LLM can see the result
    # is a non-data value (e.g. it expected a value but got a function).
    if type(value).__module__.startswith("py_mini_racer"):
        return f"<{type(value).__name__}: {value!r}>"
    return value


def _summarize_for_query(value: Any, max_chars: int = 1500) -> str:
    """Render ``value`` for a ``query`` tool result with token-budget
    awareness. Three modes:

    - small (rendered ≤ ``max_chars``): full JSON dump.
    - structured-and-large (dict/list with rendered > ``max_chars``):
      one-level structural summary — for dicts, ``{key: <type-or-len>}``;
      for lists, ``[len=N, item-shape-summary]``. Tells the LLM to
      drill in with a more specific pointer.
    - flat-and-large (string value > ``max_chars``): hard-truncate
      with a length marker — same behaviour as ``_truncate_for_prompt``.

    The summary is concise enough to keep tool-call results small while
    still giving the LLM the info it needs to choose the next pointer.
    """
    full = json.dumps(value, ensure_ascii=False, indent=2)
    if len(full) <= max_chars:
        return full
    if isinstance(value, dict):
        parts: list[str] = []
        for k, v in value.items():
            parts.append(f"  {json.dumps(k, ensure_ascii=False)}: {_describe_shape(v)}")
        body = ",\n".join(parts)
        return (
            "{\n"
            f"  // structural summary — full payload {len(full)} chars; "
            f"query a deeper pointer to inspect a specific field.\n"
            f"{body}\n"
            "}"
        )
    if isinstance(value, list):
        if not value:
            return "[]"
        sample_shape = _describe_shape(value[0])
        return (
            f"[len={len(value)}, // structural summary — full {len(full)} chars\n"
            f"  // each item: {sample_shape}\n"
            f"  // query <pointer>/N to inspect item at index N]"
        )
    # Scalar but huge — hard truncate (same as legacy _truncate_for_prompt).
    return full[:max_chars] + f"\n… (truncated, full size {len(full)} chars)"


def _describe_shape(value: Any) -> str:
    """One-line type-and-size description of a value for use inside the
    query summary. dict → ``object{N keys: …}``, list → ``list[N]``,
    string → ``string(len=N)``, others → ``<type-name>``.
    """
    if isinstance(value, dict):
        keys = list(value.keys())
        preview = ", ".join(repr(k) for k in keys[:5])
        more = f", +{len(keys) - 5} more" if len(keys) > 5 else ""
        return f"object{{{len(keys)} keys: {preview}{more}}}"
    if isinstance(value, list):
        if not value:
            return "list[0]"
        inner = type(value[0]).__name__
        return f"list[{len(value)}]<{inner}>"
    if isinstance(value, str):
        if len(value) <= 60:
            return json.dumps(value, ensure_ascii=False)
        return f"string(len={len(value)}): {json.dumps(value[:50], ensure_ascii=False)}…"
    return type(value).__name__


class SurgicalPatcher:
    """Apply patch requirements against a graph dict.

    Constructor binds to a graph (mutated in place) and an OpenAI-compatible
    client. Pass `client=None` to use the project's shared `_get_client()`.

    Per requirement we run one short tool-calling loop:
      1. seed messages with intent + context + target value
      2. let the model `query` more if needed and emit a `patch`
      3. apply each emitted op against the graph
      4. exit on empty `patch(ops=[])` or `max_steps`
    """

    def __init__(
        self,
        graph: dict,
        *,
        client: Any = None,
        model: str | None = None,
        # Default sourced from JCL_PATCHER_MAX_STEPS env (P56) —
        # historically 10 (P7C) → 15 (P51) → 30 (P54). See _config.py
        # for the corresponding read-only bail tuning.
        max_steps: int = DEFAULT_PATCHER_MAX_STEPS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.2,
        root_schema: dict | None = None,
    ) -> None:
        self.graph = graph
        # ``client`` must be an ``LLMClient`` Protocol-shape (has
        # ``chat_complete``). The library doesn't bundle a default —
        # callers construct one (see
        # ``an OpenAI adapter``) and inject.
        # Test placeholders (``object()`` for schema-only tests) are
        # accepted with a deferred check — the first real call will
        # raise a clear AttributeError if the placeholder is misused.
        if client is None:
            raise ValueError(
                "SurgicalPatcher requires an LLMClient (chat_complete); "
                "got None. Construct one in your application bootstrap "
                "and pass it in."
            )
        self.client = client
        self.model = model or DEFAULT_MODEL
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.temperature = temperature
        # Optional JSON Schema covering `graph`. When provided, the LLM can
        # call `get_schema(pointer)` to learn required fields BEFORE patching.
        # Hoist nested $defs so $ref paths resolve from the wrapper root —
        # adapters typically wrap multiple Pydantic schemas, each carrying
        # its own $defs subtree.
        self.root_schema = _hoist_defs(root_schema) if root_schema else None
        # Replaced per call inside _apply_one; defined here so attribute
        # access never AttributeErrors before the first requirement runs.
        self._loop_state: dict[str, Any] = {}

    # Public ------------------------------------------------------------

    def apply(self, requirements: list[PatchRequest]) -> list[PatchResult]:
        results: list[PatchResult] = []
        for req in requirements:
            result = self._apply_one(req)
            results.append(result)
        return results

    # Internal ----------------------------------------------------------

    def _apply_one(self, req: PatchRequest) -> PatchResult:
        calls: list[ToolCallRecord] = []
        # Verbose: surface the critic's ask BEFORE any sub-agent or
        # tool-call line so the operator can see what was requested
        # vs what the patcher actually did. Without this header, the
        # log shows raw tool calls with no context for "why".
        intent_line = (req.intent or "").replace("\n", " ").strip()
        if len(intent_line) > 280:
            intent_line = intent_line[:280] + "…"
        log_verbose(
            f"  [bold]▸ {req.requirement_id}[/bold]  "
            f"[dim]target={req.target_pointer}[/dim]"
        )
        if intent_line:
            log_verbose(f"    [yellow]intent:[/yellow] {intent_line}")
        # Pre-loop: request_validator sub-agent. Refuses obviously
        # unpatchable requirements (target_missing / too_broad) before
        # the patch loop burns its full budget. Skips cleanly when the
        # current state already satisfies the intent.
        subagent_traces: list[SubAgentTrace] = []
        early_result = self._validate_request(req, subagent_traces)
        if early_result is not None:
            return early_result
        try:
            target_value = _resolve(self.graph, req.target_pointer)
        except (KeyError, IndexError, ValueError) as exc:
            # Pre-add target — let the model see it's missing rather than fail.
            target_value = f"<not present: {exc}>"

        # P22: pre-loop template_filler. When the target is an empty
        # container AND the intent enumerates items, run a focused
        # sub-agent to assemble the complete value. The result is
        # injected as a seed in the main patcher prompt so the
        # patcher LLM can drop it in with a single replace op instead
        # of getting stuck emitting one item at a time.
        template_seed_block = self._invoke_template_filler(
            req, target_value, subagent_traces,
        )

        ctx_lines: list[str] = []
        for ptr in req.context_pointers:
            try:
                val = _resolve(self.graph, ptr)
                ctx_lines.append(f"## {ptr}\n{_truncate_for_prompt(val, 2000)}")
            except (KeyError, IndexError, ValueError) as exc:
                ctx_lines.append(f"## {ptr}\n<not present: {exc}>")
        context_block = "\n\n".join(ctx_lines) if ctx_lines else "(none)"

        constraints_line = ""
        if req.constraints:
            constraints_line = (
                "\n# Constraints\n" + json.dumps(req.constraints, ensure_ascii=False, indent=2)
            )

        # P17: propagate validator's already-performed investigation as
        # path hints. When the validator queried a deep sub-path to verify
        # a critic claim, that pointer is almost certainly where the patch
        # should land — without this, the main LLM re-discovers it from
        # scratch (and often fails to).
        subagent_hints_block = _extract_subagent_hints(subagent_traces)

        user_prompt = _USER_TMPL.format(
            rid=req.requirement_id,
            target=req.target_pointer,
            intent=req.intent,
            subagent_hints_block=subagent_hints_block,
            template_seed_block=template_seed_block,
            context_block=context_block,
            target_value=_truncate_for_prompt(target_value),
        ) + constraints_line

        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        # Per-loop transient state, accessed by tool dispatch (notably
        # `diff` and `undo`). Stored on self so handlers can mutate it
        # without threading it through every method signature. Each
        # `_apply_one` call replaces the dict — no cross-requirement leak.
        graph_snapshot = copy.deepcopy(self.graph)  # for rollback on hard failure
        self._loop_state: dict[str, Any] = {
            "applied_ops": [],
            # Snapshot of `self.graph` taken just BEFORE each successful op.
            # Parallel to applied_ops; undo restores from index `-count`.
            "pre_op_snapshots": [],
            # Loop-start snapshot used as the "before" side for diff().
            "initial_snapshot": graph_snapshot,
            # Cumulative count of ops rejected for scope-escape — drives
            # the auto-stop in the outer loop (P3C). Incremented inside
            # _apply_ops on each "outside target_pointer" rejection.
            "scope_rejections": 0,
            # Out-of-scope paths the LLM tried (parallel to scope_rejections
            # increments). Used to populate CriticError.attempted_out_of_scope_paths
            # when P3C trips. Capped at 10 entries.
            "out_of_scope_paths": [],
            # Counter for the P7B no-progress auto-stop. Incremented
            # when a step ends without applying any op AND without an
            # explicit ``patch(ops=[])``. Reset whenever a step makes
            # progress.
            "consecutive_no_applied_steps": 0,
            # P11: diff-before-converge. The LLM commonly issues a
            # plausible-looking patch then immediately calls
            # ``patch(ops=[])`` to declare done — even when the patch
            # didn't actually address the intent (or applied to the
            # wrong field, or stamped a no-op merge). Force at least
            # one ``diff`` audit between the latest successful patch
            # and convergence so the LLM has to LOOK at what it did.
            # Reset to False whenever a new in-scope op lands; flipped
            # to True on any ``diff`` call. Convergence is refused
            # while False AND applied_ops > 0.
            "diff_verified_since_last_patch": False,
            # Per-op sub-agent traces (path_finder + patch_evaluator).
            # Drained into PatchResult.subagent_traces at end of loop.
            "op_subagent_traces": [],
        }
        llm_calls = 0
        addressed = False
        reason: str | None = None
        # P49: one-shot retry latch when the model returns zero tool calls.
        no_tool_call_retried = False

        try:
            for step in range(self.max_steps):
                # Per-step bookkeeping for the auto-converge / auto-stop
                # checks at the bottom of this iteration.
                pre_step_graph = copy.deepcopy(self.graph)
                pre_step_applied_count = len(self._loop_state["applied_ops"])
                pre_step_eval_rollbacks = self._loop_state.get("evaluator_rollbacks", 0)
                # P7A: budget warning — at the last 2 steps, push the LLM
                # to converge if it has applied work. Without this, the
                # LLM commonly burns its remaining steps re-retrying a
                # tool that already succeeded (Mode A in the analyses).
                steps_left = self.max_steps - step
                if steps_left <= 2 and self._loop_state["applied_ops"]:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"BUDGET WARNING: only {steps_left} step(s) remaining. "
                            f"You have applied {len(self._loop_state['applied_ops'])} "
                            f"op(s) so far. If those changes resolve the requirement, "
                            f"call `patch(ops=[])` NOW to converge. If a previous "
                            f"`list_*_where` failed with 'no item' it likely succeeded "
                            f"in an earlier step — don't retry it; call `diff` to "
                            f"audit, then `patch(ops=[])`."
                        ),
                    })
                resp = _chat_with_retry(
                    self.client,
                    model=self.model,
                    messages=messages,
                    tools=_active_tool_schemas(),
                    tool_choice="auto",
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                llm_calls += 1
                tool_calls = resp.tool_calls

                assistant_msg: dict = {"role": "assistant", "content": resp.content}
                if tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ]
                messages.append(assistant_msg)

                if not tool_calls:
                    if resp.finish_reason == "length":
                        reason = (
                            f"finish_reason=length with no tool_calls — "
                            f"max_tokens={self.max_tokens} likely too low for emission"
                        )
                        break
                    # P49: model returned no tool calls. Some models
                    # (DeepSeek-V3.2 in particular on certain prompts)
                    # occasionally emit a free-text response instead of
                    # invoking a tool — typically saying "the change
                    # would not improve the document" or describing what
                    # they'd do without doing it. Inject a directive
                    # nudge ONCE and retry; if the model declines again,
                    # then break. Without this, an entire batch (~32
                    # traces in bunsikjip-noir world) fell off with 0
                    # steps and 0 ops, and the requirement was wrongly
                    # marked unaddressed.
                    if not no_tool_call_retried:
                        no_tool_call_retried = True
                        messages.append({
                            "role": "user",
                            "content": (
                                "You did not call any tool. Either:\n"
                                "(a) Apply the change with set_field / merge / "
                                "list_*_where, or\n"
                                "(b) If you believe no change is needed, call "
                                "patch with intent='no change needed' and "
                                "ops=[] to converge."
                            ),
                        })
                        continue
                    reason = "model returned no tool calls"
                    break

                empty_patch_seen = False
                step_had_patch_attempt = False
                for tc in tool_calls:
                    name = tc.function.name
                    if name in ("patch", "set_field", "list_append",
                                "list_replace_at", "list_insert_at",
                                "list_replace_where", "list_remove_where",
                                "list_set_where"):
                        step_had_patch_attempt = True
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    # vLLM tool-call args occasionally arrive as a bare JSON
                    # value (string / null / list / number) instead of an
                    # object. _dispatch assumes args is a dict and calls
                    # .get(...) on it — coerce to {} so we surface a clean
                    # error to the LLM instead of crashing the whole loop.
                    if not isinstance(args, dict):
                        tool_result = {
                            "content": json.dumps(
                                {"error": f"tool args must be a JSON object, "
                                          f"got {type(args).__name__}; pass "
                                          f"arguments as {{...}} not a bare value"},
                                ensure_ascii=False,
                            ),
                            "applied": [],
                        }
                        args = {}
                    else:
                        tool_result = self._dispatch(name, args, req)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result["content"],
                    })
                    summary = tool_result.get("content", "") or ""
                    if len(summary) > 400:
                        summary = summary[:400] + "…"
                    calls.append(ToolCallRecord(
                        tool=name,
                        args=args,
                        result_summary=summary,
                    ))
                    # P27: identical-call-with-identical-error detector.
                    # When the LLM submits the same (tool, normalized
                    # args) twice in a row AND both attempts fail with
                    # the same error, inject a sharp user message so the
                    # next step doesn't repeat blindly. Without this,
                    # weak models (small/local) stay stuck in a 3-step
                    # loop on schema-mismatched ops until P7B bails
                    # with no progress.
                    sig = (name, json.dumps(args, ensure_ascii=False, sort_keys=True))
                    looks_like_failure = (
                        '"error"' in summary
                        or '"rejected"' in summary
                        and '"applied": 0' in summary
                    )
                    last_failed_sigs = self._loop_state.setdefault(
                        "last_failed_sigs", []
                    )
                    if looks_like_failure:
                        last_failed_sigs.append(sig)
                        # Keep at most the last 4 entries.
                        if len(last_failed_sigs) > 4:
                            del last_failed_sigs[:-4]
                        # Two identical failed sigs in a row → inject
                        # corrective guidance once per repetition.
                        if (
                            len(last_failed_sigs) >= 2
                            and last_failed_sigs[-1] == last_failed_sigs[-2]
                            and not self._loop_state.get(
                                "p27_warned_for", ""
                            ) == repr(sig)
                        ):
                            self._loop_state["p27_warned_for"] = repr(sig)
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"You just called `{name}` with the "
                                    f"SAME arguments and got the SAME error "
                                    f"twice. Stop repeating. Read the error "
                                    f"text — it usually names the expected "
                                    f"type or the available keys. Either "
                                    f"(a) call `get_schema(pointer=...)` to "
                                    f"discover the correct value type, "
                                    f"(b) call `query(pointer=...)` to see "
                                    f"the available list keys, "
                                    f"(c) emit a different op with corrected "
                                    f"args, or (d) call `patch(ops=[])` to "
                                    f"give up on this requirement."
                                ),
                            })
                    else:
                        last_failed_sigs.clear()
                    # Verbose mode: stream each tool invocation inline so
                    # the operator can watch the patcher in real time
                    # instead of waiting for the per-iteration summary.
                    ptr = args.get("path") or args.get("pointer") or args.get("keyword") or ""
                    head = summary if len(summary) <= 80 else summary[:80] + "…"
                    log_verbose(f"      [dim]· {name:<18} {str(ptr)[:40]:<40} → {head}[/dim]")
                    # Patch-emitting tools push directly to
                    # self._loop_state["applied_ops"] inside _apply_ops, so
                    # we don't need to copy from tool_result["applied"].
                    # `diff` and `undo` likewise mutate that list in place
                    # — reading the latest state here is correct after
                    # both append and rollback.
                    if name == "patch":
                        ops = args.get("ops") or []
                        if not ops:
                            empty_patch_seen = True

                if empty_patch_seen and self._loop_state["applied_ops"]:
                    # P11: refuse convergence until the LLM audits its
                    # own work via diff(). Local-model patches commonly
                    # land on the wrong field or stamp no-op merges,
                    # then the LLM declares done — the next critic
                    # round re-flags the same target. Forcing a diff
                    # makes the LLM look at what actually changed
                    # before claiming done. The flag flips True on any
                    # diff() call and resets to False on each new
                    # successful op.
                    if not self._loop_state.get("diff_verified_since_last_patch"):
                        messages.append({
                            "role": "user",
                            "content": (
                                "Before declaring done, call `diff()` to "
                                "audit what actually changed in the graph. "
                                "Confirm the diff resolves the requirement's "
                                "intent. If it does, then call `patch(ops=[])` "
                                "again to converge. If the diff reveals the "
                                "patch missed the intent, fix it with another "
                                "patch / set_field / list_* call instead."
                            ),
                        })
                        # Not converging yet — keep the loop running.
                        empty_patch_seen = False
                        continue
                    # P21: convergence gate — refuse to flip
                    # addressed=True while there's an unresolved
                    # negative evaluator verdict. "Unresolved" =
                    # the most recent evaluator verdict on each
                    # touched path is in (no-op/off-target/partial)
                    # at high/medium confidence. The patcher LLM has
                    # to either fix the offending path or explain via
                    # a fresh op before converge.
                    unresolved = self._unresolved_negative_eval_paths()
                    if unresolved:
                        # P26: cumulative re-eval against initial snapshot.
                        # Per-op partial verdicts can mislead when the LLM
                        # built up the fix across several ops — each op
                        # in isolation looks partial but cumulatively the
                        # intent is addressed. Ask the evaluator once on
                        # the full diff. If "addressed", let convergence
                        # proceed; otherwise keep the original gate.
                        cumulative_ok = self._cumulative_eval_addressed(req)
                        if cumulative_ok:
                            addressed = True
                            reason = (
                                "auto-converged: cumulative re-eval addressed "
                                f"({len(self._loop_state['applied_ops'])} ops)"
                            )
                            break
                        messages.append({
                            "role": "user",
                            "content": (
                                "Cannot converge yet — patch_evaluator flagged "
                                "the following changes as not addressing the "
                                "intent: " + ", ".join(
                                    f"{p} ({v})" for p, v in unresolved[:5]
                                )
                                + ". Either re-patch with a corrected value or "
                                "call diff() and explain why the current state "
                                "does meet the intent before retrying patch(ops=[])."
                            ),
                        })
                        empty_patch_seen = False
                        continue
                    addressed = True
                    break
                if empty_patch_seen and not self._loop_state["applied_ops"]:
                    reason = "model converged without emitting any ops"
                    break

                # P3A: auto-converge when this step "applied" ops but
                # produced zero net state change (LLM redoing already-
                # applied work, or replace with identical value). Most
                # common Mode-A failure was the LLM never issuing the
                # explicit ``patch(ops=[])`` despite being done.
                new_applied_in_step = (
                    len(self._loop_state["applied_ops"]) - pre_step_applied_count
                )
                if new_applied_in_step > 0 and self.graph == pre_step_graph:
                    # P19: auto-converge would mask evaluator-driven
                    # rollbacks. If we rolled back any op this step
                    # because the evaluator said no-op/off-target/
                    # partial, the "no net state change" is BY DESIGN —
                    # the LLM's work was rejected. Don't flip addressed.
                    rollbacks_this_step = (
                        self._loop_state.get("evaluator_rollbacks", 0)
                        - pre_step_eval_rollbacks
                    )
                    if rollbacks_this_step == 0:
                        addressed = True
                        reason = (
                            "auto-converged: ops in this step produced no net "
                            "state change (likely re-applying already-resolved "
                            "edits — treating as done)"
                        )
                        break

                # P7B counter update (before checks below). Only count
                # steps that ATTEMPTED a patch (patch, set_field, list_*)
                # but landed nothing — read-only exploration steps
                # (query / get_schema / find_*) are excluded so legit
                # exploration before the first patch isn't punished.
                if step_had_patch_attempt and new_applied_in_step == 0:
                    self._loop_state["consecutive_no_applied_steps"] += 1
                elif new_applied_in_step > 0:
                    self._loop_state["consecutive_no_applied_steps"] = 0
                # P38: track consecutive read-only steps separately. Some
                # models (DeepSeek-V3.2 observed) get stuck in pure
                # exploration mode — querying repeatedly without ever
                # emitting a patch op. The original consecutive_no_applied
                # counter excludes read-only steps, so the loop runs to
                # max_steps with zero applied ops. Bail after
                # ``DEFAULT_PATCHER_READ_ONLY_BAIL`` consecutive query-only
                # steps with no patches applied. Threshold history:
                # 5 → 10 (P51) → 20 (P56) — bulk-DAG-revise intents
                # routinely need 10+ read-only passes before the LLM has
                # enough info to commit a single op; cutting at 10
                # reproduced the original max_steps failure in regression.
                # Tunable via ``JCL_PATCHER_READ_ONLY_BAIL`` env.
                if not step_had_patch_attempt:
                    self._loop_state["consecutive_query_only_steps"] = (
                        self._loop_state.get("consecutive_query_only_steps", 0) + 1
                    )
                else:
                    self._loop_state["consecutive_query_only_steps"] = 0
                if (
                    self._loop_state.get("consecutive_query_only_steps", 0)
                        >= DEFAULT_PATCHER_READ_ONLY_BAIL
                    and not self._loop_state["applied_ops"]
                ):
                    reason = (
                        f"abandoned: "
                        f"{self._loop_state['consecutive_query_only_steps']} "
                        f"consecutive read-only steps without ever attempting "
                        f"a patch — model stuck in exploration mode"
                    )
                    break

                # P3C: cumulative scope-escape stop. Checked BEFORE P7B
                # so that when both conditions hold, the more specific
                # scope-related reason wins.
                if self._loop_state["scope_rejections"] >= 2:
                    # P3+: attempt one auto-widen before bailing. When the
                    # LLM has repeatedly tried siblings of target_pointer
                    # and zero in-scope ops have landed, the critic's
                    # target_pointer was probably too narrow (e.g. one
                    # /key_events/N slot when the fix needs to span the
                    # whole list to deduplicate). Widen target_pointer to
                    # the deepest common ancestor of {original_target,
                    # *out_of_scope_paths} once and resume the loop.
                    if (
                        not self._loop_state["applied_ops"]
                        and not self._loop_state.get("auto_widened")
                    ):
                        all_paths = (
                            [req.target_pointer]
                            + list(self._loop_state["out_of_scope_paths"])
                        )
                        new_target = _common_parent_pointer(all_paths)
                        # Only widen if the new pointer is non-root and
                        # actually broader (strict prefix of original).
                        if (
                            new_target
                            and new_target != req.target_pointer
                            and req.target_pointer.startswith(new_target + "/")
                        ):
                            old_target = req.target_pointer
                            req.target_pointer = new_target
                            self._loop_state["auto_widened"] = True
                            self._loop_state["original_target_pointer"] = old_target
                            self._loop_state["scope_rejections"] = 0
                            self._loop_state["out_of_scope_paths"] = []
                            self._loop_state["consecutive_no_applied_steps"] = 0
                            logger.info(
                                "patcher auto-widened target_pointer %s → %s for %s",
                                old_target, new_target, req.requirement_id,
                            )
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"SCOPE WIDENED: target_pointer expanded from "
                                    f"{old_target!r} to {new_target!r} because previous "
                                    f"ops needed sibling-level changes. Continue applying "
                                    f"the requirement under the new scope; call "
                                    f"`patch(ops=[])` when the issues are resolved."
                                ),
                            })
                            continue
                    if self._loop_state["applied_ops"]:
                        addressed = True
                        reason = (
                            f"auto-converged best-effort: "
                            f"{self._loop_state['scope_rejections']} ops rejected "
                            f"as out-of-scope (intent likely needs sibling changes); "
                            f"{len(self._loop_state['applied_ops'])} in-scope ops applied"
                        )
                    else:
                        reason = (
                            f"abandoned: {self._loop_state['scope_rejections']} "
                            f"ops outside target_pointer; nothing in-scope landed"
                        )
                    break

                # P7B: 3 consecutive failed-patch-attempt steps → bail.
                # Bumped from 2 → 3 because plot-phase critics often steer
                # the LLM through scope_rejection → dict-hint → set_field
                # in three turns; cutting at 2 strands the LLM right after
                # it received the hint that names the exact correct call
                # (snowed-pension-six tr-da9b55bb, bunsikjip-noir tr-c11ee277).
                if self._loop_state["consecutive_no_applied_steps"] >= 3:
                    if self._loop_state["applied_ops"]:
                        addressed = True
                        reason = (
                            f"auto-converged best-effort: "
                            f"{self._loop_state['consecutive_no_applied_steps']} "
                            f"consecutive no-progress steps after "
                            f"{len(self._loop_state['applied_ops'])} successful op(s); "
                            f"likely already done"
                        )
                    else:
                        reason = (
                            f"abandoned: "
                            f"{self._loop_state['consecutive_no_applied_steps']} "
                            f"consecutive steps with no op applied; nothing landed"
                        )
                    break
            else:
                reason = f"hit max_steps={self.max_steps} without convergence"
        except Exception as exc:
            # Roll the graph back so a half-applied requirement doesn't poison
            # downstream requirements in the same batch.
            self.graph.clear()
            self.graph.update(graph_snapshot)
            self._loop_state["applied_ops"] = []
            self._loop_state["pre_op_snapshots"] = []
            addressed = False
            reason = f"exception during patch loop: {exc!r}"
            logger.exception("patcher failed for %s", req.requirement_id)

        applied_ops = list(self._loop_state["applied_ops"])
        scope_rejs = self._loop_state.get("scope_rejections", 0)
        out_of_scope_paths = list(self._loop_state.get("out_of_scope_paths", []))

        # P8: when the loop ended unceremoniously (max_steps hit, LLM
        # emission truncated by length, or LLM went silent with no
        # tool calls) but applied_ops > 0, the LLM clearly DID make
        # progress — it just never issued the explicit ``patch(ops=[])``
        # convergence call. Treat as best-effort addressed=True so the
        # critic loop doesn't re-flag work that already landed. The
        # original ``reason`` is preserved so reviewers see what
        # happened.
        if (
            not addressed
            and applied_ops
            and reason is not None
            and (
                "max_steps" in reason
                or "finish_reason=length" in reason
                or "no tool calls" in reason
            )
        ):
            addressed = True
            reason = (
                f"best-effort: applied {len(applied_ops)} op(s) before "
                f"loop exit ({reason})"
            )

        # Diagnose critic-level mistakes from the loop's outcome.
        # Two distinct shapes:
        #   - scope_mismatch: 2+ out-of-scope rejections accumulated
        #     (regardless of whether some in-scope ops also landed —
        #     a partial fix still flags the critic for asking too much).
        #   - unfulfillable_intent: loop exhausted (max_steps hit OR
        #     LLM returned no tool calls) with zero applied ops AND
        #     zero scope rejections — the intent points at something
        #     the LLM couldn't even attempt to patch in scope.
        critic_error: CriticErrorRecord | None = None
        if scope_rejs >= 2:
            critic_error = CriticErrorRecord(
                kind="scope_mismatch",
                summary=(
                    f"{scope_rejs} ops attempted outside target_pointer "
                    f"{req.target_pointer!r} — fix likely needs sibling "
                    f"changes; consider widening target_pointer or "
                    f"splitting the requirement"
                ),
                attempted_out_of_scope_paths=out_of_scope_paths,
            )
        elif scope_rejs >= 1 and not applied_ops:
            # Even one out-of-scope rejection with zero in-scope work
            # landed is a critic-fault: the LLM tried to escape and the
            # remaining attempts couldn't fulfill the intent within the
            # given target_pointer. Without this fallback, runs that
            # bail via P7B (consecutive no-applied) after just one
            # scope rejection went un-tagged (academy-bottom-mage
            # tr-2977d0fa: critic flagged key-evt-9 but the dedupe
            # target was at /key_events/10).
            critic_error = CriticErrorRecord(
                kind="scope_mismatch",
                summary=(
                    f"{scope_rejs} op attempted outside target_pointer "
                    f"{req.target_pointer!r} and no in-scope op landed — "
                    f"intent likely targets the wrong slot or needs "
                    f"sibling-level changes"
                ),
                attempted_out_of_scope_paths=out_of_scope_paths,
            )
        elif (
            not applied_ops
            and scope_rejs == 0
            and reason is not None
            and (
                "max_steps" in reason
                or "no tool calls" in reason
                or "converged without emitting any ops" in reason
                or "abandoned" in reason  # P7B's "consecutive no-applied"
            )
        ):
            critic_error = CriticErrorRecord(
                kind="unfulfillable_intent",
                summary=(
                    f"loop exhausted ({llm_calls} LLM calls) without "
                    f"applying any op or escaping scope — intent likely "
                    f"points at something not present in target_pointer "
                    f"{req.target_pointer!r}"
                ),
            )

        # Capture P3+ auto-widen origin (if it fired) before we drop loop state.
        auto_widened_from = self._loop_state.get("original_target_pointer")
        # Collect per-op subagent traces written during the loop.
        subagent_traces.extend(self._loop_state.get("op_subagent_traces", []))
        # Drop the per-loop state so a stray reference can't leak into the
        # next requirement. Handlers always re-establish it on entry.
        self._loop_state = {}
        return PatchResult(
            requirement_id=req.requirement_id,
            addressed=addressed,
            target_pointer=req.target_pointer,
            intent=req.intent,
            applied_ops=applied_ops,
            reason=reason,
            llm_calls=llm_calls,
            calls=calls,
            critic_error=critic_error,
            auto_widened_from=auto_widened_from,
            subagent_traces=subagent_traces,
        )

    # Tool dispatch ----------------------------------------------------

    def _dispatch(self, name: str, args: dict, req: PatchRequest) -> dict:
        """Returns {'content': str, 'applied': list[op]} (applied empty unless name=='patch')."""
        if name == "query":
            ptr = args.get("pointer", "")
            try:
                val = _resolve(self.graph, ptr)
            except (KeyError, IndexError, ValueError) as exc:
                # Surface the graph's TOP-LEVEL keys when a root-adjacent
                # lookup fails — the LLM commonly hallucinates paths like
                # ``/characters`` or ``/__settings__`` that don't exist
                # in this graph. Showing what IS there saves another
                # exploratory round-trip.
                payload: dict[str, Any] = {"error": str(exc)}
                if isinstance(self.graph, dict):
                    payload["available_top_level"] = sorted(self.graph.keys())[:30]
                return {
                    "content": json.dumps(payload, ensure_ascii=False),
                    "applied": [],
                }
            return {"content": _summarize_for_query(val, max_chars=1500), "applied": []}

        if name == "find_paths":
            keyword = args.get("keyword", "") or ""
            try:
                cap = max(1, int(args.get("max_results", 30) or 30))
            except (TypeError, ValueError):
                cap = 30
            paths = _find_paths_by_key(self.graph, keyword, max_results=cap)
            payload: dict[str, Any] = {
                "keyword": keyword,
                "n_results": len(paths),
                "paths": paths,
            }
            if not paths:
                payload["note"] = "no dict key contains the keyword"
            return {"content": json.dumps(payload, ensure_ascii=False), "applied": []}

        if name == "find_values":
            keyword = args.get("keyword", "") or ""
            try:
                cap = max(1, int(args.get("max_results", 30) or 30))
            except (TypeError, ValueError):
                cap = 30
            hits = _find_values(self.graph, keyword, max_results=cap)
            payload = {
                "keyword": keyword,
                "n_results": len(hits),
                "hits": hits,
            }
            if not hits:
                payload["note"] = "no leaf string value contains the keyword"
            return {"content": json.dumps(payload, ensure_ascii=False), "applied": []}

        if name == "find_index":
            return self._find_index(args)

        if name == "jq":
            return self._jq(args)

        if name == "python":
            return self._python(args)

        if name == "javascript":
            return self._javascript(args)

        if name == "get_schema":
            ptr = args.get("pointer", "")
            if not self.root_schema:
                return {
                    "content": json.dumps(
                        {"note": "no schema available for this graph"},
                        ensure_ascii=False,
                    ),
                    "applied": [],
                }
            try:
                node = _resolve_schema_at(self.root_schema, ptr)
            except (KeyError, ValueError) as exc:
                return {
                    "content": json.dumps({"error": str(exc)}, ensure_ascii=False),
                    "applied": [],
                }
            trimmed = _trim_schema_for_llm(node)
            return {"content": json.dumps(trimmed, ensure_ascii=False), "applied": []}

        if name == "patch":
            ops = args.get("ops") or []
            if not ops:
                return {"content": json.dumps({"ok": True, "applied": 0}), "applied": []}
            return self._apply_ops(ops, req)

        # Narrow tools — translate to a single op then run through _apply_ops
        # so scope-guard / error handling stay uniform.
        if name == "set_field":
            ptr = args.get("pointer", "")
            # Use replace if path exists, add if not.
            try:
                _resolve(self.graph, ptr)
                op = {"op": "replace", "path": ptr, "value": args.get("value")}
            except (KeyError, IndexError, ValueError):
                op = {"op": "add", "path": ptr, "value": args.get("value")}
            return self._apply_ops([op], req)

        if name == "list_append":
            ptr = (args.get("pointer", "") or "").rstrip("/")
            op = {"op": "add", "path": f"{ptr}/-", "value": args.get("item")}
            return self._apply_ops([op], req)

        if name == "list_replace_at":
            ptr = (args.get("pointer", "") or "").rstrip("/")
            idx = args.get("index")
            if idx is None:
                return {
                    "content": json.dumps(
                        {"error": "list_replace_at requires 'index' (int); "
                                  "for identity-keyed lists call list_replace_where instead"},
                        ensure_ascii=False,
                    ),
                    "applied": [],
                }
            op = {"op": "replace", "path": f"{ptr}/{int(idx)}", "value": args.get("item")}
            return self._apply_ops([op], req)

        if name == "list_insert_at":
            ptr = (args.get("pointer", "") or "").rstrip("/")
            idx = args.get("index")
            if idx is None:
                return {
                    "content": json.dumps(
                        {"error": "list_insert_at requires 'index' (int); "
                                  "use list_append for end-insertion"},
                        ensure_ascii=False,
                    ),
                    "applied": [],
                }
            op = {"op": "add", "path": f"{ptr}/{int(idx)}", "value": args.get("item")}
            return self._apply_ops([op], req)

        if name in ("list_replace_where", "list_remove_where", "list_set_where"):
            return self._list_where(name, args, req)

        if name == "list_str_remove":
            return self._list_str_remove(args, req)

        if name == "list_move_at":
            return self._list_move_at(args, req)

        if name == "diff":
            # P11: any diff() call satisfies the verify-before-converge
            # gate. We also append an explicit "next action" hint to the
            # diff result so the LLM doesn't waste budget on more diffs
            # or exploratory queries — observed: gemma4:26b called diff
            # twice then queried then went silent.
            if self._loop_state:
                self._loop_state["diff_verified_since_last_patch"] = True
            result = self._diff()
            applied_count = len(self._loop_state.get("applied_ops") or []) if self._loop_state else 0
            if applied_count > 0:
                # Try to inject the next_action hint, but the diff
                # content may be truncated (no longer valid JSON) when
                # the diff is large. Fall back to an appended hint
                # instead of crashing the patcher loop.
                try:
                    payload = json.loads(result["content"])
                    payload["next_action"] = (
                        "If the before/after pairs above resolve the requirement's "
                        "intent, call `patch(ops=[])` NOW to converge — do NOT call "
                        "diff again or run more queries. If the diff reveals the "
                        "patch missed the intent, fix it with another patch op "
                        "instead."
                    )
                    result["content"] = json.dumps(payload, ensure_ascii=False)
                except json.JSONDecodeError:
                    result["content"] = (
                        result["content"]
                        + "\n\nnext_action: If the diff resolves the intent, "
                        "call patch(ops=[]) to converge; otherwise emit a "
                        "corrective patch op."
                    )
            return result

        if name == "undo":
            return self._undo(args)

        return {
            "content": json.dumps({"error": f"unknown tool {name!r}"}, ensure_ascii=False),
            "applied": [],
        }

    # identity-keyed list ops ------------------------------------------

    def _list_where(self, name: str, args: dict, req: PatchRequest) -> dict:
        """Resolve an identity-keyed list op into the matching index, then
        delegate to ``_apply_ops`` so scope guard + snapshot/undo bookkeeping
        stay uniform.

        Why these exist: label-keyed list-of-dicts (like ``extensions[].items[]``,
        keyed by ``label``) make ``list_remove_at(index=N)`` brittle —
        the next op's indices are silently wrong after siblings move.
        These tools take ``key_field`` + ``key_value`` and find the index
        AT CALL TIME against the current ``self.graph``, so the LLM
        never has to track indices across ops.

        Errors that surface back to the LLM:
        - target not a list / target missing
        - zero matches → tells the LLM the key_value it tried doesn't exist
        - multiple matches → ambiguous; LLM should pick a more specific key
        """
        ptr = (args.get("pointer", "") or "").rstrip("/")
        key_field = args.get("key_field")
        key_value = args.get("key_value")
        if not ptr or not key_field:
            return {
                "content": json.dumps(
                    {"error": f"{name} requires 'pointer' and 'key_field'"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        # Early scope check on the pointer itself. The narrow tools never
        # build an op when target isn't a list / no match — but a true
        # scope escape (LLM picked the wrong subtree, e.g. critic asked
        # to fix subplots/1 but the LLM tried character_arcs) should
        # still count toward scope_rejections so the post-loop
        # critic_error classifier tags it as scope_mismatch instead of
        # letting it look like a patcher bug.
        #
        # Permitted relationships between ``ptr`` (the LLM-supplied list
        # pointer) and ``req.target_pointer``:
        #   - ptr is in/at target — direct subtree match (`_op_in_scope`).
        #   - ptr is an ancestor of target — the LLM is trying to resolve
        #     a single list-item by id (e.g. ptr=/key_events,
        #     target=/key_events/6 with key_value=key-evt-7). This is
        #     normal usage; the resolved op's path will be re-checked
        #     downstream by ``_apply_ops`` so a wrong-index resolution
        #     still gets caught as a scope rejection there.
        target = req.target_pointer
        ptr_is_ancestor_of_target = bool(target) and target.startswith(ptr + "/")
        if not (
            self._op_in_scope({"path": ptr}, target) or ptr_is_ancestor_of_target
        ):
            state = self._loop_state
            if state:
                state["scope_rejections"] = state.get("scope_rejections", 0) + 1
                paths = state.setdefault("out_of_scope_paths", [])
                if len(paths) < 10:
                    paths.append(ptr)
            return {
                "content": json.dumps(
                    {"error": f"pointer {ptr!r} is outside target_pointer "
                              f"{target!r}; this requirement only "
                              f"allows edits inside that subtree"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        try:
            target_list = _resolve(self.graph, ptr)
        except (KeyError, IndexError, ValueError) as exc:
            return {
                "content": json.dumps(
                    {"error": f"list at {ptr!r} not found: {exc}"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        if not isinstance(target_list, list):
            # Hint at the right tool when target is a dict — the LLM
            # commonly reaches for list_set_where on dict-keyed
            # collections (e.g. /plot/character_arcs is dict[name → arc])
            # OR on list-item dicts (e.g. /plot/subplots/2 is one
            # subplot dict). A generic "not a list" message left it
            # looping on the same broken call. We can give an action-
            # specific suggestion that names the exact set_field call
            # the LLM almost certainly meant.
            actual_type = type(target_list).__name__
            if actual_type == "dict":
                # Decide which dict-shape this is:
                #   list-item dict — pointer ends in a numeric segment AND
                #     the LLM-supplied field is a direct key on that dict.
                #     Suggest set_field <pointer>/<field>.
                #   dict-keyed collection — every value is itself a dict
                #     AND the LLM's key_value names one of those keys.
                #     Suggest set_field <pointer>/<key_value>/<field>.
                #   neither — fall back to the generic merge/set_field hint.
                field = (args.get("field") if name == "list_set_where" else None)
                if field and field in target_list:
                    suggested_ptr = f"{ptr}/{_encode_pointer_token(field)}"
                    hint = (
                        f"target at {ptr!r} is a dict (one item, not a list of "
                        f"items). The list_*_where tools only work on lists. "
                        f"To set this dict's {field!r} field, call "
                        f"set_field(pointer={suggested_ptr!r}, value=...) "
                        f"(or merge for multiple fields at once)."
                    )
                elif (
                    target_list
                    and all(isinstance(v, dict) for v in target_list.values())
                    and key_value in target_list
                ):
                    inner_ptr = f"{ptr}/{_encode_pointer_token(str(key_value))}"
                    if field and field in target_list[key_value]:
                        leaf = f"{inner_ptr}/{_encode_pointer_token(field)}"
                        hint = (
                            f"target at {ptr!r} is a dict-keyed collection "
                            f"(dict[K, V]), not a list. To set {key_value!r}'s "
                            f"{field!r} field, call set_field(pointer={leaf!r}, "
                            f"value=...) (or merge on {inner_ptr!r} for several "
                            f"fields at once). The list_*_where tools only work "
                            f"on lists of dicts."
                        )
                    else:
                        hint = (
                            f"target at {ptr!r} is a dict-keyed collection "
                            f"(dict[K, V]), not a list. To update {key_value!r}, "
                            f"use merge(pointer={inner_ptr!r}, value={{...}}) or "
                            f"set_field on {inner_ptr}/<field>. The list_*_where "
                            f"tools only work on lists of dicts."
                        )
                else:
                    hint = (
                        f"target at {ptr!r} is a dict, not a list. "
                        f"For dict-shaped targets use set_field with a "
                        f"fully-keyed pointer (op=replace), or merge to update "
                        f"several fields at once. The list_*_where tools only "
                        f"work on lists of dicts."
                    )
            else:
                hint = f"target at {ptr!r} is not a list (got {actual_type})"
            return {
                "content": json.dumps(
                    {"error": hint},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        matches = [
            i for i, item in enumerate(target_list)
            if isinstance(item, dict) and item.get(key_field) == key_value
        ]
        if not matches:
            seen = sorted({
                str(it.get(key_field)) for it in target_list
                if isinstance(it, dict) and key_field in it
            })
            # P30: weak-model recovery — when key_value is bare int/bool
            # but the actual labels are strings, scan the intent for any
            # of the available labels and substitute. Matches the common
            # tool-call corruption mode where the LLM intended a string
            # label (e.g. "한수진") but emitted ``key_value: 1``.
            if (
                isinstance(key_value, (int, bool))
                and isinstance(req.intent, str)
                and seen
            ):
                # Find labels that appear in the intent text. When the
                # critic mentions multiple labels (e.g. naming both
                # "박동훈-한수진" and the relationship's other parties),
                # pick the one whose first occurrence comes earliest —
                # the intent typically leads with the target label.
                hits = [
                    (req.intent.find(s), s) for s in seen
                    if s and s in req.intent
                ]
                if hits:
                    hits.sort()
                    key_value = hits[0][1]
                    matches = [
                        i for i, item in enumerate(target_list)
                        if isinstance(item, dict) and item.get(key_field) == key_value
                    ]
            if not matches:
                return {
                    "content": json.dumps(
                        {
                            "error": f"no item with {key_field}={key_value!r}",
                            "available_values": seen[:30],
                        },
                        ensure_ascii=False,
                    ),
                    "applied": [],
                }
        # P9: match_policy resolves multi-match into a deterministic
        # set of target indices. Default ``single`` keeps the strict
        # behaviour (good for label-uniqueness invariants); ``all``
        # is the right tool for deduplication. ``first``/``last``
        # let the LLM pick a specific one without re-`query`'ing for
        # exact indices.
        policy = args.get("match_policy") or "single"
        if policy == "single":
            if len(matches) > 1:
                return {
                    "content": json.dumps(
                        {
                            "error": (
                                f"multiple items match {key_field}={key_value!r}; "
                                f"pass match_policy='all' to dedupe, or "
                                f"'first'/'last' to target one"
                            ),
                            "indices": matches,
                        },
                        ensure_ascii=False,
                    ),
                    "applied": [],
                }
            target_indices: list[int] = [matches[0]]
        elif policy == "first":
            target_indices = [matches[0]]
        elif policy == "last":
            target_indices = [matches[-1]]
        elif policy == "all":
            target_indices = list(matches)
        else:
            return {
                "content": json.dumps(
                    {"error": f"unknown match_policy={policy!r}; "
                              f"expected one of single/first/last/all"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }

        if name == "list_set_where":
            field = args.get("field")
            value = args.get("value")
            if not field:
                return {
                    "content": json.dumps(
                        {"error": "list_set_where requires 'field'"},
                        ensure_ascii=False,
                    ),
                    "applied": [],
                }

        ops: list[dict] = []
        # P47: when removing list-of-dict items, scan the rest of the
        # graph for stale ID references in list[str] fields and prepend
        # ``_cascade``-flagged remove ops so referential integrity is
        # restored in the same patch. Without this the LLM has to chain
        # ``list_str_remove`` calls — and historically gave up after a
        # few rounds, leaving dangling refs.
        cascade_ops: list[dict] = []
        if name == "list_remove_where":
            cascade_ids: set[str] = set()
            for idx in target_indices:
                item = target_list[idx]
                if not isinstance(item, dict):
                    continue
                for fname in (key_field, "id"):
                    v = item.get(fname)
                    if isinstance(v, str) and v:
                        cascade_ids.add(v)
            if cascade_ids:
                # Exclude the specific items being removed from the
                # scan — any self-ref inside an about-to-be-removed
                # dict would just generate wasted cleanup ops. The
                # rest of the parent list (sibling items) MUST be
                # scanned — that's where the dangling refs typically
                # live (e.g. /key_events/B/prerequisite_event_ids
                # naming /key_events/A's id).
                excludes = {f"{ptr}/{idx}" for idx in target_indices}
                cascade_ops = self._scan_id_refs_for_cascade(
                    cascade_ids, exclude_subtrees=excludes,
                )

        # Build ops in DESCENDING index order so multi-target removes
        # don't suffer from index shift mid-batch.
        for idx in sorted(target_indices, reverse=(name == "list_remove_where")):
            path_at_idx = f"{ptr}/{idx}"
            if name == "list_remove_where":
                ops.append({"op": "remove", "path": path_at_idx})
            elif name == "list_replace_where":
                ops.append({"op": "replace", "path": path_at_idx, "value": args.get("item")})
            else:  # list_set_where
                field = args["field"]
                value = args.get("value")
                sub_op = "replace" if field in target_list[idx] else "add"
                ops.append({"op": sub_op, "path": f"{path_at_idx}/{field}", "value": value})
        # Cascade cleanup runs FIRST so the primary remove (which may
        # shift sibling indices in the parent list) doesn't invalidate
        # cascade paths. Cascade paths target unrelated list[str] fields,
        # so they don't conflict with the primary path.
        return self._apply_ops(cascade_ops + ops, req)

    def _scan_id_refs_for_cascade(
        self, ids: set[str], *, exclude_subtrees: set[str] | None = None,
    ) -> list[dict]:
        """Walk ``self.graph`` for list[str] fields containing any of the
        given ids; return RFC 6902 ``remove`` ops marked ``_cascade=True``.

        Only matches lists whose entries are ALL strings — list-of-dict
        traversal continues normally so we don't mis-fire on shape-mixed
        data. Within each matching list, indices are emitted descending
        so a single list with multiple stale refs cleans correctly.
        Across lists, ordering doesn't matter (paths are disjoint).
        """
        excludes = exclude_subtrees or set()
        hits: list[tuple[str, int]] = []

        def walk(node: Any, ptr: str) -> None:
            for ex in excludes:
                if ptr == ex or ptr.startswith(ex + "/"):
                    return
            if isinstance(node, list):
                if node and all(isinstance(x, str) for x in node):
                    for i, s in enumerate(node):
                        if s in ids:
                            hits.append((ptr, i))
                else:
                    for i, child in enumerate(node):
                        walk(child, f"{ptr}/{i}")
            elif isinstance(node, dict):
                for k, v in node.items():
                    walk(v, f"{ptr}/{_encode_pointer_token(str(k))}")

        walk(self.graph, "")

        ops: list[dict] = []
        # Group by list path so reverse-index ordering is preserved
        # within each list; across lists, paths are disjoint so order
        # is irrelevant.
        by_list: dict[str, list[int]] = {}
        for list_ptr, idx in hits:
            by_list.setdefault(list_ptr, []).append(idx)
        for list_ptr, idxs in by_list.items():
            for idx in sorted(idxs, reverse=True):
                ops.append({
                    "op": "remove",
                    "path": f"{list_ptr}/{idx}",
                    "_cascade": True,
                })
        return ops

    def _find_index(self, args: dict) -> dict:
        """Read-only lookup: index of a list-of-dicts item by identity field.

        P51: addresses the abandoned-on-insert pattern where the LLM
        burned its read-only budget chasing "what index is kev-X at?"
        across `find_values` (returns paths like '/key_events/5/id'
        that the model has to mentally slice) and `query` calls. With
        a direct id→int helper, the typical insert flow drops from
        4-5 read steps to 2.

        Returns ``{"index": N}`` for a single match,
        ``{"indices": [N, M, ...]}`` for multi-match (information only,
        not an error — caller can pick), and ``{"error": ...,
        "available_values": [...]}`` for a miss (so the LLM can spot a
        typo'd id without another `query` round trip).
        """
        ptr = (args.get("pointer", "") or "").rstrip("/")
        key_field = args.get("key_field")
        # ``key_value`` is allowed to be ``None`` in principle but in
        # practice we hard-error since matching None against absent
        # fields produces brittle results.
        if not ptr or not key_field or "key_value" not in args:
            return {
                "content": json.dumps(
                    {"error": "find_index requires 'pointer', 'key_field', and 'key_value'"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        try:
            target_list = _resolve(self.graph, ptr)
        except (KeyError, IndexError, ValueError) as exc:
            return {
                "content": json.dumps(
                    {"error": f"list at {ptr!r} not found: {exc}"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        if not isinstance(target_list, list):
            return {
                "content": json.dumps(
                    {"error": f"target at {ptr!r} is not a list (got {type(target_list).__name__})"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        key_value = args.get("key_value")
        matches = [
            i for i, item in enumerate(target_list)
            if isinstance(item, dict) and item.get(key_field) == key_value
        ]
        if not matches:
            seen = sorted({
                str(it.get(key_field)) for it in target_list
                if isinstance(it, dict) and key_field in it
            })
            payload = {
                "error": f"no item with {key_field}={key_value!r}",
                "available_values": seen[:30],
            }
            return {"content": json.dumps(payload, ensure_ascii=False), "applied": []}
        if len(matches) == 1:
            return {
                "content": json.dumps({"index": matches[0]}, ensure_ascii=False),
                "applied": [],
            }
        return {
            "content": json.dumps({"indices": matches}, ensure_ascii=False),
            "applied": [],
        }

    def _jq(self, args: dict) -> dict:
        """Run a jq expression against the graph (or scoped subtree).

        Designed to collapse the multi-`query` exploration loop. The
        most common LLM exploration pattern was: ``query(/list)`` →
        14× ``query(/list/i)`` to learn each item's id+summary →
        commit. With jq, the same info comes in one call:
        ``map({id, summary})`` against the list.

        Errors (compile / runtime) come back as JSON ``{"error": ...}``
        with the offending expression so the LLM can self-correct on
        the next call. Result size is bounded by the same 1500-char
        budget as ``query`` to keep tool-result tokens predictable.
        """
        expression = args.get("expression")
        if not expression or not isinstance(expression, str):
            return {
                "content": json.dumps(
                    {"error": "jq requires non-empty 'expression' (string)"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        ptr = (args.get("pointer") or "").rstrip("/")
        try:
            target = _resolve(self.graph, ptr) if ptr else self.graph
        except (KeyError, IndexError, ValueError) as exc:
            return {
                "content": json.dumps(
                    {"error": f"pointer {ptr!r} not found: {exc}"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        try:
            import jq as _jq_lib  # local import — keeps cold import cheap
        except ImportError:
            return {
                "content": json.dumps(
                    {"error": "jq library not installed; jq tool unavailable"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        try:
            program = _jq_lib.compile(expression)
        except ValueError as exc:
            return {
                "content": json.dumps(
                    {
                        "error": f"jq compile error: {exc}",
                        "expression": expression,
                        "hint": "verify jq syntax — common pitfalls: "
                                "string slicing uses `.[0:60]` not `[:60]`, "
                                "field projection uses `{a, b}` shorthand "
                                "or `{a: .a, b: .b}`",
                    },
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        try:
            results = program.input(target).all()
        except Exception as exc:
            return {
                "content": json.dumps(
                    {
                        "error": f"jq runtime error: {exc}",
                        "expression": expression,
                    },
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        # jq 'output stream' semantics: even single-value results come back as
        # a one-element list. Unwrap when len==1 so callers get the value
        # directly, but keep the list when there are multiple emissions
        # (typical for `.[]` style iteration).
        payload = results[0] if len(results) == 1 else results
        return {"content": _summarize_for_query(payload, max_chars=1500), "applied": []}

    def _python(self, args: dict) -> dict:
        """Execute Python code against the graph (or a scoped subtree).

        Counterpart to ``_jq`` — read-only language tool with broader
        expressiveness (loops, conditionals, comprehensions). Per
        jqBench (Parnin et al. ICLR 2026), code-language tools beat
        jq on harder JSON read tasks.

        Two execution modes (auto-detected):
          - single Python expression → ``eval``, value returned
          - statements → ``exec``, ``result`` variable returned

        Sandbox: a curated builtins subset (no I/O, no os, no eval/exec
        within the snippet, no imports beyond ``json`` pre-bound).
        ``data`` is a deep copy so accidental mutation can't leak into
        ``self.graph`` — commits MUST go through ``patch`` / ``set_field``
        / ``list_*`` tools.
        """
        code = args.get("code")
        if not code or not isinstance(code, str):
            return {
                "content": json.dumps(
                    {"error": "python requires non-empty 'code' (string)"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        ptr = (args.get("pointer") or "").rstrip("/")
        try:
            target = _resolve(self.graph, ptr) if ptr else self.graph
        except (KeyError, IndexError, ValueError) as exc:
            return {
                "content": json.dumps(
                    {"error": f"pointer {ptr!r} not found: {exc}"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        # Restricted builtins. Whitelisted utility functions only — no
        # I/O, no introspection, no dynamic code (eval/exec/compile).
        safe_builtins = {
            n: getattr(__builtins__, n, None) if not isinstance(__builtins__, dict)
            else __builtins__.get(n)
            for n in (
                "abs", "all", "any", "bool", "dict", "divmod", "enumerate",
                "filter", "float", "frozenset", "getattr", "hasattr", "hash",
                "int", "isinstance", "issubclass", "iter", "len", "list",
                "map", "max", "min", "next", "ord", "pow", "print", "range",
                "repr", "reversed", "round", "set", "slice", "sorted", "str",
                "sum", "tuple", "type", "zip", "True", "False", "None",
            )
        }
        # Filter out None entries (in case __builtins__ varies between
        # interpreters). Also strip dunders defensively.
        safe_builtins = {k: v for k, v in safe_builtins.items() if v is not None}
        ns: dict[str, Any] = {
            "__builtins__": safe_builtins,
            "data": copy.deepcopy(target),
            "json": json,
        }
        # Auto-detect single expression vs statements. ast.parse to
        # decide: pure-expression → eval, statements → exec all-but-last
        # with eval of trailing expression (Jupyter-cell semantics), or
        # fall back to ``result`` variable when no trailing expression.
        import ast as _ast
        try:
            tree = _ast.parse(code, mode="exec")
        except SyntaxError as exc:
            return {
                "content": json.dumps(
                    {
                        "error": f"python syntax error: {exc.msg} (line {exc.lineno})",
                        "code": code[:200],
                    },
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        try:
            if (
                len(tree.body) == 1
                and isinstance(tree.body[0], _ast.Expression.__bases__[0])
                and isinstance(tree.body[0], _ast.Expr)
            ):
                # Single statement that IS an Expression (eval-able).
                ret = eval(  # noqa: S307 — sandboxed
                    compile(_ast.Expression(tree.body[0].value), "<python tool>", "eval"),
                    ns,
                )
            elif tree.body and isinstance(tree.body[-1], _ast.Expr):
                # Multi-statement with trailing expression — Jupyter-cell
                # semantics: run preceding statements, eval last for return.
                head = _ast.Module(body=tree.body[:-1], type_ignores=[])
                tail = _ast.Expression(tree.body[-1].value)
                exec(  # noqa: S102 — sandboxed
                    compile(head, "<python tool>", "exec"), ns,
                )
                ret = eval(  # noqa: S307
                    compile(tail, "<python tool>", "eval"), ns,
                )
            else:
                # Pure statements — require ``result`` convention.
                exec(compile(tree, "<python tool>", "exec"), ns)  # noqa: S102
                if "result" not in ns:
                    return {
                        "content": json.dumps(
                            {
                                "error": "python: code with no trailing "
                                         "expression must assign to "
                                         "``result`` (e.g. ``result = [...]``)",
                                "code": code[:200],
                            },
                            ensure_ascii=False,
                        ),
                        "applied": [],
                    }
                ret = ns["result"]
        except Exception as exc:
            return {
                "content": json.dumps(
                    {
                        "error": f"python runtime error: {type(exc).__name__}: {exc}",
                        "code": code[:200],
                    },
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        # Serialize result. Fallback to repr for non-JSON-serializable
        # values (set, tuple, custom objects).
        try:
            return {"content": _summarize_for_query(ret, max_chars=1500), "applied": []}
        except Exception:
            return {
                "content": json.dumps(
                    {"value_repr": repr(ret)[:1500], "note": "result not JSON-serializable"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }

    def _javascript(self, args: dict) -> dict:
        """Execute JavaScript code against the graph (or scoped subtree)
        via embedded V8 (py-mini-racer).

        Counterpart to ``_python`` for jqBench-style comparison. The
        graph is JSON-marshalled into V8 as the global ``data``;
        the eval'd value (single expression) or a ``result`` variable
        (multi-statement) is JSON-marshalled back.

        Mutations to ``data`` inside V8 don't reach ``self.graph``
        because the JSON round-trip is one-way.
        """
        code = args.get("code")
        if not code or not isinstance(code, str):
            return {
                "content": json.dumps(
                    {"error": "javascript requires non-empty 'code' (string)"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        ptr = (args.get("pointer") or "").rstrip("/")
        try:
            target = _resolve(self.graph, ptr) if ptr else self.graph
        except (KeyError, IndexError, ValueError) as exc:
            return {
                "content": json.dumps(
                    {"error": f"pointer {ptr!r} not found: {exc}"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        try:
            from py_mini_racer import MiniRacer, JSEvalException, JSParseException
        except ImportError:
            return {
                "content": json.dumps(
                    {"error": "mini-racer not installed; javascript tool unavailable"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        # Fresh isolate per call — no persistent state. ~5-10ms cost,
        # negligible vs LLM round-trip latency.
        ctx = MiniRacer()
        # Marshal graph subset as the global ``data``. JSON round-trip
        # gives us a deep-copied snapshot (mutations stay in V8).
        try:
            ctx.eval(f"globalThis.data = {json.dumps(target, ensure_ascii=False)};")
        except (JSEvalException, JSParseException) as exc:
            return {
                "content": json.dumps(
                    {"error": f"javascript graph-marshal failed: {exc}"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        # V8 returns the value of the last completion-yielding statement
        # in a script — naturally giving Jupyter-cell semantics. Single
        # expression, multi-statement with trailing expression, and
        # ``result = ...`` (assignment expression returns its rhs) all
        # work without wrapping. No ``result`` convention needed.
        try:
            ret = ctx.eval(code)
        except (JSParseException, JSEvalException) as exc:
            return {
                "content": json.dumps(
                    {
                        "error": f"javascript runtime error: {exc}",
                        "code": code[:200],
                    },
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        # mini-racer returns lazy proxies for objects/arrays; convert
        # them to plain Python via recursive walk so json.dumps works.
        ret = _js_to_python(ret)
        return {"content": _summarize_for_query(ret, max_chars=1500), "applied": []}

    def _list_str_remove(self, args: dict, req: PatchRequest) -> dict:
        """Strip every occurrence of ``value`` from a list[str] at ``pointer``.

        Built for the dangling-id-reference pattern: when an item gets
        removed elsewhere (e.g. /key_events/N), other items' id-list
        slots (prerequisite_event_ids, assigned_event_ids) still
        reference the removed id. set_field with whole-list re-emit
        was the only prior path and reliably caused the LLM to abandon
        after 5 read-only steps, since "carry every other entry verbatim
        but drop one specific string" is awkward to encode as a list
        replacement intent.

        The op resolves the list at call time, builds an indexed-remove
        chain (highest index first so subsequent removes don't shift),
        and runs through ``_apply_ops`` for uniform scope guard / undo
        bookkeeping. No-op (returns ok with removed=0) when the value
        isn't present — the LLM's intent was idempotent regardless.
        """
        ptr = (args.get("pointer", "") or "").rstrip("/")
        value = args.get("value")
        if not ptr or value is None:
            return {
                "content": json.dumps(
                    {"error": "list_str_remove requires 'pointer' and 'value'"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        if not isinstance(value, str):
            return {
                "content": json.dumps(
                    {"error": f"list_str_remove 'value' must be a string, got {type(value).__name__}"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        # Scope guard mirrors _list_where: pointer must be inside (or an
        # ancestor of) target_pointer, otherwise count as scope rejection.
        target = req.target_pointer
        ptr_is_ancestor_of_target = bool(target) and target.startswith(ptr + "/")
        if not (
            self._op_in_scope({"path": ptr}, target) or ptr_is_ancestor_of_target
        ):
            state = self._loop_state
            if state:
                state["scope_rejections"] = state.get("scope_rejections", 0) + 1
                paths = state.setdefault("out_of_scope_paths", [])
                if len(paths) < 10:
                    paths.append(ptr)
            return {
                "content": json.dumps(
                    {"error": f"pointer {ptr!r} is outside target_pointer "
                              f"{target!r}; this requirement only allows "
                              f"edits inside that subtree"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        try:
            target_list = _resolve(self.graph, ptr)
        except (KeyError, IndexError, ValueError) as exc:
            return {
                "content": json.dumps(
                    {"error": f"list at {ptr!r} not found: {exc}"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        if not isinstance(target_list, list):
            return {
                "content": json.dumps(
                    {"error": f"target at {ptr!r} is not a list "
                              f"({type(target_list).__name__}); "
                              f"list_str_remove only operates on list[str]"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        bad_types = [type(it).__name__ for it in target_list if not isinstance(it, str)]
        if bad_types:
            return {
                "content": json.dumps(
                    {"error": f"list at {ptr!r} contains non-string elements "
                              f"({bad_types[:3]}); list_str_remove only operates "
                              f"on list[str]. Use list_remove_where with key_field "
                              f"for list-of-dicts."},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        # Find indices to drop. Build remove ops in reverse-index order so
        # each removal doesn't shift subsequent target indices.
        to_remove = [i for i, it in enumerate(target_list) if it == value]
        if not to_remove:
            # Idempotent no-op — explicit ok signal so the LLM doesn't
            # interpret silence as "still need to retry".
            return {
                "content": json.dumps(
                    {"ok": True, "removed": 0,
                     "note": f"value {value!r} not present in list at {ptr!r}; nothing to do"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        ops = [{"op": "remove", "path": f"{ptr}/{i}"} for i in reversed(to_remove)]
        result = self._apply_ops(ops, req)
        # Decorate the response so the LLM sees an explicit removed-count.
        try:
            payload = json.loads(result["content"])
            if isinstance(payload, dict):
                payload["removed"] = len(to_remove)
                result["content"] = json.dumps(payload, ensure_ascii=False)
        except (json.JSONDecodeError, KeyError):
            pass
        return result

    def _list_move_at(self, args: dict, req: PatchRequest) -> dict:
        """Move one list element from ``from_index`` to ``to_index``.

        Why a dedicated op: the only prior path was ``remove`` +
        ``insert_at``, which (a) tripped the scope guard twice for one
        logical intent and (b) made the LLM compute the post-removal
        destination index, a known foot-gun. ``to_index`` here is the
        FINAL position the moved item occupies in the result list —
        e.g. moving 0→3 in [a,b,c,d,e] yields [b,c,d,a,e]. The helper
        translates that to the indexed remove + add pair internally.
        """
        ptr = (args.get("pointer", "") or "").rstrip("/")
        from_idx = args.get("from_index")
        to_idx = args.get("to_index")
        if not ptr or from_idx is None or to_idx is None:
            return {
                "content": json.dumps(
                    {"error": "list_move_at requires 'pointer', 'from_index', and 'to_index'"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        try:
            from_idx = int(from_idx)
            to_idx = int(to_idx)
        except (TypeError, ValueError):
            return {
                "content": json.dumps(
                    {"error": f"list_move_at indices must be integers; got "
                              f"from={from_idx!r}, to={to_idx!r}"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        # Scope guard parity with other _list_* helpers.
        target = req.target_pointer
        ptr_is_ancestor_of_target = bool(target) and target.startswith(ptr + "/")
        if not (
            self._op_in_scope({"path": ptr}, target) or ptr_is_ancestor_of_target
        ):
            state = self._loop_state
            if state:
                state["scope_rejections"] = state.get("scope_rejections", 0) + 1
                paths = state.setdefault("out_of_scope_paths", [])
                if len(paths) < 10:
                    paths.append(ptr)
            return {
                "content": json.dumps(
                    {"error": f"pointer {ptr!r} is outside target_pointer "
                              f"{target!r}; this requirement only allows "
                              f"edits inside that subtree"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        try:
            target_list = _resolve(self.graph, ptr)
        except (KeyError, IndexError, ValueError) as exc:
            return {
                "content": json.dumps(
                    {"error": f"list at {ptr!r} not found: {exc}"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        if not isinstance(target_list, list):
            return {
                "content": json.dumps(
                    {"error": f"target at {ptr!r} is not a list "
                              f"({type(target_list).__name__}); "
                              f"list_move_at only operates on lists"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        n = len(target_list)
        if not (0 <= from_idx < n) or not (0 <= to_idx < n):
            return {
                "content": json.dumps(
                    {"error": f"index out of range: list at {ptr!r} has length {n}, "
                              f"got from={from_idx}, to={to_idx}"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        if from_idx == to_idx:
            return {
                "content": json.dumps(
                    {"ok": True, "moved": 0, "note": "from_index == to_index — no-op"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        # to_idx is the FINAL position. After popping from_idx, the list
        # has length n-1 and indices ≥ from_idx have shifted down by 1.
        # The insert position in the popped list that yields the desired
        # final position is to_idx itself when to_idx ≤ from_idx, and
        # to_idx (in the popped list, which now indexes one less) when
        # to_idx > from_idx — but since the popped list is what the
        # insert acts on, and final position == post-insert index, the
        # correct post-pop insert index is just to_idx in both cases
        # (it lands at exactly to_idx in the result regardless of
        # direction, because pop+insert is commutative around the gap).
        item = target_list[from_idx]
        ops = [
            {"op": "remove", "path": f"{ptr}/{from_idx}"},
            {"op": "add", "path": f"{ptr}/{to_idx}", "value": item},
        ]
        result = self._apply_ops(ops, req)
        try:
            payload = json.loads(result["content"])
            if isinstance(payload, dict):
                payload["moved"] = 1
                payload["from_index"] = from_idx
                payload["to_index"] = to_idx
                result["content"] = json.dumps(payload, ensure_ascii=False)
        except (json.JSONDecodeError, KeyError):
            pass
        return result

    # diff / undo ------------------------------------------------------

    def _diff(self) -> dict:
        """Render before/after at every path the loop has patched so far.

        Dedupes by path in op order — if the LLM patched ``/a/b`` twice,
        the diff lists ``/a/b`` once with the first-call's "before" and
        the latest "after". Paths created by an ``add`` show ``before:
        "<not present initially>"``. Paths that were ``remove``'d show
        ``after: "<removed>"``.
        """
        state = self._loop_state
        ops: list[dict] = state.get("applied_ops") or []
        if not ops:
            return {
                "content": json.dumps(
                    {"changed": False, "note": "no ops applied yet"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        initial = state.get("initial_snapshot") or {}
        seen: set[str] = set()
        items: list[dict] = []
        for op in ops:
            path = op.get("path", "")
            if path in seen:
                continue
            seen.add(path)
            try:
                before: Any = _resolve(initial, path)
            except (KeyError, IndexError, ValueError):
                before = "<not present initially>"
            try:
                after: Any = _resolve(self.graph, path)
            except (KeyError, IndexError, ValueError):
                after = "<removed>"
            items.append({"path": path, "before": before, "after": after})

        payload = {
            "changed": True,
            "n_ops": len(ops),
            "n_paths": len(items),
            "diff": items,
        }
        return {"content": _truncate_for_prompt(payload, 4000), "applied": []}

    def _undo(self, args: dict) -> dict:
        """Roll back the trailing ``count`` ops (or all when ``all=true``).

        Uses the per-op pre-snapshot stack: restoring to
        ``pre_op_snapshots[-count]`` reverts both the deepest op and
        every op layered on top, in one shot. Both stacks are then
        truncated so subsequent diffs / convergence checks see the
        post-undo state.
        """
        state = self._loop_state
        applied: list[dict] = state.get("applied_ops") or []
        snapshots: list[dict] = state.get("pre_op_snapshots") or []
        n_total = len(applied)
        if n_total == 0:
            return {
                "content": json.dumps(
                    {"undone": 0, "note": "no ops to undo"},
                    ensure_ascii=False,
                ),
                "applied": [],
            }
        if args.get("all"):
            count = n_total
        else:
            try:
                count = int(args.get("count", 1) or 1)
            except (TypeError, ValueError):
                count = 1
        count = max(1, min(count, n_total))

        snapshot = snapshots[-count]
        self.graph.clear()
        self.graph.update(copy.deepcopy(snapshot))
        undone_ops = applied[-count:]
        del applied[-count:]
        del snapshots[-count:]

        payload = {
            "undone": count,
            "remaining_ops": len(applied),
            "undone_ops": undone_ops,
        }
        return {"content": _truncate_for_prompt(payload, 2000), "applied": []}

    def _apply_ops(self, ops: list[dict], req: PatchRequest) -> dict:
        """Apply a list of ops with scope guard + per-op error capture.
        Used by both `patch` and the narrow tools (`set_field`, `list_*`).

        Each successful op also pushes a pre-op deepcopy of ``self.graph``
        onto ``self._loop_state["pre_op_snapshots"]`` so ``undo`` can
        restore prior state. We snapshot inside the per-op try/except so
        rejected ops don't leave dangling snapshots.

        Returns {'content': str, 'applied': list[op]}.
        """
        state = self._loop_state
        applied: list[dict] = []
        errors: list[str] = []
        notes: list[str] = []
        # P25: for multi-op patches, defer per-op patch_evaluator. The
        # evaluator sees only the marginal effect of each op — for
        # additive intents ("add A and B") each op individually scores
        # "partial" and used to be rolled back, deadlocking the loop.
        # Instead we evaluate the patch as a whole post-application.
        # Snapshot taken before any op in the batch lands.
        is_multi_op = len(ops) > 1
        pre_batch_snapshot = copy.deepcopy(self.graph) if is_multi_op else None
        for i, op in enumerate(ops):
            try:
                # P14: any per-op pre-check (translate, path-finder,
                # scope guard, _split_pointer in helpers) may raise on
                # malformed input (e.g. bare-string paths like
                # 'sensory' that don't start with '/'). Catch broadly
                # so a single bad op becomes a per-op rejection rather
                # than killing the whole patch loop.
                translated, note, hard_error = _translate_tool_op_to_rfc(op)
                if hard_error:
                    # Tool name used as op that we can't auto-fix — record as
                    # a regular rejection (NOT counted as scope-rejection,
                    # because the LLM's intent might be perfectly in scope).
                    errors.append(f"op[{i}]: {hard_error}")
                    continue
                if note:
                    notes.append(f"op[{i}]: {note}")
                op = translated
                # Path-finder sub-agent (auto-invoked, before scope check).
                # Confirms or corrects the proposed path. May rewrite op["path"].
                # Skipped for cascade ops — their paths are computed from a
                # deterministic graph scan (P47), not LLM proposal, so
                # asking an LLM to second-guess them just burns tokens
                # and risks rewriting a verified-correct pointer.
                if not op.get("_cascade"):
                    finder_note = self._invoke_path_finder(i, op, req, state)
                    if finder_note:
                        notes.append(f"op[{i}]: {finder_note}")
                # ``_cascade``: True is set internally when a higher-level
                # tool (currently ``list_remove_where``) auto-emits referential
                # cleanup ops alongside its primary remove. The cleanup ops
                # touch paths outside the original target_pointer by design
                # (they reach into other items' id-list slots) and would
                # otherwise be rejected as scope escapes. We trust the
                # cascade because it is BUILT BY THE PATCHER ITSELF, not
                # the LLM, on the basis of the now-removed item's id.
                if not op.get("_cascade") and not self._op_in_scope(op, req.target_pointer):
                    errors.append(
                        f"op[{i}] path {op.get('path')!r} outside target_pointer "
                        f"{req.target_pointer!r}"
                    )
                    if state:
                        state["scope_rejections"] = state.get("scope_rejections", 0) + 1
                        paths = state.setdefault("out_of_scope_paths", [])
                        if len(paths) < 10:
                            paths.append(op.get("path", ""))
                    continue
                # P15: enum/const pre-flight against root_schema. Stops
                # Literal-typed field violations from poisoning the
                # whole iteration via post-iteration pydantic validation.
                schema_err = self._validate_op_value_against_schema(op, req)
                if schema_err:
                    errors.append(f"op[{i}] schema-rejected at {op.get('path')!r}: {schema_err}")
                    continue
                snapshot = copy.deepcopy(self.graph)
                _apply_op(self.graph, op)
            except Exception as exc:
                errors.append(f"op[{i}] {op.get('op')} {op.get('path')!r}: {exc}")
                continue
            applied.append(op)
            # Track on loop state so diff/undo can see it. Skip when
            # state is empty (which only happens if _apply_ops is somehow
            # invoked outside an _apply_one loop — defensive only).
            if state:
                state["applied_ops"].append(op)
                state["pre_op_snapshots"].append(snapshot)
                # P11: any new in-scope landing invalidates prior diff
                # verification — must re-audit before declaring done.
                state["diff_verified_since_last_patch"] = False
            # P25: skip per-op evaluator for multi-op patches (handled
            # once after the loop). For single-op patches, keep per-op
            # evaluation but only roll back on "off-target".
            # "partial" means real progress; "no-op" verdicts have a
            # high false-positive rate (the evaluator misses small but
            # real text additions like adding a substring), so trust
            # the change and let the convergence gate's cumulative
            # re-eval (P26) catch genuine no-progress.
            if is_multi_op:
                continue
            eval_note, eval_verdict, eval_conf = self._invoke_patch_evaluator(
                i, op, req, snapshot, state,
            )
            if eval_note:
                notes.append(f"op[{i}]: {eval_note}")
            should_rollback = (
                eval_verdict == "off-target"
                and eval_conf in ("high", "medium")
            )
            if should_rollback:
                # Restore graph to pre-op snapshot and untrack.
                self.graph.clear()
                self.graph.update(snapshot)
                if state:
                    if state["applied_ops"] and state["applied_ops"][-1] is op:
                        state["applied_ops"].pop()
                    if state["pre_op_snapshots"]:
                        state["pre_op_snapshots"].pop()
                    # P19: track rollbacks per step so the auto-converge
                    # path can refuse to flip addressed=True when the
                    # evaluator just rejected our work.
                    state["evaluator_rollbacks"] = state.get("evaluator_rollbacks", 0) + 1
                if applied and applied[-1] is op:
                    applied.pop()
                errors.append(
                    f"op[{i}] rolled back by patch_evaluator: "
                    f"{eval_verdict} ({eval_conf}) — emit a different op"
                )
        # P25: batch evaluation for multi-op patches. Evaluate the whole
        # patch's net effect against the intent, not each op in isolation.
        if is_multi_op and applied:
            synthetic_op = {
                "op": "batch",
                "path": req.target_pointer,
                "n_ops": len(applied),
            }
            eval_note, eval_verdict, eval_conf = self._invoke_patch_evaluator(
                0, synthetic_op, req, pre_batch_snapshot, state,
            )
            if eval_note:
                notes.append(f"batch: {eval_note}")
            # Roll back the entire batch only when the evaluator says
            # the whole patch missed the intent (off-target/no-op).
            # "partial" is fine — the LLM made progress; let convergence
            # gate ask for more.
            if eval_verdict in ("no-op", "off-target") and eval_conf in ("high", "medium"):
                self.graph.clear()
                self.graph.update(pre_batch_snapshot)
                if state:
                    n_to_pop = len(applied)
                    for _ in range(n_to_pop):
                        if state["applied_ops"]:
                            state["applied_ops"].pop()
                        if state["pre_op_snapshots"]:
                            state["pre_op_snapshots"].pop()
                    state["evaluator_rollbacks"] = state.get("evaluator_rollbacks", 0) + 1
                applied.clear()
                errors.append(
                    f"batch rolled back by patch_evaluator: "
                    f"{eval_verdict} ({eval_conf}) — try a different approach"
                )
        payload: dict[str, Any] = {"applied": len(applied), "rejected": errors}
        if notes:
            payload["notes"] = notes
        content = json.dumps(payload, ensure_ascii=False)
        return {"content": content, "applied": applied}

    # ── Sub-agent integrations ────────────────────────────────────────

    def _validate_request(
        self,
        req: PatchRequest,
        traces_out: list,
    ) -> PatchResult | None:
        """Run request_validator sub-agent. If verdict refuses the
        requirement (target_missing / too_broad / already_satisfied),
        short-circuit with an early PatchResult. Otherwise return None
        and let the main loop run."""
        from json_correction_loop.request_validator import (
            validate_request, ValidateRequestResult,
        )
        try:
            r: ValidateRequestResult = validate_request(
                self.graph, req, client=self.client, model=self.model,
            )
        except Exception as exc:
            # P16: record exception in trace so the inspector shows
            # *something* happened — silent failures looked like
            # "sub-agent never ran" in production audit.
            logger.warning("request_validator raised: %s — proceeding without it", exc)
            traces_out.append(SubAgentTrace(
                kind="request_validator",
                verdict="error",
                confidence="low",
                rationale=f"raised: {exc!r}"[:300],
            ))
            return None
        traces_out.append(SubAgentTrace(
            kind="request_validator",
            verdict=r.verdict,
            confidence=r.confidence,
            rationale=r.rationale,
            calls=[{"tool": c.tool, "args": c.args, "result_summary": c.result_summary} for c in r.calls],
        ))
        log_verbose(f"    [dim]· request_validator → {r.verdict} ({r.confidence}) {r.rationale[:80]}[/dim]")
        if r.verdict == "valid" or r.confidence == "low":
            return None
        if r.verdict == "already_satisfied":
            return PatchResult(
                requirement_id=req.requirement_id,
                addressed=True,
                target_pointer=req.target_pointer,
                intent=req.intent,
                applied_ops=[],
                reason=f"request_validator: already_satisfied — {r.rationale}",
                llm_calls=r.llm_calls,
                calls=[],
                subagent_traces=list(traces_out),
            )
        # Refusal verdicts — all surface as critic_error so the upstream
        # critic issue gets marked invalidated and isn't re-flagged next
        # round. Each carries its own kind so the inspector can colour-code.
        refuse_kinds = {
            "spurious": "spurious_critic",
            "bad_suggestion": "bad_suggestion",
            "target_missing": "unfulfillable_intent",
            "too_broad": "unfulfillable_intent",
        }
        if r.verdict in refuse_kinds:
            return PatchResult(
                requirement_id=req.requirement_id,
                addressed=False,
                target_pointer=req.target_pointer,
                intent=req.intent,
                applied_ops=[],
                reason=f"request_validator: {r.verdict} — {r.rationale}",
                llm_calls=r.llm_calls,
                calls=[],
                critic_error=CriticErrorRecord(
                    kind=refuse_kinds[r.verdict],
                    summary=f"validator: {r.verdict} — {r.rationale}",
                ),
                subagent_traces=list(traces_out),
            )
        # ambiguous / unknown: let the loop try.
        return None

    def _invoke_path_finder(
        self, op_idx: int, op: dict, req: PatchRequest, state: dict,
    ) -> str | None:
        """Run path_finder on this op. Rewrites op['path'] in place if the
        finder returns a different (high/medium-confidence) pointer.
        Returns a one-line note for inclusion in the tool result, or None
        when the finder confirmed the proposal as-is."""
        from json_correction_loop.path_finder import find_target, FindTargetResult
        proposed = op.get("path", "")
        if not proposed:
            return None
        try:
            r: FindTargetResult = find_target(
                self.graph,
                intent=req.intent,
                proposed_path=proposed,
                target_pointer=req.target_pointer,
                op_kind=op.get("op", "replace"),
                proposed_value=op.get("value"),
                client=self.client,
                model=self.model,
            )
        except Exception as exc:
            logger.warning("path_finder raised: %s — using proposal as-is", exc)
            if state is not None:
                state.setdefault("op_subagent_traces", []).append(SubAgentTrace(
                    kind="path_finder",
                    verdict="error",
                    confidence="low",
                    rationale=f"raised: {exc!r}"[:300],
                    op_index=op_idx,
                    proposed_path=proposed,
                    final_path=proposed,
                ))
            return None
        final_path = r.pointer if r.pointer is not None else proposed
        if state is not None:
            state.setdefault("op_subagent_traces", []).append(SubAgentTrace(
                kind="path_finder",
                verdict=r.confidence,
                confidence=r.confidence,
                rationale=r.rationale,
                op_index=op_idx,
                proposed_path=proposed,
                final_path=final_path,
                calls=[{"tool": c.tool, "args": c.args, "result_summary": c.result_summary} for c in r.calls],
            ))
        arrow = "→" if (r.pointer and r.pointer != proposed) else "✓"
        log_verbose(f"    [dim]· path_finder {arrow} {r.pointer or '<none>'} ({r.confidence}) {r.rationale[:60]}[/dim]")
        if r.pointer is None and r.confidence != "low":
            # Should not happen — finder said no pointer with non-low conf.
            return None
        if r.pointer is None:
            # Couldn't locate. Leave proposal alone; scope guard will catch
            # if it's invalid.
            return None
        if r.pointer == proposed:
            return None
        # Rewrite the op's path with the corrected pointer.
        op["path"] = r.pointer
        return f"path_finder corrected {proposed!r} → {r.pointer!r} ({r.confidence}: {r.rationale[:80]})"

    def _invoke_patch_evaluator(
        self, op_idx: int, op: dict, req: PatchRequest,
        graph_before: dict, state: dict,
    ) -> tuple[str | None, str, str]:
        """Run patch_evaluator after a successful op.

        Returns ``(note, verdict, confidence)``:
        - ``note``: one-line warning for the tool result when verdict
          isn't ``addressed`` (so the patcher LLM sees the warning),
          else ``None``.
        - ``verdict``: one of ``addressed | partial | off-target |
          no-op | error``.
        - ``confidence``: ``high | medium | low``.

        The 3-tuple is what P18 needs in ``_apply_ops`` to decide
        whether to rollback the op when the evaluator rejects it.
        """
        from json_correction_loop.patch_evaluator import (
            evaluate_patch, EvaluatePatchResult,
        )
        try:
            r: EvaluatePatchResult = evaluate_patch(
                graph_before, self.graph,
                intent=req.intent,
                patched_path=op.get("path", ""),
                op_kind=op.get("op", "replace"),
                client=self.client, model=self.model,
            )
        except Exception as exc:
            logger.warning("patch_evaluator raised: %s — accepting op", exc)
            if state is not None:
                state.setdefault("op_subagent_traces", []).append(SubAgentTrace(
                    kind="patch_evaluator",
                    verdict="error",
                    confidence="low",
                    rationale=f"raised: {exc!r}"[:300],
                    op_index=op_idx,
                ))
            return None, "error", "low"
        if state is not None:
            state.setdefault("op_subagent_traces", []).append(SubAgentTrace(
                kind="patch_evaluator",
                verdict=r.verdict,
                confidence=r.confidence,
                rationale=r.rationale,
                op_index=op_idx,
                calls=[{"tool": c.tool, "args": c.args, "result_summary": c.result_summary} for c in r.calls],
            ))
        log_verbose(f"    [dim]· patch_evaluator: {r.verdict} ({r.confidence}) {r.rationale[:80]}[/dim]")
        if r.verdict == "addressed":
            return None, r.verdict, r.confidence
        return (
            f"patch_evaluator: {r.verdict} ({r.confidence}) — {r.rationale[:80]}",
            r.verdict,
            r.confidence,
        )

    def _invoke_template_filler(
        self,
        req: PatchRequest,
        target_value: Any,
        traces_out: list,
    ) -> str:
        """P22: pre-loop fill-template sub-agent.

        Run only when (a) target_value is an empty list/dict, and
        (b) ``req.intent`` carries an enumeration pattern. The
        sub-agent extracts the items from intent and returns a
        complete assembled value. We surface that value as a SEED
        block in the main patcher's user prompt — the patcher LLM
        is free to use it verbatim or override.

        Returns the seed block string (with leading ``\\n`` for
        prompt formatting), or an empty string when the agent didn't
        run or returned low-confidence / unusable output.
        """
        from json_correction_loop.template_filler import (
            fill_template, FillTemplateResult,
            has_enumeration_pattern, is_empty_container,
        )
        if not is_empty_container(target_value):
            return ""
        if not has_enumeration_pattern(req.intent or ""):
            return ""

        # Reference pointers: try sibling entries of the same parent
        # so the filler can match shape (e.g. /plot/character_arcs/X
        # is empty → look at /plot/character_arcs/Y for shape clues).
        ref_ptrs: list[str] = []
        try:
            tokens = _split_pointer(req.target_pointer)
            if len(tokens) >= 1:
                parent_ptr = "/" + "/".join(tokens[:-1])
                parent_val = _resolve(self.graph, parent_ptr) if parent_ptr else self.graph
                last_token = tokens[-1]
                if isinstance(parent_val, dict):
                    for k in list(parent_val.keys())[:3]:
                        if k != last_token:
                            ref_ptrs.append(f"{parent_ptr}/{k}")
        except (KeyError, IndexError, ValueError):
            pass

        # Field schema (item shape if available).
        field_schema: dict | None = None
        if self.root_schema:
            try:
                field_schema = _resolve_schema_at(self.root_schema, req.target_pointer)
            except Exception:
                field_schema = None

        try:
            r: FillTemplateResult = fill_template(
                req.target_pointer,
                target_value,
                req.intent,
                graph=self.graph,
                field_schema=field_schema,
                reference_pointers=ref_ptrs,
                client=self.client,
                model=self.model,
            )
        except Exception as exc:
            logger.warning("template_filler raised: %s — skipping seed", exc)
            traces_out.append(SubAgentTrace(
                kind="template_filler",
                verdict="error",
                confidence="low",
                rationale=f"raised: {exc!r}"[:300],
            ))
            return ""

        traces_out.append(SubAgentTrace(
            kind="template_filler",
            verdict=("seeded" if r.value is not None and r.confidence in ("high", "medium") else "skipped"),
            confidence=r.confidence,
            rationale=r.rationale,
            calls=[{"tool": c.tool, "args": c.args, "result_summary": c.result_summary} for c in r.calls],
        ))
        if r.value is not None and r.confidence in ("high", "medium"):
            log_verbose(f"    [dim]· template_filler seeded {r.item_count} items ({r.confidence}): {r.rationale[:60]}[/dim]")
        else:
            log_verbose(f"    [dim]· template_filler skipped ({r.confidence}): {r.rationale[:60]}[/dim]")

        if r.value is None or r.confidence not in ("high", "medium"):
            return ""

        try:
            seed_json = json.dumps(r.value, ensure_ascii=False, indent=2)
        except Exception:
            return ""
        # Cap to keep prompt compact.
        if len(seed_json) > 4000:
            seed_json = seed_json[:4000] + "\n…(truncated)"
        return (
            "\n# Template-filler seed (sub-agent assembled this from the intent — "
            f"use it verbatim with `replace` on target_pointer if it looks correct, "
            f"or adjust as needed)\n```json\n{seed_json}\n```\n"
        )

    def _cumulative_eval_addressed(self, req: PatchRequest) -> bool:
        """P26: re-evaluate the full applied-ops diff against the intent.

        Per-op patch_evaluator sees only marginal effects. For additive
        intents, each op individually is "partial" even when the cumulative
        effect addresses the intent. Before refusing convergence on
        unresolved partials, ask the evaluator once on the whole
        (initial_snapshot → current_graph) diff.
        """
        from json_correction_loop.patch_evaluator import (
            evaluate_patch, EvaluatePatchResult,
        )
        initial = self._loop_state.get("initial_snapshot")
        if initial is None:
            return False
        try:
            r: EvaluatePatchResult = evaluate_patch(
                initial, self.graph,
                intent=req.intent,
                patched_path=req.target_pointer or "",
                op_kind="batch",
                client=self.client, model=self.model,
            )
        except Exception as exc:
            logger.warning("cumulative patch_evaluator raised: %s", exc)
            return False
        log_verbose(
            f"    [dim]· cumulative patch_evaluator: {r.verdict} "
            f"({r.confidence}) {r.rationale[:80]}[/dim]"
        )
        return r.verdict == "addressed" and r.confidence in ("high", "medium")

    def _unresolved_negative_eval_paths(self) -> list[tuple[str, str]]:
        """P21: per-path latest evaluator verdict. Return paths whose
        most recent verdict is negative (no-op/off-target/partial) at
        high/medium confidence AND no later applied op overwrote them.

        Conservative: a low-confidence negative doesn't gate
        convergence (the LLM might be right and the evaluator unsure).
        Used by the convergence gate before flipping addressed=True.
        """
        traces = self._loop_state.get("op_subagent_traces") or []
        applied = self._loop_state.get("applied_ops") or []
        # Map op_index → (verdict, confidence) for evaluator traces only.
        verdict_by_op: dict[int, tuple[str, str, str]] = {}
        for t in traces:
            if t.kind != "patch_evaluator":
                continue
            if t.op_index is None or t.op_index >= len(applied):
                continue
            path = applied[t.op_index].get("path", "")
            verdict_by_op[t.op_index] = (path, t.verdict, t.confidence)
        # Walk in op_index order so later verdicts on the same path
        # override earlier ones (the model might fix a partial with a
        # subsequent good patch).
        latest_per_path: dict[str, tuple[str, str]] = {}
        for op_idx in sorted(verdict_by_op):
            path, verdict, conf = verdict_by_op[op_idx]
            latest_per_path[path] = (verdict, conf)
        out: list[tuple[str, str]] = []
        for path, (verdict, conf) in latest_per_path.items():
            if verdict in ("no-op", "off-target", "partial") and conf in ("high", "medium"):
                out.append((path, verdict))
        return out

    @staticmethod
    def _op_in_scope(op: dict, target: str) -> bool:
        """Reject ops that try to escape the requirement's target pointer.

        target_pointer "" means the whole graph (rare; only for top-level
        scaffolding requirements). Otherwise the op's path must be == target
        or start with target + "/".
        """
        if not target:
            return True
        path = op.get("path", "")
        return path == target or path.startswith(target + "/")

    def _validate_op_value_against_schema(
        self, op: dict, req: PatchRequest | None = None
    ) -> str | None:
        """P15: pre-flight check that ``op['value']`` satisfies enum/const
        constraints in ``self.root_schema`` at ``op['path']``.

        Catches Pydantic ``Literal``-type violations (e.g. ``structure_hint``
        must be one of N strings) at op-emission time, BEFORE the op
        lands in the graph. Without this, a single bad value ends up in
        the dict, the iteration completes, and the model's full
        re-validation explodes at save time — wiping ALL ops in that
        iteration, including the good ones.

        Returns a short error string when the op should be rejected, or
        ``None`` when the schema imposes no enum/const constraint at
        that path (or the constraint is satisfied, or the path can't be
        resolved against the schema).
        """
        if not self.root_schema:
            return None
        if op.get("op") not in ("add", "replace"):
            return None
        if "value" not in op:
            return None
        path = op.get("path", "")
        try:
            node = _resolve_schema_at(self.root_schema, path)
        except Exception:
            return None
        if not isinstance(node, dict):
            return None
        # For ``add`` to an array (path ending in ``/-`` or ``/<int>``),
        # _resolve_schema_at already descended into ``items`` — the
        # node we have is the item schema. For ``replace`` on a leaf,
        # node is that leaf's schema. Either way, check enum/const here.
        value = op["value"]
        enum = node.get("enum")
        if isinstance(enum, list) and enum:
            if value not in enum:
                # Render allowed values compactly so the LLM (which sees
                # this in the tool result) can self-correct on the next
                # tool call.
                allowed = ", ".join(repr(e) for e in enum[:8])
                if len(enum) > 8:
                    allowed += f", … ({len(enum) - 8} more)"
                return f"value {value!r} not in enum [{allowed}]"
        const = node.get("const")
        if const is not None and value != const:
            return f"value {value!r} != const {const!r}"
        # P20: JSON Schema type checks (extends P15). Catches obvious
        # mismatches like the "subplot end_state expects string but the
        # patcher emitted -1" failure observed in plot.critic traces.
        # We support both single-type and ``["string", "null"]`` forms.
        type_err = _typecheck_json_schema(node, value)
        if type_err:
            # P29: auto-coerce list[str] → ", ".join(...) when schema
            # expects string. Some models persistently emit a JSON list
            # for fields whose schema is `string` (typical_occupants,
            # routine content) and won't self-correct even after the
            # tool result and P27's warning. Coercing in-place lets the
            # patch land instead of bailing on schema rejection.
            schema_types = node.get("type")
            if isinstance(schema_types, str):
                allowed = {schema_types}
            elif isinstance(schema_types, list):
                allowed = set(schema_types)
            else:
                allowed = set()
            if (
                "string" in allowed
                and isinstance(value, list)
                and all(isinstance(x, (str, int, float)) for x in value)
            ):
                # Guard: refuse to coerce when the list looks like a
                # mis-emitted JSON Patch op flattened into a value list
                # (model bug: ``value: ["op", "replace", "path", ...]``).
                # Coercing would write nonsense like "op, replace, path"
                # into the field. Detect by a JSON-Patch keyword in
                # any of the first few entries.
                jp_kw = {"op", "replace", "add", "remove", "path", "from", "test"}
                first_few = {str(x).strip().lower() for x in value[:6]}
                if not (first_few & jp_kw):
                    op["value"] = ", ".join(str(x) for x in value)
                    return None
            # P30: when schema expects string but the LLM emitted a
            # bare int / bool (commonly observed as ``value: 1`` from
            # corrupted tool-call args on weak models), try to recover
            # by extracting the first single-quoted string from the
            # requirement's intent. The intent reliably carries the
            # author's literal target value as ``... '<value>' ...``
            # in the suggestion (e.g. "X를 '대상값'으로 변경").
            if (
                "string" in allowed
                and isinstance(value, (int, bool))
                and req is not None
                and isinstance(req.intent, str)
            ):
                import re
                m = re.search(r"'([^']{2,300})'", req.intent)
                if m:
                    op["value"] = m.group(1)
                    return None
            return type_err
        return None
