"""Path-finder sub-agent.

Auto-invoked by ``SurgicalPatcher._apply_ops`` before each patch op.
DISCOVERS the correct JSON pointer for a patch intent by actively
investigating the graph. Treats the LLM-proposed pointer (and the
critic's target_pointer) as HINTS only — if investigation lands
elsewhere, returns the discovered pointer.

Why a separate agent: searching the graph ("where is the field that
matches this intent?") is a different cognitive task from emitting a
patch ("what value should land there?"). Mixing both in one LLM loop
is the dominant source of wrong-path patches in production traces
(60+ identical critic re-flags on /plot/character_arcs/<x> across
projects). Splitting discovery into a small read-only sub-agent with
a dedicated prompt keeps the main patcher focused on values.

Contract:
- ``find_target(graph, intent, proposed_path, target_pointer, model)``
  runs a short read-only LLM session that investigates the intent
  against the graph and returns the path it concludes is correct.
- Tools available to the sub-agent: ``query``, ``get_schema``,
  ``find_paths``, ``find_values``. NO mutation tools.
- Returns ``FindTargetResult`` with the discovered pointer, a
  confidence band, and the evidence trail (calls made).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from json_correction_loop.llm import TransientLLMError

from json_correction_loop._config import DEFAULT_MODEL

logger = logging.getLogger(__name__)


# ── Public types ────────────────────────────────────────────────────────────


@dataclass
class PathFinderCall:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    result_summary: str = ""


@dataclass
class FindTargetResult:
    """Outcome of one ``find_target`` invocation.

    ``confidence`` is the sub-agent's self-reported certainty:
      - ``"high"``: strong match (verified the field, confident the
        change goes here).
      - ``"medium"``: plausible best guess but not fully verified.
      - ``"low"``: couldn't locate confidently — caller should refuse
        the op and surface the intent for human inspection.
    ``rationale`` is one short line explaining the choice.
    ``calls`` is the read-only tool trail (for inspector / audit).
    """
    pointer: str | None
    confidence: str  # "high" | "medium" | "low"
    rationale: str
    calls: list[PathFinderCall] = field(default_factory=list)
    llm_calls: int = 0


# ── Tool schemas (read-only subset) ─────────────────────────────────────────


_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "query",
            "description": (
                "Read the value at a JSON Pointer. Use this to inspect "
                "candidate fields and confirm they hold what the intent "
                "describes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pointer": {"type": "string", "description": "RFC 6901 JSON Pointer."},
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
                "Inspect the schema (field names + descriptions) at a "
                "pointer. Useful when you don't know which sub-field "
                "matches the intent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pointer": {"type": "string"},
                },
                "required": ["pointer"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_paths",
            "description": (
                "Search the graph for JSON Pointers whose dict-key "
                "segments contain ``keyword``. Use when you know a "
                "field name from the intent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
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
                "Search the graph for leaf-string values containing "
                "``keyword``. Use when the intent quotes a known phrase "
                "from the data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "jq",
            "description": (
                "Run a jq expression against the graph (or a subtree). "
                "Use this for bulk projection / filtering when the "
                "intent points at many items at once. Examples:\n"
                "  • `.key_events | map({id, summary: .summary[:60]})` "
                "→ id+brief for every event in one call\n"
                "  • `.key_events | map(select(.act_number == 2)) | "
                "map(.id)` → ids of every act-2 event\n"
                "  • `.spaces | keys` → list every space label.\n"
                "Read-only. Errors come back with the offending "
                "expression so you can correct on the next call."
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
                            "(empty = whole graph)."
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
            "name": "answer",
            "description": (
                "Final answer. Call exactly once when you've identified "
                "the pointer (or determined the intent isn't locatable)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pointer": {
                        "type": ["string", "null"],
                        "description": "The confirmed/corrected JSON Pointer, or null if not locatable.",
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "rationale": {
                        "type": "string",
                        "description": "One short line explaining the choice.",
                    },
                },
                "required": ["pointer", "confidence", "rationale"],
            },
        },
    },
]


_SYSTEM_PROMPT = """\
You are a path-finder sub-agent for a JSON patcher.

# Your job
DISCOVER the JSON Pointer where a patch should land. The patcher LLM
gave you a ``proposed_path`` and the critic gave a ``target_pointer``,
but treat BOTH as hints — they are frequently wrong. Your authority
is the INTENT (and the actual graph). Investigate, then call
``answer`` with the pointer YOU conclude is correct.

# How to investigate
1. Read the intent. Identify the entities/fields it names — character
   names, ids, field names, quoted phrases. These are your search
   keys.
2. Use ``find_paths`` (key search) or ``find_values`` (string search)
   on those keys to enumerate candidate locations IN THE GRAPH. Don't
   trust the proposed/target paths until you've checked.
3. ``query`` the candidates to confirm one holds what the intent
   describes.
4. ``get_schema`` when you need to know what fields a parent supports
   (e.g. for an ``add`` op into an array or a sibling key).
5. Pick the path most precisely matching the intent's *operation*:
   - "delete X" → the path of X itself, not its parent.
   - "rename A → B" → A's path.
   - "change X.foo to Y" → the path of ``foo`` under X.
   - "add Z to list" → the list's path (op=add will append).
6. Call ``answer(pointer, confidence, rationale)``.

# Hint disagreement
- proposed_path and target_pointer often AGREE. When they do, still
  verify with ``query`` before accepting.
- When they DISAGREE with each other or with what you find, the
  intent + actual graph wins. State the disagreement in rationale.

# Confidence rules
- ``high``: you read the value at the pointer; it precisely matches
  the intent's described entity/field/value.
- ``medium``: the pointer's location/shape fits but verification was
  partial.
- ``low``: couldn't locate a fitting target. Pass ``pointer=null``.

# Scope hint
- ``target_pointer`` is a SCOPE HINT from the critic — usually right,
  occasionally too narrow or wrong. Prefer staying at-or-under it,
  but if the intent clearly needs a sibling/cousin path, use that
  and explain in rationale (the patcher's scope guard will widen).

# Brevity
- 2–5 tool calls is typical. NEVER mutate. NEVER call ``patch`` or
  ``set_field`` (not available).
"""


_USER_TMPL = """\
# Patch about to land
- intent: {intent}
- proposed op: {op_kind}

# Hints (treat as starting points, NOT authoritative)
- critic's target_pointer (scope hint): {scope}
- patcher's proposed_path (op hint):    {proposed}

# Proposed value preview
{value_preview}

# Current value at proposed_path (if it resolves)
{existing_preview}

INVESTIGATE the graph and DISCOVER the correct pointer. Call ``answer``
when you've concluded.
"""


# ── Public entry ────────────────────────────────────────────────────────────


def find_target(
    graph: dict,
    intent: str,
    proposed_path: str,
    target_pointer: str = "",
    op_kind: str = "replace",
    proposed_value: Any = None,
    *,
    client: Any = None,
    model: str | None = None,
    max_steps: int = 7,
) -> FindTargetResult:
    """Discover the correct pointer for ``intent`` via active investigation.

    Uses the ``SurgicalPatcher`` read-only tool implementations
    (re-imported here to avoid duplication). The sub-agent runs its
    own LLM loop with a small read-only toolset and ends by calling
    ``answer`` with the final pointer + confidence.
    """
    # Lazy-import the tool helpers to avoid a circular module load.
    from json_correction_loop.patcher import (
        _resolve,
        _find_paths_by_key,
        _find_values,
        _resolve_schema_at,
        _summarize_for_query,
        _trim_schema_for_llm,
    )

    if os.environ.get("JCL_PATH_FINDER_ENABLED", "1").strip() not in ("1", "true", "True", "yes"):
        # Disabled — pass through the proposal as-is.
        return FindTargetResult(
            pointer=proposed_path,
            confidence="medium",
            rationale="path_finder disabled — passing proposal through",
        )

    # Hint fast-path: skip the LLM agent loop when patcher and critic
    # agree on a scalar pointer whose key/value substring-matches an
    # intent token. Conservative on purpose — false-positives here cause
    # off-target patches downstream. Disable with
    # ``JCL_PATH_FINDER_FASTPATH=0``.
    if os.environ.get("JCL_PATH_FINDER_FASTPATH", "1").strip() in ("1", "true", "True", "yes"):
        fp = _hint_fastpath(graph, intent, proposed_path, target_pointer)
        if fp is not None:
            return fp

    chosen_model = (
        model
        or os.environ.get("JCL_PATH_FINDER_MODEL", "").strip()
        or DEFAULT_MODEL
    )
    if client is None:
        raise ValueError(
            "path_finder requires an LLMClient (chat_complete) — got None. "
            "Caller must construct one (e.g. OpenAILLMClient) and pass it."
        )
    cli = client

    # Build a "current value at proposed_path" preview for the prompt.
    try:
        existing = _resolve(graph, proposed_path)
        existing_preview = _summarize_for_query(existing, max_chars=800)
    except (KeyError, IndexError, ValueError):
        existing_preview = "<not present>"

    if proposed_value is not None:
        try:
            value_preview = json.dumps(proposed_value, ensure_ascii=False, indent=2)[:800]
        except Exception:
            value_preview = repr(proposed_value)[:800]
    else:
        value_preview = "<no value (remove op)>"

    user_prompt = _USER_TMPL.format(
        intent=intent or "(none)",
        scope=target_pointer or "/",
        proposed=proposed_path,
        op_kind=op_kind,
        value_preview=value_preview,
        existing_preview=existing_preview,
    )
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    calls: list[PathFinderCall] = []
    llm_calls = 0
    answered: dict | None = None

    for _step in range(max_steps):
        try:
            resp = cli.chat_complete(
                model=chosen_model,
                messages=messages,
                tools=_TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=2048,
                extra=({"reasoning_effort": e} if (e := (os.environ.get("JCL_REASONING_EFFORT", "none").strip() or "none")) and e != "none" else {}),
            )
        except TransientLLMError as exc:
            logger.warning("path_finder transient error: %s", exc)
            return FindTargetResult(
                pointer=proposed_path,
                confidence="low",
                rationale=f"path_finder backend error: {exc!r}",
                calls=calls,
                llm_calls=llm_calls,
            )
        llm_calls += 1
        tool_calls = resp.tool_calls
        assistant_msg: dict = {"role": "assistant", "content": resp.content}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
        messages.append(assistant_msg)

        if not tool_calls:
            # Model went silent without answering — treat as inconclusive.
            return FindTargetResult(
                pointer=proposed_path,
                confidence="low",
                rationale="path_finder produced no tool call",
                calls=calls,
                llm_calls=llm_calls,
            )

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                content = json.dumps({"error": "args must be a JSON object"}, ensure_ascii=False)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
                continue

            if name == "answer":
                # Capture and prepare to exit after this step.
                answered = args
                # Pretend success so the protocol message is consistent.
                content = json.dumps({"ok": True}, ensure_ascii=False)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
                calls.append(PathFinderCall(tool="answer", args=args, result_summary=content[:200]))
                continue

            content = _dispatch_readonly(
                name, args, graph,
                _resolve=_resolve,
                _find_paths_by_key=_find_paths_by_key,
                _find_values=_find_values,
                _resolve_schema_at=_resolve_schema_at,
                _summarize_for_query=_summarize_for_query,
                _trim_schema_for_llm=_trim_schema_for_llm,
            )
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
            preview = content if len(content) <= 200 else content[:200] + "…"
            calls.append(PathFinderCall(tool=name, args=args, result_summary=preview))

        if answered is not None:
            break

    if answered is None:
        return FindTargetResult(
            pointer=proposed_path,
            confidence="low",
            rationale=f"path_finder hit max_steps={max_steps} without answer",
            calls=calls,
            llm_calls=llm_calls,
        )

    pointer = answered.get("pointer")
    if pointer is not None and not isinstance(pointer, str):
        pointer = None
    confidence = str(answered.get("confidence", "low")).lower()
    if confidence not in ("high", "medium", "low"):
        confidence = "low"
    rationale = str(answered.get("rationale", ""))[:300]

    return FindTargetResult(
        pointer=pointer,
        confidence=confidence,
        rationale=rationale,
        calls=calls,
        llm_calls=llm_calls,
    )


# ── Internal: dispatch read-only tools ──────────────────────────────────────


def _dispatch_readonly(
    name: str,
    args: dict,
    graph: dict,
    *,
    _resolve,
    _find_paths_by_key,
    _find_values,
    _resolve_schema_at,
    _summarize_for_query,
    _trim_schema_for_llm,
) -> str:
    if name == "query":
        ptr = args.get("pointer", "")
        try:
            val = _resolve(graph, ptr)
        except (KeyError, IndexError, ValueError) as exc:
            return json.dumps({"error": f"resolve failed at {ptr!r}: {exc}"}, ensure_ascii=False)
        return _summarize_for_query(val, max_chars=1500)

    if name == "get_schema":
        ptr = args.get("pointer", "")
        # Sub-agent works without a root_schema (we don't pass one in
        # to keep this module independent). When unavailable, return
        # the value's STRUCTURAL shape from the data instead — close
        # enough for the sub-agent's purpose ("what fields exist here").
        try:
            val = _resolve(graph, ptr)
        except (KeyError, IndexError, ValueError) as exc:
            return json.dumps({"error": f"resolve failed at {ptr!r}: {exc}"}, ensure_ascii=False)
        if isinstance(val, dict):
            shape = {k: type(v).__name__ for k, v in val.items()}
            return json.dumps({"shape": "dict", "fields": shape}, ensure_ascii=False)
        if isinstance(val, list):
            shape = {"len": len(val), "item_type": type(val[0]).__name__ if val else "?"}
            return json.dumps({"shape": "list", "info": shape}, ensure_ascii=False)
        return json.dumps({"shape": type(val).__name__}, ensure_ascii=False)

    if name == "find_paths":
        keyword = args.get("keyword", "")
        results = _find_paths_by_key(graph, keyword, max_results=20)
        return json.dumps({"matches": results}, ensure_ascii=False)

    if name == "find_values":
        keyword = args.get("keyword", "")
        results = _find_values(graph, keyword, max_results=20)
        return json.dumps({"matches": results}, ensure_ascii=False)

    if name == "jq":
        expression = args.get("expression")
        if not expression or not isinstance(expression, str):
            return json.dumps(
                {"error": "jq requires non-empty 'expression' (string)"},
                ensure_ascii=False,
            )
        ptr = (args.get("pointer") or "").rstrip("/")
        try:
            target = _resolve(graph, ptr) if ptr else graph
        except (KeyError, IndexError, ValueError) as exc:
            return json.dumps(
                {"error": f"pointer {ptr!r} not found: {exc}"},
                ensure_ascii=False,
            )
        try:
            import jq as _jq_lib
        except ImportError:
            return json.dumps(
                {"error": "jq library not installed"},
                ensure_ascii=False,
            )
        try:
            program = _jq_lib.compile(expression)
        except ValueError as exc:
            return json.dumps(
                {"error": f"jq compile error: {exc}", "expression": expression},
                ensure_ascii=False,
            )
        try:
            results = program.input(target).all()
        except Exception as exc:
            return json.dumps(
                {"error": f"jq runtime error: {exc}", "expression": expression},
                ensure_ascii=False,
            )
        payload = results[0] if len(results) == 1 else results
        return _summarize_for_query(payload, max_chars=1500)

    return json.dumps({"error": f"unknown tool {name!r}"}, ensure_ascii=False)


# ── Hint fast-path ──────────────────────────────────────────────────────────


# Token regex covers ASCII alphanumerics plus CJK (Hangul + CJK Unified
# Ideographs) so Korean intents tokenize meaningfully without a heavier
# morphological analyzer.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_가-힯㐀-鿿]+")


def _intent_tokens(intent: str, *, min_len: int = 2) -> list[str]:
    """Split ``intent`` into ≥``min_len`` substring-matchable tokens."""
    if not intent:
        return []
    return [t for t in _TOKEN_RE.findall(intent) if len(t) >= min_len]


def _hint_fastpath(
    graph: dict,
    intent: str,
    proposed_path: str,
    target_pointer: str,
) -> "FindTargetResult | None":
    """Return a high-confidence result without calling the LLM when:

      1. ``proposed_path`` and ``target_pointer`` agree (both non-empty,
         non-root).
      2. ``proposed_path`` resolves in ``graph``.
      3. Resolved value is a scalar (``str``/``int``/``float``/``bool``/
         ``None``) — container values may need a deeper drill that only
         the agent loop can determine.
      4. At least one ≥2-char token from ``intent`` substring-matches a
         non-numeric key segment of ``proposed_path``, OR the resolved
         scalar's string form.

    Returns ``None`` when any condition fails — caller falls through to
    the normal LLM agent loop. Designed conservatively: the cost of a
    false-positive (off-target patch) is much higher than the cost of
    one extra agent invocation.
    """
    from json_correction_loop.patcher import _resolve

    if not proposed_path or not target_pointer:
        return None
    if proposed_path == "/" or target_pointer == "/":
        return None
    if proposed_path != target_pointer:
        return None
    try:
        value = _resolve(graph, proposed_path)
    except (KeyError, IndexError, ValueError):
        return None
    if isinstance(value, (dict, list)):
        return None

    tokens = _intent_tokens(intent)
    if not tokens:
        return None

    segments = [s for s in proposed_path.split("/") if s and not s.isdigit()]
    seg_text = " ".join(segments).lower()
    val_text = "" if value is None else str(value).lower()

    matched: str | None = None
    for tok in tokens:
        low = tok.lower()
        if low in seg_text or (val_text and low in val_text):
            matched = tok
            break
    if matched is None:
        return None

    return FindTargetResult(
        pointer=proposed_path,
        confidence="high",
        rationale=(
            "hint-fastpath: proposed_path == target_pointer, resolves to "
            f"scalar, intent token {matched!r} matches"
        ),
        calls=[],
        llm_calls=0,
    )
