"""Request-validator sub-agent.

Auto-invoked by ``SurgicalPatcher.apply`` BEFORE the patch loop runs.
Reads the critic's ``PatchRequest`` (intent + target_pointer + context)
against the actual graph and judges, by ACTIVE INVESTIGATION, whether:

  (a) the critique's claim about the data is actually true
  (b) the proposed fix is sensible given the surrounding data

This is *substance* judgment, not form-checking. A critic may claim
"X duplicates Y" when X is in fact unique, or propose "merge A into B"
when B's schema can't hold A. The validator catches both before the
patcher burns LLM budget on a doomed loop.

Why: production traces show ~10–20% of critic requirements are
spurious (claim is wrong), bad-suggestion (fix would worsen things),
or out-of-scope (target_missing / too_broad / already_satisfied).

Contract:
- ``validate_request(graph, requirement, model)`` runs a short
  read-only LLM session.
- Tools: ``query``, ``get_schema``, ``find_paths``, ``find_values``.
- Returns ``ValidateRequestResult`` with verdict + confidence + reasons.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from json_correction_loop.llm import TransientLLMError

from json_correction_loop.path_finder import PathFinderCall  # reuse trace shape
from json_correction_loop._config import DEFAULT_MODEL

logger = logging.getLogger(__name__)


@dataclass
class ValidateRequestResult:
    """Outcome of one ``validate_request`` invocation.

    ``verdict``:
      - ``"valid"``: critique's claim about the data is true AND the
        proposed fix is sensible. Patcher should proceed.
      - ``"spurious"``: the critic's CLAIM about the data is false
        (e.g. "X duplicates Y" but X is unique; "field is missing" but
        it's there). Patcher should refuse with critic_error.
      - ``"bad_suggestion"``: claim is real, but the PROPOSED FIX
        would make things worse / conflicts with surrounding data /
        is structurally impossible. Refuse with critic_error.
      - ``"already_satisfied"``: the current state already meets the
        intent — no patch needed. Patcher marks addressed=True.
      - ``"target_missing"``: target_pointer doesn't resolve AND no
        equivalent path exists in the graph. Refuse with critic_error.
      - ``"too_broad"``: intent requires bigger structural changes
        than a single localized patch can do. Refuse.
      - ``"ambiguous"``: intent is unclear about WHAT or HOW. Let the
        loop try (it may resolve), but flag low confidence.
    ``confidence``: ``"high"|"medium"|"low"``.
    """
    verdict: str
    confidence: str
    rationale: str
    calls: list[PathFinderCall] = field(default_factory=list)
    llm_calls: int = 0


_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "query",
            "description": "Read the value at a JSON Pointer.",
            "parameters": {
                "type": "object",
                "properties": {"pointer": {"type": "string"}},
                "required": ["pointer"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_schema",
            "description": "Inspect the structural shape (fields + types) at a pointer.",
            "parameters": {
                "type": "object",
                "properties": {"pointer": {"type": "string"}},
                "required": ["pointer"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_paths",
            "description": "Search dict-key segments containing keyword.",
            "parameters": {
                "type": "object",
                "properties": {"keyword": {"type": "string"}},
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_values",
            "description": "Search leaf-string values containing keyword.",
            "parameters": {
                "type": "object",
                "properties": {"keyword": {"type": "string"}},
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verdict",
            "description": "Final verdict. Call exactly once when done.",
            "parameters": {
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": [
                            "valid",
                            "spurious",
                            "bad_suggestion",
                            "already_satisfied",
                            "target_missing",
                            "too_broad",
                            "ambiguous",
                        ],
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "rationale": {
                        "type": "string",
                        "description": "One short line explaining the verdict.",
                    },
                },
                "required": ["verdict", "confidence", "rationale"],
            },
        },
    },
]


_SYSTEM_PROMPT = """\
You are a request-validator sub-agent for a JSON patcher.

# Your job
A critic produced a patch requirement. The intent typically has the
shape ``<problem claim> | 제안: <suggested fix>`` (or English
equivalent). Your job is SUBSTANTIVE judgment — by actively reading
the graph, decide:

  1. Is the critic's CLAIM about the data actually true? (Does the
     problem really exist as described?)
  2. Is the SUGGESTED FIX sensible given what's actually in the data
     and surrounding fields? (Would it make things worse, conflict
     with siblings, violate the schema, be redundant?)

End with exactly one ``verdict`` call.

# Verdicts (in priority order — pick the FIRST that fits)
- ``spurious``: the critic's claim is wrong. Example claims that turn
  out to be false: "X is duplicated 3 times" (it's unique), "field is
  missing" (it's present), "Y conflicts with Z" (no actual conflict).
- ``target_missing``: claim might be valid but the named target_pointer
  doesn't resolve AND no equivalent path exists for that thing.
- ``already_satisfied``: claim is moot — the data already meets what
  the suggestion asks for.
- ``bad_suggestion``: claim is REAL, but the proposed fix is wrong —
  e.g. "delete X" but X is referenced by Y; "merge into B" but B's
  schema can't hold it; the fix would create a new inconsistency.
- ``too_broad``: claim is real, but a single localized patch at the
  given target_pointer cannot resolve it. This is the MOST COMMON
  miss in production — the patcher LLM tries to fix one field and
  the evaluator keeps saying ``partial``. Detect this verdict early.
  Strong signals to recognise it:
    * The intent compares two fields/sections by name — e.g. "X와
      Y가 안 맞음", "A's outcome conflicts with B's content", "subplot
      Z duplicates main_arc's climax".
    * The suggestion says "수정하여 일관성 확보" / "make consistent" /
      "differentiate" / "align" — coordinated edit across siblings.
    * The fix would only make sense if BOTH the target AND another
      section change. Patching one alone leaves the other dangling
      (evaluator will flag as partial).
  When you see these, return ``too_broad`` even if the literal
  surface text could be patched at target_pointer alone.
- ``ambiguous``: intent doesn't say clearly WHAT to change or HOW.
- ``valid``: claim is true, suggestion is sensible. Default when none
  of the above apply.

# How to investigate
1. Read intent. Identify (a) the claim and (b) the suggestion. Note
   key terms (entity names, field names, JSON-pointer-like fragments).
2. ``query`` the target_pointer to see what's actually there. If
   doesn't resolve, ``find_paths`` for the claim's key terms.
3. **Cross-section check (FIRST priority)**: does the claim compare
   two sections / fields by name? If yes:
     - ``find_paths`` / ``query`` BOTH sides.
     - If fixing only the target_pointer would leave the other side
       contradicting → ``too_broad``. Do not let the literal target
       fool you: the patcher can only edit ONE pointer, so a
       comparison-shaped intent is structurally cross-section even
       when the suggestion is phrased as a single-field edit.
4. To verify the claim, check the graph IS what the critic alleges:
   - "duplicated 3 times" → ``find_values`` or ``find_paths`` for the
     allegedly duplicated id/value, count occurrences.
   - "missing X" → ``find_paths`` with X.
   - "conflicts with Y" → ``query`` Y and compare.
5. To judge the suggestion, look at sibling/related data:
   - "delete X" → is X referenced from elsewhere?
   - "change content to Z" → does Z fit the field's schema/style?
   - "rename A → B" → is B already used elsewhere?
6. Call ``verdict``. Rationale must cite WHAT you observed (not just
   restate the claim) — e.g. "verified key-evt-10-5 appears 3× at
   /key_events/9, /key_events/11, /key_events/15 — claim true; merge
   into single id is structurally feasible — valid".

# Scope discipline
- DO NOT propose specific patch ops. Your job is verdict, not surgery.
- 2–5 tool calls is typical. Spend them on verifying the claim, not
  re-querying things you've already seen.
"""


_USER_TMPL = """\
# Patch requirement
- requirement_id: {rid}
- target_pointer: {target}
- intent: {intent}

# Current value at target_pointer
{target_value}

# Context (other relevant pointers)
{context_block}

Decide whether this is a valid, patchable requirement. Call ``verdict``.
"""


def validate_request(
    graph: dict,
    requirement: Any,  # PatchRequest — typed loosely to avoid circular import
    *,
    client: Any = None,
    model: str | None = None,
    max_steps: int = 6,
) -> ValidateRequestResult:
    """Judge whether a patch requirement is valid + patchable."""
    from json_correction_loop.patcher import (
        _resolve,
        _find_paths_by_key,
        _find_values,
        _summarize_for_query,
    )

    if os.environ.get("JCL_REQUEST_VALIDATOR_ENABLED", "1").strip() not in ("1", "true", "True", "yes"):
        return ValidateRequestResult(
            verdict="valid",
            confidence="low",
            rationale="request_validator disabled — accepting without check",
        )

    chosen_model = (
        model
        or os.environ.get("JCL_REQUEST_VALIDATOR_MODEL", "").strip()
        or DEFAULT_MODEL
    )
    if client is None:
        raise ValueError(
            "request_validator requires an LLMClient (chat_complete) — got None. "
            "Caller must construct one and pass it."
        )
    cli = client

    target_pointer = getattr(requirement, "target_pointer", "")
    intent = getattr(requirement, "intent", "")
    rid = getattr(requirement, "requirement_id", "")
    context_pointers = getattr(requirement, "context_pointers", []) or []

    # Pre-fetch target value
    try:
        tv = _resolve(graph, target_pointer)
        target_preview = _summarize_for_query(tv, max_chars=1500)
    except (KeyError, IndexError, ValueError) as exc:
        target_preview = f"<not present: {exc}>"

    ctx_lines: list[str] = []
    for ptr in context_pointers[:3]:
        try:
            v = _resolve(graph, ptr)
            ctx_lines.append(f"## {ptr}\n{_summarize_for_query(v, max_chars=600)}")
        except (KeyError, IndexError, ValueError):
            ctx_lines.append(f"## {ptr}\n<not present>")
    context_block = "\n\n".join(ctx_lines) or "(none)"

    user_prompt = _USER_TMPL.format(
        rid=rid,
        target=target_pointer or "/",
        intent=intent or "(none)",
        target_value=target_preview,
        context_block=context_block,
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
                extra={"reasoning_effort": os.environ.get("JCL_REASONING_EFFORT", "none").strip() or "none"},
            )
        except TransientLLMError as exc:
            logger.warning("request_validator transient error: %s", exc)
            return ValidateRequestResult(
                verdict="valid",
                confidence="low",
                rationale=f"request_validator backend error: {exc!r}",
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
            return ValidateRequestResult(
                verdict="valid",
                confidence="low",
                rationale="request_validator produced no tool call",
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

            if name == "verdict":
                answered = args
                content = json.dumps({"ok": True}, ensure_ascii=False)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
                calls.append(PathFinderCall(tool="verdict", args=args, result_summary=content[:200]))
                continue

            content = _dispatch_validate(
                name, args, graph,
                _resolve=_resolve,
                _find_paths_by_key=_find_paths_by_key,
                _find_values=_find_values,
                _summarize_for_query=_summarize_for_query,
            )
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
            preview = content if len(content) <= 200 else content[:200] + "…"
            calls.append(PathFinderCall(tool=name, args=args, result_summary=preview))

        if answered is not None:
            break

    if answered is None:
        return ValidateRequestResult(
            verdict="valid",
            confidence="low",
            rationale=f"request_validator hit max_steps={max_steps} without verdict",
            calls=calls,
            llm_calls=llm_calls,
        )

    verdict = str(answered.get("verdict", "valid")).lower()
    if verdict not in (
        "valid", "spurious", "bad_suggestion", "already_satisfied",
        "target_missing", "too_broad", "ambiguous",
    ):
        verdict = "valid"
    confidence = str(answered.get("confidence", "low")).lower()
    if confidence not in ("high", "medium", "low"):
        confidence = "low"
    rationale = str(answered.get("rationale", ""))[:300]

    return ValidateRequestResult(
        verdict=verdict,
        confidence=confidence,
        rationale=rationale,
        calls=calls,
        llm_calls=llm_calls,
    )


def _dispatch_validate(
    name: str,
    args: dict,
    graph: dict,
    *,
    _resolve,
    _find_paths_by_key,
    _find_values,
    _summarize_for_query,
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
        try:
            val = _resolve(graph, ptr)
        except (KeyError, IndexError, ValueError) as exc:
            return json.dumps({"error": f"resolve failed at {ptr!r}: {exc}"}, ensure_ascii=False)
        if isinstance(val, dict):
            return json.dumps({"shape": "dict", "fields": {k: type(v).__name__ for k, v in val.items()}}, ensure_ascii=False)
        if isinstance(val, list):
            return json.dumps({"shape": "list", "len": len(val), "item_type": type(val[0]).__name__ if val else "?"}, ensure_ascii=False)
        return json.dumps({"shape": type(val).__name__}, ensure_ascii=False)

    if name == "find_paths":
        keyword = args.get("keyword", "")
        return json.dumps({"matches": _find_paths_by_key(graph, keyword, max_results=20)}, ensure_ascii=False)

    if name == "find_values":
        keyword = args.get("keyword", "")
        return json.dumps({"matches": _find_values(graph, keyword, max_results=20)}, ensure_ascii=False)

    return json.dumps({"error": f"unknown tool {name!r}"}, ensure_ascii=False)
