"""Patch-evaluator sub-agent.

Auto-invoked by ``SurgicalPatcher._apply_ops`` AFTER each op lands.
Reads the before/after pair at the patched pointer and judges whether
the change actually addresses the patch intent — or missed (wrong
field, wrong value, partial, no-op).

Why a separate agent: the main patcher LLM has a confirmation bias
("I picked this op, of course it works"). A fresh evaluator with no
stake in the choice catches off-target patches that the patcher would
otherwise rubber-stamp via ``patch(ops=[])``. Combined with P11's
mandatory ``diff()``, this gives the patcher TWO independent voices
on whether the work landed.

Contract:
- ``evaluate_patch(graph_before, graph_after, intent, patched_path,
  op_kind, model)`` runs a short read-only LLM session.
- Tools: ``query``, ``get_schema``, ``find_paths``, ``find_values``
  (same read-only subset as ``path_finder``).
- Returns ``EvaluatePatchResult`` with verdict + confidence + evidence.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any


from json_correction_loop.path_finder import PathFinderCall  # reuse trace shape
from json_correction_loop._config import DEFAULT_MODEL
from json_correction_loop.llm import TransientLLMError

logger = logging.getLogger(__name__)


@dataclass
class EvaluatePatchResult:
    """Outcome of one ``evaluate_patch`` invocation.

    ``verdict``:
      - ``"addressed"``: change directly resolves the intent.
      - ``"partial"``: change is in the right direction but incomplete
        — caller should keep iterating.
      - ``"off-target"``: change is in the wrong place or wrong field.
        Caller should consider rolling back.
      - ``"no-op"``: change happened but state didn't meaningfully
        differ (LLM re-emitted the same value, or merge collapsed).
    ``confidence``: ``"high"|"medium"|"low"`` — same as path_finder.
    """
    verdict: str  # "addressed" | "partial" | "off-target" | "no-op"
    confidence: str
    rationale: str
    calls: list[PathFinderCall] = field(default_factory=list)
    llm_calls: int = 0


_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "query_after",
            "description": "Read the AFTER value at any JSON Pointer in the post-patch graph.",
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
            "name": "query_before",
            "description": "Read the BEFORE value at any JSON Pointer in the pre-patch graph.",
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
            "description": "Search dict-key segments containing keyword (in the AFTER graph).",
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
            "description": "Search leaf-string values containing keyword (in the AFTER graph).",
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
            "description": (
                "Final verdict. Call exactly once when you've decided "
                "whether the patch addressed the intent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "verdict": {
                        "type": "string",
                        "enum": ["addressed", "partial", "off-target", "no-op"],
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
You are a patch-evaluator sub-agent for a JSON patcher.

# Your job
A patch just landed. Compare the before/after pair at the patched
pointer and judge whether the change actually addresses the patch
intent. End with exactly one ``verdict`` call.

# Tools
- ``query_before(pointer)`` — read pre-patch value.
- ``query_after(pointer)`` — read post-patch value.
- ``find_paths/find_values`` — search the after-graph for related fields.

# Verdicts
- ``addressed``: the change directly resolves what the intent asked for.
  The right field was edited and the new value satisfies the intent.
- ``partial``: change is in the right direction but doesn't fully
  resolve the intent (e.g., addressed one of two issues, or the new
  value is closer but still off).
- ``off-target``: the change is in the WRONG place — wrong field,
  wrong slot, or doesn't relate to the intent at all. The caller
  should roll back and try again.
- ``no-op``: a change happened but the state didn't meaningfully
  differ (same value re-written, merge collapsed, or whitespace-only
  edit).

# How to work
1. Read intent + patched_path.
2. Compare before vs after at patched_path with query_before/query_after.
3. If intent references siblings or related fields, briefly check those
   too — does the change disturb them or leave them stale?
4. Call ``verdict``.

# Brevity
- 1–3 calls is enough for most cases. Don't over-explore.
"""


_USER_TMPL = """\
# Patch just applied
- intent: {intent}
- patched_path: {path}
- op_kind: {op_kind}

# Before value at patched_path
{before}

# After value at patched_path
{after}

Decide whether the patch addressed the intent. Call ``verdict`` to finish.
"""


def evaluate_patch(
    graph_before: dict,
    graph_after: dict,
    intent: str,
    patched_path: str,
    op_kind: str = "replace",
    *,
    client: Any = None,
    model: str | None = None,
    max_steps: int = 4,
) -> EvaluatePatchResult:
    """Judge whether the patch at ``patched_path`` addresses ``intent``."""
    from json_correction_loop.patcher import (
        _resolve,
        _find_paths_by_key,
        _find_values,
        _summarize_for_query,
    )

    if os.environ.get("JCL_PATCH_EVALUATOR_ENABLED", "1").strip() not in ("1", "true", "True", "yes"):
        return EvaluatePatchResult(
            verdict="addressed",
            confidence="low",
            rationale="patch_evaluator disabled — accepting without check",
        )

    chosen_model = (
        model
        or os.environ.get("JCL_PATCH_EVALUATOR_MODEL", "").strip()
        or DEFAULT_MODEL
    )
    if client is None:
        raise ValueError(
            "patch_evaluator requires an LLMClient (chat_complete) — got None. "
            "Caller must construct one and pass it."
        )
    cli = client

    def _preview(g: dict) -> str:
        try:
            return _summarize_for_query(_resolve(g, patched_path), max_chars=800)
        except (KeyError, IndexError, ValueError) as exc:
            return f"<not present: {exc}>"

    user_prompt = _USER_TMPL.format(
        intent=intent or "(none)",
        path=patched_path,
        op_kind=op_kind,
        before=_preview(graph_before),
        after=_preview(graph_after),
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
            logger.warning("patch_evaluator transient error: %s", exc)
            return EvaluatePatchResult(
                verdict="addressed",
                confidence="low",
                rationale=f"patch_evaluator backend error: {exc!r}",
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
            return EvaluatePatchResult(
                verdict="addressed",
                confidence="low",
                rationale="patch_evaluator produced no tool call",
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

            content = _dispatch_eval(
                name, args, graph_before, graph_after,
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
        return EvaluatePatchResult(
            verdict="addressed",
            confidence="low",
            rationale=f"patch_evaluator hit max_steps={max_steps} without verdict",
            calls=calls,
            llm_calls=llm_calls,
        )

    verdict = str(answered.get("verdict", "addressed")).lower()
    if verdict not in ("addressed", "partial", "off-target", "no-op"):
        verdict = "addressed"
    confidence = str(answered.get("confidence", "low")).lower()
    if confidence not in ("high", "medium", "low"):
        confidence = "low"
    rationale = str(answered.get("rationale", ""))[:300]

    return EvaluatePatchResult(
        verdict=verdict,
        confidence=confidence,
        rationale=rationale,
        calls=calls,
        llm_calls=llm_calls,
    )


def _dispatch_eval(
    name: str,
    args: dict,
    graph_before: dict,
    graph_after: dict,
    *,
    _resolve,
    _find_paths_by_key,
    _find_values,
    _summarize_for_query,
) -> str:
    if name in ("query_before", "query_after"):
        ptr = args.get("pointer", "")
        g = graph_before if name == "query_before" else graph_after
        try:
            val = _resolve(g, ptr)
        except (KeyError, IndexError, ValueError) as exc:
            return json.dumps({"error": f"resolve failed at {ptr!r}: {exc}"}, ensure_ascii=False)
        return _summarize_for_query(val, max_chars=1500)

    if name == "find_paths":
        keyword = args.get("keyword", "")
        results = _find_paths_by_key(graph_after, keyword, max_results=20)
        return json.dumps({"matches": results}, ensure_ascii=False)

    if name == "find_values":
        keyword = args.get("keyword", "")
        results = _find_values(graph_after, keyword, max_results=20)
        return json.dumps({"matches": results}, ensure_ascii=False)

    return json.dumps({"error": f"unknown tool {name!r}"}, ensure_ascii=False)
