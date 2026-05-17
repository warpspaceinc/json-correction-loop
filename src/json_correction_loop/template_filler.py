"""Template-filler sub-agent (structured-output variant).

Auto-invoked by ``SurgicalPatcher`` BEFORE the main patch loop when:
- ``target_pointer`` resolves to an EMPTY array/object, AND
- the critic's intent contains enumerated items (``1. ... 2. ...``,
  ``- ...``, ``첫째 ... 둘째 ...``, etc).

Why a separate agent: production traces show the main patcher LLM
struggles with "extract N items from prose, build a list of N
matching the schema" when it's mid-loop and juggling tool calls
+ value decisions + format alignment all at once. The pattern is
visible in plot.character_arcs.threshold_chain failures: the LLM
emits ``replace`` with a single ``"1.1"`` step ten times in a row
and never produces the full 5-step chain.

This sub-agent runs in **structured-output mode** (NOT tool-calling),
so the LLM is structurally constrained to emit a value matching the
target field's actual JSON Schema. We wrap the field schema inside
a small envelope (``value`` + ``confidence`` + ``rationale``) and
pass the whole thing as ``response_format`` to the LLM. xgrammar /
OpenAI structured-output enforces shape during generation — wrong
type / missing required field is impossible by construction.

Falls back to ``json_object`` (free-form JSON) when the backend
rejects the schema (vLLM xgrammar doesn't implement every JSON
Schema feature), mirroring ``call_llm_json``'s pattern.

Contract:
- ``fill_template(target_pointer, current_value, intent, field_schema,
  ...)`` runs a SINGLE structured LLM call.
- NO tools. References are pre-fetched into the prompt.
- Returns ``FillTemplateResult`` with the assembled value, item count,
  confidence band.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from json_correction_loop.llm import (
    SchemaRejectedError,
    TransientLLMError,
)

from json_correction_loop.path_finder import PathFinderCall  # reuse trace shape
from json_correction_loop._config import DEFAULT_MODEL

logger = logging.getLogger(__name__)


# ── Public types ────────────────────────────────────────────────────────────


@dataclass
class FillTemplateResult:
    """Outcome of one ``fill_template`` invocation.

    ``value``: the assembled value to drop into the target. ``None``
    when the agent couldn't extract a structured fill (low confidence,
    no enumeration found, backend error).
    ``item_count``: number of entries in the value (len of list / dict).
    ``confidence``:
      - ``"high"``: enumeration unambiguous, all items extracted, schema-matched.
      - ``"medium"``: extractable but some ambiguity / partial schema match.
      - ``"low"``: couldn't extract a clean fill — caller falls back to
        the main patcher loop.
    ``rationale``: one short line explaining the choice.
    ``calls``: a single-entry "structured_output" record (no tool trail
    in this design — kept for inspector parity with other sub-agents).
    """
    value: Any
    item_count: int
    confidence: str
    rationale: str
    calls: list[PathFinderCall] = field(default_factory=list)
    llm_calls: int = 0


# ── Trigger detection (deterministic) ───────────────────────────────────────


_ENUM_PATTERNS = [
    re.compile(r"\b\d+\.\s"),                # "1. ", "2. "
    re.compile(r"\b\d+\)\s"),                # "1) ", "2) "
    re.compile(r"^[-*•]\s", re.MULTILINE),   # "- foo", "* bar"
    re.compile(r"\b(first|second|third|fourth|fifth)\b", re.IGNORECASE),
    re.compile(r"(첫째|둘째|셋째|넷째|다섯째)"),
    re.compile(r"단계\s*\d"),                # "단계 1", "단계 2"
]


def has_enumeration_pattern(text: str) -> bool:
    """True when ``text`` looks like it enumerates discrete items.

    Conservative: must hit at least 2 markers from any single pattern,
    OR ≥2 markers across patterns. A single ``1. X`` could be a section
    header — not worth invoking the sub-agent for.
    """
    if not text:
        return False
    hits = 0
    for pat in _ENUM_PATTERNS:
        matches = pat.findall(text)
        if len(matches) >= 2:
            return True
        hits += len(matches)
    return hits >= 2


def is_empty_container(value: Any) -> bool:
    """True iff value is an empty list or empty dict."""
    return (isinstance(value, list) and len(value) == 0) or (
        isinstance(value, dict) and len(value) == 0
    )


# ── Schema helpers ──────────────────────────────────────────────────────────


def _schema_has_property_names(node: Any) -> bool:
    """Walk a JSON Schema looking for ``propertyNames`` keys.

    vLLM xgrammar doesn't support ``propertyNames`` (auto-generated for
    ``dict[K, V]`` types). When found, we route directly to
    ``json_object`` mode instead of eating a 400 first. Mirrors
    ``llm._schema_has_property_names``.
    """
    if isinstance(node, dict):
        if "propertyNames" in node:
            return True
        return any(_schema_has_property_names(v) for v in node.values())
    if isinstance(node, list):
        return any(_schema_has_property_names(v) for v in node)
    return False


def _build_envelope_schema(field_schema: dict | None, fallback_shape: str) -> dict:
    """Wrap ``field_schema`` in an envelope so we can require
    ``confidence`` + ``rationale`` alongside the actual value.

    When ``field_schema`` is unavailable, fall back to a permissive
    array/object based on ``fallback_shape``.
    """
    if isinstance(field_schema, dict) and field_schema:
        # Strip $defs that reference the parent root — they'd resolve
        # against the wrong root. Best-effort: pass through, the
        # backend usually handles inline $ref dereferencing.
        value_schema = field_schema
    else:
        if fallback_shape == "list":
            value_schema = {"type": "array"}
        else:
            value_schema = {"type": "object"}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "value": value_schema,
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
            },
            "rationale": {
                "type": "string",
            },
        },
        "required": ["value", "confidence", "rationale"],
    }


# ── Prompts ─────────────────────────────────────────────────────────────────


_SYSTEM_PROMPT = """\
You are a template-filler sub-agent for a JSON patcher.

# Your job
The patch target is an EMPTY container (list or dict). The critic's
intent enumerates the items that should fill it (e.g. "1. X, 2. Y,
3. Z, 4. A, 5. B"). Build the complete value.

# Output
Return JSON with three fields:
- ``value``: the complete assembled value matching the target's schema.
  - List target → list with ONE entry per enumerated item in the intent.
  - Dict target → dict with the keys/structure described in the intent.
- ``confidence``: ``high`` (unambiguous, all items map cleanly) /
  ``medium`` (some ambiguity / partial match) / ``low`` (couldn't
  extract a clean N-item structure — emit best partial).
- ``rationale``: one short line explaining the assembly.

# Rules
- Stay close to the user's wording; don't paraphrase specific phrases.
- Don't invent items beyond what the intent enumerates.
- Match the target's schema precisely (string vs object-with-fields, etc.).
- If the target is a list of objects, fill in plausible values for
  each required field per item (use the schema and references).
"""


_USER_TMPL = """\
# Target
- pointer: {pointer}
- current value (empty): {current_value}

# Field schema (the structured-output schema enforces this — your value
# field will be validated against it)
{schema_block}

# Intent (read carefully — the enumerated items here are what to fill in)
{intent}

# Reference (sibling values, if any, for shape/length calibration)
{reference_block}
"""


# ── Public entry ────────────────────────────────────────────────────────────


def fill_template(
    target_pointer: str,
    current_value: Any,
    intent: str,
    *,
    graph: dict | None = None,
    field_schema: dict | None = None,
    reference_pointers: list[str] | None = None,
    min_items: int = 1,  # accepted for API compat; unused in structured-output mode
    client: Any = None,
    model: str | None = None,
    max_attempts: int = 2,
) -> FillTemplateResult:
    """Extract an N-item structure from ``intent`` via a single
    structured-output LLM call.

    Returns ``FillTemplateResult.value=None`` when disabled, when the
    backend errors transiently, or when the LLM outputs low-confidence.
    """
    from json_correction_loop.patcher import (
        _resolve,
        _summarize_for_query,
    )

    if os.environ.get("JCL_TEMPLATE_FILLER_ENABLED", "1").strip() not in ("1", "true", "True", "yes"):
        return FillTemplateResult(
            value=None,
            item_count=0,
            confidence="low",
            rationale="template_filler disabled — caller should fall back",
        )

    chosen_model = (
        model
        or os.environ.get("JCL_TEMPLATE_FILLER_MODEL", "").strip()
        or DEFAULT_MODEL
    )
    if client is None:
        raise ValueError(
            "template_filler requires an LLMClient (chat_complete) — got None. "
            "Caller must construct one and pass it."
        )
    cli = client

    # Pre-fetch references for the prompt (no tool loop here — single call).
    ref_lines: list[str] = []
    if graph is not None and reference_pointers:
        for ptr in reference_pointers[:3]:
            try:
                v = _resolve(graph, ptr)
                ref_lines.append(f"## {ptr}\n{_summarize_for_query(v, max_chars=600)}")
            except (KeyError, IndexError, ValueError):
                ref_lines.append(f"## {ptr}\n<not present>")
    reference_block = "\n\n".join(ref_lines) or "(none)"

    if isinstance(field_schema, dict) and field_schema:
        try:
            schema_block = json.dumps(field_schema, ensure_ascii=False, indent=2)[:1500]
        except Exception:
            schema_block = repr(field_schema)[:1500]
    else:
        schema_block = "(unavailable — emit best-effort)"

    fallback_shape = "list" if isinstance(current_value, list) else "object"
    envelope_schema = _build_envelope_schema(field_schema, fallback_shape)

    user_prompt = _USER_TMPL.format(
        pointer=target_pointer or "/",
        current_value=json.dumps(current_value, ensure_ascii=False)[:200],
        schema_block=schema_block,
        intent=intent or "(none)",
        reference_block=reference_block,
    )
    base_messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Gemini's responseSchema only accepts a narrow OpenAPI 3.0 subset
    # ($defs/$ref, anyOf, additionalProperties:false all 400). Rewrite up
    # front so the first attempt actually succeeds — saves the mandatory
    # ~1.5s schema-rejection retry per call on Gemini-routed runs.
    from json_correction_loop.llm.schema_compat import is_gemini_model, sanitize_for_gemini
    if is_gemini_model(chosen_model):
        envelope_schema = sanitize_for_gemini(envelope_schema)

    # If schema has unsupported features, route directly to json_object.
    skip_json_schema = _schema_has_property_names(envelope_schema)

    last_exc: Exception | None = None
    raw_content = ""
    for attempt in range(max_attempts):
        use_structured = (attempt == 0 and not skip_json_schema)
        try:
            if use_structured:
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "FilledTemplate",
                        "schema": envelope_schema,
                        "strict": False,
                    },
                }
            else:
                response_format = {"type": "json_object"}
                # In fallback mode, append a textual schema hint so the LLM
                # still produces the right shape.
                base_messages = [
                    {"role": "system", "content": (
                        _SYSTEM_PROMPT
                        + "\n\nRespond with JSON exactly matching this envelope:\n"
                        + json.dumps(envelope_schema, ensure_ascii=False, indent=2)
                    )},
                    {"role": "user", "content": user_prompt},
                ]

            resp = cli.chat_complete(
                model=chosen_model,
                messages=base_messages,
                temperature=0.1,
                max_tokens=4096,
                response_format=response_format,
                extra=({"reasoning_effort": e} if (e := (os.environ.get("JCL_REASONING_EFFORT", "none").strip() or "none")) and e != "none" else {}),
            )
        except SchemaRejectedError as exc:
            last_exc = exc
            # vLLM xgrammar 400. Retry as json_object on the next attempt.
            if use_structured and attempt < max_attempts - 1:
                logger.warning("template_filler schema rejected — falling back to json_object")
                continue
            return FillTemplateResult(
                value=None, item_count=0, confidence="low",
                rationale=f"schema rejected: {exc!r}"[:200],
            )
        except TransientLLMError as exc:
            last_exc = exc
            return FillTemplateResult(
                value=None, item_count=0, confidence="low",
                rationale=f"backend transient: {exc!r}"[:200],
            )

        raw_content = resp.content or ""
        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                continue
            return FillTemplateResult(
                value=None, item_count=0, confidence="low",
                rationale=f"parse failure: {exc}; raw={raw_content[:200]!r}",
            )
        break
    else:
        return FillTemplateResult(
            value=None, item_count=0, confidence="low",
            rationale=f"exhausted {max_attempts} attempts: {last_exc!r}"[:200],
        )

    if not isinstance(parsed, dict):
        return FillTemplateResult(
            value=None, item_count=0, confidence="low",
            rationale=f"response was not an object: {type(parsed).__name__}",
        )

    value = parsed.get("value")
    confidence = str(parsed.get("confidence", "low")).lower()
    if confidence not in ("high", "medium", "low"):
        confidence = "low"
    rationale = str(parsed.get("rationale", ""))[:300]

    if not isinstance(value, (list, dict)):
        return FillTemplateResult(
            value=None, item_count=0, confidence="low",
            rationale=f"value was not a list/dict: {type(value).__name__}",
        )

    item_count = len(value)
    # One dummy "call" entry so the inspector trace renders consistently.
    calls = [PathFinderCall(
        tool="structured_output",
        args={"response_format": "json_schema" if not skip_json_schema else "json_object"},
        result_summary=raw_content[:200],
    )]

    return FillTemplateResult(
        value=value,
        item_count=item_count,
        confidence=confidence,
        rationale=rationale,
        calls=calls,
        llm_calls=1,
    )
