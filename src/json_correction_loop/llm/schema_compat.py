"""Provider-specific schema compatibility transforms.

Some LLM providers reject parts of the OpenAPI/JSON-Schema dialect that
Pydantic generates by default. Rather than eating a 400 and downgrading
to ``json_object`` (which loses structured-output enforcement), we
detect the target provider and rewrite the schema up front into a
subset that provider accepts.

Currently covers:

* **Google Gemini** (``google/gemini-*`` and direct ``gemini-*``) —
  ``responseSchema`` supports only a narrow OpenAPI 3.0 subset
  (type / format / description / nullable / enum / properties / required
  / items / maxItems / minItems / uniqueItems). ``$defs``, ``$ref``,
  ``anyOf``, ``additionalProperties: false`` all produce
  ``400 INVALID_ARGUMENT``. :func:`sanitize_for_gemini` rewrites them
  away while preserving semantics.
"""

from __future__ import annotations

from typing import Any


def is_gemini_model(model: str | None) -> bool:
    """Google Gemini family detection — accepts both the OpenRouter
    prefix ``google/gemini-*`` and the direct ``gemini-*`` name."""
    m = (model or "").lower()
    return m.startswith("google/gemini") or m.startswith("gemini")


def sanitize_for_gemini(schema: dict) -> dict:
    """Rewrite a JSON Schema into Gemini's supported subset.

    Three transforms (all semantics-preserving):

    1. **Inline ``$defs`` / ``$ref``.** Walk every
       ``{"$ref": "#/$defs/Foo"}`` occurrence and substitute the resolved
       ``$defs[Foo]`` body. Drop the top-level ``$defs`` block once
       everything is inlined.
    2. **Nullable ``anyOf`` collapse.**
       ``{"anyOf": [{"type": T}, {"type": "null"}]}`` →
       ``{"type": T, "nullable": true}``. Pydantic generates the anyOf
       form for every ``X | None`` field; Gemini wants ``nullable``.
    3. **Strip ``additionalProperties: false``.** Gemini's subset
       doesn't allow the keyword.
    """
    defs = schema.get("$defs", {}) or {}

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            # 1. $ref → inline. Only handle local ``#/$defs/Name`` refs.
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref.split("/", 2)[2]
                target = defs.get(name)
                if isinstance(target, dict):
                    return _resolve(target)
                return node
            # 2. Nullable anyOf collapse.
            any_of = node.get("anyOf")
            if isinstance(any_of, list) and len(any_of) == 2:
                null_idx = next(
                    (i for i, opt in enumerate(any_of)
                     if isinstance(opt, dict) and opt.get("type") == "null"),
                    None,
                )
                if null_idx is not None:
                    other = any_of[1 - null_idx]
                    other_resolved = _resolve(other) if isinstance(other, dict) else other
                    if isinstance(other_resolved, dict):
                        merged = {k: v for k, v in node.items() if k != "anyOf"}
                        merged.update(other_resolved)
                        merged["nullable"] = True
                        return _resolve(merged)
            # General recursion. Strip additionalProperties:false (3),
            # drop the $defs block at any nesting level.
            out: dict = {}
            for k, v in node.items():
                if k == "additionalProperties" and v is False:
                    continue
                if k == "$defs":
                    continue
                out[k] = _resolve(v)
            return out
        if isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    return _resolve(schema)
