"""Tests for ``json_correction_loop.llm.schema_compat`` — provider
detection + Gemini schema sanitizer."""

from __future__ import annotations

from json_correction_loop.llm.schema_compat import (
    is_gemini_model,
    sanitize_for_gemini,
)


# ── is_gemini_model ─────────────────────────────────────────────────────────


def test_detection_openrouter_prefix():
    assert is_gemini_model("google/gemini-3.1-flash-lite") is True
    assert is_gemini_model("GOOGLE/Gemini-1.5-Pro") is True  # case-insensitive


def test_detection_direct_name():
    assert is_gemini_model("gemini-1.5-pro") is True
    assert is_gemini_model("Gemini-2.0-Flash") is True


def test_detection_non_gemini():
    assert is_gemini_model("openai/gpt-4o") is False
    assert is_gemini_model("anthropic/claude-sonnet-4-6") is False
    assert is_gemini_model("") is False
    assert is_gemini_model(None) is False


# ── sanitize_for_gemini ─────────────────────────────────────────────────────


def test_inline_top_level_ref():
    schema = {
        "$defs": {
            "Inner": {"type": "object", "properties": {"x": {"type": "integer"}}}
        },
        "type": "object",
        "properties": {"item": {"$ref": "#/$defs/Inner"}},
    }
    out = sanitize_for_gemini(schema)
    assert "$defs" not in out
    assert out["properties"]["item"] == {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
    }


def test_inline_nested_ref():
    """Refs inside arrays / nested objects must also resolve."""
    schema = {
        "$defs": {"Item": {"type": "object", "properties": {"n": {"type": "integer"}}}},
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"$ref": "#/$defs/Item"},
            },
        },
    }
    out = sanitize_for_gemini(schema)
    assert out["properties"]["items"]["items"] == {
        "type": "object",
        "properties": {"n": {"type": "integer"}},
    }


def test_unresolvable_ref_left_alone():
    """A $ref pointing at a missing def is preserved (best-effort)."""
    schema = {"$defs": {}, "type": "object", "properties": {"x": {"$ref": "#/$defs/Missing"}}}
    out = sanitize_for_gemini(schema)
    assert out["properties"]["x"] == {"$ref": "#/$defs/Missing"}


def test_collapse_nullable_anyof_string_first():
    schema = {
        "type": "object",
        "properties": {
            "opt": {"anyOf": [{"type": "string"}, {"type": "null"}], "title": "Opt"},
        },
    }
    out = sanitize_for_gemini(schema)
    field = out["properties"]["opt"]
    assert field["type"] == "string"
    assert field["nullable"] is True
    assert "anyOf" not in field
    # outer keys (title) preserved
    assert field["title"] == "Opt"


def test_collapse_nullable_anyof_null_first():
    schema = {"anyOf": [{"type": "null"}, {"type": "integer"}]}
    assert sanitize_for_gemini(schema) == {"type": "integer", "nullable": True}


def test_anyof_without_null_left_alone():
    """A real polymorphic anyOf (no null branch) is not the nullable
    shorthand and must pass through unchanged."""
    schema = {"anyOf": [{"type": "string"}, {"type": "integer"}]}
    assert sanitize_for_gemini(schema) == schema


def test_strip_additional_properties_false():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"x": {"type": "string"}},
    }
    out = sanitize_for_gemini(schema)
    assert "additionalProperties" not in out


def test_keep_additional_properties_when_schema():
    """``additionalProperties`` set to a schema (allowing extra fields of a
    type) is legitimate OpenAPI and must pass through."""
    schema = {"type": "object", "additionalProperties": {"type": "string"}}
    out = sanitize_for_gemini(schema)
    assert out["additionalProperties"] == {"type": "string"}


def test_combined_transforms_end_to_end():
    """All three transforms cooperate on one schema."""
    schema = {
        "$defs": {
            "Inner": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "opt": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
            },
        },
        "type": "object",
        "additionalProperties": False,
        "properties": {"inner": {"$ref": "#/$defs/Inner"}},
    }
    out = sanitize_for_gemini(schema)

    def walk(node, found):
        if isinstance(node, dict):
            if "$defs" in node: found["defs"] = True
            if "$ref" in node: found["ref"] = True
            if "anyOf" in node: found["anyOf"] = True
            if node.get("additionalProperties") is False: found["addProp"] = True
            for v in node.values():
                walk(v, found)
        elif isinstance(node, list):
            for item in node:
                walk(item, found)

    found = {"defs": False, "ref": False, "anyOf": False, "addProp": False}
    walk(out, found)
    assert found == {"defs": False, "ref": False, "anyOf": False, "addProp": False}
    # Nullable shorthand actually landed.
    assert out["properties"]["inner"]["properties"]["opt"]["nullable"] is True
