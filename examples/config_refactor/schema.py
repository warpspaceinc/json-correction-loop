"""JSON Schema definition + a critic that flags schema violations.

The schema is small but realistic for a service config. The critic
is a thin wrapper around ``jsonschema.validators`` that converts each
``ValidationError`` into a :class:`CriticIssue` with a JSON-pointer
target.
"""
from __future__ import annotations

from typing import Any

try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover
    Draft202012Validator = None  # type: ignore[assignment]

from json_correction_loop import CriticIssue, CriticReport


SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["app_name", "version", "services"],
    "properties": {
        "app_name": {"type": "string", "minLength": 1},
        "version": {"type": "string", "pattern": r"^\d+\.\d+\.\d+$"},
        "services": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "required": ["image", "port", "replicas"],
                "properties": {
                    "image":    {"type": "string", "minLength": 1},
                    "port":     {"type": "integer", "minimum": 1, "maximum": 65535},
                    "replicas": {"type": "integer", "minimum": 1, "maximum": 32},
                    "env":      {"type": "string", "enum": ["dev", "staging", "prod"]},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}


def _pointer(error) -> str:
    """Convert a jsonschema absolute_path deque into RFC 6901 pointer."""
    if not error.absolute_path:
        return ""
    parts = ["" + str(p).replace("~", "~0").replace("/", "~1") for p in error.absolute_path]
    return "/" + "/".join(parts)


def _severity(error) -> str:
    # Required-field violations are critical; type/enum/format are
    # major; everything else is minor.
    if error.validator in {"required"}:
        return "critical"
    if error.validator in {"type", "enum", "minimum", "maximum", "minLength", "pattern"}:
        return "major"
    return "minor"


def schema_critic(state: Any, iteration: int = 0, model: str | None = None) -> list[CriticReport]:
    if Draft202012Validator is None:
        raise RuntimeError("install jsonschema: pip install jsonschema")
    validator = Draft202012Validator(SCHEMA)
    issues: list[CriticIssue] = []
    for err in validator.iter_errors(state):
        issues.append(CriticIssue(
            target_id=_pointer(err),
            severity=_severity(err),
            issue_type=err.validator or "schema_violation",
            description=err.message,
        ))
    score = 10 if not issues else max(1, 10 - len(issues))
    return [CriticReport(
        id=f"schema-iter{iteration}",
        level="config",
        iteration=iteration,
        issues=issues,
        score=score,
        overall_assessment=("clean" if not issues else f"{len(issues)} schema violation(s)"),
    )]
