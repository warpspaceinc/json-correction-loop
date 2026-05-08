"""Library-internal defaults.

Sub-agents and the patcher read these as fallback values when the
caller doesn't override per-call. Env-var prefix is ``JCL_*``
(json-correction-loop). Callers can override either by passing
parameters explicitly or by setting the env var.
"""
from __future__ import annotations

import os


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Default max output tokens per LLM call. Sized for ~32K context
# backends (vLLM gemma4 / ollama gemma4:26b) leaving prompt headroom.
DEFAULT_MAX_TOKENS = _int_env("JCL_MAX_TOKENS", 24576)

# Default model identifier. Empty string means "caller must pass
# explicit model"; sub-agents that try to use this without a caller-
# supplied override will surface a clear error from the LLM provider.
DEFAULT_MODEL = os.environ.get("JCL_MODEL", "")
