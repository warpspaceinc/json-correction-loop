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

# Per-requirement patcher loop budgets. Both are tunable per environment
# (e.g. raise PATCHER_MAX_STEPS for harder domains, lower
# PATCHER_READ_ONLY_BAIL for tighter exploration cutoffs).
#
# - PATCHER_MAX_STEPS: hard cap on LLM calls inside one ``apply()`` for
#   one PatchRequest. Loop terminates with reason ``hit max_steps`` if
#   reached without convergence.
# - PATCHER_READ_ONLY_BAIL: number of consecutive read-only steps
#   (query / find_* / get_schema / diff) without any patch attempt
#   before the loop bails with reason ``consecutive read-only steps``.
#   Designed to catch models stuck in pure exploration. Must be < MAX
#   for the bail to fire before MAX hits.
DEFAULT_PATCHER_MAX_STEPS = _int_env("JCL_PATCHER_MAX_STEPS", 30)
DEFAULT_PATCHER_READ_ONLY_BAIL = _int_env("JCL_PATCHER_READ_ONLY_BAIL", 20)
