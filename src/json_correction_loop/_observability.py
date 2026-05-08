"""Verbose-mode side channel for the library.

Sub-agents emit one-line progress traces (path_finder corrections,
evaluator verdicts, …) when ``JCL_VERBOSE`` is set. The library
doesn't ship a formatter — by default it writes to stderr. Callers
that want richer output (rich console, log file, structured JSON)
plug in their own writer via :func:`set_verbose_writer`.

Stays library-internal: callers don't import from this module
directly. ``the host application`` calls
``json_correction_loop.observability.set_verbose_writer(...)`` at startup
to route through its rich console.
"""
from __future__ import annotations

import os
import sys
from typing import Callable

_writer: Callable[[str], None] | None = None


def set_verbose_writer(fn: Callable[[str], None] | None) -> None:
    """Install / clear a sink for sub-agent verbose lines.

    ``fn`` is called with one already-formatted message per event.
    Pass ``None`` to revert to the default (stderr).
    """
    global _writer
    _writer = fn


def is_verbose() -> bool:
    return os.environ.get("JCL_VERBOSE", "0").strip() in ("1", "true", "True", "yes")


def log_verbose(msg: str) -> None:
    """Emit a verbose-mode trace line. No-op when ``JCL_VERBOSE`` is off.

    Sub-agents call this directly. Failures in the writer are
    swallowed — verbose telemetry must never break a pipeline run.
    """
    if not is_verbose():
        return
    if _writer is not None:
        try:
            _writer(msg)
            return
        except Exception:
            pass
    print(msg, file=sys.stderr)
