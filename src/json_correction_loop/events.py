"""Event sink — replaces ``console.print`` calls inside the library.

The library emits structured events at every salient point (iteration
start, critic verdict, planner output, executor result, convergence).
Callers wire whichever sink they want — Rich console, log, telemetry,
silent — without the library knowing about any specific UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Event:
    """One observation from the loop driver.

    ``kind`` is a stable enum-like string. ``data`` carries event-
    specific payload. The event taxonomy:

      - ``"iter_start"`` — { iter, max_loops, level }
      - ``"critic_report"`` — { level, score, critical, major, minor, assessment }
      - ``"converged"`` — { reason }
      - ``"hardcap"`` — { cap }
      - ``"approved"`` — { iter }
      - ``"planner_skipped"`` — { count, rationale }
      - ``"planner_empty"`` — { rationale }
      - ``"planner_failed"`` — { error }
      - ``"executor_result"`` — { addressed, total }
      - ``"executor_failed"`` — { error }
      - ``"loop_end"`` — { reason ("approved"|"converged"|"hardcap"|"max_loops") }
    """
    kind: str
    data: dict[str, Any] = field(default_factory=dict)


class EventSink(Protocol):
    """Where the loop sends progress events."""
    def emit(self, event: Event) -> None: ...


class NullEventSink:
    """Drops every event. Default when the caller doesn't care about
    progress reporting (e.g. inside tests)."""
    def emit(self, event: Event) -> None:  # noqa: D401, ARG002
        pass
