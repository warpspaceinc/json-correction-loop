"""Executors apply a CorrectionPlan to the caller's state object.

The library doesn't ship a real executor — every domain wires its own
edit mechanic (JSON Patch, model rewrite, list mutation, ...). What the
library DOES provide is a thin factory + result type so the loop driver
gets uniform feedback.

The most common shape (used by every the host application level) wraps a
side-effecting callback that returns a list of trace records. See
:func:`make_callback_executor` for that helper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from json_correction_loop.models import CorrectionPlan


@runtime_checkable
class PatcherTraceLike(Protocol):
    """Structural type the loop reads from each trace.

    Domain trace classes (e.g. the host application's ``PatcherTrace``) satisfy
    this Protocol structurally — no inheritance needed. The loop only
    touches these attributes; everything else is opaque.
    """
    id: str
    requirement_id: str
    addressed: bool
    reason: str


@dataclass
class ExecuteResult:
    """What the executor returns to the loop.

    ``traces`` get attached to the iteration's stored record (the loop
    relays them through the StorageBackend without inspecting their
    contents beyond the Protocol). ``addressed`` and
    ``skipped_target_ids`` are convenience summaries the loop uses to
    populate manifest fields.
    """
    traces: list[Any] = field(default_factory=list)  # PatcherTraceLike
    addressed: list[str] = field(default_factory=list)
    skipped_target_ids: list[str] = field(default_factory=list)


ExecutorFn = Callable[[Any, CorrectionPlan, "str | None"], ExecuteResult]
"""(state, plan, model) → ExecuteResult."""


def make_callback_executor(
    apply_corrections: Callable[
        [Any, list[str], dict[str, str], "str | None"],
        list[Any],
    ],
) -> ExecutorFn:
    """Build an executor from a callback that takes the legacy
    ``(state, flagged_paths, feedback_by_path, model)`` signature and
    returns the list of traces it produced.

    ``apply_corrections`` is the domain's existing patch function —
    e.g. the host application's ``_world_surgical_v2`` wrapped to return
    its trace stash. The factory translates the CorrectionPlan into
    the ``flagged_paths`` / ``feedback_by_path`` arguments most legacy
    patch functions expect, then collects ``addressed`` / ``skipped``
    from the returned traces.
    """
    def _execute(state, plan: CorrectionPlan, model=None) -> ExecuteResult:
        if not plan.corrections:
            return ExecuteResult()
        flagged_paths = [c.requirement_id for c in plan.corrections]
        feedback_by_path = {
            c.requirement_id: c.intent for c in plan.corrections if c.intent
        }
        traces = apply_corrections(state, flagged_paths, feedback_by_path, model)
        addressed = [t.requirement_id for t in traces if getattr(t, "addressed", False)]
        skipped = [
            t.requirement_id for t in traces
            if not getattr(t, "addressed", False)
        ]
        return ExecuteResult(
            traces=list(traces),
            addressed=addressed,
            skipped_target_ids=skipped,
        )
    return _execute
