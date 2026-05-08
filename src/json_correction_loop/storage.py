"""StorageBackend Protocol — persists each iteration of the loop.

The library does NOT prescribe a storage shape. It hands the
StorageBackend a per-iteration record (critic reports, plan, traces,
addressed/skipped summaries) and trusts the backend to translate that
into whatever the domain wants (MongoDB snapshots, file commits,
in-memory ring buffer, ...).

A second hook, :meth:`StorageBackend.stamp_outcome`, lets the loop
retroactively mark every trace from this run as ``loop_converged=True``
or ``False`` once the loop's outcome is known. Domain backends update
their stored records accordingly; the in-library
:class:`NullStorageBackend` no-ops.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from json_correction_loop.models import CorrectionPlan, CriticReport


@dataclass
class IterationRecord:
    """Snapshot of one iteration handed to the StorageBackend.

    Domain backends typically:
      - append the reports to ``state.critic_history`` (or equivalent),
      - write the manifest (correction_plan + addressed + traces) to a
        snapshot store keyed by (level, iteration),
      - emit any human-readable side effects (commit message, log line).
    """
    level: str
    iteration: int
    reports: list[CriticReport]
    correction_plan: CorrectionPlan | None = None
    addressed_target_ids: list[str] = field(default_factory=list)
    skipped_target_ids: list[str] = field(default_factory=list)
    traces: list[Any] = field(default_factory=list)  # PatcherTraceLike
    commit_msg: str = ""


class StorageBackend(Protocol):
    """Per-iteration persistence hook."""
    def save_iteration(self, record: IterationRecord) -> None: ...
    def stamp_outcome(self, level: str, trace_ids: list[str], converged: bool) -> None: ...


class NullStorageBackend:
    """No-op backend used in tests where persistence is irrelevant."""
    def save_iteration(self, record: IterationRecord) -> None:  # noqa: ARG002
        pass

    def stamp_outcome(self, level: str, trace_ids: list[str], converged: bool) -> None:  # noqa: ARG002
        pass
