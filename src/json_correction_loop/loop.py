"""Universal correction loop driver.

Drives the ``gather → plan → execute`` cycle for any JSON-shaped
domain state. The library doesn't import any domain code — every
side effect (persistence, console printing, LLM calls) flows through
caller-supplied callbacks (``gather_fn`` / ``plan_fn`` / ``execute_fn``)
or Protocols (``StorageBackend`` / ``EventSink``).

Convergence policy is composable: by default a
:class:`QualityStablePolicy` decides early termination, with a
hardcap as a separate guard. Both are checked each iteration; either
firing terminates the loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from json_correction_loop.convergence import (
    ConvergencePolicy,
    HardcapPolicy,
    HistoryEntry,
    QualityStablePolicy,
)
from json_correction_loop.events import Event, EventSink, NullEventSink
from json_correction_loop.executors import ExecutorFn
from json_correction_loop.models import (
    CorrectionPlan,
    CriticReport,
)
from json_correction_loop.planners import PlannerFn
from json_correction_loop.storage import (
    IterationRecord,
    NullStorageBackend,
    StorageBackend,
)


GatherFn = Callable[[Any, int, "str | None"], list[CriticReport]]
"""(state, iter, model) → list[CriticReport]. Domain code wraps one or
more critic calls and returns their reports for this iteration."""


HistoryProviderFn = Callable[[Any, str], list[CriticReport]]
"""(state, level) → list[CriticReport]. Returns the per-level slice of
the project's prior critic history (most-recent-last) so the planner
can detect oscillation. Defaults to ``lambda st, lvl: []`` — i.e. no
history available."""


@dataclass
class CorrectionLoopConfig:
    """All loop knobs in one place.

    Required fields are wired by the domain caller; the rest have
    sensible defaults (NullEventSink, NullStorageBackend, identity
    history provider, QualityStablePolicy + HardcapPolicy(20)).
    """
    level: str
    max_loops: int
    hardcap: int = 20
    accept_score: int = 7
    stable_n: int = 3
    storage: StorageBackend = field(default_factory=NullStorageBackend)
    events: EventSink = field(default_factory=NullEventSink)
    history_provider: HistoryProviderFn | None = None
    quality_policy: ConvergencePolicy | None = None  # defaults to QualityStablePolicy
    history_window: int = 5  # how many recent reports to forward to the planner


def _severity_counts(report: CriticReport) -> tuple[int, int, int]:
    c = sum(1 for iss in report.issues if iss.severity == "critical")
    m = sum(1 for iss in report.issues if iss.severity == "major")
    n = sum(1 for iss in report.issues if iss.severity == "minor")
    return c, m, n


def run_correction_loop(
    state: Any,
    config: CorrectionLoopConfig,
    *,
    gather_fn: GatherFn,
    plan_fn: PlannerFn,
    execute_fn: ExecutorFn,
    model: str | None = None,
) -> bool:
    """Run the ``gather → plan → execute`` loop until convergence or
    ``max_loops``. Returns ``True`` on graceful exit (approval,
    quality-stable, hardcap, or natural max_loops); never raises on
    convergence — exceptions propagate from gather/plan/execute only
    when the caller's callbacks raise.

    The loop:

      1. Calls ``gather_fn`` for fresh critic reports.
      2. Computes severity counts + records a history tuple.
      3. Checks the convergence policy + hardcap.
      4. Persists the iteration via ``StorageBackend`` (with
         ``correction_plan=None`` for the critic-only snapshot).
      5. Bails out early on approval / quality-stable / hardcap.
      6. Calls ``plan_fn`` for the iteration's CorrectionPlan.
      7. If the plan has no corrections, persists an empty manifest
         (capturing planner skips/rationale) and continues.
      8. Calls ``execute_fn``; persists the result manifest with traces.
      9. Loops.

    The convergence outcome (True/False) is stamped onto every trace
    this loop produced via ``StorageBackend.stamp_outcome`` so
    downstream analytics can distinguish "patch landed in iter N" from
    "loop these patches were part of actually converged".
    """
    quality_policy = config.quality_policy or QualityStablePolicy(
        stable_n=config.stable_n, accept_score=config.accept_score,
    )
    hardcap_policy = HardcapPolicy(cap=config.hardcap)
    history_provider: HistoryProviderFn = config.history_provider or (lambda _st, _lvl: [])

    convergence_hist: list[HistoryEntry] = []
    loop_trace_ids: list[str] = []
    loop_converged_outcome = False

    def _stamp_outcome():
        if loop_trace_ids:
            config.storage.stamp_outcome(
                config.level, loop_trace_ids, loop_converged_outcome,
            )

    for i in range(1, config.max_loops + 1):
        config.events.emit(Event("iter_start", {
            "iter": i, "max_loops": config.max_loops, "level": config.level,
        }))

        # ── Stage 1: gather ──────────────────────────────────────────
        reports = gather_fn(state, i, model)
        if not isinstance(reports, list):
            reports = [reports]
        for r in reports:
            r.level = config.level
            r.iteration = i

        merged_critical = sum(_severity_counts(r)[0] for r in reports)
        merged_major = sum(_severity_counts(r)[1] for r in reports)
        merged_minor = sum(_severity_counts(r)[2] for r in reports)
        merged_score = min((r.score for r in reports), default=10)
        total = merged_critical + merged_major + merged_minor

        target_set = frozenset(
            iss.target_id for r in reports for iss in r.issues if iss.target_id
        )

        convergence_tag = ""
        early_exit_reason = ""
        if total > 0:
            convergence_hist.append(
                (merged_score, merged_critical, merged_major, target_set)
            )
            converged, reason = quality_policy.check(convergence_hist)
            if converged:
                early_exit_reason = reason
                convergence_tag = "converged"
            else:
                hard_done, hard_reason = hardcap_policy.check(convergence_hist)
                if hard_done:
                    early_exit_reason = hard_reason
                    convergence_tag = "hardcap"
            if early_exit_reason:
                for r in reports:
                    r.convergence_reason = early_exit_reason

        commit_msg = (
            f"critic:{config.level} iter={i} score={merged_score}"
            f" C={merged_critical} M={merged_major} m={merged_minor}"
        )
        if convergence_tag:
            commit_msg += f" {convergence_tag}"

        config.storage.save_iteration(IterationRecord(
            level=config.level,
            iteration=i,
            reports=reports,
            correction_plan=None,
            commit_msg=commit_msg,
        ))

        for r in reports:
            c, m, n = _severity_counts(r)
            config.events.emit(Event("critic_report", {
                "level": r.level or config.level,
                "score": r.score,
                "critical": c, "major": m, "minor": n,
                "assessment": (r.overall_assessment or "")[:120],
            }))

        if total == 0:
            config.events.emit(Event("approved", {"iter": i}))
            loop_converged_outcome = True
            _stamp_outcome()
            config.events.emit(Event("loop_end", {"reason": "approved"}))
            return True

        if convergence_tag == "converged":
            config.events.emit(Event("converged", {"reason": early_exit_reason}))
            loop_converged_outcome = True
            _stamp_outcome()
            config.events.emit(Event("loop_end", {"reason": "converged"}))
            return True

        if convergence_tag == "hardcap":
            config.events.emit(Event("hardcap", {"cap": config.hardcap}))
            loop_converged_outcome = False
            _stamp_outcome()
            config.events.emit(Event("loop_end", {"reason": "hardcap"}))
            return True

        # ── Stage 2: plan ────────────────────────────────────────────
        history = list(history_provider(state, config.level))[-config.history_window:]
        try:
            plan = plan_fn(state, reports, history, model)
        except Exception as exc:
            config.events.emit(Event("planner_failed", {"error": str(exc)}))
            continue
        if not isinstance(plan, CorrectionPlan):
            config.events.emit(Event("planner_failed", {
                "error": f"non-CorrectionPlan {type(plan).__name__}",
            }))
            continue

        if plan.skipped:
            config.events.emit(Event("planner_skipped", {
                "count": len(plan.skipped),
                "rationale": plan.rationale[:120],
            }))

        if not plan.corrections:
            config.events.emit(Event("planner_empty", {"rationale": plan.rationale}))
            config.storage.save_iteration(IterationRecord(
                level=config.level,
                iteration=i,
                reports=reports,
                correction_plan=plan,
                addressed_target_ids=[],
                skipped_target_ids=[s.target_id for s in plan.skipped],
                commit_msg=(
                    f"revise:{config.level} iter={i} planner-empty"
                ),
            ))
            continue

        # ── Stage 3: execute ─────────────────────────────────────────
        try:
            result = execute_fn(state, plan, model)
        except Exception as exc:
            config.events.emit(Event("executor_failed", {"error": str(exc)}))
            continue

        config.events.emit(Event("executor_result", {
            "addressed": len(result.addressed),
            "total": len(plan.corrections),
            # Forward the traces so verbose sinks can print per-op
            # detail (target_id, target_pointer, calls, critic_error).
            # Casts to dict via model_dump for consumers that don't
            # want a hard dependency on PatcherTrace.
            "traces": [
                t.model_dump() if hasattr(t, "model_dump") else t
                for t in (result.traces or [])
            ],
        }))

        loop_trace_ids.extend(getattr(t, "id", "") for t in result.traces if getattr(t, "id", None))

        config.storage.save_iteration(IterationRecord(
            level=config.level,
            iteration=i,
            reports=reports,
            correction_plan=plan,
            addressed_target_ids=list(result.addressed),
            skipped_target_ids=list(result.skipped_target_ids),
            traces=list(result.traces),
            commit_msg=(
                f"revise:{config.level} iter={i}"
                f" addressed={len(result.addressed)}/{len(plan.corrections)}"
                f" planner={plan.planner_kind}"
            ),
        ))

    # Natural max_loops exit.
    loop_converged_outcome = False
    _stamp_outcome()
    config.events.emit(Event("loop_end", {"reason": "max_loops"}))
    return True
