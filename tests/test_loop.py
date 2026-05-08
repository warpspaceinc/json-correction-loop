"""Library-level smoke tests for json_correction_loop.run_correction_loop.

These tests use only the library — no host application imports — to confirm
the loop driver, planner, executor, convergence policies, and storage /
event Protocols compose correctly into a working 3-stage cycle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from json_correction_loop import (
    CorrectionLoopConfig,
    CriticIssue,
    CriticReport,
    Event,
    HardcapPolicy,
    IterationRecord,
    QualityStablePolicy,
    make_callback_executor,
    make_identity_planner,
    run_correction_loop,
)


@dataclass
class _FakeTrace:
    id: str
    requirement_id: str
    addressed: bool
    reason: str = ""


class _CapturingSink:
    def __init__(self):
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)


@dataclass
class _CapturingStorage:
    saves: list[IterationRecord] = field(default_factory=list)
    stamps: list[tuple[str, list[str], bool]] = field(default_factory=list)

    def save_iteration(self, record: IterationRecord) -> None:
        self.saves.append(record)

    def stamp_outcome(self, level: str, trace_ids: list[str], converged: bool) -> None:
        self.stamps.append((level, list(trace_ids), converged))


def _parse_target_ids(issues):
    paths: list[str] = []
    fb: dict[str, str] = {}
    for iss in issues:
        tid = (iss.target_id or "").strip()
        if tid and tid not in fb:
            paths.append(tid)
            fb[tid] = iss.description
    return paths, fb


def _make_report(iteration: int, issues: list[tuple[str, str]]) -> CriticReport:
    return CriticReport(
        id=f"cr-iter{iteration}",
        level="test",
        iteration=iteration,
        issues=[
            CriticIssue(target_id=tid, severity=sev, description=f"fix {tid}")
            for tid, sev in issues
        ],
        score=10 if not issues else 5,
    )


def test_run_correction_loop_approves_immediately_on_zero_issues():
    sink = _CapturingSink()
    storage = _CapturingStorage()
    planner = make_identity_planner(_parse_target_ids)
    executor = make_callback_executor(
        lambda state, paths, fb, model: []
    )
    cfg = CorrectionLoopConfig(
        level="test", max_loops=3, storage=storage, events=sink,
    )
    state: dict[str, Any] = {}
    ok = run_correction_loop(
        state, cfg,
        gather_fn=lambda st, i, m: _make_report(i, []),
        plan_fn=planner,
        execute_fn=executor,
    )
    assert ok is True
    assert any(e.kind == "approved" for e in sink.events)
    assert any(e.kind == "loop_end" and e.data["reason"] == "approved" for e in sink.events)
    # One iteration's snapshot was stored.
    assert len(storage.saves) == 1


def test_run_correction_loop_drives_planner_and_executor_until_clean():
    """Iter 1: 1 issue. Iter 2: 0 issues → approved."""
    sink = _CapturingSink()
    storage = _CapturingStorage()
    planner = make_identity_planner(_parse_target_ids)
    apply_calls: list[list[str]] = []

    def _apply(state, paths, fb, model):
        apply_calls.append(list(paths))
        return [_FakeTrace(id=f"tr-{p}", requirement_id=p, addressed=True) for p in paths]

    executor = make_callback_executor(_apply)
    state: dict[str, Any] = {"iter": 0}
    reports = [
        _make_report(1, [("scene-1", "major")]),
        _make_report(2, []),
    ]

    def _gather(st, i, m):
        return reports[i - 1]

    cfg = CorrectionLoopConfig(
        level="test", max_loops=5, storage=storage, events=sink,
    )
    ok = run_correction_loop(
        state, cfg, gather_fn=_gather, plan_fn=planner, execute_fn=executor,
    )
    assert ok is True
    # Executor called once with [scene-1]; second iter approved before plan.
    assert apply_calls == [["scene-1"]]
    # Three save_iteration calls — iter-1 critic-only snapshot, iter-1
    # post-execute manifest, iter-2 critic-only approval.
    assert len(storage.saves) == 3
    assert storage.saves[0].correction_plan is None
    assert storage.saves[1].correction_plan is not None
    assert storage.saves[1].traces and storage.saves[1].traces[0].id == "tr-scene-1"
    assert storage.saves[2].correction_plan is None  # approval snapshot
    # Outcome stamped with the executor's trace.
    assert storage.stamps == [("test", ["tr-scene-1"], True)]


def test_run_correction_loop_hardcap_exits_with_outcome_false():
    sink = _CapturingSink()
    storage = _CapturingStorage()
    planner = make_identity_planner(_parse_target_ids)

    def _apply(state, paths, fb, model):
        return [_FakeTrace(id=f"tr-{i}-{p}", requirement_id=p, addressed=False, reason="stuck")
                for i, p in enumerate(paths)]

    executor = make_callback_executor(_apply)

    def _gather(st, i, m):
        # Always 1 critical issue — never converges on quality.
        return _make_report(i, [("scene-1", "critical")])

    cfg = CorrectionLoopConfig(
        level="test", max_loops=20, hardcap=3, storage=storage, events=sink,
    )
    ok = run_correction_loop(
        {}, cfg, gather_fn=_gather, plan_fn=planner, execute_fn=executor,
    )
    assert ok is True
    assert any(e.kind == "hardcap" for e in sink.events)
    # Stamp outcome=False because hardcap is structural failure.
    assert storage.stamps and storage.stamps[0][2] is False


def test_run_correction_loop_quality_stable_short_circuits():
    """Three consecutive iters with score=9, no critical/major → converged."""
    sink = _CapturingSink()
    storage = _CapturingStorage()
    planner = make_identity_planner(_parse_target_ids)
    executor = make_callback_executor(
        lambda state, paths, fb, model: [
            _FakeTrace(id=f"tr-{p}", requirement_id=p, addressed=True) for p in paths
        ]
    )
    # Each iter has a single minor issue → score stays high but issue list non-empty.
    def _gather(st, i, m):
        r = CriticReport(
            id=f"cr-{i}",
            level="test",
            iteration=i,
            issues=[CriticIssue(target_id="scene-1", severity="minor", description="nit")],
            score=9,
        )
        return r

    cfg = CorrectionLoopConfig(
        level="test", max_loops=10, stable_n=3, accept_score=8,
        storage=storage, events=sink,
    )
    ok = run_correction_loop(
        {}, cfg, gather_fn=_gather, plan_fn=planner, execute_fn=executor,
    )
    assert ok is True
    converged = [e for e in sink.events if e.kind == "converged"]
    assert converged and "안정" in converged[0].data["reason"]


def test_quality_stable_policy_requires_zero_critical_and_major():
    p = QualityStablePolicy(stable_n=2, accept_score=7)
    # Three iters all with 0 critical, 0 major, score 9 → converged.
    history = [
        (9, 0, 0, frozenset()),
        (9, 0, 0, frozenset()),
    ]
    assert p.check(history)[0] is True
    # Major present blocks convergence.
    history2 = [
        (9, 0, 1, frozenset(["x"])),
        (9, 0, 1, frozenset(["x"])),
    ]
    assert p.check(history2)[0] is False


def test_hardcap_policy_fires_at_cap():
    p = HardcapPolicy(cap=3)
    assert p.check([(1, 1, 0, frozenset())] * 2)[0] is False
    assert p.check([(1, 1, 0, frozenset())] * 3)[0] is True
