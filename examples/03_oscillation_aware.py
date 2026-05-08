"""Oscillation-aware planner — break out of a stuck critic loop.

Sometimes a critic and a patcher disagree philosophically: the critic
keeps re-flagging the same target every iteration, the patcher keeps
"fixing" it the same wrong way, and the loop spins forever. The
oscillation-aware planner detects this — any target re-flagged for
``threshold`` consecutive iterations is dropped from subsequent plans
with reason ``contradicts_higher``.

This example contrives that situation: a critic that ALWAYS flags
``/disputed`` no matter what, and a patcher that toggles a value
between two states (so the critic always finds something to complain
about).

Run::

    python examples/03_oscillation_aware.py
"""
from __future__ import annotations

import sys

from json_correction_loop import (
    CorrectionLoopConfig,
    CriticIssue,
    CriticReport,
    HardcapPolicy,
    QualityStablePolicy,
    make_callback_executor,
    make_identity_planner,
    make_oscillation_aware_planner,
    run_correction_loop,
)


def stubborn_critic(state, iteration, model):
    """Always flag /disputed. The patcher will toggle it; the critic
    will flag it again. Without an oscillation policy, this never
    terminates."""
    issues = [
        CriticIssue(
            target_id="/disputed",
            severity="major",
            issue_type="never_satisfied",
            description=f"value is currently {state['disputed']!r} (critic disagrees)",
        ),
    ]
    return [CriticReport(issues=issues, score=4)]


def toggling_patcher(state, flagged_paths, feedback_by_path, model):
    """Flip /disputed between True/False."""
    traces = []
    for path in flagged_paths:
        if path == "/disputed":
            state["disputed"] = not state["disputed"]
            traces.append(type("T", (), {
                "id": f"t-{state['disputed']}",
                "requirement_id": path,
                "addressed": True,
                "reason": f"toggled to {state['disputed']}",
            })())
    return traces


def parse(issues):
    return (
        [iss.target_id for iss in issues],
        {iss.target_id: iss.description for iss in issues},
    )


def run_with(planner_factory, *, label: str) -> bool:
    state = {"disputed": False, "stable_value": "untouched"}
    cfg = CorrectionLoopConfig(
        level="disputed",
        max_loops=8,
        hardcap=8,
        # Use a permissive QualityStable so it doesn't terminate on
        # quality alone — we want to see the oscillation policy fire.
        quality_policy=QualityStablePolicy(stable_n=10, accept_score=10),
    )
    print(f"\n── {label} ──")
    ok = run_correction_loop(
        state, cfg,
        gather_fn=stubborn_critic,
        plan_fn=planner_factory(parse),
        execute_fn=make_callback_executor(toggling_patcher),
    )
    print(f"  loop returned: {ok}")
    print(f"  final state: {state}")
    return ok


def main():
    print("Without oscillation awareness — runs to hardcap, patcher")
    print("flips /disputed every iteration.")
    run_with(make_identity_planner, label="identity planner")

    print("\nWith oscillation awareness (threshold=3) — after the")
    print("target has been flagged 3 iterations in a row, the planner")
    print("drops it on the 4th iteration and the loop converges.")
    run_with(
        lambda parse_fn: make_oscillation_aware_planner(parse_fn, threshold=3),
        label="oscillation-aware planner",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
