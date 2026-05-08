"""End-to-end config-refactor example — JSON Schema critic + simple patcher."""
from __future__ import annotations

import json
import sys

from json_correction_loop import (
    CorrectionLoopConfig,
    Event,
    EventSink,
    QualityStablePolicy,
    make_callback_executor,
    make_identity_planner,
    run_correction_loop,
)

from config import broken, clean
from patcher import make_patcher
from schema import schema_critic


def parse(issues):
    seen, paths, fb = set(), [], {}
    for iss in issues:
        tid = (iss.target_id or "").strip()
        if not tid or tid in seen:
            continue
        seen.add(tid)
        paths.append(tid)
        fb[tid] = iss.description
    return paths, fb


class TerseSink(EventSink):
    def emit(self, event: Event) -> None:
        if event.kind == "iter_start":
            print(f"\n── iter {event.data['iter']}/{event.data['max_loops']} ──")
        elif event.kind == "critic_report":
            d = event.data
            print(f"  critic score={d['score']} C={d['critical']} M={d['major']} m={d['minor']}")
        elif event.kind == "approved":
            print(f"  ✓ approved at iter {event.data['iter']}")


def main():
    state = broken()
    print("== broken config ==")
    print(json.dumps(state, indent=2))

    cfg = CorrectionLoopConfig(
        level="config",
        max_loops=4,
        events=TerseSink(),
        quality_policy=QualityStablePolicy(stable_n=2, accept_score=8),
    )

    converged = run_correction_loop(
        state, cfg,
        gather_fn=schema_critic,
        plan_fn=make_identity_planner(parse),
        execute_fn=make_callback_executor(make_patcher()),
    )

    print("\n== after correction ==")
    print(json.dumps(state, indent=2))

    final_issues = sum(len(r.issues) for r in schema_critic(state, iteration=999))
    matches_clean = json.dumps(state, sort_keys=True) == json.dumps(clean(), sort_keys=True)
    print(f"\nconverged: {converged}")
    print(f"residual schema violations: {final_issues}")
    print(f"matches clean baseline: {matches_clean}")
    return 0 if converged and final_issues == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
