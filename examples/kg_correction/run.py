"""End-to-end KG correction example — no LLM required.

Uses the deterministic oracle patcher to demonstrate how the json-
correction loop wires gather → plan → execute on a structured KG
state. Swap ``oracle.OraclePatcher`` for an LLM-driven RFC 6902
patcher to get the surgical-edit setup measured in the paper.
"""
from __future__ import annotations

import argparse
import copy
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

from critic import kg_critic
from kg import clean_kg, kg_size
from oracle import OraclePatcher
from perturb import apply_perturbations


def kg_target_parser(issues):
    """Map critic issues → (flagged_paths, feedback_by_path).

    Each ``target_id`` is already a JSON pointer; we deduplicate and
    aggregate descriptions per pointer.
    """
    seen: set[str] = set()
    paths: list[str] = []
    feedback: dict[str, str] = {}
    for iss in issues:
        tid = (iss.target_id or "").strip()
        if not tid:
            continue
        if tid not in seen:
            seen.add(tid)
            paths.append(tid)
            feedback[tid] = iss.description
        else:
            feedback[tid] = (feedback[tid] + " | " + iss.description).strip(" |")
    return paths, feedback


class ConsoleSink(EventSink):
    def emit(self, event: Event) -> None:
        kind, d = event.kind, event.data
        if kind == "iter_start":
            print(f"\n── iter {d['iter']}/{d['max_loops']} (level={d['level']}) ──")
        elif kind == "critic_report":
            print(
                f"  critic score={d['score']} "
                f"C={d['critical']} M={d['major']} m={d['minor']} "
                f":: {d['assessment']}"
            )
        elif kind == "approved":
            print(f"  ✓ approved at iter {d['iter']}")


def critic_tag_counts(reports) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in reports:
        for iss in r.issues:
            counts[iss.issue_type] = counts.get(iss.issue_type, 0) + 1
    return counts


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-defects", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-loops", type=int, default=5)
    args = p.parse_args(argv)

    clean = clean_kg()
    n_e, n_x = kg_size(clean)
    print(f"[setup] clean KG: {n_e} entities, {n_x} edges")

    perturbed, defects = apply_perturbations(clean, n=args.n_defects, seed=args.seed)
    print(f"[setup] injected {len(defects)} defects")

    initial_reports = kg_critic(perturbed, iteration=0)
    print(
        f"[critic-pre] {critic_tag_counts(initial_reports)}, "
        f"score={initial_reports[0].score}"
    )

    state = copy.deepcopy(perturbed)
    patcher = OraclePatcher(defects)
    cfg = CorrectionLoopConfig(
        level="kg",
        max_loops=args.max_loops,
        hardcap=args.max_loops,
        accept_score=8,
        stable_n=2,
        events=ConsoleSink(),
        quality_policy=QualityStablePolicy(stable_n=2, accept_score=8),
    )

    converged = run_correction_loop(
        state, cfg,
        gather_fn=kg_critic,
        plan_fn=make_identity_planner(kg_target_parser),
        execute_fn=make_callback_executor(patcher),
    )

    final_reports = kg_critic(state, iteration=999)
    final_tags = critic_tag_counts(final_reports)
    print(f"\n[critic-post] {final_tags}, score={final_reports[0].score}")
    print(f"[loop] converged={converged}")

    initial_total = sum(critic_tag_counts(initial_reports).values())
    final_total = sum(final_tags.values())
    resolved = max(0, initial_total - final_total)
    fix_rate = (resolved / initial_total) if initial_total else 1.0
    print(f"[metric] structural fix rate: {resolved}/{initial_total} = {fix_rate:.0%}")

    drift = 0 if json.dumps(clean, sort_keys=True) == json.dumps(state, sort_keys=True) else 1
    print(f"[metric] state drift vs clean: {drift} (0 = byte-identical)")

    return 0 if converged and fix_rate == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
