"""Custom StorageBackend + EventSink — the Protocol pattern in action.

Demonstrates how to plug your own observability into the loop without
the library knowing what you're doing. Same trivial domain as
``01_quickstart.py``, but every iteration is persisted to a JSONL
file and every event is printed via a Rich-flavored console sink (no
Rich dependency — just ANSI colors).

Run::

    python examples/02_observability.py
    cat /tmp/jcl-iterations.jsonl

What this shows:

  - ``StorageBackend`` and ``EventSink`` are Protocols. Any class
    with the right method signatures works — no inheritance required.
  - ``IterationRecord`` carries everything you need to reconstruct a
    full timeline of a run for debugging or analytics.
  - The library calls ``stamp_outcome`` once at loop end so storage
    backends can retroactively mark whether a run converged.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from json_correction_loop import (
    CorrectionLoopConfig,
    CriticIssue,
    CriticReport,
    Event,
    EventSink,
    IterationRecord,
    StorageBackend,
    make_callback_executor,
    make_identity_planner,
    run_correction_loop,
)


# ── Observability backends ────────────────────────────────────────────────


class JsonlStorage(StorageBackend):
    """Append each iteration as one JSON line to a file."""

    def __init__(self, path: Path):
        self.path = path
        self.path.write_text("")  # truncate
        self._trace_outcomes: dict[str, bool] = {}

    def save_iteration(self, record: IterationRecord) -> None:
        line = {
            "level": record.level,
            "iteration": record.iteration,
            "n_reports": len(record.reports),
            "n_issues": sum(len(r.issues) for r in record.reports),
            "score": min((r.score for r in record.reports), default=10),
            "addressed": list(record.addressed_target_ids),
            "skipped": list(record.skipped_target_ids),
            "n_traces": len(record.traces),
            "commit_msg": record.commit_msg,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(line) + "\n")

    def stamp_outcome(self, level: str, trace_ids: list[str], converged: bool) -> None:
        for tid in trace_ids:
            self._trace_outcomes[tid] = converged
        with self.path.open("a") as f:
            f.write(json.dumps({
                "_meta": "stamp_outcome",
                "level": level,
                "n_traces": len(trace_ids),
                "converged": converged,
            }) + "\n")


# ANSI helpers — keep dependency-free.
_DIM = "\033[2m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_RESET = "\033[0m"


class AnsiSink(EventSink):
    """Print events with light ANSI styling. Drop-in for production
    sinks that go to log/Rich/structlog/etc.
    """

    def emit(self, event: Event) -> None:
        kind, d = event.kind, event.data
        if kind == "iter_start":
            print(f"\n{_BOLD}── iter {d['iter']}/{d['max_loops']} (level={d['level']}){_RESET}")
        elif kind == "critic_report":
            color = _GREEN if d["score"] >= 8 else _YELLOW if d["score"] >= 4 else _BLUE
            print(
                f"  {_DIM}critic{_RESET} score={color}{d['score']}{_RESET} "
                f"C={d['critical']} M={d['major']} m={d['minor']}: "
                f"{d['assessment'][:60]}"
            )
        elif kind == "approved":
            print(f"  {_GREEN}✓ approved at iter {d['iter']}{_RESET}")
        elif kind == "executor_result":
            print(
                f"  {_DIM}executor:{_RESET} "
                f"addressed {d['addressed']}/{d['total']}"
            )
        elif kind == "loop_end":
            print(f"  {_DIM}loop_end:{_RESET} {d['reason']}")


# ── Domain (same as 01_quickstart) ────────────────────────────────────────


def gather(state, iteration, model):
    issues = [
        CriticIssue(
            target_id=f"/items/{i}/ok",
            severity="major",
            issue_type="needs_fix",
            description=f"set ok=True on item {item['id']}",
        )
        for i, item in enumerate(state["items"]) if not item["ok"]
    ]
    return [CriticReport(issues=issues, score=10 if not issues else 4)]


def apply_one(state, flagged_paths, feedback_by_path, model):
    traces = []
    for path in flagged_paths:
        idx = int(path.strip("/").split("/")[1])
        state["items"][idx]["ok"] = True
        traces.append(type("T", (), {
            "id": f"t-{idx}", "requirement_id": path,
            "addressed": True, "reason": "set ok=True",
        })())
    return traces


def parse(issues):
    return (
        [iss.target_id for iss in issues],
        {iss.target_id: iss.description for iss in issues},
    )


def main():
    state = {"items": [
        {"id": "a", "ok": False},
        {"id": "b", "ok": True},
        {"id": "c", "ok": False},
    ]}
    log_path = Path("/tmp/jcl-iterations.jsonl")
    print(f"writing iteration log → {log_path}")

    cfg = CorrectionLoopConfig(
        level="items",
        max_loops=5,
        storage=JsonlStorage(log_path),
        events=AnsiSink(),
    )
    ok = run_correction_loop(
        state, cfg,
        gather_fn=gather,
        plan_fn=make_identity_planner(parse),
        execute_fn=make_callback_executor(apply_one),
    )
    print(f"\nfinal state: {state}")
    print(f"converged: {ok}")
    print(f"\n--- {log_path} contents ---")
    print(log_path.read_text())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
