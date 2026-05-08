"""Minimal end-to-end example — no LLM required.

Wires a tiny critic + executor through ``run_correction_loop`` to show
how the pieces compose. Run with: ``python examples/01_quickstart.py``.
"""
from json_correction_loop import (
    CorrectionLoopConfig,
    CriticIssue, CriticReport,
    make_callback_executor,
    make_identity_planner,
    run_correction_loop,
)


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
    print(f"before: {state}")
    cfg = CorrectionLoopConfig(level="items", max_loops=5)
    ok = run_correction_loop(
        state, cfg,
        gather_fn=gather,
        plan_fn=make_identity_planner(parse),
        execute_fn=make_callback_executor(apply_one),
    )
    print(f"after : {state}")
    print(f"converged: {ok}")
    assert ok and all(item["ok"] for item in state["items"])


if __name__ == "__main__":
    main()
