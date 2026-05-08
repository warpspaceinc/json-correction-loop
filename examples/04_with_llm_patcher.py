"""End-to-end real-LLM example — emit RFC 6902 patches via OpenAI API.

This is the smallest example that calls a real LLM. The domain is
intentionally tiny so the token cost is low and the demo runs in a few
seconds.

Setup
-----

Set one of these in your environment::

    export OPENAI_API_KEY=sk-...                 # OpenAI direct
    # or
    export OPENROUTER_API_KEY=sk-or-...          # OpenRouter

Then::

    pip install openai
    python examples/04_with_llm_patcher.py

What this shows
---------------

  - A minimal LLM-backed executor: prompt the model for a JSON Patch
    op array, validate with the ``jsonpatch`` library, apply.
  - The same loop driver as the no-LLM examples — only the executor
    changes. ``gather_fn`` and ``plan_fn`` are identical to
    ``01_quickstart.py``.
  - The "happy path" only: no path_finder, no narrowing, no
    sub-agents. For the full stack on a non-trivial KG, see the
    EXPERIMENTS.md harness.

Notes
-----

  - We use ``response_format={"type": "json_object"}`` for portability;
    OpenRouter and most providers accept it.
  - Errors and retries are deliberately simple — production code
    would use the ``json_correction_loop.tracking.TrackingLLMClient``
    wrapper for cost accounting and structured retries.
"""
from __future__ import annotations

import json
import os
import sys

try:
    import jsonpatch  # type: ignore[import-not-found]
    from openai import OpenAI
except ImportError as exc:  # noqa: BLE001
    print(
        "missing dependency:", exc,
        "\nrun: pip install openai jsonpatch",
        file=sys.stderr,
    )
    sys.exit(1)

from json_correction_loop import (
    CorrectionLoopConfig,
    CriticIssue,
    CriticReport,
    QualityStablePolicy,
    make_callback_executor,
    make_identity_planner,
    run_correction_loop,
)


# ── Tiny domain: a "team" object whose members must have a non-empty
#    name and a role from a fixed enum. ──────────────────────────────────

ALLOWED_ROLES = {"engineer", "manager", "designer"}


def critic(state, iteration, model):
    issues: list[CriticIssue] = []
    for i, m in enumerate(state.get("members", [])):
        if not m.get("name"):
            issues.append(CriticIssue(
                target_id=f"/members/{i}/name",
                severity="critical",
                issue_type="empty_name",
                description=f"member {i} has an empty name; replace with a placeholder name like 'member-{i}'.",
            ))
        if m.get("role") not in ALLOWED_ROLES:
            issues.append(CriticIssue(
                target_id=f"/members/{i}/role",
                severity="major",
                issue_type="bad_role",
                description=(
                    f"member {i}'s role {m.get('role')!r} is not in the "
                    f"allowed set {sorted(ALLOWED_ROLES)}. Replace with one of these."
                ),
            ))
    score = 10 if not issues else max(1, 10 - len(issues))
    return [CriticReport(issues=issues, score=score, overall_assessment=f"{len(issues)} issues")]


def parse(issues):
    return (
        [iss.target_id for iss in issues],
        {iss.target_id: iss.description for iss in issues},
    )


# ── LLM-backed executor ──────────────────────────────────────────────────


def make_llm_executor():
    """Returns a callback executor that asks the LLM for RFC 6902 ops."""
    base_url = os.environ.get("OPENROUTER_API_KEY") and "https://openrouter.ai/api/v1" or None
    api_key = (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        print("set OPENROUTER_API_KEY or OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(api_key=api_key, **({"base_url": base_url} if base_url else {}))
    model = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini" if base_url else "gpt-4o-mini")

    def _apply(state, flagged_paths, feedback_by_path, model_arg=None):
        if not flagged_paths:
            return []

        issues_block = "\n".join(
            f"  - {p}: {feedback_by_path.get(p, '')}" for p in flagged_paths
        )
        sys_msg = (
            "You are editing a JSON document. Emit a JSON Patch (RFC 6902) "
            "as an array of ops that fixes ALL flagged issues with the "
            "MINIMUM possible edit footprint. Touch only fields the issues "
            "name. Allowed roles: engineer, manager, designer.\n\n"
            "Respond as: {\"ops\": [{\"op\": \"replace\", \"path\": \"...\", \"value\": ...}, ...]}"
        )
        user_msg = (
            f"Document:\n```json\n{json.dumps(state, indent=2)}\n```\n\n"
            f"Issues:\n{issues_block}"
        )
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = json.loads(resp.choices[0].message.content or "{}")
        ops = raw.get("ops", []) if isinstance(raw, dict) else []
        applied = 0
        traces = []
        for op in ops:
            try:
                jsonpatch.JsonPatch([op]).apply(state, in_place=True)
                applied += 1
            except Exception as exc:  # noqa: BLE001
                traces.append(type("T", (), {
                    "id": f"t-fail-{len(traces)}",
                    "requirement_id": op.get("path", "?"),
                    "addressed": False,
                    "reason": f"invalid op: {exc!s}"[:120],
                })())
        for path in flagged_paths:
            traces.append(type("T", (), {
                "id": f"t-{path.replace('/', '-')}",
                "requirement_id": path,
                "addressed": applied > 0,
                "reason": f"applied {applied}/{len(ops)} ops",
            })())
        print(f"  [llm] {len(ops)} ops proposed, {applied} applied")
        return traces

    return _apply


def main():
    state = {
        "members": [
            {"name": "Alice",  "role": "engineer"},
            {"name": "",       "role": "engineer"},     # empty name
            {"name": "Carol",  "role": "wizard"},       # bad role
        ],
    }
    print("before:")
    print(json.dumps(state, indent=2))

    cfg = CorrectionLoopConfig(
        level="team", max_loops=4,
        quality_policy=QualityStablePolicy(stable_n=2, accept_score=8),
    )
    ok = run_correction_loop(
        state, cfg,
        gather_fn=critic,
        plan_fn=make_identity_planner(parse),
        execute_fn=make_callback_executor(make_llm_executor()),
    )
    print("\nafter:")
    print(json.dumps(state, indent=2))
    print(f"\nconverged: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
