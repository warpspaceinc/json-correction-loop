"""Planners turn a list of CriticReports into a CorrectionPlan.

Two deterministic planners ship with the library:

  * :func:`make_identity_planner` — 1-issue → 1-correction. The
    baseline; equivalent to the legacy "every flagged target gets a
    revise pass" behavior.
  * :func:`make_oscillation_aware_planner` — identity + a streak
    detector. Drops any target_id that has been flagged in the last
    N consecutive iterations (i.e. the patcher has tried to fix it N
    times running and the critic keeps re-flagging — almost certainly
    philosophical disagreement, not a real defect).

LLM planners stay outside the library — callers supply their own
``PlannerFn`` by closing over their domain's prompt + LLM client.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

from json_correction_loop.models import (
    Correction,
    CorrectionPlan,
    CriticIssue,
    CriticReport,
    SkippedIssue,
)


PlannerFn = Callable[
    [Any, list[CriticReport], list[CriticReport], "str | None"],
    CorrectionPlan,
]
"""(state, current_reports, history, model) → CorrectionPlan.

``state`` is the caller's domain state object — opaque to the library.
``history`` is the per-level critic_history slice (most-recent-last)
the loop driver passes in. ``model`` is forwarded so LLM planners can
pick the same backend as the rest of the level; deterministic planners
ignore it.
"""


# Type alias used by both planners — converts a list of issues into
# (sorted_slot_ids, slot_id → aggregated_feedback). Domain-specific.
TargetIdParser = Callable[
    [Sequence[CriticIssue]],
    tuple[list[str], dict[str, str]],
]


def make_identity_planner(target_id_parser: TargetIdParser) -> PlannerFn:
    """Identity planner — maps each unique issue ``target_id`` to one
    :class:`Correction`. No LLM call.

    ``target_id_parser`` is the caller-supplied function that extracts
    addressable slot ids from raw critic issues. It returns
    ``(flagged_paths, feedback_by_path)`` so the planner can attach
    aggregated feedback as the Correction's ``intent``.
    """
    def _plan(state, reports, history, model=None):
        all_issues: list[CriticIssue] = []
        critic_id_by_issue: dict[int, str] = {}
        for r in reports:
            for iss in r.issues:
                critic_id_by_issue[id(iss)] = r.id or ""
                all_issues.append(iss)
        if not all_issues:
            return CorrectionPlan(rationale="no issues", planner_kind="identity")
        flagged_paths, fb_by_path = target_id_parser(all_issues)
        if not flagged_paths:
            return CorrectionPlan(
                rationale="all issues unparseable into addressable slots",
                planner_kind="identity",
            )
        first_issue_by_target: dict[str, CriticIssue] = {}
        for iss in all_issues:
            for raw in iss.target_ids or []:
                tid = (raw or "").strip()
                if not tid:
                    continue
                first_issue_by_target.setdefault(tid, iss)
        corrections: list[Correction] = []
        for slot in flagged_paths:
            iss = first_issue_by_target.get(slot)
            corrections.append(
                Correction(
                    requirement_id=slot,
                    intent=fb_by_path.get(slot, "") or (iss.description if iss else ""),
                    source_critic_id=critic_id_by_issue.get(id(iss)) if iss else None,
                    source_severity=iss.severity if iss else "",
                )
            )
        return CorrectionPlan(
            corrections=corrections,
            rationale="identity planner — every issue → one correction",
            planner_kind="identity",
        )
    return _plan


def make_oscillation_aware_planner(
    target_id_parser: TargetIdParser,
    *,
    threshold: int = 3,
) -> PlannerFn:
    """Identity planner + deterministic oscillation filter.

    Wraps :func:`make_identity_planner`. After producing the baseline
    corrections, it inspects ``history`` and drops any correction whose
    ``requirement_id`` has been re-flagged in the last ``threshold``
    consecutive iterations. The drop is recorded in
    :attr:`CorrectionPlan.skipped` with reason ``contradicts_higher``
    and an explanation noting the consecutive-flag count.

    threshold=3 (default): a target is dropped on its FOURTH flag —
    i.e. after three rounds of failed correction.
    """
    base = make_identity_planner(target_id_parser)

    def _plan(state, reports, history, model=None):
        plan = base(state, reports, history, model)
        if not plan.corrections or not history:
            return plan
        recent = list(history)[-threshold:]
        if len(recent) < threshold:
            return plan
        consecutive_targets: set[str] = set()
        for c in plan.corrections:
            tid = c.requirement_id
            present_in_all = all(
                any(
                    tid in {(t or "").strip() for t in (iss.target_ids or [])}
                    for iss in (h.issues or [])
                )
                for h in recent
            )
            if present_in_all:
                consecutive_targets.add(tid)
        if not consecutive_targets:
            return plan
        kept_corrections = [
            c for c in plan.corrections if c.requirement_id not in consecutive_targets
        ]
        added_skips = [
            SkippedIssue(
                target_id=tid,
                reason="contradicts_higher",
                explanation=(
                    f"oscillation: re-flagged {threshold + 1}x consecutive "
                    f"(prior {threshold} iterations addressed but not converged); "
                    f"dropping to break the loop"
                ),
            )
            for tid in sorted(consecutive_targets)
        ]
        return CorrectionPlan(
            corrections=kept_corrections,
            skipped=list(plan.skipped) + added_skips,
            rationale=(
                plan.rationale
                + f"; oscillation filter dropped {len(added_skips)} target(s)"
            ),
            planner_kind="identity",
        )
    return _plan
