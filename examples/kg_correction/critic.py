"""Structural critic — deterministic, ground-truth-free.

Detects four defect classes by inspecting the KG state alone:

  - ``dangling_ref``    edge points to an entity not in entities (covers
                        ``entity_drop`` and the ``dangling_ref`` operator)
  - ``type_violation``  edge predicate's expected (subj_type, obj_type)
                        doesn't match the actual entity types (covers
                        ``relation_swap`` and ``type_violation``)
  - ``duplicate_label`` two distinct entity IDs share the same label and
                        type (covers ``duplicate``)
  - ``schema_invalid``  predicate not in ALLOWED_PREDICATES

``entity_paraphrase`` is invisible here — Phase B's semantic (LLM)
critic handles it. We deliberately omit it so structural-only fix-rate
metrics are honest.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from json_correction_loop import CriticIssue, CriticReport

from kg import ALLOWED_PREDICATES


def kg_critic(state: dict[str, Any], iteration: int = 0, model: str | None = None) -> list[CriticReport]:
    issues: list[CriticIssue] = []

    # ── dangling_ref + schema_invalid + type_violation ───────────────────
    for idx, edge in enumerate(state.get("edges", [])):
        pred = edge.get("predicate")
        subj = edge.get("subject")
        obj = edge.get("object")

        if pred not in ALLOWED_PREDICATES:
            issues.append(CriticIssue(
                target_id=f"/edges/{idx}",
                severity="critical",
                issue_type="schema_invalid",
                description=f"edge {edge.get('id')} uses unknown predicate '{pred}'",
            ))
            continue

        subj_ent = state["entities"].get(subj)
        obj_ent = state["entities"].get(obj)
        if subj_ent is None:
            issues.append(CriticIssue(
                target_id=f"/edges/{idx}",
                severity="critical",
                issue_type="dangling_ref",
                description=f"edge {edge.get('id')} subject {subj!r} not in entities",
            ))
        if obj_ent is None:
            issues.append(CriticIssue(
                target_id=f"/edges/{idx}",
                severity="critical",
                issue_type="dangling_ref",
                description=f"edge {edge.get('id')} object {obj!r} not in entities",
            ))
        if subj_ent and obj_ent:
            want_s, want_o = ALLOWED_PREDICATES[pred]
            if subj_ent.get("type") != want_s or obj_ent.get("type") != want_o:
                issues.append(CriticIssue(
                    target_id=f"/edges/{idx}",
                    severity="major",
                    issue_type="type_violation",
                    description=(
                        f"edge {edge.get('id')} predicate '{pred}' requires "
                        f"({want_s}, {want_o}) but got "
                        f"({subj_ent.get('type')}, {obj_ent.get('type')})"
                    ),
                ))

    # ── duplicate_label ───────────────────────────────────────────────────
    by_signature: dict[tuple[str, str], list[str]] = defaultdict(list)
    for eid, ent in state.get("entities", {}).items():
        by_signature[(ent.get("label", ""), ent.get("type", ""))].append(eid)
    for (label, _t), eids in by_signature.items():
        if len(eids) > 1:
            for eid in eids[1:]:
                issues.append(CriticIssue(
                    target_id=f"/entities/{eid}",
                    severity="major",
                    issue_type="duplicate_label",
                    description=f"entity {eid} duplicates label {label!r} (also on {eids[0]})",
                ))

    score = 10 if not issues else max(1, 10 - len(issues))
    overall = "clean" if not issues else f"{len(issues)} structural issue(s)"
    return [CriticReport(
        id=f"kg-structural-iter{iteration}",
        level="kg",
        iteration=iteration,
        issues=issues,
        score=score,
        overall_assessment=overall,
    )]
