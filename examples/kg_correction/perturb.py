"""Perturbation operators — inject ground-truthed defects into a clean KG.

Six operators, each (a) mutates the KG in place on a deep copy and (b)
returns a ``Defect`` capturing the change so an oracle patcher (or, in
later phases, a metric harness) can verify reversal.

Determinism: every operator takes a ``random.Random`` so an outer
``apply_perturbations`` driver gives reproducible defect lists per seed.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from kg import ALLOWED_PREDICATES


@dataclass
class Defect:
    """One ground-truthed perturbation. ``before`` is the slice of the KG
    that existed before the operator ran; ``after`` is what replaced it.
    For ``entity_drop`` ``after`` is None; for ``dangling_ref`` /
    ``duplicate`` ``before`` is None.
    """
    defect_id: str
    operator: str
    target_pointer: str          # RFC 6901 pointer into the KG
    before: Any | None
    after: Any | None
    notes: str = ""
    # Tags the structural critic is expected to attach when it spots
    # this defect. Used in evaluation to bucket recall.
    expected_critic_tags: list[str] = field(default_factory=list)


# ── Operators ───────────────────────────────────────────────────────────────


def relation_swap(kg: dict[str, Any], rng: random.Random) -> Defect | None:
    """Replace one edge's predicate with a different allowed predicate
    that creates a type violation. The critic catches this via the
    type-mismatch check, not by predicate-name alone.
    """
    if not kg["edges"]:
        return None
    idx = rng.randrange(len(kg["edges"]))
    edge = kg["edges"][idx]
    orig = edge["predicate"]
    # Pick a different predicate whose (subj_type, obj_type) does NOT
    # match the existing subject/object types — guarantees a violation.
    subj_t = kg["entities"].get(edge["subject"], {}).get("type")
    obj_t = kg["entities"].get(edge["object"], {}).get("type")
    bad = [
        p for p, (s, o) in ALLOWED_PREDICATES.items()
        if p != orig and (s != subj_t or o != obj_t)
    ]
    if not bad:
        return None
    new_pred = rng.choice(bad)
    before = copy.deepcopy(edge)
    edge["predicate"] = new_pred
    return Defect(
        defect_id="",
        operator="relation_swap",
        target_pointer=f"/edges/{idx}",
        before=before,
        after=copy.deepcopy(edge),
        expected_critic_tags=["type_violation"],
        notes=f"{orig} -> {new_pred}",
    )


def entity_drop(kg: dict[str, Any], rng: random.Random) -> Defect | None:
    """Remove an entity that has at least one incoming edge — leaves
    dangling references the critic must detect.
    """
    referenced = {e["subject"] for e in kg["edges"]} | {e["object"] for e in kg["edges"]}
    candidates = [eid for eid in kg["entities"] if eid in referenced]
    if not candidates:
        return None
    eid = rng.choice(candidates)
    before = copy.deepcopy(kg["entities"][eid])
    del kg["entities"][eid]
    return Defect(
        defect_id="",
        operator="entity_drop",
        target_pointer=f"/entities/{eid}",
        before=before,
        after=None,
        expected_critic_tags=["dangling_ref"],
        notes=f"dropped {eid}",
    )


def entity_paraphrase(kg: dict[str, Any], rng: random.Random) -> Defect | None:
    """Mutate an entity's label without changing its ID. The structural
    critic cannot detect this — it's a 'semantic' defect retained for
    completeness; semantic critic in Phase B will catch it.
    """
    eids = list(kg["entities"].keys())
    if not eids:
        return None
    eid = rng.choice(eids)
    before = copy.deepcopy(kg["entities"][eid])
    kg["entities"][eid] = {**before, "label": before["label"] + " [perturbed]"}
    return Defect(
        defect_id="",
        operator="entity_paraphrase",
        target_pointer=f"/entities/{eid}/label",
        before=before["label"],
        after=kg["entities"][eid]["label"],
        expected_critic_tags=[],  # invisible to structural critic
        notes="label paraphrase",
    )


def type_violation(kg: dict[str, Any], rng: random.Random) -> Defect | None:
    """Swap an entity's type. Many predicate type-checks then break."""
    eids = list(kg["entities"].keys())
    if not eids:
        return None
    eid = rng.choice(eids)
    before = copy.deepcopy(kg["entities"][eid])
    flipped = "Film" if before["type"] == "Person" else "Person"
    kg["entities"][eid] = {**before, "type": flipped}
    return Defect(
        defect_id="",
        operator="type_violation",
        target_pointer=f"/entities/{eid}/type",
        before=before["type"],
        after=flipped,
        expected_critic_tags=["type_violation"],
        notes=f"{before['type']} -> {flipped}",
    )


def dangling_ref(kg: dict[str, Any], rng: random.Random) -> Defect | None:
    """Add an edge whose object is a non-existent entity ID."""
    persons = [eid for eid, ent in kg["entities"].items() if ent["type"] == "Person"]
    if not persons:
        return None
    bad_id = f"Q{9000 + rng.randrange(1000)}"
    while bad_id in kg["entities"]:
        bad_id = f"Q{9000 + rng.randrange(1000)}"
    new_id = f"e{rng.randrange(10_000_000)}"
    edge = {
        "id": new_id,
        "subject": rng.choice(persons),
        "predicate": "directed",
        "object": bad_id,
    }
    kg["edges"].append(edge)
    return Defect(
        defect_id="",
        operator="dangling_ref",
        target_pointer=f"/edges/{len(kg['edges']) - 1}",
        before=None,
        after=copy.deepcopy(edge),
        expected_critic_tags=["dangling_ref"],
        notes=f"object {bad_id} not in entities",
    )


def duplicate(kg: dict[str, Any], rng: random.Random) -> Defect | None:
    """Add a duplicate of an existing entity under a new ID, keeping the
    same label. The critic flags it via duplicate-label check.
    """
    eids = list(kg["entities"].keys())
    if not eids:
        return None
    eid = rng.choice(eids)
    new_id = f"Q{8000 + rng.randrange(1000)}"
    while new_id in kg["entities"]:
        new_id = f"Q{8000 + rng.randrange(1000)}"
    kg["entities"][new_id] = copy.deepcopy(kg["entities"][eid])
    return Defect(
        defect_id="",
        operator="duplicate",
        target_pointer=f"/entities/{new_id}",
        before=None,
        after=copy.deepcopy(kg["entities"][new_id]),
        expected_critic_tags=["duplicate_label"],
        notes=f"duplicates {eid}",
    )


OPERATORS: dict[str, Callable[[dict[str, Any], random.Random], Defect | None]] = {
    "relation_swap":     relation_swap,
    "entity_drop":       entity_drop,
    "entity_paraphrase": entity_paraphrase,
    "type_violation":    type_violation,
    "dangling_ref":      dangling_ref,
    "duplicate":         duplicate,
}


def apply_perturbations(
    clean: dict[str, Any],
    *,
    n: int,
    seed: int,
    operators: list[str] | None = None,
) -> tuple[dict[str, Any], list[Defect]]:
    """Apply ``n`` perturbations from the given (or all) operators on a
    deep copy of ``clean``. Returns (perturbed_kg, ordered_defect_log).
    """
    rng = random.Random(seed)
    pool = operators or list(OPERATORS.keys())
    kg = copy.deepcopy(clean)
    defects: list[Defect] = []
    attempts = 0
    while len(defects) < n and attempts < n * 10:
        attempts += 1
        op_name = rng.choice(pool)
        d = OPERATORS[op_name](kg, rng)
        if d is None:
            continue
        d.defect_id = f"def-{len(defects) + 1:03d}"
        defects.append(d)
    return kg, defects
