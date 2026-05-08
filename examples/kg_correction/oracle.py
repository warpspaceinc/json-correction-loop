"""Oracle patcher — deterministic reversal of known defects.

Phase A patcher: receives the planned corrections, looks each up in the
ground-truth defect log, and applies the inverse. This is the
*pipeline-validation* patcher — its purpose is to prove that the
gather/plan/execute loop wires together correctly. Phase B will swap it
for an LLM-based surgical patcher emitting RFC 6902 ops.

Each correction's ``requirement_id`` is the issue's ``target_id`` (a
JSON pointer like ``/edges/3`` or ``/entities/Q10``). We match defects
by pointer to find the inverse to apply.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

from perturb import Defect


@dataclass
class OracleTrace:
    id: str
    requirement_id: str
    addressed: bool
    reason: str


class OraclePatcher:
    """Build a callback executor closure: ``(state, paths, feedback, model) -> traces``.

    The closure mutates state in place (matching the host application's
    convention) and returns a list of trace objects the loop can
    introspect via the ``PatcherTraceLike`` Protocol.
    """

    def __init__(self, defects: list[Defect]):
        self.by_pointer: dict[str, list[Defect]] = {}
        for d in defects:
            self.by_pointer.setdefault(d.target_pointer, []).append(d)
        self._next_id = 0

    def __call__(
        self,
        state,
        flagged_paths: list[str],
        feedback_by_path: dict[str, str],
        model: str | None,
    ) -> list[OracleTrace]:
        traces: list[OracleTrace] = []
        for path in flagged_paths:
            # path looks like "/edges/3" or "/entities/Q10".
            addressed, reason = self._reverse_at(state, path)
            self._next_id += 1
            traces.append(OracleTrace(
                id=f"oracle-{self._next_id}",
                requirement_id=path,
                addressed=addressed,
                reason=reason,
            ))
        return traces

    def _reverse_at(self, state, pointer: str) -> tuple[bool, str]:
        # Strategy: if a defect was logged at the pointer, apply its
        # inverse. We also try to "heal" critic issues that resulted
        # from a sibling defect (e.g. dangling_ref caused by an
        # entity_drop logged at /entities/<id>).
        defects = self.by_pointer.get(pointer)
        if not defects:
            return self._heal_fallback(state, pointer)
        applied_any = False
        for d in defects:
            if self._apply_inverse(state, d):
                applied_any = True
        if applied_any:
            return True, "oracle: inverse applied"
        return False, "oracle: no inverse rule"

    def _apply_inverse(self, state, d: Defect) -> bool:
        op = d.operator
        ptr = d.target_pointer
        if op == "relation_swap":
            idx = int(ptr.rsplit("/", 1)[-1])
            if 0 <= idx < len(state["edges"]):
                state["edges"][idx] = copy.deepcopy(d.before)
                return True
        if op == "entity_drop":
            eid = ptr.rsplit("/", 1)[-1]
            state["entities"][eid] = copy.deepcopy(d.before)
            return True
        if op == "type_violation":
            # ptr = /entities/<id>/type
            parts = ptr.strip("/").split("/")
            eid = parts[1]
            state["entities"][eid]["type"] = d.before
            return True
        if op == "dangling_ref":
            idx = int(ptr.rsplit("/", 1)[-1])
            if 0 <= idx < len(state["edges"]) and state["edges"][idx]["id"] == d.after["id"]:
                state["edges"].pop(idx)
                return True
            for i, e in enumerate(list(state["edges"])):
                if e["id"] == d.after["id"]:
                    state["edges"].pop(i)
                    return True
        if op == "duplicate":
            eid = ptr.rsplit("/", 1)[-1]
            if eid in state["entities"]:
                del state["entities"][eid]
                return True
        if op == "entity_paraphrase":
            parts = ptr.strip("/").split("/")
            eid = parts[1]
            state["entities"][eid]["label"] = d.before
            return True
        return False

    def _heal_fallback(self, state, pointer: str) -> tuple[bool, str]:
        # The critic typically flags downstream SYMPTOMS (edges) while
        # the defect log records the ROOT CAUSE (entity drop / type flip).
        # Walk the defect log to find the matching root cause and apply
        # its inverse. (In Phase B this becomes the path_finder sub-agent.)
        if pointer.startswith("/edges/"):
            try:
                idx = int(pointer.rsplit("/", 1)[-1])
            except ValueError:
                return False, "oracle: bad pointer"
            if not (0 <= idx < len(state["edges"])):
                return False, "oracle: edge oob"
            edge = state["edges"][idx]
            # Try entity_drop first.
            for defects in self.by_pointer.values():
                for d in defects:
                    if d.operator == "entity_drop":
                        eid = d.target_pointer.rsplit("/", 1)[-1]
                        if eid in (edge["subject"], edge["object"]) and eid not in state["entities"]:
                            state["entities"][eid] = copy.deepcopy(d.before)
                            return True, f"oracle: restored {eid}"
            # Try type_violation (entity type flipped → cascade flagged on edge).
            for defects in self.by_pointer.values():
                for d in defects:
                    if d.operator == "type_violation":
                        parts = d.target_pointer.strip("/").split("/")
                        eid = parts[1]
                        if eid in (edge["subject"], edge["object"]):
                            cur = state["entities"].get(eid)
                            if cur and cur.get("type") == d.after:
                                cur["type"] = d.before
                                return True, f"oracle: reverted type on {eid}"
            # Try relation_swap (predicate flipped on this exact edge).
            for defects in self.by_pointer.values():
                for d in defects:
                    if d.operator == "relation_swap" and d.target_pointer == pointer:
                        # already handled in _apply_inverse path; here a dup
                        state["edges"][idx] = copy.deepcopy(d.before)
                        return True, "oracle: reverted swap"
        return False, "oracle: unknown fix"
