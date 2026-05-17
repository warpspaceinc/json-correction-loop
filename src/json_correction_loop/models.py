"""Domain-neutral models for the correction loop.

These mirror the corresponding models that historically lived in
``the host application models``. They were lifted out of the domain package so
the loop driver, planners, and executors can import them without the
domain package being installed.

The the host application side keeps a thin re-export shim so existing call
sites continue to work without import changes.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


SeverityStr = Literal["", "critical", "major", "minor"]
"""Severity label used by critics. The empty string is allowed so
freshly-constructed CriticIssues default cleanly when the LLM omits
severity."""


SkippedReason = Literal["upstream_cure", "duplicate", "contradicts_higher"]
"""Why a planner deliberately dropped a critic issue. Constrained to a
small structural enum (NOT severity) so audit reviews can confirm the
drop is justified by graph structure rather than discomfort."""


PlannerKindStr = Literal["identity", "llm"]
"""Whether a CorrectionPlan came from a deterministic mapping (identity
/ oscillation filter) or an LLM call."""


_MAX_TARGET_IDS = 20
"""Server-side ceiling on issue.target_ids length. Mirrors the schema-level
``maxItems`` so degenerate LLM responses that ignore the schema cap (or
land in a json_object-fallback path where it was dropped) still get
clamped after the fact."""


class CriticIssue(BaseModel):
    """One defect a critic flagged.

    ``target_ids`` is a list so a single defect can name several affected
    slots without forcing the LLM to pack them into one string (the old
    ``target_id: str`` shape kept tempting models to comma-join ids and
    then expand the comma-list into a degenerate output loop). The
    before-validator tolerates legacy scalar / comma-string emits and
    enforces both uniqueness and the ``_MAX_TARGET_IDS`` ceiling so the
    list-form can't be abused as a different degenerate path
    (e.g. emitting the same id thousands of times).
    """
    target_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Slot ids this defect targets. Emit as a JSON array of UNIQUE "
            "ids. A single id is still a 1-element list. Never comma-join "
            "ids inside one element — host enums constrain each item to "
            "its catalog. Do not repeat the same id; do not emit more than "
            "20 ids in one issue (split into multiple issues if needed)."
        ),
        max_length=20,
        json_schema_extra={"uniqueItems": True, "maxItems": 20},
    )
    severity: SeverityStr = ""
    issue_type: str = ""
    description: str = ""
    suggestions: list[str] = Field(default_factory=list)
    # Set by the surgical patcher / pipeline AFTER the issue is processed
    # when downstream judged the issue itself was wrong (target doesn't
    # exist, intent unpatchable, etc.). Future critic loops should skip
    # re-flagging an invalidated issue. ``invalidation_reason`` carries
    # the diagnosis (e.g. "request_validator: target_missing — ...",
    # "patcher: scope_mismatch — ...").
    invalidated: bool = False
    invalidation_reason: str = ""

    @field_validator("target_ids", mode="before")
    @classmethod
    def _coerce_target_ids(cls, v: Any) -> Any:
        # Accept legacy scalar / comma-separated string emits so backends
        # whose schema-enum constraint was downgraded to free-form (e.g.
        # json_object fallback) still parse cleanly. List-of-strings is
        # the canonical shape. Dedup while preserving first-occurrence
        # order, and clamp to ``_MAX_TARGET_IDS`` so degenerate LLM
        # responses that emit the same id thousands of times can't bypass
        # the schema-level ``uniqueItems`` / ``maxItems`` constraints.
        if v is None or v == "":
            return []
        if isinstance(v, str):
            items = [s.strip() for s in v.split(",") if s.strip()]
        elif isinstance(v, list):
            items = [s.strip() for s in v if isinstance(s, str) and s.strip()]
        else:
            return v
        seen: set[str] = set()
        deduped: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
            if len(deduped) >= _MAX_TARGET_IDS:
                break
        return deduped

    @model_validator(mode="before")
    @classmethod
    def _absorb_legacy_target_id(cls, data: Any) -> Any:
        # Phase-by-phase migration: until every host prompt is rewritten
        # to emit ``target_ids: [...]``, accept ``target_id: "..."`` from
        # legacy emissions and promote it to a singleton list. Once all
        # host phases are migrated, this can be deleted.
        if isinstance(data, dict) and "target_ids" not in data and "target_id" in data:
            v = data.pop("target_id")
            if isinstance(v, str) and v.strip():
                data["target_ids"] = [v.strip()]
        return data

    @property
    def target_id(self) -> str:
        """Backward-compat read accessor for callers that still expect a
        single ``target_id``. Returns the first element or empty string.
        New code should iterate ``target_ids`` directly.
        """
        return self.target_ids[0] if self.target_ids else ""


class CriticReport(BaseModel):
    """One critic's verdict for one iteration.

    The library treats this as a passive carrier — fields are read but
    none mutated by the loop. Domain code can subclass freely; the loop
    only depends on the attribute names listed here.
    """
    id: str | None = None
    level: str = ""
    iteration: int = 0
    issues: list[CriticIssue] = Field(default_factory=list)
    overall_assessment: str = ""
    # Required so the LLM can't silently omit it (which made the field
    # default to 0 across every report, hiding real critic verdicts).
    # 1–10 scale: 1-3 broken, 4-5 significant issues, 6-7 minor only,
    # 8-10 strong. Domain code synthesizes empty-issues fallbacks with
    # explicit ``score=7`` (= "minor only").
    score: int = Field(..., ge=0, le=10, description="REQUIRED. 1–10 quality score. 1-3=broken, 4-5=significant issues, 6-7=minor only, 8-10=strong.")
    # ``None`` = loop is mid-flight or naturally converged via approval.
    # A non-empty string = early-exit guard fired; the value is the
    # human-readable reason (preserved verbatim in stored data).
    convergence_reason: str | None = None
    approved: bool = False

    @classmethod
    def __get_pydantic_json_schema__(cls, schema, handler):
        # Force ``issues`` and ``approved`` into the LLM-facing
        # ``required`` list. Python defaults stay so existing
        # constructors (synthetic empty-issues fallbacks etc.) keep
        # working — but the LLM can no longer silently omit either
        # field, which previously let "approved=false, issues=()"
        # responses slip through as auto-approval (``total==0`` gate
        # ignored the missing ``approved`` flag).
        s = handler(schema)
        req = list(s.get("required", []))
        for k in ("issues", "approved"):
            if k not in req:
                req.append(k)
        s["required"] = req
        return s


class SkippedIssue(BaseModel):
    """One issue the planner dropped, with structural justification."""
    target_id: str = Field(description="The issue's ``target_id`` (slot the planner refused to act on).")
    reason: SkippedReason = Field(description="Structural reason for dropping. Not severity — that's not a valid drop reason.")
    explanation: str = Field(default="", description="Short specific explanation, e.g. 'cured by directive on scene-7' or 'oscillation: re-flagged 4x consecutive'.")


class Correction(BaseModel):
    """A single edit request — universal across all critic levels.

    Stage 2 produces a list of these; Stage 3 hands them to the level's
    executor. The shape is intentionally close to a JSON-Patch / RFC-6902
    request so executors that wrap a JSON-Pointer-based patcher can
    consume Corrections with a one-line conversion.

    ``op`` lets a single executor route several edit modes (revise,
    insert_after, stub_merge, ...) — the executor switches on it.
    Levels whose executor doesn't recognize a particular op silently
    drop those corrections.
    """
    requirement_id: str = Field(description="Slot id this correction targets (caller-defined, e.g. ``act-2.summary``).")
    intent: str = Field(default="", description="Human-readable instruction passed to the executor.")
    op: str = Field(default="revise", description="Executor mode: revise (default), insert_after, stub_merge, stub_split, ...")
    target_pointer: str | None = Field(default=None, description="Optional pre-resolved JSON Pointer; if None the executor resolves from requirement_id.")
    context_pointers: list[str] = Field(default_factory=list, description="Adjacent JSON Pointers the executor LLM should also be able to read for context.")
    source_critic_id: str | None = Field(default=None, description="Critic report this correction was derived from.")
    source_severity: SeverityStr = Field(default="", description="Severity of the originating critic issue.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Op-specific metadata.")


class CorrectionPlan(BaseModel):
    """Stage 2 output — what to apply this iteration, what to skip, why.

    For deterministic planners (identity, oscillation_aware) the
    ``corrections`` list is computed without an LLM. For LLM planners
    the same shape carries the model's drop / dedup / prioritization
    decisions.
    """
    corrections: list[Correction] = Field(default_factory=list)
    skipped: list[SkippedIssue] = Field(default_factory=list)
    rationale: str = Field(default="")
    planner_kind: PlannerKindStr = Field(default="identity")
