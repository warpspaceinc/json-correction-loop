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

from pydantic import BaseModel, Field


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


class CriticIssue(BaseModel):
    """One defect a critic flagged."""
    target_id: str = ""
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
