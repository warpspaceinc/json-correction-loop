"""json_correction_loop — domain-neutral critic-correction loop for JSON objects.

Wraps a 3-stage ``gather → plan → execute`` cycle around any JSON-shaped
state object. The library is intentionally agnostic about WHAT is being
corrected: callers supply the critics (``gather_fn``), the planner
(``plan_fn``), and the executor (``execute_fn``) for their domain.

Public surface:

  * Models: :class:`Correction`, :class:`CorrectionPlan`, :class:`SkippedIssue`,
    :class:`CriticIssue`, :class:`CriticReport`
  * Planners: :func:`make_identity_planner`, :func:`make_oscillation_aware_planner`
  * Execution: :class:`ExecuteResult`, :func:`make_callback_executor`
  * Loop driver: :func:`run_correction_loop`
  * Convergence: :class:`QualityStablePolicy`, :class:`HardcapPolicy`
  * Storage / events: :class:`StorageBackend`, :class:`EventSink` Protocols

The loop emits structured events (``EventSink.emit``) for progress
reporting and persists each iteration through a ``StorageBackend`` —
both are Protocols so callers plug in console / log / DB / file
adapters without the library importing any specific backend.
"""
from json_correction_loop.models import (
    Correction,
    CorrectionPlan,
    CriticIssue,
    CriticReport,
    PlannerKindStr,
    SeverityStr,
    SkippedIssue,
    SkippedReason,
)
from json_correction_loop.planners import (
    PlannerFn,
    make_identity_planner,
    make_oscillation_aware_planner,
)
from json_correction_loop.executors import (
    ExecuteResult,
    ExecutorFn,
    PatcherTraceLike,
    make_callback_executor,
)
from json_correction_loop.convergence import (
    ConvergencePolicy,
    HardcapPolicy,
    QualityStablePolicy,
)
from json_correction_loop.events import Event, EventSink, NullEventSink
from json_correction_loop.storage import IterationRecord, NullStorageBackend, StorageBackend
from json_correction_loop.loop import CorrectionLoopConfig, run_correction_loop
# Patcher + sub-agents (Phase 3 move).
from json_correction_loop.patcher import (
    CriticErrorRecord,
    PatchRequest,
    PatchResult,
    SubAgentTrace,
    SurgicalPatcher,
    ToolCallRecord,
)
from json_correction_loop.path_finder import FindTargetResult, PathFinderCall, find_target
from json_correction_loop.patch_evaluator import EvaluatePatchResult, evaluate_patch
from json_correction_loop.request_validator import ValidateRequestResult, validate_request
from json_correction_loop.template_filler import (
    FillTemplateResult,
    fill_template,
    has_enumeration_pattern,
    is_empty_container,
)
# Tracking (Phase 5).
from json_correction_loop.tracking import (
    CallRecord,
    CostLookupFn,
    UsageTracker,
)
from json_correction_loop.llm.tracking import TrackingLLMClient


__all__ = [
    # models
    "Correction",
    "CorrectionPlan",
    "CriticIssue",
    "CriticReport",
    "PlannerKindStr",
    "SeverityStr",
    "SkippedIssue",
    "SkippedReason",
    # planners
    "PlannerFn",
    "make_identity_planner",
    "make_oscillation_aware_planner",
    # executors
    "ExecuteResult",
    "ExecutorFn",
    "PatcherTraceLike",
    "make_callback_executor",
    # convergence
    "ConvergencePolicy",
    "HardcapPolicy",
    "QualityStablePolicy",
    # events
    "Event",
    "EventSink",
    "NullEventSink",
    # storage
    "IterationRecord",
    "NullStorageBackend",
    "StorageBackend",
    # loop
    "CorrectionLoopConfig",
    "run_correction_loop",
    # patcher
    "CriticErrorRecord",
    "PatchRequest",
    "PatchResult",
    "SubAgentTrace",
    "SurgicalPatcher",
    "ToolCallRecord",
    # sub-agents
    "EvaluatePatchResult",
    "FillTemplateResult",
    "FindTargetResult",
    "PathFinderCall",
    "ValidateRequestResult",
    "evaluate_patch",
    "fill_template",
    "find_target",
    "has_enumeration_pattern",
    "is_empty_container",
    "validate_request",
    # tracking
    "CallRecord",
    "CostLookupFn",
    "TrackingLLMClient",
    "UsageTracker",
]
