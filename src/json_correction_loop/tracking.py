"""LLM usage tracking — domain-neutral.

Tracks every LLM call (model, tokens, level/purpose) for the cost
report and per-call request log. Phase 5 of the library extraction
moved this out of ``the host application LLM module`` so the patcher / sub-agents
could ship as part of ``json_correction_loop`` without dragging in
provider-specific cost-lookup code.

Provider-specific bits (e.g. OpenRouter's per-request cost API) live
in the caller and plug in via the ``cost_lookup_fn`` hook on
:class:`UsageTracker`. Library code never imports an SDK or hits a
specific API endpoint.
"""
from __future__ import annotations

import json
import logging
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


# ── Per-call record ─────────────────────────────────────────────────────────


@dataclass
class CallRecord:
    """One LLM interaction's metadata + telemetry.

    The library only populates the token / timing / prompt fields; the
    optional ``actual_cost`` field is filled by the caller's
    ``cost_lookup_fn`` (e.g. OpenRouter API).
    """
    request_id: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    level: str = ""        # full phase id, e.g. "scenes.subdivide"
    purpose: str = ""      # sub-step within the phase, e.g. "subdivide.outline"
    actual_cost: float | None = None  # filled by cost_lookup_fn
    timestamp: str = ""           # call-end (recorded in record())
    started_at: str = ""          # call-start ISO, set by caller via record(start_ts=...)
    latency_ms: int | None = None  # end − start in milliseconds; None when start unknown
    system_prompt: str = ""
    user_prompt: str = ""
    response_content: str = ""
    finish_reason: str = ""
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: dict | None = None
    response_type: str = ""
    requested_schema: dict | None = None


# ── Context-scoped attribution ──────────────────────────────────────────────
#
# ContextVars give concurrent LLM calls (asyncio tasks / threads) their
# own level / purpose attribution. ``asyncio.to_thread`` and
# ``asyncio.gather`` both copy ContextVars into child contexts, so a
# fan-out node's children each see their own purpose label without
# stepping on siblings'.

_current_level: ContextVar[str] = ContextVar("_current_level", default="")
_current_purpose: ContextVar[str] = ContextVar("_current_purpose", default="")


# ── Tracker ─────────────────────────────────────────────────────────────────


CostLookupFn = Callable[["UsageTracker"], float]


@dataclass
class UsageTracker:
    """Accumulates :class:`CallRecord` instances and emits cost reports.

    Single-process singleton in typical use (``tracker = UsageTracker()``
    at module load), but instantiable for tests / multi-tenant scenarios.

    ``cost_lookup_fn``: optional hook the caller installs to fetch
    actual per-request costs from a provider API. Library never calls
    a specific provider directly. When unset, :meth:`query_actual_costs`
    returns 0.0 and ``report()`` omits the actual-cost line.
    """

    calls: list[CallRecord] = field(default_factory=list)
    log_path: Path | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    cost_lookup_fn: CostLookupFn | None = None

    # ── Context-scoped attribution helpers ────────────────────────────

    @property
    def current_level(self) -> str:
        return _current_level.get()

    @current_level.setter
    def current_level(self, value: str) -> None:
        _current_level.set(value)

    @property
    def current_purpose(self) -> str:
        return _current_purpose.get()

    @current_purpose.setter
    def current_purpose(self, value: str) -> None:
        _current_purpose.set(value)

    def purpose(self, label: str):
        """Context manager that scopes ``current_purpose`` for a block of
        related calls. ContextVar-backed so concurrent tasks don't race."""

        @contextmanager
        def _ctx():
            token = _current_purpose.set(label)
            try:
                yield
            finally:
                _current_purpose.reset(token)
        return _ctx()

    # ── Per-call file logging ─────────────────────────────────────────

    def set_log_path(self, path: Path) -> None:
        """Configure a directory where each call gets a JSON file. The
        directory is created if missing. File names sort by timestamp
        and include the level for grep filtering."""
        path.mkdir(parents=True, exist_ok=True)
        self.log_path = path

    def record_error(
        self, model: str, exception: Exception,
        system_prompt: str = "", user_prompt: str = "",
        attempt: int = 0,
        temperature: float | None = None, max_tokens: int | None = None,
        request_id: str = "", response_content: str = "",
        kind: str = "exception",
        extra: dict | None = None,
        start_ts: datetime | None = None,
    ) -> None:
        """Persist a failed LLM interaction. ``kind`` distinguishes
        exception / empty_choices / parse_failure / truncated; files are
        prefixed with ``ERROR_`` for grep-ability."""
        if self.log_path is None:
            return
        import traceback
        try:
            end_ts = datetime.now(timezone.utc)
            ts = end_ts.isoformat()
            ts_safe = ts.replace(":", "-").replace(".", "-")
            started_at_iso = ""
            latency_ms: int | None = None
            if start_ts is not None:
                if start_ts.tzinfo is None:
                    start_ts = start_ts.replace(tzinfo=timezone.utc)
                started_at_iso = start_ts.isoformat()
                delta = (end_ts - start_ts).total_seconds() * 1000
                if delta >= 0:
                    latency_ms = int(round(delta))
            short = (request_id or f"attempt-{attempt}").replace("/", "_")[:16]
            lvl = (self.current_level or "unknown").replace("/", "_")
            purpose_seg = f"_{self.current_purpose.replace('/', '_')}" if self.current_purpose else ""
            fname = f"ERROR_{ts_safe}_{lvl}{purpose_seg}_{kind}_{short}.json"
            exc_info = {
                "type": type(exception).__name__,
                "module": type(exception).__module__,
                "message": str(exception),
                "traceback": "".join(traceback.format_exception(type(exception), exception, exception.__traceback__)),
            }
            for attr in ("status_code", "code", "param", "response", "body"):
                val = getattr(exception, attr, None)
                if val is not None:
                    exc_info[attr] = repr(val)[:1000]
            payload = {
                "timestamp": ts,
                "started_at": started_at_iso,
                "latency_ms": latency_ms,
                "level": self.current_level,
                "purpose": self.current_purpose,
                "kind": kind,
                "attempt": attempt,
                "model": model,
                "request_id": request_id,
                "exception": exc_info,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "response_content": response_content,
            }
            if extra:
                payload["extra"] = extra
            (self.log_path / fname).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"failed to write error log file: {e}")

    def record(
        self, request_id: str, model: str, usage,
        system_prompt: str = "", user_prompt: str = "",
        response_content: str = "", finish_reason: str = "",
        temperature: float | None = None, max_tokens: int | None = None,
        response_format: dict | None = None, response_type: str = "",
        requested_schema: dict | None = None,
        start_ts: datetime | None = None,
    ) -> None:
        end_ts = datetime.now(timezone.utc)
        started_at = start_ts.isoformat() if start_ts is not None else ""
        latency_ms: int | None = None
        if start_ts is not None:
            # Tolerate naive datetimes by assuming UTC — caller bugs
            # shouldn't poison the whole record.
            if start_ts.tzinfo is None:
                start_ts = start_ts.replace(tzinfo=timezone.utc)
            delta = (end_ts - start_ts).total_seconds() * 1000
            if delta >= 0:
                latency_ms = int(round(delta))
        rec = CallRecord(
            request_id=request_id,
            model=model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            level=self.current_level,
            purpose=self.current_purpose,
            timestamp=end_ts.isoformat(),
            started_at=started_at,
            latency_ms=latency_ms,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_content=response_content,
            finish_reason=finish_reason,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            response_type=response_type,
            requested_schema=requested_schema,
        )
        with self._lock:
            self.calls.append(rec)
        if self.log_path is not None:
            try:
                ts = rec.timestamp.replace(":", "-").replace(".", "-")
                short = (request_id or "no-id").replace("/", "_")[:16]
                lvl = (rec.level or "unknown").replace("/", "_")
                purpose_seg = f"_{rec.purpose.replace('/', '_')}" if rec.purpose else ""
                fname = f"{ts}_{lvl}{purpose_seg}_{short}.json"
                payload = {
                    "timestamp": rec.timestamp,
                    "started_at": rec.started_at,
                    "latency_ms": rec.latency_ms,
                    "level": rec.level,
                    "purpose": rec.purpose,
                    "model": rec.model,
                    "request_id": rec.request_id,
                    "finish_reason": rec.finish_reason,
                    "temperature": rec.temperature,
                    "max_tokens": rec.max_tokens,
                    "prompt_tokens": rec.prompt_tokens,
                    "completion_tokens": rec.completion_tokens,
                    "total_tokens": rec.total_tokens,
                    "response_type": rec.response_type,
                    "response_format": rec.response_format,
                    "requested_schema": rec.requested_schema,
                    "system_prompt": rec.system_prompt,
                    "user_prompt": rec.user_prompt,
                    "response_content": rec.response_content,
                }
                (self.log_path / fname).write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            except Exception as e:
                logger.warning(f"failed to write request log file: {e}")

    # ── Aggregations ──────────────────────────────────────────────────

    @property
    def total_prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def total_completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls)

    @property
    def total_calls(self) -> int:
        return len(self.calls)

    @property
    def total_actual_cost(self) -> float | None:
        costs = [c.actual_cost for c in self.calls if c.actual_cost is not None]
        return sum(costs) if costs else None

    def by_level(self) -> dict[str, dict]:
        levels: dict[str, dict] = {}
        for c in self.calls:
            lv = c.level or "other"
            if lv not in levels:
                levels[lv] = {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "actual_cost": 0.0,
                }
            levels[lv]["calls"] += 1
            levels[lv]["prompt_tokens"] += c.prompt_tokens
            levels[lv]["completion_tokens"] += c.completion_tokens
            levels[lv]["total_tokens"] += c.total_tokens
            if c.actual_cost is not None:
                levels[lv]["actual_cost"] += c.actual_cost
        return levels

    def query_actual_costs(self) -> float:
        """Delegate to ``cost_lookup_fn`` if installed; else 0.0.

        Library never knows about specific providers; callers register
        their own lookup at startup (see
        ``the host application cost lookup``).
        """
        if self.cost_lookup_fn is None:
            return 0.0
        return self.cost_lookup_fn(self)

    def save_request_log(self, path: Path) -> None:
        """Save a JSON summary of every request id + tokens + cost."""
        records = []
        for c in self.calls:
            records.append({
                "request_id": c.request_id,
                "model": c.model,
                "level": c.level,
                "prompt_tokens": c.prompt_tokens,
                "completion_tokens": c.completion_tokens,
                "total_tokens": c.total_tokens,
                "actual_cost": c.actual_cost,
            })
        path.write_text(json.dumps(records, indent=2, ensure_ascii=False))

    def report(self, include_actual: bool = True) -> str:
        """Render a textual cost report.

        Library-version covers totals + per-level breakdown + (when
        ``cost_lookup_fn`` is installed) actual cost. Provider-specific
        sections (e.g. "Self-hosted endpoint" labels, hypothetical
        OpenRouter-rate estimates) belong in caller-side wrappers.
        """
        actual_total: float | None = None
        if include_actual and self.calls and self.cost_lookup_fn is not None:
            actual_total = self.query_actual_costs()

        lines = [
            "=" * 65,
            "COST REPORT",
            "=" * 65,
            f"Total API calls: {self.total_calls}",
            f"Total tokens: {self.total_tokens:,}",
            f"  Prompt:     {self.total_prompt_tokens:,}",
            f"  Completion: {self.total_completion_tokens:,}",
        ]
        if actual_total is not None and actual_total > 0:
            lines.append(f"Actual cost: ${actual_total:.4f}")

        lines.append("")
        lines.append(f"{'Level':<20s} {'Calls':>5s} {'Tokens':>10s} {'In':>10s} {'Out':>9s} {'Cost':>10s}")
        lines.append("-" * 65)
        for lv, data in sorted(self.by_level().items()):
            cost_str = f"${data['actual_cost']:.4f}" if data['actual_cost'] > 0 else "-"
            lines.append(
                f"{lv:<20s} {data['calls']:>5d} {data['total_tokens']:>10,} "
                f"{data['prompt_tokens']:>10,} {data['completion_tokens']:>9,} {cost_str:>10s}"
            )
        lines.append("=" * 65)
        return "\n".join(lines)
