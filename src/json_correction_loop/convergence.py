"""Convergence policies — when to stop iterating early.

A policy reads the loop's per-iteration history (score, severity
counts, target_id set) and answers ``(converged: bool, reason: str)``.
The default ships in :class:`QualityStablePolicy`; callers can swap in
their own.

History tuple shape (loop-internal, but documented here):
    ``(score, n_critical, n_major, target_id_frozenset)``
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


HistoryEntry = tuple[int, int, int, frozenset[str]]


class ConvergencePolicy(Protocol):
    """Decide when to stop iterating early."""
    def check(self, history: list[HistoryEntry]) -> tuple[bool, str]:
        """Return ``(converged, reason)``. When ``converged=True`` the
        loop exits after this iteration. ``reason`` is a short audit
        string written into the report's ``convergence_reason`` field."""
        ...


@dataclass
class QualityStablePolicy:
    """Converge when the last ``stable_n`` iterations all hit
    ``score >= accept_score`` AND zero critical AND zero major.

    Mirrors the policy historically inlined in the host application's
    ``_check_critic_convergence``: the loop only exits early on real
    quality stability, not on oscillation or score thrash. Minor
    issues are tolerated within the stable window — they're not a
    convergence blocker.
    """
    stable_n: int = 3
    accept_score: int = 7

    def check(self, history: list[HistoryEntry]) -> tuple[bool, str]:
        if len(history) < self.stable_n:
            return False, ""
        recent = history[-self.stable_n:]
        if all(
            s >= self.accept_score and c == 0 and m == 0
            for s, c, m, _ in recent
        ):
            return True, (
                f"품질 안정 — 최근 {self.stable_n}회 score>={self.accept_score}, "
                "critical/major=0"
            )
        return False, ""


@dataclass
class HardcapPolicy:
    """Force convergence when the loop has run ``cap`` iterations.

    Composed with :class:`QualityStablePolicy` by the loop driver via
    config — :class:`HardcapPolicy` alone would never converge while
    the issue list is non-empty.
    """
    cap: int

    def check(self, history: list[HistoryEntry]) -> tuple[bool, str]:
        if len(history) >= self.cap:
            return True, f"{self.cap}회 미수렴 hardcap"
        return False, ""
