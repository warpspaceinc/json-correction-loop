"""LLM client wrapper that records every call to a tracker.

Decorator-pattern adapter: any object that satisfies the
:class:`json_correction_loop.llm.LLMClient` Protocol can be wrapped with
:class:`TrackingLLMClient` to attach usage tracking. Stays separate
from the underlying adapter (e.g. OpenAI) so tests can pass raw
adapters without tracking, and production wraps with this.
"""
from __future__ import annotations

import logging
from typing import Any

from json_correction_loop.llm.client import ChatResponse

logger = logging.getLogger(__name__)


class TrackingLLMClient:
    """Wrap an ``LLMClient``; record every successful response to ``tracker``.

    The inner client must satisfy the :class:`LLMClient` Protocol
    (``chat_complete`` returning :class:`ChatResponse`). Tracker must
    expose ``record(...)`` per :class:`UsageTracker`.
    """

    def __init__(self, inner: Any, tracker: Any) -> None:
        self._inner = inner
        self._tracker = tracker

    def chat_complete(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        response_format: dict | None = None,
        temperature: float = 0.5,
        max_tokens: int = 4096,
        extra: dict | None = None,
    ) -> ChatResponse:
        resp = self._inner.chat_complete(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            extra=extra,
        )
        if resp.usage and resp.usage.total_tokens > 0:
            system_prompt = ""
            user_prompt = ""
            for m in messages:
                if not isinstance(m, dict):
                    continue
                if m.get("role") == "system" and not system_prompt:
                    system_prompt = m.get("content") or ""
                elif m.get("role") == "user" and not user_prompt:
                    user_prompt = m.get("content") or ""
                if system_prompt and user_prompt:
                    break
            try:
                self._tracker.record(
                    resp.id or "",
                    model,
                    resp.usage,
                    system_prompt=system_prompt[:2000],
                    user_prompt=user_prompt[:2000],
                    response_content=(resp.content or "")[:2000],
                    finish_reason=resp.finish_reason or "",
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception:
                logger.warning("tracker.record failed", exc_info=True)
        return resp
