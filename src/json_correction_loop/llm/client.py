"""LLM client Protocol + normalized response types.

The Protocol defines the minimal surface json_correction_loop's sub-agents
and patcher need from an LLM provider. Adapters translate their
provider's native API into this shape so the library never imports a
specific SDK.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class Usage:
    """Token usage for one chat completion. Mirrors the OpenAI shape
    that every modern provider exposes (vLLM, Ollama, OpenRouter,
    Anthropic, etc.)."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ToolCallFunction:
    """The function-call portion of a tool call. Arguments are the raw
    JSON string the LLM emitted — caller parses them. We keep the
    string form (not pre-parsed dict) because adapters don't have
    enough context to know whether malformed JSON should be repaired
    or surfaced as-is."""
    name: str
    arguments: str


@dataclass
class ToolCall:
    """One tool the LLM asked to invoke. The library's sub-agents and
    patcher dispatch on ``function.name`` to route to the matching
    handler."""
    id: str
    type: str = "function"
    function: ToolCallFunction = field(default_factory=lambda: ToolCallFunction("", ""))


@dataclass
class ChatResponse:
    """Normalized chat-completion response.

    Mirrors the subset of OpenAI's response that the patcher and
    sub-agents actually consume:

      * ``id`` — request id (for usage tracking / log correlation).
      * ``content`` — assistant message text. May be ``None`` when the
        model only emitted tool calls.
      * ``tool_calls`` — empty list when the model didn't request any
        (most non-tool calls).
      * ``finish_reason`` — ``stop`` / ``length`` / ``tool_calls`` /
        provider-specific. Library uses this to detect truncation.
      * ``usage`` — token counts for cost tracking.
      * ``raw`` — the original SDK response object, kept for ad-hoc
        debugging. Library code should NOT depend on this; adapters
        may not always populate it.
    """
    id: str = ""
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: Usage = field(default_factory=Usage)
    raw: Any = None


@runtime_checkable
class LLMClient(Protocol):
    """The minimal contract any LLM adapter must satisfy.

    Adapters (e.g. ``OpenAILLMClient`` in the host application) implement
    ``chat_complete`` and translate their provider's response into a
    :class:`ChatResponse`. Errors translate to the exception classes
    in :mod:`json_correction_loop.llm.exceptions` so library retry/fallback
    logic stays provider-neutral.
    """

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
        """Issue one chat-completion request.

        ``extra`` carries provider-specific knobs (``reasoning_effort``,
        ``seed``, vLLM-only ``guided_choice``, …) that the adapter
        routes into the right place — typically OpenAI SDK's
        ``extra_body`` — without the library having to know what each
        one means.

        Errors must map to the library's exception hierarchy:
          * Transient failures (timeout, 5xx, connection) →
            :class:`~json_correction_loop.llm.exceptions.TransientLLMError`
          * Schema rejection (vLLM xgrammar 400 on unsupported
            features) →
            :class:`~json_correction_loop.llm.exceptions.SchemaRejectedError`
          * JSON parse failures →
            :class:`~json_correction_loop.llm.exceptions.ParseError`

        Adapters MAY raise their native exceptions for non-recognised
        cases, but library code only catches the classes above.
        """
        ...
