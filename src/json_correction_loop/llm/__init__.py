"""LLM client Protocol for json_correction_loop.

The library's sub-agents and patcher main loop call ``client.chat_complete()``
through this Protocol — no direct OpenAI / Anthropic / LiteLLM SDK
imports inside the library. Callers supply a concrete adapter.

Why a Protocol (not a base class):
- Library carries no SDK dependencies.
- Callers can inject mocks (ducktyping) without inheritance.
- Multiple adapters (OpenAI-compat, LiteLLM, …) coexist trivially —
  the library only sees the contract.
"""
from json_correction_loop.llm.client import (
    ChatResponse,
    LLMClient,
    ToolCall,
    ToolCallFunction,
    Usage,
)
from json_correction_loop.llm.exceptions import (
    LLMError,
    ParseError,
    SchemaRejectedError,
    TransientLLMError,
)

__all__ = [
    "ChatResponse",
    "LLMClient",
    "LLMError",
    "ParseError",
    "SchemaRejectedError",
    "ToolCall",
    "ToolCallFunction",
    "TransientLLMError",
    "Usage",
]
