"""Adapter-agnostic LLM error classes.

Adapters (OpenAILLMClient, LiteLLMClient, …) translate provider-specific
exceptions into these so the library's retry / fallback logic doesn't
have to import any SDK to recognise them.
"""
from __future__ import annotations


class LLMError(Exception):
    """Base for any error raised by an ``LLMClient`` adapter."""


class TransientLLMError(LLMError):
    """The provider had a transient failure (timeout, 502/503/524, etc.).

    Library retry loops should catch this and back off. Adapters MUST
    map their own ``APITimeoutError`` / ``APIConnectionError`` /
    ``InternalServerError`` / ``APIStatusError(429|502|503|524)`` to
    this so retry logic is provider-neutral.
    """


class SchemaRejectedError(LLMError):
    """The provider rejected the requested ``response_format`` schema.

    Most often this is vLLM's xgrammar engine erroring out on a JSON
    Schema feature it doesn't implement (``propertyNames``, complex
    ``$ref`` chains, …) — emitted as 400. Callers (json_correction_loop's
    structured-output sub-agents) catch this and fall back to
    ``json_object`` mode with a textual schema hint.
    """


class ParseError(LLMError):
    """The provider returned a response we couldn't parse as JSON.

    Distinct from ``TransientLLMError`` because retrying the same
    request usually doesn't help — the LLM emitted bad JSON. Caller
    decides whether to retry with adjusted prompt or surface the raw
    content for debugging.
    """
