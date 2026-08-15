"""The contract every LLM provider implements.

The pipeline speaks one neutral dialect — a system prompt plus a list of
``{"role": "user"|"assistant", "content": str}`` dicts — and each provider
translates it to its own SDK. Nothing above this layer knows whether a tenant
runs on Anthropic or Gemini; that is the entire point of the abstraction, and
it is why conversation history is stored in this neutral shape rather than in
any SDK's message classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Completion:
    """One model response, plus the metadata about how it was produced."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: Optional[str]


class ChatProvider(ABC):
    """A chat model bound to one tenant's credential and model choice."""

    name: str = "base"

    def __init__(self, api_key: str, model: str, max_tokens: int) -> None:
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens

    @abstractmethod
    def complete(self, system: str, messages: List[Dict[str, str]]) -> Completion:
        """Answer the conversation, grounded in ``system``.

        ``messages`` alternate user/assistant and end with the user's turn.
        Implementations must raise ``ProviderError`` (or a subclass) on
        failure so the web layer can map it to a status code.
        """
