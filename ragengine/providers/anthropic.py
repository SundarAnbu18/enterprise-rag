"""Claude, through the official Anthropic SDK.

The SDK is imported inside the method so that ``import ragengine`` stays fast
and a Gemini-only deployment never needs the package at all. No temperature is
ever sent — current Claude models reject a non-default value with a 400, and
the default is the right choice for grounded question answering anyway.
"""

from __future__ import annotations

from typing import Dict, List

from ..exceptions import ProviderError, ProviderNotInstalledError
from .base import ChatProvider, Completion

# Shown instead of an empty answer when Claude's safety layer declines a turn.
REFUSAL_TEXT = "I can't help with that request."


class AnthropicProvider(ChatProvider):
    name = "anthropic"

    def __init__(self, api_key: str, model: str, max_tokens: int) -> None:
        super().__init__(api_key, model, max_tokens)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - environment-dependent
                raise ProviderNotInstalledError(
                    "The 'anthropic' package is not installed. Run: pip install anthropic"
                ) from exc
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def complete(self, system: str, messages: List[Dict[str, str]]) -> Completion:
        import anthropic

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            )
        except anthropic.AuthenticationError as exc:
            raise ProviderError("Anthropic rejected the tenant's API key") from exc
        except anthropic.RateLimitError as exc:
            raise ProviderError("Anthropic rate limit reached for this tenant") from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(f"Anthropic request failed ({exc.status_code})") from exc
        except anthropic.APIConnectionError as exc:
            raise ProviderError("Could not reach the Anthropic API") from exc

        # A refusal is a successful HTTP response with empty or partial
        # content — never index into content before checking for it.
        text = "".join(block.text for block in response.content if block.type == "text")
        if response.stop_reason == "refusal" and not text.strip():
            text = REFUSAL_TEXT

        usage = getattr(response, "usage", None)
        return Completion(
            text=text,
            model=response.model,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            stop_reason=response.stop_reason,
        )
