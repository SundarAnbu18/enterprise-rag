"""Provider registry: from a tenant's configuration to a working chat model.

Adding a provider means writing one module implementing ``ChatProvider`` and
adding one line to the registry below — nothing else in the system changes.
The provider classes themselves are imported lazily so an unused SDK is never
loaded.
"""

from __future__ import annotations

from ..exceptions import UnknownProviderError
from .base import ChatProvider, Completion

__all__ = ["ChatProvider", "Completion", "build_provider"]


def build_provider(provider: str, api_key: str, model: str, max_tokens: int) -> ChatProvider:
    """Instantiate the right provider for a tenant."""
    if provider == "anthropic":
        from .anthropic import AnthropicProvider

        return AnthropicProvider(api_key, model, max_tokens)
    if provider == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider(api_key, model, max_tokens)
    raise UnknownProviderError(f"Unknown provider {provider!r}")
