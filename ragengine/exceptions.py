"""Errors the engine raises, so callers can tell them apart.

The Django views map these onto HTTP status codes, and the CLI turns them
into readable messages instead of tracebacks. Every error a caller might want
to branch on gets its own class; everything inherits from ``RagError`` so a
blanket "the engine failed" handler stays possible.
"""


class RagError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(RagError):
    """Something required is missing from the environment (e.g. the admin key)."""


class TenantNotFoundError(RagError):
    """No tenant with that slug exists."""


class TenantExistsError(RagError):
    """A tenant with that slug already exists."""


class AuthenticationError(RagError):
    """The presented API key does not match any tenant."""


class IndexNotBuiltError(RagError):
    """The tenant's vector index has not been built yet."""


class VectorDBError(RagError):
    """The configured vector database is unreachable, missing or misconfigured."""


class NoDocumentsError(RagError):
    """The tenant has no readable documents to index."""


class ProviderError(RagError):
    """The tenant's LLM provider rejected the request or failed."""


class ProviderNotInstalledError(ProviderError):
    """The SDK for the tenant's provider is not importable in this environment."""


class UnknownProviderError(ProviderError):
    """The tenant is configured with a provider this build does not know."""
