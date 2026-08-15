"""A multi-tenant retrieval-augmented generation engine.

Each tenant brings their own documents, their own LLM provider (Anthropic or
Gemini) and their own API key for it; retrieval runs locally over a per-tenant
FAISS index of sentence-transformer embeddings shared through one encoder.
Nothing in here knows about Django, so the same code backs the web app, the
CLI and the tests.

    from ragengine import get_tenant_store, answer_question
    tenant = get_tenant_store().get("acme")
    answer_question(tenant, "what is the refund policy?")

The web layer lives in ``ragapi`` and ``web``; this package is the engine.
"""

from .chunking import Chunk, chunk_document, split_paragraphs
from .config import Settings, get_settings
from .exceptions import (
    AuthenticationError,
    ConfigurationError,
    IndexNotBuiltError,
    NoDocumentsError,
    ProviderError,
    ProviderNotInstalledError,
    RagError,
    TenantExistsError,
    TenantNotFoundError,
    UnknownProviderError,
)
from .history import forget, is_valid_conversation_id
from .indexing import build_index, delete_document, load_chunks, save_document
from .pipeline import Answer, TenantPipeline, answer_question, clear_pipelines, get_pipeline
from .providers import Completion, build_provider
from .store import SearchResult, VectorStore
from .tenants import DEFAULT_MODELS, PROVIDERS, Tenant, TenantStore, get_tenant_store, slugify

__all__ = [
    "Answer",
    "AuthenticationError",
    "Chunk",
    "Completion",
    "ConfigurationError",
    "DEFAULT_MODELS",
    "IndexNotBuiltError",
    "NoDocumentsError",
    "PROVIDERS",
    "ProviderError",
    "ProviderNotInstalledError",
    "RagError",
    "SearchResult",
    "Settings",
    "Tenant",
    "TenantExistsError",
    "TenantNotFoundError",
    "TenantPipeline",
    "TenantStore",
    "UnknownProviderError",
    "VectorStore",
    "answer_question",
    "build_index",
    "build_provider",
    "chunk_document",
    "clear_pipelines",
    "delete_document",
    "forget",
    "get_pipeline",
    "get_settings",
    "get_tenant_store",
    "is_valid_conversation_id",
    "load_chunks",
    "save_document",
    "slugify",
    "split_paragraphs",
]
