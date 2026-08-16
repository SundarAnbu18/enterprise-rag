"""Retrieve, then generate — joined per tenant.

Nothing here loads at import time. Each tenant's index and provider are pulled
in on first use and cached against the index file's mtime, so a reindex done
by another worker (or the CLI) is picked up on the next request without any
restart or cross-process signalling — the filesystem is the coordination
mechanism, same as the tenant registry.

Two model calls happen per turn once a conversation is under way, both on the
tenant's own key: the first rewrites the follow-up into a standalone question
*for retrieval only*, the second answers the user's original wording with the
prior turns in context. Without the rewrite, "and their refund policy?" gets
embedded as that fragment and retrieves nothing useful — which is how
multi-turn RAG usually falls apart.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .config import Settings, get_settings
from .history import recent_messages, record_turn
from .providers import ChatProvider, build_provider
from .store import SearchResult
from .tenants import Tenant, TenantStore, get_tenant_store
from .vectordb import index_stamp_path, load_store

SYSTEM_TEMPLATE = """You are the assistant for {tenant_name}, answering questions about the
documents excerpted below.

Answer from the context when the question is about the documents. You may also
refer to the conversation so far when the user asks about it. If the answer is
in neither, say you don't know — never answer from outside knowledge.

Context:
{context}"""

CONDENSE_TEMPLATE = """Given the conversation so far and a follow-up message, rewrite the \
follow-up as a standalone question that makes sense without the conversation. \
If it is already standalone, return it unchanged. Return only the question.

Conversation:
{history}

Follow-up: {question}"""


@dataclass(frozen=True)
class Answer:
    """An answer, plus the metadata about how it was produced."""

    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: Optional[str]
    sources: List[SearchResult]

    def to_dict(self) -> dict:
        """The JSON payload the API returns."""
        return {
            "answer": self.text,
            "provider": self.provider,
            "model": self.model,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            },
            "stop_reason": self.stop_reason,
            "sources": [
                {
                    "source": result.source,
                    "distance": round(result.distance, 3),
                    "preview": result.text[:120],
                }
                for result in self.sources
            ],
        }


def build_system_prompt(tenant: Tenant, results: Sequence[SearchResult]) -> str:
    """Lay the retrieved passages out as context for the model."""
    context = "\n".join(result.text for result in results)
    return SYSTEM_TEMPLATE.format(tenant_name=tenant.name, context=context)


def _transcript(history: Sequence[dict]) -> str:
    """Render prior turns as plain text, for the condensing prompt."""
    speaker = {"user": "User", "assistant": "Assistant"}
    return "\n".join(
        f"{speaker.get(m['role'], m['role'])}: {m['content']}" for m in history
    )


class TenantPipeline:
    """Answers questions against one tenant's index with one tenant's model."""

    def __init__(
        self,
        tenant: Tenant,
        provider: ChatProvider,
        store,  # anything with .search(query, k) and __len__ — see vectordb.py
        settings: Optional[Settings] = None,
    ) -> None:
        self.tenant = tenant
        self.provider = provider
        self.store = store
        self.settings = settings or get_settings()

    def retrieve(self, question: str, k: Optional[int] = None) -> List[SearchResult]:
        """The passages most similar to ``question``."""
        return self.store.search(question, k or self.tenant.top_k)

    def condense(self, question: str, history: Sequence[dict]) -> str:
        """Rewrite a follow-up so it stands on its own — retrieval only."""
        if not history:
            return question
        prompt = CONDENSE_TEMPLATE.format(history=_transcript(history), question=question)
        completion = self.provider.complete(
            system="You rewrite follow-up messages into standalone questions.",
            messages=[{"role": "user", "content": prompt}],
        )
        # A blank rewrite is worse than no rewrite.
        return completion.text.strip() or question

    def answer(
        self,
        question: str,
        k: Optional[int] = None,
        conversation_id: Optional[str] = None,
    ) -> Answer:
        """Retrieve context for ``question`` and have the tenant's model answer.

        Given a ``conversation_id`` the earlier turns are loaded, the follow-up
        is condensed for retrieval, and the exchange is recorded once answered.
        Without one the turn is stateless.
        """
        history: List[dict] = []
        if conversation_id:
            history = recent_messages(self.tenant.slug, conversation_id, self.settings)

        results = self.retrieve(self.condense(question, history), k)

        messages = list(history) + [{"role": "user", "content": question}]
        completion = self.provider.complete(build_system_prompt(self.tenant, results), messages)

        if conversation_id:
            record_turn(self.tenant.slug, conversation_id, question, completion.text, self.settings)

        return Answer(
            text=completion.text,
            provider=self.provider.name,
            model=completion.model,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            stop_reason=completion.stop_reason,
            sources=list(results),
        )


# ---------------------------------------------------------------------------
# The per-tenant pipeline cache.
# ---------------------------------------------------------------------------

# slug -> (index mtime, tenant.json mtime, pipeline). Both mtimes are part of
# the key so a reindex *or* a config change (say, switching provider) rebuilds
# the pipeline; anything else reuses the loaded index and provider client.
_PIPELINES: Dict[str, Tuple[float, float, TenantPipeline]] = {}
_PIPELINES_LOCK = threading.Lock()


def _mtimes(tenant: Tenant) -> Tuple[float, float]:
    try:
        # For FAISS this is index.faiss itself; for Qdrant it is the stamp
        # file every rebuild rewrites — either way, a reindex moves the mtime.
        index_mtime = index_stamp_path(tenant).stat().st_mtime
    except OSError:
        index_mtime = 0.0
    try:
        config_mtime = (tenant.home / "tenant.json").stat().st_mtime
    except OSError:
        config_mtime = 0.0
    return index_mtime, config_mtime


def get_pipeline(tenant: Tenant, tenant_store: Optional[TenantStore] = None) -> TenantPipeline:
    """The cached pipeline for a tenant, rebuilt when their files change."""
    index_mtime, config_mtime = _mtimes(tenant)

    with _PIPELINES_LOCK:
        cached = _PIPELINES.get(tenant.slug)
        if cached and cached[0] == index_mtime and cached[1] == config_mtime:
            return cached[2]

    tenant_store = tenant_store or get_tenant_store()
    provider = build_provider(
        tenant.provider,
        tenant_store.provider_api_key(tenant),
        tenant.model,
        tenant.max_tokens,
    )
    store = load_store(tenant)
    pipeline = TenantPipeline(tenant, provider, store)

    with _PIPELINES_LOCK:
        _PIPELINES[tenant.slug] = (index_mtime, config_mtime, pipeline)
    return pipeline


def clear_pipelines() -> None:
    """Drop every cached pipeline (used by tests and after bulk changes)."""
    with _PIPELINES_LOCK:
        _PIPELINES.clear()


def answer_question(
    tenant: Tenant,
    question: str,
    k: Optional[int] = None,
    conversation_id: Optional[str] = None,
) -> Answer:
    """Convenience entry point used by the web app and the CLI."""
    return get_pipeline(tenant).answer(question, k, conversation_id)
