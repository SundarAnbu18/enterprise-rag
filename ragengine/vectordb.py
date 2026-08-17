"""Where a tenant's vectors live: local FAISS files or a Qdrant server.

Two backends, one contract. Callers ask this module to save an index for a
tenant or to load a searchable store for one; whether the vectors sit in
``index.faiss`` next to the documents or in a Qdrant collection named after
the tenant is a deployment decision (``ENTERPRISE_VECTOR_BACKEND``), not
something the pipeline, the views or the CLI ever branch on.

FAISS stays the default because it needs nothing running beside the app.
Qdrant is for deployments that want a real database underneath — snapshots,
replication, a dashboard, memory that doesn't grow with the Django process.
The isolation story is identical in both: one collection per tenant mirrors
one index file per tenant, and there is still no shared index to filter.

The filesystem remains the cross-worker coordination mechanism even with a
server-side backend: every Qdrant rebuild stamps ``index/qdrant.json`` in the
tenant's directory, and the pipeline cache keys on that stamp's mtime exactly
as it keys on ``index.faiss`` for FAISS. The stamp also records how many
vectors were written, so ``load_store`` can refuse a collection that has
drifted out of step — the same in-step contract ``VectorStore.load`` enforces
between ``index.faiss`` and ``chunks.json``.

``qdrant_client`` is imported lazily, like every heavy dependency here, and a
deployment that never selects the backend never needs it installed.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence

from .chunking import Chunk
from .config import get_settings
from .embeddings import Embedder, get_embedder
from .exceptions import IndexNotBuiltError, VectorDBError
from .store import SearchResult, VectorStore

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from .tenants import Tenant

# Stamped into the tenant's index/ directory after every Qdrant rebuild; its
# existence means "built", its mtime invalidates pipeline caches, its counts
# feed the console without loading any vectors.
MARKER_FILENAME = "qdrant.json"


def collection_name(tenant: "Tenant") -> str:
    """The Qdrant collection holding this tenant's vectors."""
    return f"erag-{tenant.slug}"


def index_stamp_path(tenant: "Tenant") -> Path:
    """The file whose existence and mtime mean "this tenant's index changed".

    Everything that used to stat ``index.faiss`` — readiness checks, the
    pipeline cache — stats this instead, so backends stay interchangeable.
    """
    if get_settings().vector_backend == "qdrant":
        return tenant.index_dir / MARKER_FILENAME
    return tenant.index_path


def index_location(tenant: "Tenant") -> str:
    """A human-readable "where did the vectors go" for CLI output."""
    settings = get_settings()
    if settings.vector_backend == "qdrant":
        return f"qdrant collection {collection_name(tenant)!r} at {settings.qdrant_url}"
    return str(tenant.index_path)


@lru_cache(maxsize=1)
def get_qdrant_client():
    """One client per process (call ``.cache_clear()`` in tests).

    ``:memory:`` runs the whole database inside this process — that is what
    keeps the test suite free of servers. A cached client is also what makes
    the in-memory mode coherent: build and search must see the same instance.
    """
    settings = get_settings()
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:  # degrade with instructions, not a traceback
        raise VectorDBError(
            "ENTERPRISE_VECTOR_BACKEND=qdrant but the qdrant-client package "
            "is not installed. Run: pip install qdrant-client"
        ) from exc

    if settings.qdrant_url == ":memory:":
        return QdrantClient(":memory:")
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)


class QdrantVectorStore:
    """The Qdrant twin of ``store.VectorStore``: ``search`` and ``len``.

    Chunk text and source travel as point payloads, so the collection is
    self-contained — there is no chunks.json to keep in step, only the count
    recorded in the stamp file.
    """

    def __init__(self, client, collection: str, count: int) -> None:
        self._client = client
        self._collection = collection
        self._count = count

    def __len__(self) -> int:
        return self._count

    def search(self, query: str, k: int, embedder: Optional[Embedder] = None) -> List[SearchResult]:
        """Return the ``k`` nearest chunks to ``query``, closest first."""
        if not self._count:
            return []
        embedder = embedder or get_embedder()
        vector = embedder.encode_query(query)[0].tolist()
        try:
            points = self._client.query_points(
                self._collection,
                query=vector,
                limit=max(1, min(k, self._count)),
                with_payload=True,
            ).points
        except Exception as exc:
            raise VectorDBError(_unreachable(exc)) from exc
        return [
            SearchResult(
                Chunk(
                    text=(point.payload or {}).get("text", ""),
                    source=(point.payload or {}).get("source", ""),
                ),
                float(point.score),
            )
            for point in points
        ]


def _unreachable(exc: Exception) -> str:
    return (
        f"Qdrant at {get_settings().qdrant_url} is unreachable or refused the "
        f"request ({exc.__class__.__name__}). Is the server running?"
    )


def _write_stamp(tenant: "Tenant", payload: dict) -> None:
    """Atomic, like every other JSON this project writes."""
    path = tenant.index_dir / MARKER_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def save_index(tenant: "Tenant", chunks: Sequence[Chunk], embedder: Optional[Embedder] = None):
    """Embed ``chunks`` and persist them wherever this deployment keeps vectors.

    Returns a store the caller can ``len()`` — the API reports chunk counts
    from it right after an upload.
    """
    chunks = list(chunks)
    if get_settings().vector_backend != "qdrant":
        store = VectorStore.build(chunks, embedder=embedder)
        store.save(tenant.index_path, tenant.chunks_path)
        return store

    embedder = embedder or get_embedder()
    vectors = embedder.encode([chunk.text for chunk in chunks])
    client = get_qdrant_client()
    from qdrant_client import models

    name = collection_name(tenant)
    try:
        # Rebuild from scratch, mirroring how the FAISS index is rewritten
        # whole: no stale points can survive a reindex.
        if client.collection_exists(name):
            client.delete_collection(name)
        client.create_collection(
            name,
            vectors_config=models.VectorParams(
                size=int(vectors.shape[1]), distance=models.Distance.EUCLID
            ),
        )
        client.upsert(
            name,
            points=models.Batch(
                ids=list(range(len(chunks))),
                vectors=vectors.tolist(),
                payloads=[chunk.to_dict() for chunk in chunks],
            ),
        )
    except Exception as exc:
        raise VectorDBError(_unreachable(exc)) from exc

    _write_stamp(
        tenant,
        {
            "collection": name,
            "vectors": len(chunks),
            "sources": dict(Counter(chunk.source for chunk in chunks)),
            "embedding_model": getattr(embedder, "model_name", ""),
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    return QdrantVectorStore(client, name, len(chunks))


def load_store(tenant: "Tenant"):
    """A searchable store for one tenant, or ``IndexNotBuiltError``."""
    if get_settings().vector_backend != "qdrant":
        return VectorStore.load(tenant.index_path, tenant.chunks_path)

    stamp = tenant.index_dir / MARKER_FILENAME
    if not stamp.is_file():
        raise IndexNotBuiltError(
            f"No index at {stamp.parent}. Upload documents and reindex first."
        )
    marker = json.loads(stamp.read_text(encoding="utf-8"))
    name = marker.get("collection") or collection_name(tenant)

    client = get_qdrant_client()
    try:
        exists = client.collection_exists(name)
        count = client.count(name, exact=True).count if exists else 0
    except Exception as exc:
        raise VectorDBError(_unreachable(exc)) from exc

    if not exists:
        raise IndexNotBuiltError(
            f"Collection {name!r} is missing from Qdrant. Reindex to repair."
        )
    if count != int(marker.get("vectors", -1)):
        raise IndexNotBuiltError(
            f"Qdrant holds {count} vectors but the stamp records "
            f"{marker.get('vectors')}. Reindex to repair."
        )
    return QdrantVectorStore(client, name, count)


def live_vector_counts() -> dict:
    """slug -> exact point count, straight from the live Qdrant server.

    The stamp files say what *was* built; this says what is *there now*. The
    console listing shows both so a tenant whose collection is missing from
    the connected cluster (fresh cluster, changed URL, wiped volume) is
    visible at a glance instead of failing at question time. One request for
    the names plus one per collection — fine at operator-console scale.
    """
    client = get_qdrant_client()
    try:
        names = [c.name for c in client.get_collections().collections]
        return {
            name[len("erag-") :]: client.count(name, exact=True).count
            for name in names
            if name.startswith("erag-")
        }
    except Exception as exc:
        raise VectorDBError(_unreachable(exc)) from exc


def index_info(tenant: "Tenant") -> Optional[dict]:
    """Vector and per-document chunk counts without loading any vectors.

    ``None`` means no index has been built. Cheap on purpose — the console
    calls this once per tenant on every listing.
    """
    if get_settings().vector_backend == "qdrant":
        stamp = tenant.index_dir / MARKER_FILENAME
        if not stamp.is_file():
            return None
        marker = json.loads(stamp.read_text(encoding="utf-8"))
        return {
            "vectors": int(marker.get("vectors", 0)),
            "sources": dict(marker.get("sources", {})),
        }

    if not (tenant.index_path.is_file() and tenant.chunks_path.is_file()):
        return None
    payload = json.loads(tenant.chunks_path.read_text(encoding="utf-8"))
    return {
        "vectors": len(payload),
        "sources": dict(Counter(item.get("source", "") for item in payload)),
    }
