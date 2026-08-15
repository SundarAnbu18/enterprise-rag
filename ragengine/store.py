"""The per-tenant FAISS index: building, saving, loading, searching.

Two files make up a saved index and they must stay in step: ``index.faiss``
holds the vectors, ``chunks.json`` holds the passages in the same order. Row
*i* of the index is chunk *i* of the JSON — that is the whole contract, and
``load`` refuses an index that violates it.

Tenant isolation is structural: a ``VectorStore`` only ever knows the two
paths it was given, and those come from one tenant's directory. There is no
shared index to filter, so a bug cannot leak one customer's passages into
another's retrieval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

from .chunking import Chunk
from .embeddings import Embedder, get_embedder
from .exceptions import IndexNotBuiltError


class SearchResult:
    """A retrieved chunk and how far it sat from the query."""

    __slots__ = ("chunk", "distance")

    def __init__(self, chunk: Chunk, distance: float) -> None:
        self.chunk = chunk
        self.distance = distance

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def source(self) -> str:
        return self.chunk.source

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SearchResult(source={self.source!r}, distance={self.distance:.3f})"


class VectorStore:
    """An in-memory FAISS index plus the chunks its rows correspond to."""

    def __init__(self, index, chunks: Sequence[Chunk]) -> None:
        self._index = index
        self.chunks: List[Chunk] = list(chunks)

    def __len__(self) -> int:
        return len(self.chunks)

    @classmethod
    def build(cls, chunks: Sequence[Chunk], embedder: Optional[Embedder] = None) -> "VectorStore":
        """Embed every chunk and load the vectors into a fresh flat L2 index."""
        import faiss

        chunks = list(chunks)
        embedder = embedder or get_embedder()
        vectors = embedder.encode([chunk.text for chunk in chunks])
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
        return cls(index, chunks)

    @classmethod
    def load(cls, index_path: Path, chunks_path: Path) -> "VectorStore":
        """Read a previously built index off disk."""
        import faiss

        if not index_path.is_file() or not chunks_path.is_file():
            raise IndexNotBuiltError(
                f"No index at {index_path.parent}. Upload documents and reindex first."
            )

        index = faiss.read_index(str(index_path))
        payload = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = [Chunk.from_dict(item) for item in payload]

        if index.ntotal != len(chunks):
            raise IndexNotBuiltError(
                f"Index holds {index.ntotal} vectors but chunks.json holds "
                f"{len(chunks)} passages. Reindex to repair."
            )
        return cls(index, chunks)

    def save(self, index_path: Path, chunks_path: Path) -> None:
        """Write both halves of the index, creating the directory if needed."""
        import faiss

        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path))
        chunks_path.write_text(
            json.dumps([chunk.to_dict() for chunk in self.chunks], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def search(self, query: str, k: int, embedder: Optional[Embedder] = None) -> List[SearchResult]:
        """Return the ``k`` nearest chunks to ``query``, closest first."""
        if not self.chunks:
            return []

        embedder = embedder or get_embedder()
        vector = embedder.encode_query(query)
        distances, indices = self._index.search(vector, max(1, min(k, len(self.chunks))))

        # FAISS pads with -1 when it finds fewer neighbours than asked for.
        return [
            SearchResult(self.chunks[position], float(distance))
            for distance, position in zip(distances[0], indices[0])
            if position != -1
        ]
