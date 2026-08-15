"""Turning one tenant's uploaded corpus into their saved index.

This is the offline half of the system. Documents arrive through the API (or
are dropped into the tenant's ``documents/`` directory by hand); rebuilding is
cheap at knowledge-base scale, so uploads reindex synchronously and the change
is queryable on the very next request — the pipeline notices the new index
mtime without any restart.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from .chunking import Chunk, chunk_document
from .exceptions import ConfigurationError, NoDocumentsError
from .store import VectorStore
from .tenants import Tenant

DOCUMENT_SUFFIXES = (".txt", ".md")

# Uploaded filenames become paths inside the tenant directory, so they are
# validated rather than trusted — no separators, no leading dots.
FILENAME_RE = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._ -]{0,120}\Z")

MAX_DOCUMENT_BYTES = 2 * 1024 * 1024


def iter_document_paths(documents_dir: Path) -> List[Path]:
    """Every indexable file under ``documents_dir``, in a stable order."""
    if not documents_dir.is_dir():
        return []
    return sorted(
        path
        for path in documents_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in DOCUMENT_SUFFIXES
    )


def load_chunks(tenant: Tenant) -> List[Chunk]:
    """Read and chunk the tenant's whole corpus."""
    paths = iter_document_paths(tenant.documents_dir)
    if not paths:
        raise NoDocumentsError(
            f"Tenant {tenant.slug!r} has no {' or '.join(DOCUMENT_SUFFIXES)} documents yet"
        )

    chunks: List[Chunk] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks.extend(chunk_document(text, source=path.name))

    if not chunks:
        raise NoDocumentsError(f"Tenant {tenant.slug!r} documents contain no readable text")
    return chunks


def save_document(tenant: Tenant, filename: str, text: str) -> Path:
    """Store one uploaded document in the tenant's corpus."""
    filename = (filename or "").strip()
    if not FILENAME_RE.match(filename):
        raise ConfigurationError(
            "filename must be a plain name like 'handbook.md' (letters, digits, dot, dash)"
        )
    if Path(filename).suffix.lower() not in DOCUMENT_SUFFIXES:
        raise ConfigurationError(f"filename must end in {' or '.join(DOCUMENT_SUFFIXES)}")
    if not (text or "").strip():
        raise ConfigurationError("document text is empty")
    if len(text.encode("utf-8")) > MAX_DOCUMENT_BYTES:
        raise ConfigurationError("document exceeds the 2 MB limit")

    tenant.documents_dir.mkdir(parents=True, exist_ok=True)
    path = tenant.documents_dir / filename
    path.write_text(text, encoding="utf-8")
    return path


def delete_document(tenant: Tenant, filename: str) -> bool:
    """Remove one document from the corpus. Returns whether it existed."""
    if not FILENAME_RE.match((filename or "").strip()):
        raise ConfigurationError("invalid filename")
    path = tenant.documents_dir / filename
    if not path.is_file():
        return False
    path.unlink()
    return True


def build_index(tenant: Tenant, embedder=None) -> VectorStore:
    """Chunk, embed and persist the tenant's corpus. Returns the store it wrote."""
    chunks = load_chunks(tenant)
    store = VectorStore.build(chunks, embedder=embedder)
    store.save(tenant.index_path, tenant.chunks_path)
    return store
