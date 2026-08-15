"""Turning source documents into the units that get embedded and retrieved.

One paragraph — text between blank lines — is one chunk. Knowledge-base
documents are written as self-contained statements, so paragraphs retrieve far
more precisely than fixed-size windows, which tend to cut sentences in half.
Very long paragraphs are split on sentence boundaries as a backstop so a single
wall of text can't blow past the embedding model's useful input length.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

# Beyond this, a "paragraph" is really several ideas glued together and dilutes
# its own embedding. Roughly two to three hundred tokens.
MAX_CHUNK_CHARS = 1200

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    """A retrievable passage, and the file it came from."""

    text: str
    source: str

    def to_dict(self) -> dict:
        return {"text": self.text, "source": self.source}

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(text=data["text"], source=data.get("source", ""))


def split_paragraphs(text: str) -> List[str]:
    """Split on blank lines and collapse the whitespace inside each paragraph."""
    return [" ".join(part.split()) for part in text.split("\n\n") if part.strip()]


def _split_long(paragraph: str) -> List[str]:
    """Break an oversized paragraph on sentence boundaries."""
    if len(paragraph) <= MAX_CHUNK_CHARS:
        return [paragraph]

    pieces: List[str] = []
    current = ""
    for sentence in _SENTENCE_END.split(paragraph):
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > MAX_CHUNK_CHARS and current:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def chunk_document(text: str, source: str) -> List[Chunk]:
    """Chunk one document, tagging every chunk with its filename."""
    chunks: List[Chunk] = []
    for paragraph in split_paragraphs(text):
        for body in _split_long(paragraph):
            chunks.append(Chunk(text=body, source=source))
    return chunks
