from __future__ import annotations

import re
from typing import Iterable, List


class TextChunker:
    """Splits raw text into overlapping chunks that work well with embedding models."""

    def __init__(self, *, chunk_size: int = 512, chunk_overlap: int = 128):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        # Collapse multiple blank lines and trim whitespace so chunking is consistent.
        cleaned = self._normalise_whitespace(text)
        if not cleaned:
            return []

        chunks: List[str] = []
        start = 0
        while start < len(cleaned):
            end = start + self.chunk_size
            chunk_text = cleaned[start:end]
            chunks.append(chunk_text.strip())
            start += self.chunk_size - self.chunk_overlap
        return [chunk for chunk in chunks if chunk]

    @staticmethod
    def _normalise_whitespace(text: str) -> str:
        without_extra_spaces = re.sub(r"[ \t]+", " ", text)
        return re.sub(r"\n{3,}", "\n\n", without_extra_spaces).strip()


def chunk_documents(chunker: TextChunker, texts: Iterable[str]) -> List[List[str]]:
    """Convenience helper for splitting multiple documents in one call."""
    return [chunker.split(text) for text in texts]

