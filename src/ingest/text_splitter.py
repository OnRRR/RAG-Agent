from __future__ import annotations

import re
from typing import Iterable, List


class TextChunker:
    """Splits text into overlapping token windows while preserving section boundaries."""

    def __init__(self, *, chunk_size_tokens: int = 400, chunk_overlap_tokens: int = 80):
        if chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be positive")
        if chunk_overlap_tokens < 0:
            raise ValueError("chunk_overlap_tokens must be non-negative")
        if chunk_overlap_tokens >= chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk size")

        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens

    def split(self, text: str) -> List[str]:
        collapsed = self._normalise_whitespace(text)
        if not collapsed:
            return []

        tokens = collapsed.split()
        if not tokens:
            return []

        chunks: List[str] = []
        step = self.chunk_size_tokens - self.chunk_overlap_tokens
        for start in range(0, len(tokens), step):
            window = tokens[start : start + self.chunk_size_tokens]
            if not window:
                continue
            chunk_text = self._restore_spacing(window)
            chunks.append(chunk_text)
        return chunks

    @staticmethod
    def _normalise_whitespace(text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _restore_spacing(tokens: List[str]) -> str:
        return " ".join(tokens)


def chunk_documents(chunker: TextChunker, texts: Iterable[str]) -> List[List[str]]:
    """Convenience helper for splitting multiple documents in one call."""
    return [chunker.split(text) for text in texts]
