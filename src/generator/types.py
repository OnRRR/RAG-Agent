from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from src.retriever.faiss_retriever import RetrievedChunk


@dataclass(slots=True)
class Citation:
    chunk_id: str
    source: str | None
    score: float
    text_preview: str


@dataclass(slots=True)
class GeneratedAnswer:
    answer: str
    citations: Sequence[Citation]


class GeneratorProtocol(Protocol):
    def generate(self, query: str, chunks: Iterable[RetrievedChunk]) -> GeneratedAnswer:
        ...

