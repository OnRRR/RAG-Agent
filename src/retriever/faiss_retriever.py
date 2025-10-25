from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from src.index.embedding import SentenceTransformerEmbedder
from src.index.faiss_store import FaissDocumentIndex
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict


class FaissRetriever:
    """Encodes queries, searches FAISS, and returns scored chunks."""

    def __init__(
        self,
        *,
        index_dir: Path,
        embedder: SentenceTransformerEmbedder | None = None,
        top_k: int = 5,
    ) -> None:
        self.index_dir = index_dir
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.top_k = top_k
        self._index: FaissDocumentIndex | None = None

    def _load_index(self) -> FaissDocumentIndex:
        if self._index is None:
            logger.info("Loading FAISS index from %s", self.index_dir)
            self._index = FaissDocumentIndex.load(storage_dir=self.index_dir)
        return self._index

    def retrieve(self, query: str, *, top_k: int | None = None) -> List[RetrievedChunk]:
        if not query.strip():
            raise ValueError("Query must not be empty")
        index = self._load_index()
        embedder = self.embedder
        query_embedding = embedder.embed([query])
        neighbors = index.search(query_embedding, top_k or self.top_k)
        return [self._to_chunk(result) for result in neighbors]

    @staticmethod
    def _to_chunk(result: dict) -> RetrievedChunk:
        metadata = dict(result.get("metadata", {}))
        metadata["rank"] = result["rank"]
        metadata["query_index"] = result["query_index"]
        if result.get("section_name") and "section" not in metadata:
            metadata["section"] = result["section_name"]
        if result.get("section_title") and "section_title" not in metadata:
            metadata["section_title"] = result["section_title"]
        if result.get("page_start") is not None:
            metadata.setdefault("page_start", result["page_start"])
        if result.get("page_end") is not None:
            metadata.setdefault("page_end", result["page_end"])
        return RetrievedChunk(
            chunk_id=result["chunk_id"],
            document_id=result["document_id"],
            text=result["text"],
            score=result["score"],
            metadata=metadata,
        )
