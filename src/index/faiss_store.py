from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import faiss
import numpy as np

from src.ingest.models import DocumentChunk


class FaissDocumentIndex:
    """Stores embeddings inside FAISS while keeping chunk metadata alongside."""

    def __init__(self, dim: int, *, storage_dir: Path) -> None:
        self.dim = dim
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index = faiss.IndexFlatIP(dim)
        self._metadata: List[Dict] = []

    @property
    def size(self) -> int:
        return self.index.ntotal

    def add(self, embeddings: np.ndarray, chunks: Sequence[DocumentChunk]) -> None:
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Embeddings and chunks lengths do not match")
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dim}, got {embeddings.shape[1]}")

        self.index.add(embeddings)
        self._metadata.extend(self._serialise_chunk(chunk) for chunk in chunks)

    def persist(self) -> None:
        faiss.write_index(self.index, str(self.storage_dir / "faiss.index"))
        metadata_path = self.storage_dir / "metadata.jsonl"
        with metadata_path.open("w", encoding="utf-8") as fh:
            for entry in self._metadata:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def search(self, query_embeddings: np.ndarray, top_k: int = 5) -> List[Dict]:
        if query_embeddings.shape[1] != self.dim:
            raise ValueError(f"Query embedding dimension mismatch: expected {self.dim}, got {query_embeddings.shape[1]}")
        scores, indices = self.index.search(query_embeddings, top_k)
        results: List[Dict] = []
        for query_idx, neighbors in enumerate(indices):
            for rank, chunk_idx in enumerate(neighbors):
                if chunk_idx == -1:
                    continue
                chunk_meta = self._metadata[chunk_idx]
                results.append(
                    {
                        "query_index": query_idx,
                        "rank": rank,
                        "score": float(scores[query_idx, rank]),
                        **chunk_meta,
                    }
                )
        return results

    def _serialise_chunk(self, chunk: DocumentChunk) -> Dict:
        return {
            "document_id": chunk.document_id,
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "metadata": chunk.metadata,
        }

    @classmethod
    def load(cls, *, storage_dir: Path) -> "FaissDocumentIndex":
        index_path = storage_dir / "faiss.index"
        metadata_path = storage_dir / "metadata.jsonl"
        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError("Index files not found. Have you run ingestion yet?")

        index = faiss.read_index(str(index_path))
        dim = index.d
        instance = cls(dim=dim, storage_dir=storage_dir)
        instance.index = index
        with metadata_path.open("r", encoding="utf-8") as fh:
            instance._metadata = [json.loads(line) for line in fh]
        return instance

    @property
    def metadata(self) -> List[Dict]:
        return self._metadata
