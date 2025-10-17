from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
import uuid


@dataclass(slots=True)
class Document:
    """Represents a source document that will be ingested into the RAG index."""

    path: Path
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        # Store only the filename by default to avoid leaking absolute paths.
        self.metadata.setdefault("source", self.path.name)


@dataclass(slots=True)
class DocumentChunk:
    """Represents a single chunk of text derived from a source document."""

    document_id: str
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_document(
        cls, document: Document, chunk_text: str, chunk_index: int, *, extra_metadata: Dict[str, Any] | None = None
    ) -> "DocumentChunk":
        metadata: Dict[str, Any] = {"chunk_index": chunk_index, "source": document.metadata.get("source", document.path.name)}
        if extra_metadata:
            metadata.update(extra_metadata)
        return cls(
            document_id=document.document_id,
            chunk_id=f"{document.document_id}::chunk::{chunk_index}",
            text=chunk_text,
            metadata=metadata,
        )


ChunkBatch = List[DocumentChunk]

