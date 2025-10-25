from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid


@dataclass(slots=True)
class DocumentSection:
    """Represents a semantically coherent section within a document."""

    name: str
    title: str
    text: str
    page_start: int
    page_end: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Document:
    """Represents a source document that will be ingested into the RAG index."""

    path: Path
    text: str
    title: Optional[str] = None
    sections: List[DocumentSection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    document_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        self.metadata.setdefault("source", self.path.name)
        if self.title:
            self.metadata.setdefault("title", self.title)


@dataclass(slots=True)
class DocumentChunk:
    """Represents a single chunk of text derived from a source document."""

    document_id: str
    chunk_id: str
    text: str
    section_name: Optional[str] = None
    section_title: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_section(
        cls,
        document: Document,
        section: DocumentSection,
        chunk_text: str,
        *,
        section_index: int,
        chunk_index: int,
        extra_metadata: Dict[str, Any] | None = None,
    ) -> "DocumentChunk":
        metadata: Dict[str, Any] = {
            "chunk_index": chunk_index,
            "section_index": section_index,
            "source": document.metadata.get("source", document.path.name),
            "section": section.name,
            "section_title": section.title,
            "page_start": section.page_start,
            "page_end": section.page_end,
        }
        metadata["section_type"] = section.metadata.get("section_type", section.name)
        if section.metadata.get("is_reference"):
            metadata["is_reference"] = True
        if document.metadata.get("title"):
            metadata["title"] = document.metadata["title"]
        if extra_metadata:
            metadata.update(extra_metadata)
        return cls(
            document_id=document.document_id,
            chunk_id=f"{document.document_id}::sec::{section_index}::chunk::{chunk_index}",
            text=chunk_text,
            section_name=section.name,
            section_title=section.title,
            page_start=section.page_start,
            page_end=section.page_end,
            metadata=metadata,
        )


ChunkBatch = List[DocumentChunk]
