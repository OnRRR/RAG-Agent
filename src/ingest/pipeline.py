from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from src.index.embedding import SentenceTransformerEmbedder
from src.index.faiss_store import FaissDocumentIndex
from src.utils.logger import get_logger

from .document_loader import DocumentLoader
from .models import Document, DocumentChunk, DocumentSection
from .text_splitter import TextChunker

logger = get_logger(__name__)


class IngestionPipeline:
    """Coordinates document loading, chunking, embedding, and indexing."""

    def __init__(
        self,
        *,
        loader: DocumentLoader | None = None,
        chunker: TextChunker | None = None,
        embedder: SentenceTransformerEmbedder | None = None,
        index_storage_dir: Path,
    ) -> None:
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or TextChunker()
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.index = FaissDocumentIndex(dim=self.embedder.dimension, storage_dir=index_storage_dir)

    def ingest_paths(self, paths: Sequence[Path]) -> None:
        documents = list(self.loader.load_documents(paths))
        chunks = self._chunk_documents(documents)
        if not chunks:
            logger.warning("No text content extracted from provided documents")
            return

        all_chunk_texts = [chunk.text for chunk in chunks]
        logger.info("Embedding %d chunks", len(all_chunk_texts))
        embeddings = self.embedder.embed(all_chunk_texts)
        self.index.add(embeddings, chunks)
        self.index.persist()
        logger.info("Successfully ingested %d documents into index %s", len(documents), self.index.storage_dir)

    def ingest_directory(self, root: Path) -> None:
        from .document_loader import discover_documents

        paths = list(discover_documents(root))
        if not paths:
            logger.warning("No supported documents found under %s", root)
            return

        self.ingest_paths(paths)

    def _chunk_documents(self, documents: Iterable[Document]) -> list[DocumentChunk]:
        chunk_collection: list[DocumentChunk] = []
        for document in documents:
            page_count = document.metadata.get("page_count")
            if isinstance(page_count, int):
                fallback_page_end = page_count
            else:
                fallback_page_end = 1

            sections = document.sections or [
                DocumentSection(
                    name="body",
                    title=document.title or "Body",
                    text=document.text,
                    page_start=1,
                    page_end=fallback_page_end,
                    metadata={"section_type": "body"},
                )
            ]

            for section_index, section in enumerate(sections):
                chunk_texts = self.chunker.split(section.text)
                for chunk_index, text in enumerate(chunk_texts):
                    if not text.strip():
                        continue
                    extra_metadata = dict(section.metadata)
                    chunk = DocumentChunk.from_section(
                        document,
                        section,
                        text,
                        section_index=section_index,
                        chunk_index=chunk_index,
                        extra_metadata=extra_metadata,
                    )
                    chunk_collection.append(chunk)
        return chunk_collection
