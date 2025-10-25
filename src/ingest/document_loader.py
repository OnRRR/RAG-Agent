from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .models import Document
from .sections import SectionExtractor

TEXT_FILE_EXTENSIONS = {".txt", ".md"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = TEXT_FILE_EXTENSIONS | PDF_EXTENSIONS


class DocumentLoader:
    """Loads documents from disk, extracting structural metadata when possible."""

    def __init__(self, *, encoding: str = "utf-8", extractor: SectionExtractor | None = None) -> None:
        self.encoding = encoding
        self.extractor = extractor or SectionExtractor(encoding=encoding)

    def load_documents(self, paths: Sequence[Path]) -> Iterator[Document]:
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Document not found: {path}")

            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported file extension for ingestion: {path.suffix}")

            yield self.extractor.extract(path)


def discover_documents(root: Path) -> Iterable[Path]:
    """Recursively yields all supported documents beneath the provided root."""
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path
