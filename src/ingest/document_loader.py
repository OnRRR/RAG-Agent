from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Sequence

import pdfplumber

from .models import Document

TEXT_FILE_EXTENSIONS = {".txt", ".md"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = TEXT_FILE_EXTENSIONS | PDF_EXTENSIONS


class DocumentLoader:
    """Loads documents from disk and normalises them into plain text."""

    def __init__(self, *, encoding: str = "utf-8") -> None:
        self.encoding = encoding

    def load_documents(self, paths: Sequence[Path]) -> Iterator[Document]:
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Document not found: {path}")

            suffix = path.suffix.lower()
            if suffix in TEXT_FILE_EXTENSIONS:
                yield self._load_text_document(path)
            elif suffix in PDF_EXTENSIONS:
                yield self._load_pdf_document(path)
            else:
                raise ValueError(f"Unsupported file extension for ingestion: {path.suffix}")

    def _load_text_document(self, path: Path) -> Document:
        text = path.read_text(encoding=self.encoding)
        return Document(path=path, text=text, metadata={"content_type": "text/plain"})

    def _load_pdf_document(self, path: Path) -> Document:
        # pdfplumber tends to return raw page text; joining keeps paragraph separation readable.
        with pdfplumber.open(path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        text = "\n".join(pages)
        return Document(path=path, text=text, metadata={"content_type": "application/pdf", "page_count": len(pages)})


def discover_documents(root: Path) -> Iterable[Path]:
    """Recursively yields all supported documents beneath the provided root."""
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path

