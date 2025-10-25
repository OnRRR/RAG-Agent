from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import fitz  # PyMuPDF

from .models import Document, DocumentSection
from src.utils.logger import get_logger

logger = get_logger(__name__)


SECTION_KEYWORDS = {
    "abstract": re.compile(r"^\s*(abstract)\b", re.IGNORECASE),
    "introduction": re.compile(r"^\s*(introduction)\b", re.IGNORECASE),
    "method": re.compile(r"^\s*(methods?|methodology|approach)\b", re.IGNORECASE),
    "results": re.compile(r"^\s*(results?|findings)\b", re.IGNORECASE),
    "discussion": re.compile(r"^\s*(discussion|analysis)\b", re.IGNORECASE),
    "conclusion": re.compile(r"^\s*(conclusions?|summary)\b", re.IGNORECASE),
    "references": re.compile(r"^\s*(references|bibliography|works\s+cited)\b", re.IGNORECASE),
    "acknowledgements": re.compile(r"^\s*(acknowledg(e)?ments?)\b", re.IGNORECASE),
}


@dataclass(slots=True)
class LineInfo:
    text: str
    page: int
    font_size: float
    is_bold: bool


class SectionExtractor:
    """Extracts structural sections from PDF or text documents using heuristics."""

    def __init__(self, *, encoding: str = "utf-8") -> None:
        self.encoding = encoding

    def extract(self, path: Path) -> Document:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_from_pdf(path)
        if suffix in {".txt", ".md"}:
            return self._extract_from_text(path)
        raise ValueError(f"Unsupported file extension for section extraction: {path}")

    # ------------------------------------------------------------------
    # PDF extraction
    # ------------------------------------------------------------------

    def _extract_from_pdf(self, path: Path) -> Document:
        lines: List[LineInfo] = []
        page_count = 0
        metadata: Dict = {}

        try:
            with fitz.open(path) as doc:
                metadata = doc.metadata or {}
                for page_index, page in enumerate(doc, start=1):
                    page_dict = page.get_text("dict")
                    for block in page_dict.get("blocks", []):
                        if block.get("type") != 0:
                            continue
                        for line in block.get("lines", []):
                            spans = line.get("spans", [])
                            if not spans:
                                continue
                            text = "".join(span.get("text", "") for span in spans).strip()
                            if not text:
                                continue
                            font_size = max(span.get("size", 0) for span in spans)
                            is_bold = any(self._is_bold_span(span) for span in spans)
                            lines.append(LineInfo(text=text, page=page_index, font_size=font_size, is_bold=is_bold))
                page_count = len(doc)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("PyMuPDF failed for %s, falling back to basic extraction: %s", path.name, exc)
            return self._fallback_pdf(path)

        title = self._infer_title(lines, metadata)
        sections = self._segment_into_sections(lines)
        if not sections:
            logger.warning("Failed to segment sections for %s, falling back to single block", path.name)
            joined_text = "\n".join(line.text for line in lines)
            sections = [
                DocumentSection(
                    name="body",
                    title="Body",
                    text=joined_text,
                    page_start=1,
                    page_end=page_count,
                )
            ]

        metadata = {
            "content_type": "application/pdf",
            "page_count": page_count,
        }
        if title:
            metadata["title"] = title
        return Document(
            path=path,
            text="\n\n".join(section.text for section in sections),
            title=title,
            sections=sections,
            metadata=metadata,
        )

    @staticmethod
    def _is_bold_span(span: Dict) -> bool:
        flags = span.get("flags", 0)
        # Bitmask 2 usually indicates bold in PyMuPDF, but some fonts use 20.
        return bool(flags & 2) or "Bold" in span.get("font", "")

    @staticmethod
    def _infer_title(lines: List[LineInfo], metadata: Dict) -> Optional[str]:
        if metadata.get("title"):
            return metadata["title"]
        if not lines:
            return None
        sorted_lines = sorted(lines, key=lambda ln: ln.font_size, reverse=True)
        for candidate in sorted_lines[:10]:
            text = candidate.text.strip()
            if len(text) < 200 and len(text) > 6:
                return text
        return None

    def _segment_into_sections(self, lines: List[LineInfo]) -> List[DocumentSection]:
        if not lines:
            return []

        avg_font = sum(line.font_size for line in lines) / max(len(lines), 1)
        sections: List[DocumentSection] = []
        current_lines: List[str] = []
        current_name = "front_matter"
        current_title = "Front Matter"
        current_metadata: Dict[str, Any] = {"section_type": current_name}
        section_start_page = lines[0].page

        def flush_section(end_page: int) -> None:
            nonlocal current_lines, current_name, current_title, section_start_page, current_metadata
            content = "\n".join(current_lines).strip()
            if content:
                sections.append(
                    DocumentSection(
                        name=current_name,
                        title=current_title,
                        text=content,
                        page_start=section_start_page,
                        page_end=end_page,
                        metadata=dict(current_metadata),
                    )
                )
            current_lines = []

        for line in lines:
            if self._looks_like_heading(line, avg_font):
                flush_section(line.page - 1 if line.page > section_start_page else line.page)
                section_start_page = line.page
                normalized_name, display_title = self._normalise_heading(line.text)
                current_name = normalized_name
                current_title = display_title
                current_metadata = {"section_type": current_name}
                if current_name == "references":
                    current_metadata["is_reference"] = True
                current_lines = [line.text]
            else:
                current_lines.append(line.text)

        flush_section(lines[-1].page)
        return sections

    @staticmethod
    def _looks_like_heading(line: LineInfo, avg_font: float) -> bool:
        text = line.text.strip()
        if not text or len(text) > 160:
            return False
        if any(keyword.search(text) for keyword in SECTION_KEYWORDS.values()):
            return True
        if line.font_size >= avg_font * 1.35:
            return True
        if line.is_bold and len(text.split()) <= 12:
            return True
        if re.match(r"^[0-9]+(\.[0-9]+)*\s+\w", text) and len(text.split()) <= 15:
            return True
        if text.isupper() and len(text.split()) <= 12:
            return True
        return False

    @staticmethod
    def _normalise_heading(raw_heading: str) -> tuple[str, str]:
        cleaned = raw_heading.strip()
        cleaned = re.sub(r"^[0-9.\-\s]+", "", cleaned)
        cleaned = re.sub(r"[:\-]+$", "", cleaned).strip()
        if not cleaned:
            cleaned = "Section"
        lower = cleaned.lower()
        for key, pattern in SECTION_KEYWORDS.items():
            if pattern.search(cleaned):
                return key, cleaned
        return re.sub(r"\s+", "_", lower)[:40] or "section", cleaned

    def _fallback_pdf(self, path: Path) -> Document:
        try:
            import pdfplumber
        except ImportError as exc:
            raise RuntimeError(
                "PyMuPDF failed and pdfplumber is not installed; cannot extract text from PDF."
            ) from exc

        with pdfplumber.open(path) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        joined = "\n".join(pages)
        title = self._infer_title_from_text(joined.splitlines())
        section = DocumentSection(
            name="body",
            title=title or "Body",
            text=joined,
            page_start=1,
            page_end=len(pages),
            metadata={"section_type": "body"},
        )
        metadata = {
            "content_type": "application/pdf",
            "page_count": len(pages),
        }
        if title:
            metadata["title"] = title
        return Document(
            path=path,
            text=joined,
            title=title,
            sections=[section],
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Text/Markdown extraction
    # ------------------------------------------------------------------

    def _extract_from_text(self, path: Path) -> Document:
        raw_text = path.read_text(encoding=self.encoding)
        lines = raw_text.splitlines()
        title = self._infer_title_from_text(lines)

        sections: List[DocumentSection] = []
        current_lines: List[str] = []
        current_name = "body"
        current_title = title or "Body"
        current_metadata: Dict[str, Any] = {"section_type": current_name}
        section_start = 1

        def flush_section(section_end: int) -> None:
            nonlocal current_lines, current_name, current_title, section_start, current_metadata
            content = "\n".join(current_lines).strip()
            if content:
                sections.append(
                    DocumentSection(
                        name=current_name,
                        title=current_title,
                        text=content,
                        page_start=section_start,
                        page_end=section_end,
                        metadata=dict(current_metadata),
                    )
                )
            current_lines = []

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if self._line_is_section_heading(stripped):
                flush_section(idx - 1 if idx > section_start else idx)
                section_start = idx
                current_name, current_title = self._normalise_heading(stripped)
                current_metadata = {"section_type": current_name}
                if current_name == "references":
                    current_metadata["is_reference"] = True
                current_lines = [stripped]
            else:
                current_lines.append(line)

        flush_section(len(lines))

        if not sections:
                sections.append(
                    DocumentSection(
                        name="body",
                        title=current_title,
                        text=raw_text,
                        page_start=1,
                        page_end=len(lines),
                        metadata={"section_type": "body"},
                    )
                )

        metadata = {"content_type": "text/plain"}
        if title:
            metadata["title"] = title

        return Document(
            path=path,
            text="\n\n".join(section.text for section in sections),
            title=title,
            sections=sections,
            metadata=metadata,
        )

    @staticmethod
    def _infer_title_from_text(lines: Iterable[str]) -> Optional[str]:
        for line in lines:
            stripped = line.strip()
            if len(stripped) >= 5 and len(stripped) <= 200:
                return stripped
        return None

    @staticmethod
    def _line_is_section_heading(line: str) -> bool:
        if not line or len(line) > 140:
            return False
        if any(keyword.search(line) for keyword in SECTION_KEYWORDS.values()):
            return True
        if line.isupper() and len(line.split()) <= 10:
            return True
        if re.match(r"^#+\s+\w+", line):
            return True
        return False
