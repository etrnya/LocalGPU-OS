"""Document Classifier: distinguishes native-text PDFs from scanned PDFs and images."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pymupdf as fitz  # PyMuPDF


class DocumentKind(str, Enum):
    NATIVE_TEXT_PDF = "NATIVE_TEXT_PDF"
    SCANNED_PDF = "SCANNED_PDF"
    IMAGE = "IMAGE"
    UNSUPPORTED = "UNSUPPORTED"


_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"})


@dataclass(frozen=True)
class ClassificationResult:
    source_path: Path
    kind: DocumentKind
    page_count: int
    text_page_count: int
    total_chars: int
    avg_chars_per_page: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "kind": self.kind.value,
            "page_count": self.page_count,
            "text_page_count": self.text_page_count,
            "total_chars": self.total_chars,
            "avg_chars_per_page": round(self.avg_chars_per_page, 2),
            "reason": self.reason,
        }


def classify_document(
    source_path: Path,
    min_chars_per_page: int = 20,
    min_text_page_ratio: float = 0.5,
) -> ClassificationResult:
    """Classify input file so native-text PDFs bypass OCR completely."""
    path = Path(source_path)
    ext = path.suffix.lower()

    if ext in _IMAGE_EXTENSIONS:
        return ClassificationResult(
            source_path=path,
            kind=DocumentKind.IMAGE,
            page_count=1,
            text_page_count=0,
            total_chars=0,
            avg_chars_per_page=0.0,
            reason=f"Image file ({ext}) requires OCR pipeline.",
        )

    if ext != ".pdf":
        return ClassificationResult(
            source_path=path,
            kind=DocumentKind.UNSUPPORTED,
            page_count=0,
            text_page_count=0,
            total_chars=0,
            avg_chars_per_page=0.0,
            reason=f"Unsupported file extension: {ext!r}.",
        )

    with fitz.open(path) as doc:
        page_count = len(doc)
        if page_count == 0:
            return ClassificationResult(
                source_path=path,
                kind=DocumentKind.SCANNED_PDF,
                page_count=0,
                text_page_count=0,
                total_chars=0,
                avg_chars_per_page=0.0,
                reason="Empty PDF (0 pages).",
            )

        text_pages = 0
        total_chars = 0
        for page in doc:
            extracted = page.get_text("text").strip()
            char_len = len(extracted)
            total_chars += char_len
            if char_len >= min_chars_per_page:
                text_pages += 1

        avg_chars = total_chars / page_count
        ratio = text_pages / page_count

        if ratio >= min_text_page_ratio and total_chars >= min_chars_per_page:
            return ClassificationResult(
                source_path=path,
                kind=DocumentKind.NATIVE_TEXT_PDF,
                page_count=page_count,
                text_page_count=text_pages,
                total_chars=total_chars,
                avg_chars_per_page=avg_chars,
                reason=(
                    f"Native text detected on {text_pages}/{page_count} pages "
                    f"(avg {avg_chars:.1f} chars/page); bypassing OCR."
                ),
            )

        return ClassificationResult(
            source_path=path,
            kind=DocumentKind.SCANNED_PDF,
            page_count=page_count,
            text_page_count=text_pages,
            total_chars=total_chars,
            avg_chars_per_page=avg_chars,
            reason=(
                f"Low native text coverage ({text_pages}/{page_count} pages, "
                f"avg {avg_chars:.1f} chars/page); OCR required."
            ),
        )
