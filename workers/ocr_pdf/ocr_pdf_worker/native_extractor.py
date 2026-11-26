"""Native-text PDF extraction powered by PyMuPDF (zero OCR invocation)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pymupdf as fitz  # PyMuPDF

from ocr_pdf_worker.triage import (
    PageTriageResult,
    TriageConfig,
    evaluate_native_page_triage,
)


class ExtractionCancelledError(Exception):
    """Raised internally when the cooperative cancel signal is observed mid-document."""


@dataclass(frozen=True)
class ExtractedPage:
    page_number: int  # 1-based
    text: str
    char_count: int
    width_pt: float
    height_pt: float
    quality_triage: PageTriageResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "text": self.text,
            "char_count": self.char_count,
            "width_pt": round(self.width_pt, 2),
            "height_pt": round(self.height_pt, 2),
            "quality_triage": self.quality_triage.to_dict(),
        }


@dataclass(frozen=True)
class NativePdfExtraction:
    source_path: Path
    page_count: int
    pages: tuple[ExtractedPage, ...]
    pdf_metadata: dict[str, str]
    total_chars: int

    def to_markdown(self) -> str:
        sections: list[str] = [f"# {self.source_path.name}", ""]
        for page in self.pages:
            sections.append(f"## Page {page.page_number}")
            sections.append("")
            sections.append(page.text.strip())
            sections.append("")
        return "\n".join(sections).rstrip() + "\n"

    def to_document_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "source_file": self.source_path.name,
            "extraction_method": "pymupdf_native",
            "ocr_invoked": False,
            "page_count": self.page_count,
            "total_chars": self.total_chars,
            "pages": [p.to_dict() for p in self.pages],
        }


def extract_native_pdf(
    source_path: Path,
    cancel_check: Callable[[], bool] | None = None,
    triage_config: TriageConfig | None = None,
) -> NativePdfExtraction:
    """Extract native text page-by-page with 1-based page numbers and M2-A quality triage."""
    path = Path(source_path)
    pages: list[ExtractedPage] = []
    total_chars = 0

    with fitz.open(path) as doc:
        raw_meta = doc.metadata or {}
        pdf_metadata = {
            str(k): str(v)
            for k, v in raw_meta.items()
            if v is not None and str(v).strip() != ""
        }

        for idx, page in enumerate(doc, start=1):
            if cancel_check is not None and cancel_check():
                raise ExtractionCancelledError(f"Cancelled before processing page {idx}.")

            text = page.get_text("text")
            cleaned = text.replace("\r\n", "\n").strip()
            rect = page.rect
            char_count = len(cleaned)
            total_chars += char_count

            image_rects: list[tuple[float, float, float, float]] = []
            for img_info in page.get_image_info(xrefs=True) or ():
                bbox = img_info.get("bbox")
                if bbox is not None and len(bbox) == 4:
                    image_rects.append((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))

            try:
                vector_count = len(page.get_drawings() or ())
            except Exception:
                vector_count = 0

            triage = evaluate_native_page_triage(
                char_count=char_count,
                image_rects=image_rects,
                page_width=float(rect.width),
                page_height=float(rect.height),
                vector_drawing_count=vector_count,
                config=triage_config,
            )

            pages.append(
                ExtractedPage(
                    page_number=idx,
                    text=cleaned,
                    char_count=char_count,
                    width_pt=float(rect.width),
                    height_pt=float(rect.height),
                    quality_triage=triage,
                )
            )

        return NativePdfExtraction(
            source_path=path,
            page_count=len(doc),
            pages=tuple(pages),
            pdf_metadata=pdf_metadata,
            total_chars=total_chars,
        )
