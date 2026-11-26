"""RapidOCR offline inference engine with strict backend verification and SHA-256 model audit."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable

import pymupdf as fitz  # PyMuPDF
import numpy as np
import onnxruntime as ort
from PIL import Image
import rapidocr_onnxruntime
from rapidocr_onnxruntime import RapidOCR

from ocr_pdf_worker.native_extractor import ExtractionCancelledError


@dataclass(frozen=True)
class OcrTextBlock:
    text: str
    confidence: float
    bbox: tuple[tuple[float, float], ...]  # 4-point polygon [[x1,y1], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "bbox": [[round(pt[0], 2), round(pt[1], 2)] for pt in self.bbox],
        }


@dataclass(frozen=True)
class OcrPageResult:
    page_number: int  # 1-based
    width_px: int
    height_px: int
    text: str
    char_count: int
    blocks: tuple[OcrTextBlock, ...]
    processing_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "text": self.text,
            "char_count": self.char_count,
            "block_count": len(self.blocks),
            "processing_time_ms": round(self.processing_time_ms, 2),
            "blocks": [b.to_dict() for b in self.blocks],
        }


@dataclass(frozen=True)
class OcrBackendAudit:
    requested_backend: str
    actual_backend: str
    cuda_verified: bool
    available_providers: tuple[str, ...]
    active_providers: dict[str, tuple[str, ...]]
    model_precision: str
    model_manifest: tuple[dict[str, Any], ...]
    rapidocr_version: str
    onnxruntime_version: str
    intra_op_num_threads: int
    checksums_verified: bool = False
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_backend": self.requested_backend,
            "actual_backend": self.actual_backend,
            "cuda_verified": self.cuda_verified,
            "available_providers": list(self.available_providers),
            "active_providers": {k: list(v) for k, v in self.active_providers.items()},
            "model_precision": self.model_precision,
            "model_manifest": list(self.model_manifest),
            "checksums_verified": self.checksums_verified,
            "rapidocr_version": self.rapidocr_version,
            "onnxruntime_version": self.onnxruntime_version,
            "intra_op_num_threads": self.intra_op_num_threads,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class OcrExtraction:
    source_path: Path
    source_kind: str  # "SCANNED_PDF" | "IMAGE"
    page_count: int
    pages: tuple[OcrPageResult, ...]
    total_chars: int
    backend_audit: OcrBackendAudit
    dpi: int | None = None

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
            "source_kind": self.source_kind,
            "extraction_method": "rapidocr",
            "ocr_invoked": True,
            "actual_backend": self.backend_audit.actual_backend,
            "cuda_verified": self.backend_audit.cuda_verified,
            "page_count": self.page_count,
            "total_chars": self.total_chars,
            "pages": [p.to_dict() for p in self.pages],
        }


def collect_bundled_model_manifest() -> tuple[dict[str, Any], ...]:
    """Audit bundled offline ONNX model files and compute their SHA-256 hashes."""
    pkg_dir = Path(rapidocr_onnxruntime.__file__).resolve().parent
    entries: list[dict[str, Any]] = []
    for model_path in sorted(pkg_dir.rglob("*.onnx")):
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        entries.append(
            {
                "filename": model_path.name,
                "size_bytes": model_path.stat().st_size,
                "sha256": digest,
            }
        )
    return tuple(entries)


def _extract_active_providers(engine: RapidOCR) -> dict[str, tuple[str, ...]]:
    providers: dict[str, tuple[str, ...]] = {}
    if hasattr(engine, "text_det") and hasattr(engine.text_det, "infer"):
        providers["det"] = tuple(engine.text_det.infer.session.get_providers())
    if hasattr(engine, "text_cls") and hasattr(engine.text_cls, "infer"):
        providers["cls"] = tuple(engine.text_cls.infer.session.get_providers())
    if hasattr(engine, "text_rec") and hasattr(engine.text_rec, "session"):
        providers["rec"] = tuple(engine.text_rec.session.session.get_providers())
    return providers


def default_checksums_path() -> Path:
    return Path(__file__).resolve().parents[1] / "model_checksums.json"


def verify_bundled_models(checksums_path: Path | None = None) -> tuple[dict[str, Any], ...]:
    """Verify bundled ONNX model files against pinned SHA-256 checksums BEFORE loading.

    Fail-Closed: raises RuntimeError if checksums file is unreadable, any pinned model is
    missing, or any SHA-256 digest does not match.
    """
    lock_path = Path(checksums_path) if checksums_path is not None else default_checksums_path()
    if not lock_path.exists() or not lock_path.is_file():
        raise RuntimeError(f"Model SHA-256 checksums lock file not found: {lock_path}")

    try:
        lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
        expected_entries = lock_data["models"]
        if not isinstance(expected_entries, list) or not expected_entries:
            raise ValueError("models list must be non-empty.")
    except Exception as exc:
        raise RuntimeError(f"Invalid model checksums file {lock_path}: {exc}") from exc

    actual_manifest = collect_bundled_model_manifest()
    actual_by_name = {entry["filename"]: entry for entry in actual_manifest}

    for expected in expected_entries:
        fname = expected["filename"]
        exp_sha = str(expected["sha256"]).lower()
        if fname not in actual_by_name:
            raise RuntimeError(
                f"Model SHA-256 integrity verification failed: missing bundled model {fname!r}."
            )
        actual_sha = str(actual_by_name[fname]["sha256"]).lower()
        if actual_sha != exp_sha:
            raise RuntimeError(
                f"Model SHA-256 integrity verification failed for {fname!r}: "
                f"expected {exp_sha}, got {actual_sha}."
            )

    return actual_manifest


class RapidOcrEngine:
    """Offline RapidOCR runner with pre-load SHA-256 verification and CUDA/CPU provider audit."""

    def __init__(
        self,
        requested_backend: str = "cpu_int8",
        intra_op_num_threads: int = 2,
        strict_backend: bool = False,
        checksums_path: Path | None = None,
    ) -> None:
        self._requested_backend = requested_backend
        self._intra_op_num_threads = max(1, int(intra_op_num_threads))
        self._strict_backend = strict_backend

        # Pre-load integrity gate: verify all pinned ONNX models BEFORE initializing RapidOCR
        models = verify_bundled_models(checksums_path=checksums_path)

        available = tuple(ort.get_available_providers())
        wants_cuda = requested_backend.lower() == "cuda"
        has_cuda_ep = "CUDAExecutionProvider" in available

        if wants_cuda and has_cuda_ep and hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
            except Exception:
                pass

        fallback_reason: str | None = None
        use_cuda = wants_cuda and has_cuda_ep
        if wants_cuda and not has_cuda_ep:
            fallback_reason = (
                "CUDAExecutionProvider is not available in onnxruntime "
                f"(available={list(available)})."
            )
            if strict_backend:
                raise RuntimeError(fallback_reason)

        init_kwargs: dict[str, Any] = {
            "det_use_cuda": use_cuda,
            "cls_use_cuda": use_cuda,
            "rec_use_cuda": use_cuda,
            "intra_op_num_threads": self._intra_op_num_threads,
        }
        self._ocr = RapidOCR(**init_kwargs)
        active = _extract_active_providers(self._ocr)

        # Verify CUDA is genuinely active on det & rec sessions (never trust requested flag alone)
        cuda_verified = (
            use_cuda
            and "CUDAExecutionProvider" in active.get("det", ())
            and "CUDAExecutionProvider" in active.get("rec", ())
        )
        if wants_cuda and not cuda_verified:
            if fallback_reason is None:
                fallback_reason = (
                    "Requested CUDA but ONNX Runtime fell back to CPU "
                    f"(active_providers={active})."
                )
            if strict_backend:
                raise RuntimeError(fallback_reason)

        actual_backend = "cuda" if cuda_verified else "cpu"
        # Be honest about model precision: bundled PP-OCRv4 models are FP32 ONNX unless quantized
        is_int8_models = any("int8" in m["filename"].lower() for m in models)
        if cuda_verified:
            model_precision = "int8_onnx_cuda" if is_int8_models else "fp32_onnx_cuda"
        else:
            model_precision = "int8_onnx_cpu" if is_int8_models else "fp32_onnx_cpu"

        self._audit = OcrBackendAudit(
            requested_backend=requested_backend,
            actual_backend=actual_backend,
            cuda_verified=cuda_verified,
            available_providers=available,
            active_providers=active,
            model_precision=model_precision,
            model_manifest=models,
            checksums_verified=True,
            rapidocr_version=getattr(rapidocr_onnxruntime, "__version__", "unknown"),
            onnxruntime_version=ort.__version__,
            intra_op_num_threads=self._intra_op_num_threads,
            fallback_reason=fallback_reason,
        )

    @property
    def backend_audit(self) -> OcrBackendAudit:
        return self._audit

    def _run_on_rgb_array(self, rgb_array: np.ndarray, page_number: int) -> OcrPageResult:
        t0 = time.monotonic()
        raw_result, _ = self._ocr(rgb_array)
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        blocks: list[OcrTextBlock] = []
        lines: list[str] = []
        if raw_result:
            for item in raw_result:
                raw_box, raw_text, raw_score = item[0], str(item[1]), float(item[2])
                bbox = tuple((float(pt[0]), float(pt[1])) for pt in raw_box)
                blocks.append(OcrTextBlock(text=raw_text, confidence=raw_score, bbox=bbox))
                if raw_text.strip():
                    lines.append(raw_text.strip())

        full_text = "\n".join(lines)
        h, w = int(rgb_array.shape[0]), int(rgb_array.shape[1])
        return OcrPageResult(
            page_number=page_number,
            width_px=w,
            height_px=h,
            text=full_text,
            char_count=len(full_text),
            blocks=tuple(blocks),
            processing_time_ms=elapsed_ms,
        )

    def extract_image(
        self,
        image_path: Path,
        cancel_check: Callable[[], bool] | None = None,
    ) -> OcrExtraction:
        if cancel_check is not None and cancel_check():
            raise ExtractionCancelledError("Cancelled before image OCR.")

        path = Path(image_path)
        with Image.open(path) as img:
            rgb = np.asarray(img.convert("RGB"))

        page = self._run_on_rgb_array(rgb, page_number=1)
        return OcrExtraction(
            source_path=path,
            source_kind="IMAGE",
            page_count=1,
            pages=(page,),
            total_chars=page.char_count,
            backend_audit=self._audit,
            dpi=None,
        )

    def extract_scanned_pdf(
        self,
        pdf_path: Path,
        dpi: int = 200,
        cancel_check: Callable[[], bool] | None = None,
    ) -> OcrExtraction:
        path = Path(pdf_path)
        pages: list[OcrPageResult] = []
        total_chars = 0
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        with fitz.open(path) as doc:
            for idx, page in enumerate(doc, start=1):
                if cancel_check is not None and cancel_check():
                    raise ExtractionCancelledError(f"Cancelled before OCR page {idx}.")

                pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
                rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, 3
                )
                page_res = self._run_on_rgb_array(rgb, page_number=idx)
                pages.append(page_res)
                total_chars += page_res.char_count

            return OcrExtraction(
                source_path=path,
                source_kind="SCANNED_PDF",
                page_count=len(doc),
                pages=tuple(pages),
                total_chars=total_chars,
                backend_audit=self._audit,
                dpi=dpi,
            )
