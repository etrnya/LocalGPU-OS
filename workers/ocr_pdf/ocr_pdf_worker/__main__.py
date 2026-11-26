"""CLI & LocalGPU-OS WorkerRuntime entrypoint for `ocr_pdf` (M1-D D1 Native PDF + D2 RapidOCR)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import time
from typing import Any, Sequence

import pymupdf as fitz  # PyMuPDF

from ocr_pdf_worker import __version__
from ocr_pdf_worker.classifier import DocumentKind, classify_document
from ocr_pdf_worker.native_extractor import ExtractionCancelledError, extract_native_pdf
from ocr_pdf_worker.ocr_engine import RapidOcrEngine
from ocr_pdf_worker.output_writer import (
    export_review_pages_into_staging,
    write_job_output_atomic,
)
from ocr_pdf_worker.triage import TriageConfig, build_triage_summary


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _classify_error_kind(exc: BaseException) -> str:
    if isinstance(exc, MemoryError):
        return "OOM"
    msg = str(exc).lower()
    if any(
        token in msg
        for token in (
            "out of memory",
            "cuda out of memory",
            "cudamalloc",
            "bfc_arena",
            "failed to allocate memory",
            "std::bad_alloc",
        )
    ):
        return "OOM"
    return "RUNTIME_ERROR"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocr_pdf_worker",
        description="LocalGPU-OS Layered Offline OCR / PDF Worker (M1-D D1+D2)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root output directory for job bundles (default: <LOCALGPU_WORK_DIR>/output)",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Optional base job ID prefix (default: LOCALGPU_RUN_ID)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Rendering DPI for scanned PDF pages (default: 200)",
    )
    parser.add_argument(
        "--strict-backend",
        action="store_true",
        help="Fail instead of falling back to CPU if requested CUDA provider is unavailable",
    )
    parser.add_argument(
        "--export-review-images",
        action="store_true",
        help="Export PNGs + review_manifest.json ONLY for pages flagged needs_vision_review",
    )
    parser.add_argument(
        "--review-dpi",
        type=int,
        default=150,
        help="Rendering DPI for exported review_pages PNGs (default: 150)",
    )
    parser.add_argument(
        "--max-review-image-mib",
        type=float,
        default=15.0,
        help="Maximum allowed size in MiB per exported review PNG (default: 15.0)",
    )
    return parser


def _handle_cancel(ctx: Any, reason: str) -> int:
    ctx.cleanup_temp_dir()
    released = ctx.release_lease()
    ctx.write_cancelled({"reason": reason, "released_by_worker": released})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    from localgpu_os.sdk import WorkerContext

    parser = _build_parser()
    args = parser.parse_args(argv)
    ctx = WorkerContext.from_env()

    output_root = args.output_dir if args.output_dir is not None else (ctx.work_dir / "output")
    base_job_id = args.job_id or ctx.run_id
    max_cpu_threads = int(os.environ.get("LOCALGPU_MAX_CPU_THREADS", "2"))

    manifest = ctx.load_input_manifest()
    if not manifest or not manifest.get("files"):
        ctx.write_result(
            {
                "status": "FAILED",
                "error": "No input files provided in LOCALGPU_INPUT_MANIFEST.",
                "worker_version": __version__,
            }
        )
        return 1

    files = manifest["files"]
    jobs_summary: list[dict[str, Any]] = []
    any_failed = False
    triage_config = TriageConfig()
    max_review_bytes = max(1, int(args.max_review_image_mib * 1024 * 1024))

    # Lazy-init RapidOCR only when an input actually needs OCR (never loaded for native PDFs!)
    ocr_engine: RapidOcrEngine | None = None

    for idx, entry in enumerate(files, start=1):
        sig = ctx.check_cancelled()
        if sig is not None:
            return _handle_cancel(ctx, sig.reason)

        source_path = Path(entry["path"])
        job_id = base_job_id if len(files) == 1 else f"{base_job_id}_{idx:03d}"
        t0 = time.monotonic()

        try:
            classification = classify_document(source_path)

            if classification.kind == DocumentKind.UNSUPPORTED:
                any_failed = True
                jobs_summary.append(
                    {
                        "job_id": job_id,
                        "source_file": source_path.name,
                        "status": "UNSUPPORTED",
                        "classification": classification.to_dict(),
                        "ocr_invoked": False,
                        "message": classification.reason,
                    }
                )
                continue

            if classification.kind == DocumentKind.NATIVE_TEXT_PDF:
                extraction = extract_native_pdf(
                    source_path,
                    cancel_check=lambda: ctx.check_cancelled() is not None,
                    triage_config=triage_config,
                )
                elapsed_ms = round((time.monotonic() - t0) * 1000.0, 2)

                page_triages = [(p.page_number, p.quality_triage) for p in extraction.pages]
                review_items = [
                    (p.page_number, [r.value for r in p.quality_triage.reasons])
                    for p in extraction.pages
                    if p.quality_triage.needs_vision_review
                ]
                will_export_images = bool(args.export_review_images and review_items)
                triage_summary = build_triage_summary(
                    page_triages=page_triages,
                    config=triage_config,
                    review_images_exported=will_export_images,
                )

                metadata_payload: dict[str, Any] = {
                    "schema_version": "1.0",
                    "job_id": job_id,
                    "run_id": ctx.run_id,
                    "worker_id": ctx.worker_id,
                    "worker_version": __version__,
                    "pymupdf_version": getattr(fitz, "__doc__", "").splitlines()[0]
                    if hasattr(fitz, "__doc__") and fitz.__doc__
                    else "PyMuPDF",
                    "source_file": source_path.name,
                    "source_size_bytes": source_path.stat().st_size,
                    "classification": classification.to_dict(),
                    "extraction_method": "pymupdf_native",
                    "ocr_invoked": False,
                    "compute_backend": ctx.compute_backend,
                    "network_policy": ctx.network_policy,
                    "page_count": extraction.page_count,
                    "total_chars": extraction.total_chars,
                    "processing_time_ms": elapsed_ms,
                    "processed_at": _utc_now_iso(),
                    "pdf_metadata": extraction.pdf_metadata,
                    "triage_summary": triage_summary,
                }

                review_export_fn = (
                    (
                        lambda staging, _sp=source_path, _ri=review_items: export_review_pages_into_staging(
                            staging_dir=staging,
                            source_path=_sp,
                            source_kind="NATIVE_TEXT_PDF",
                            review_pages=_ri,
                            review_dpi=args.review_dpi,
                            max_image_bytes=max_review_bytes,
                        )
                    )
                    if will_export_images
                    else None
                )

                bundle = write_job_output_atomic(
                    output_root=output_root,
                    job_id=job_id,
                    markdown_text=extraction.to_markdown(),
                    document_payload=extraction.to_document_dict(),
                    metadata_payload=metadata_payload,
                    review_export_fn=review_export_fn,
                )

                jobs_summary.append(
                    {
                        "job_id": job_id,
                        "source_file": source_path.name,
                        "status": "SUCCEEDED",
                        "extraction_method": "pymupdf_native",
                        "ocr_invoked": False,
                        "page_count": extraction.page_count,
                        "total_chars": extraction.total_chars,
                        "vision_review_pages": triage_summary["vision_review_pages"],
                        "processing_time_ms": elapsed_ms,
                        "outputs": bundle.to_dict(),
                    }
                )
                continue

            # Scanned PDF or Image -> Lazy-load RapidOCR offline engine
            if ocr_engine is None:
                ocr_engine = RapidOcrEngine(
                    requested_backend=ctx.compute_backend,
                    intra_op_num_threads=max_cpu_threads,
                    strict_backend=args.strict_backend,
                )

            if classification.kind == DocumentKind.IMAGE:
                ocr_res = ocr_engine.extract_image(
                    source_path,
                    cancel_check=lambda: ctx.check_cancelled() is not None,
                    triage_config=triage_config,
                )
            else:
                ocr_res = ocr_engine.extract_scanned_pdf(
                    source_path,
                    dpi=args.dpi,
                    cancel_check=lambda: ctx.check_cancelled() is not None,
                    triage_config=triage_config,
                )

            elapsed_ms = round((time.monotonic() - t0) * 1000.0, 2)
            audit_dict = ocr_res.backend_audit.to_dict()

            page_triages = [(p.page_number, p.quality_triage) for p in ocr_res.pages]
            review_items = [
                (p.page_number, [r.value for r in p.quality_triage.reasons])
                for p in ocr_res.pages
                if p.quality_triage.needs_vision_review
            ]
            will_export_images = bool(args.export_review_images and review_items)
            triage_summary = build_triage_summary(
                page_triages=page_triages,
                config=triage_config,
                review_images_exported=will_export_images,
            )

            metadata_payload = {
                "schema_version": "1.0",
                "job_id": job_id,
                "run_id": ctx.run_id,
                "worker_id": ctx.worker_id,
                "worker_version": __version__,
                "source_file": source_path.name,
                "source_size_bytes": source_path.stat().st_size,
                "classification": classification.to_dict(),
                "extraction_method": "rapidocr",
                "ocr_invoked": True,
                "compute_backend": ocr_res.backend_audit.actual_backend,
                "requested_backend": ocr_res.backend_audit.requested_backend,
                "cuda_verified": ocr_res.backend_audit.cuda_verified,
                "ocr_backend_audit": audit_dict,
                "network_policy": ctx.network_policy,
                "page_count": ocr_res.page_count,
                "total_chars": ocr_res.total_chars,
                "dpi": ocr_res.dpi,
                "processing_time_ms": elapsed_ms,
                "processed_at": _utc_now_iso(),
                "triage_summary": triage_summary,
            }

            review_export_fn = (
                (
                    lambda staging, _sp=source_path, _sk=ocr_res.source_kind, _ri=review_items: export_review_pages_into_staging(
                        staging_dir=staging,
                        source_path=_sp,
                        source_kind=_sk,
                        review_pages=_ri,
                        review_dpi=args.review_dpi,
                        max_image_bytes=max_review_bytes,
                    )
                )
                if will_export_images
                else None
            )

            bundle = write_job_output_atomic(
                output_root=output_root,
                job_id=job_id,
                markdown_text=ocr_res.to_markdown(),
                document_payload=ocr_res.to_document_dict(),
                metadata_payload=metadata_payload,
                review_export_fn=review_export_fn,
            )

            jobs_summary.append(
                {
                    "job_id": job_id,
                    "source_file": source_path.name,
                    "status": "SUCCEEDED",
                    "extraction_method": "rapidocr",
                    "ocr_invoked": True,
                    "requested_backend": ocr_res.backend_audit.requested_backend,
                    "actual_backend": ocr_res.backend_audit.actual_backend,
                    "cuda_verified": ocr_res.backend_audit.cuda_verified,
                    "page_count": ocr_res.page_count,
                    "total_chars": ocr_res.total_chars,
                    "vision_review_pages": triage_summary["vision_review_pages"],
                    "processing_time_ms": elapsed_ms,
                    "outputs": bundle.to_dict(),
                }
            )
        except ExtractionCancelledError as exc:
            return _handle_cancel(ctx, str(exc))
        except Exception as exc:
            any_failed = True
            jobs_summary.append(
                {
                    "job_id": job_id,
                    "source_file": source_path.name,
                    "status": "FAILED",
                    "error_kind": _classify_error_kind(exc),
                    "error": str(exc),
                }
            )

    ctx.write_result(
        {
            "schema_version": "1.0",
            "status": "PARTIAL_OR_FAILED" if any_failed else "SUCCEEDED",
            "worker_id": ctx.worker_id,
            "worker_version": __version__,
            "compute_backend": ctx.compute_backend,
            "network_policy": ctx.network_policy,
            "output_root": str(output_root),
            "jobs": jobs_summary,
        }
    )
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
