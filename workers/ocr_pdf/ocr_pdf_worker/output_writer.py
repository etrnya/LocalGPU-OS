"""Transactional staged output writer for M1-D / M2-A ocr_pdf (preserves prior valid output)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Sequence

from PIL import Image
import pymupdf as fitz


@dataclass(frozen=True)
class JobOutputBundle:
    job_dir: Path
    markdown_path: Path
    json_path: Path
    metadata_path: Path
    review_pages_dir: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "job_dir": str(self.job_dir),
            "document_md": str(self.markdown_path),
            "document_json": str(self.json_path),
            "metadata_json": str(self.metadata_path),
        }
        if self.review_pages_dir is not None:
            data["review_pages_dir"] = str(self.review_pages_dir)
            data["review_manifest_json"] = str(self.review_pages_dir / "review_manifest.json")
        return data


def _write_text_fsync(path: Path, content: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(content)
        fh.flush()
        os.fsync(fh.fileno())


def _write_bytes_fsync(path: Path, data: bytes) -> None:
    with open(path, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def export_review_pages_into_staging(
    staging_dir: Path,
    source_path: Path,
    source_kind: str,
    review_pages: Sequence[tuple[int, Sequence[str]]],
    review_dpi: int = 150,
    max_image_bytes: int = 15 * 1024 * 1024,
) -> Path | None:
    """Stream-render ONLY needs_vision_review pages into staging_dir/review_pages/.

    Renders page-by-page from the original PDF/image and releases memory immediately.
    Enforces max_image_bytes per PNG and computes final-file SHA-256 into review_manifest.json.
    """
    if not review_pages:
        return None

    review_dir = staging_dir / "review_pages"
    review_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict[str, Any]] = []

    if source_kind == "IMAGE":
        page_num, reasons = review_pages[0]
        filename = f"page_{page_num:03d}.png"
        out_png = review_dir / filename
        with Image.open(source_path) as img:
            rgb = img.convert("RGB")
            rgb.save(out_png, format="PNG")
            width_px, height_px = rgb.size
        size_bytes = out_png.stat().st_size
        if size_bytes <= 0 or size_bytes > max_image_bytes:
            raise ValueError(
                f"Review image {filename} size ({size_bytes} bytes) exceeds limit ({max_image_bytes} bytes)."
            )
        manifest_entries.append(
            {
                "page_number": page_num,
                "filename": filename,
                "reasons": list(reasons),
                "dpi": None,
                "width_px": width_px,
                "height_px": height_px,
                "size_bytes": size_bytes,
                "sha256": _sha256_file(out_png),
            }
        )
    else:
        zoom = review_dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        with fitz.open(source_path) as doc:
            for page_num, reasons in review_pages:
                if page_num < 1 or page_num > len(doc):
                    raise ValueError(f"Review page {page_num} out of bounds for {source_path.name}.")
                page = doc[page_num - 1]
                pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
                try:
                    png_bytes = pix.tobytes("png")
                    width_px, height_px = pix.width, pix.height
                finally:
                    del pix
                if len(png_bytes) == 0 or len(png_bytes) > max_image_bytes:
                    raise ValueError(
                        f"Review image page_{page_num:03d}.png size ({len(png_bytes)} bytes) "
                        f"exceeds limit ({max_image_bytes} bytes)."
                    )
                filename = f"page_{page_num:03d}.png"
                out_png = review_dir / filename
                _write_bytes_fsync(out_png, png_bytes)
                del png_bytes
                manifest_entries.append(
                    {
                        "page_number": page_num,
                        "filename": filename,
                        "reasons": list(reasons),
                        "dpi": review_dpi,
                        "width_px": width_px,
                        "height_px": height_px,
                        "size_bytes": out_png.stat().st_size,
                        "sha256": _sha256_file(out_png),
                    }
                )

    manifest_payload = {
        "schema_version": "1.0",
        "source_file": source_path.name,
        "source_kind": source_kind,
        "review_dpi": review_dpi if source_kind != "IMAGE" else None,
        "pages": manifest_entries,
    }
    _write_text_fsync(
        review_dir / "review_manifest.json",
        json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n",
    )
    return review_dir


def _verify_staged_bundle(staging_dir: Path, expect_review_pages: bool = False) -> None:
    md_file = staging_dir / "document.md"
    doc_json = staging_dir / "document.json"
    meta_json = staging_dir / "metadata.json"

    for p in (md_file, doc_json, meta_json):
        if not p.exists() or not p.is_file() or p.stat().st_size == 0:
            raise OSError(f"Staged output validation failed for {p.name}.")

    json.loads(doc_json.read_text(encoding="utf-8"))
    json.loads(meta_json.read_text(encoding="utf-8"))

    if expect_review_pages:
        review_dir = staging_dir / "review_pages"
        manifest_path = review_dir / "review_manifest.json"
        if not manifest_path.is_file():
            raise OSError("Staged review_pages/review_manifest.json is missing.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest.get("pages", []):
            png_path = review_dir / entry["filename"]
            if not png_path.is_file() or png_path.stat().st_size == 0:
                raise OSError(f"Staged review image missing or empty: {entry['filename']}")
            if _sha256_file(png_path) != entry["sha256"]:
                raise OSError(f"Staged review image SHA-256 mismatch: {entry['filename']}")


def write_job_output_atomic(
    output_root: Path,
    job_id: str,
    markdown_text: str,
    document_payload: dict[str, Any],
    metadata_payload: dict[str, Any],
    review_export_fn: Callable[[Path], Path | None] | None = None,
) -> JobOutputBundle:
    """Transactional staged commit preserving prior valid output on any failure (Windows-safe)."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    staging_dir = root / f".{job_id}.{os.getpid()}.staging"
    final_dir = root / job_id

    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        _write_text_fsync(staging_dir / "document.md", markdown_text)
        _write_text_fsync(
            staging_dir / "document.json",
            json.dumps(document_payload, indent=2, ensure_ascii=False) + "\n",
        )
        _write_text_fsync(
            staging_dir / "metadata.json",
            json.dumps(metadata_payload, indent=2, ensure_ascii=False) + "\n",
        )

        has_review_pages = False
        if review_export_fn is not None:
            exported = review_export_fn(staging_dir)
            has_review_pages = exported is not None

        _verify_staged_bundle(staging_dir, expect_review_pages=has_review_pages)

        # Transactional directory swap: preserve existing valid output in .old until commit succeeds
        if final_dir.exists():
            backup_dir = root / f".{job_id}.{os.getpid()}.old"
            if backup_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)
            os.replace(final_dir, backup_dir)
            try:
                os.replace(staging_dir, final_dir)
            except BaseException:
                # Roll back to prior valid output
                if not final_dir.exists() and backup_dir.exists():
                    os.replace(backup_dir, final_dir)
                raise
            shutil.rmtree(backup_dir, ignore_errors=True)
        else:
            os.replace(staging_dir, final_dir)

        return JobOutputBundle(
            job_dir=final_dir,
            markdown_path=final_dir / "document.md",
            json_path=final_dir / "document.json",
            metadata_path=final_dir / "metadata.json",
            review_pages_dir=(final_dir / "review_pages") if has_review_pages else None,
        )
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise
