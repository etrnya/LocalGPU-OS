"""Atomic staged output writer for M1-D ocr_pdf (prevents partial/corrupt outputs on cancel)."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any


@dataclass(frozen=True)
class JobOutputBundle:
    job_dir: Path
    markdown_path: Path
    json_path: Path
    metadata_path: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "job_dir": str(self.job_dir),
            "document_md": str(self.markdown_path),
            "document_json": str(self.json_path),
            "metadata_json": str(self.metadata_path),
        }


def _write_text_fsync(path: Path, content: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(content)
        fh.flush()
        os.fsync(fh.fileno())


def _verify_staged_bundle(staging_dir: Path) -> None:
    md_file = staging_dir / "document.md"
    doc_json = staging_dir / "document.json"
    meta_json = staging_dir / "metadata.json"

    for p in (md_file, doc_json, meta_json):
        if not p.exists() or not p.is_file() or p.stat().st_size == 0:
            raise OSError(f"Staged output validation failed for {p.name}.")

    # Verify JSON integrity before committing
    json.loads(doc_json.read_text(encoding="utf-8"))
    json.loads(meta_json.read_text(encoding="utf-8"))


def write_job_output_atomic(
    output_root: Path,
    job_id: str,
    markdown_text: str,
    document_payload: dict[str, Any],
    metadata_payload: dict[str, Any],
) -> JobOutputBundle:
    """Write document.md, document.json, metadata.json into staging dir, validate, then commit."""
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
        _verify_staged_bundle(staging_dir)

        if final_dir.exists():
            backup_dir = root / f".{job_id}.{os.getpid()}.old"
            if backup_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)
            os.replace(final_dir, backup_dir)
            try:
                os.replace(staging_dir, final_dir)
            except BaseException:
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
        )
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise
