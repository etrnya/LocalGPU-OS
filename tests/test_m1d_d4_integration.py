"""M1-D D4 Integration & Delivery Acceptance Test Suite.

Validates delivery readiness across all M1-D acceptance dimensions:
- D4.1: End-to-end 3-path integration (Native PDF + Scanned PDF + Image in one batch)
- D4.2: Pre-load model SHA-256 integrity gate (valid lock passes, tampered digest fails closed)
- D4.3: 5-round multi-run stability & VRAM/Lease reclamation check
- D4.4: Offline reproducibility artifacts (requirements.txt, model_checksums.json, zero local paths)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

from localgpu_os import LocalGpuClient

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKERS_DIR = _REPO_ROOT / "workers"
_OCR_PDF_DIR = _WORKERS_DIR / "ocr_pdf"
_OCR_PDF_VENV_PY = (
    _OCR_PDF_DIR / ".venv" / "Scripts" / "python.exe"
    if os.name == "nt"
    else _OCR_PDF_DIR / ".venv" / "bin" / "python"
)


def _run_in_worker_venv(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(_OCR_PDF_VENV_PY), "-c", script, *args],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )


def _make_native_pdf(target: Path, text: str) -> None:
    _run_in_worker_venv(
        "import pymupdf as fitz, sys\n"
        "doc = fitz.open()\n"
        "p = doc.new_page()\n"
        "p.insert_text((72, 72), sys.argv[2])\n"
        "doc.save(sys.argv[1])\n"
        "doc.close()\n",
        str(target),
        text,
    )


def _make_scanned_pdf(target: Path, text: str) -> None:
    _run_in_worker_venv(
        "import pymupdf as fitz, sys\n"
        "tmp = fitz.open()\n"
        "tp = tmp.new_page(width=800, height=300)\n"
        "tp.insert_text((50, 110), sys.argv[2], fontsize=28)\n"
        "png = tp.get_pixmap(dpi=180).tobytes('png')\n"
        "tmp.close()\n"
        "out = fitz.open()\n"
        "p = out.new_page(width=800, height=300)\n"
        "p.insert_image(p.rect, stream=png)\n"
        "out.save(sys.argv[1])\n"
        "out.close()\n",
        str(target),
        text,
    )


def _make_image(target: Path, text: str) -> None:
    _run_in_worker_venv(
        "import pymupdf as fitz, sys\n"
        "doc = fitz.open()\n"
        "p = doc.new_page(width=800, height=300)\n"
        "p.insert_text((50, 110), sys.argv[2], fontsize=28)\n"
        "p.get_pixmap(dpi=180).save(sys.argv[1])\n"
        "doc.close()\n",
        str(target),
        text,
    )


def _query_nvidia_vram_used_mib() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5.0,
        ).strip()
        return int(out.splitlines()[0].strip())
    except Exception:
        return None


@unittest.skipUnless(
    _OCR_PDF_VENV_PY.exists(),
    "workers/ocr_pdf/.venv is not installed; skipping M1-D D4 integration tests.",
)
class TestM1DD4IntegrationDelivery(unittest.TestCase):
    """M1-D D4 delivery acceptance test suite."""

    # ------------------------------------------------------------------
    # D4.1: All 3 pipelines (Native PDF + Scanned PDF + Image) in one batch
    # ------------------------------------------------------------------
    def test_d4_1_all_three_pipelines_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "inputs"
            input_dir.mkdir()

            _make_native_pdf(
                input_dir / "01_native.pdf",
                "D4 Native PDF: Direct PyMuPDF extraction without invoking OCR.",
            )
            _make_scanned_pdf(input_dir / "02_scanned.pdf", "D4 SCANNED PDF 2026")
            _make_image(input_dir / "03_photo.png", "D4 IMAGE LABEL 4060")

            output_dir = tmp_path / "output"
            client = LocalGpuClient(workers_dir=_WORKERS_DIR, state_dir=tmp_path / "state")
            res = client.run_worker(
                "ocr_pdf",
                extra_args=["--output-dir", str(output_dir), "--job-id", "d4_all"],
                input_manifest={
                    "files": [
                        {"path": "01_native.pdf"},
                        {"path": "02_scanned.pdf"},
                        {"path": "03_photo.png"},
                    ]
                },
                allowed_root=input_dir,
                work_dir=tmp_path / "work",
                allow_cpu_fallback=True,
            )

            self.assertTrue(res.succeeded, f"stderr={res.stderr}, output={res.output_data}")
            jobs = res.output_data["jobs"]
            self.assertEqual(len(jobs), 3)

            # 1. Native PDF bypassed OCR
            self.assertEqual(jobs[0]["extraction_method"], "pymupdf_native")
            self.assertFalse(jobs[0]["ocr_invoked"])
            # 2. Scanned PDF used RapidOCR
            self.assertEqual(jobs[1]["extraction_method"], "rapidocr")
            self.assertTrue(jobs[1]["ocr_invoked"])
            # 3. Image used RapidOCR
            self.assertEqual(jobs[2]["extraction_method"], "rapidocr")
            self.assertTrue(jobs[2]["ocr_invoked"])

            for jid in ("d4_all_001", "d4_all_002", "d4_all_003"):
                jdir = output_dir / jid
                self.assertTrue((jdir / "document.md").is_file())
                self.assertTrue((jdir / "document.json").is_file())
                self.assertTrue((jdir / "metadata.json").is_file())

            # Verify checksums_verified recorded in OCR metadata
            meta_scan = json.loads((output_dir / "d4_all_002" / "metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(meta_scan["ocr_backend_audit"]["checksums_verified"])
            self.assertEqual(meta_scan["network_policy"], "RUNTIME_OFFLINE")
            self.assertEqual(len(client.vram_gate.read_registry_snapshot().active_leases()), 0)

    # ------------------------------------------------------------------
    # D4.2: Pre-load SHA-256 verification fails closed on tampered checksum
    # ------------------------------------------------------------------
    def test_d4_2_preload_model_sha256_verification_fails_closed_on_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tampered_lock = Path(tmp) / "bad_checksums.json"
            good_data = json.loads((_OCR_PDF_DIR / "model_checksums.json").read_text(encoding="utf-8"))
            good_data["models"][0]["sha256"] = "0" * 64
            tampered_lock.write_text(json.dumps(good_data, indent=2), encoding="utf-8")

            # Run inside worker venv: valid lock passes, tampered lock raises RuntimeError BEFORE load
            script = (
                "import sys\n"
                "from pathlib import Path\n"
                "sys.path.insert(0, sys.argv[1])\n"
                "from ocr_pdf_worker.ocr_engine import verify_bundled_models\n"
                "valid = verify_bundled_models()\n"
                "assert len(valid) >= 3\n"
                "try:\n"
                "    verify_bundled_models(Path(sys.argv[2]))\n"
                "except RuntimeError as exc:\n"
                "    assert 'SHA-256 integrity verification failed' in str(exc)\n"
                "    print('FAIL_CLOSED_OK')\n"
                "else:\n"
                "    raise AssertionError('Tampered checksum was not rejected!')\n"
            )
            proc = _run_in_worker_venv(script, str(_OCR_PDF_DIR), str(tampered_lock))
            self.assertIn("FAIL_CLOSED_OK", proc.stdout)

    # ------------------------------------------------------------------
    # D4.3: 5-round stability & VRAM/Lease reclamation observation
    # ------------------------------------------------------------------
    def test_d4_3_five_round_stability_and_resource_reclamation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "inputs"
            input_dir.mkdir()
            _make_native_pdf(input_dir / "native.pdf", "Round stability native PDF document.")
            _make_image(input_dir / "img.png", "STABILITY ROUND 2026")

            output_dir = tmp_path / "output"
            client = LocalGpuClient(workers_dir=_WORKERS_DIR, state_dir=tmp_path / "state")
            vram_start = _query_nvidia_vram_used_mib()

            for i in range(1, 6):
                target_file = "native.pdf" if i % 2 == 1 else "img.png"
                r = client.run_worker(
                    "ocr_pdf",
                    extra_args=["--output-dir", str(output_dir), "--job-id", f"round_{i}"],
                    input_manifest={"files": [{"path": target_file}]},
                    allowed_root=input_dir,
                    work_dir=tmp_path / f"work_{i}",
                    allow_cpu_fallback=True,
                )
                self.assertTrue(r.succeeded, f"round {i} failed: stderr={r.stderr}")
                self.assertTrue((output_dir / f"round_{i}" / "document.md").is_file())
                self.assertEqual(len(client.vram_gate.read_registry_snapshot().active_leases()), 0)

            if vram_start is not None:
                time.sleep(0.5)
                vram_end = _query_nvidia_vram_used_mib()
                self.assertIsNotNone(vram_end)
                self.assertLess(
                    abs(vram_end - vram_start),
                    256,
                    f"VRAM drift after 5 rounds: start={vram_start} MiB, end={vram_end} MiB",
                )

    # ------------------------------------------------------------------
    # D4.4: Offline reproducibility artifacts check (zero local paths)
    # ------------------------------------------------------------------
    def test_d4_4_offline_reproducibility_artifacts_clean(self) -> None:
        req_file = _OCR_PDF_DIR / "requirements.txt"
        chk_file = _OCR_PDF_DIR / "model_checksums.json"
        readme_file = _OCR_PDF_DIR / "README.md"

        self.assertTrue(req_file.is_file())
        self.assertTrue(chk_file.is_file())
        self.assertTrue(readme_file.is_file())

        req_text = req_file.read_text(encoding="utf-8")
        # Must pin versions cleanly with zero local file:// or absolute drive paths
        self.assertNotIn("file://", req_text)
        self.assertNotIn(":\\", req_text)
        self.assertIn("rapidocr-onnxruntime==", req_text)
        self.assertIn("pymupdf==", req_text)

        chk_data = json.loads(chk_file.read_text(encoding="utf-8"))
        self.assertEqual(chk_data["schema_version"], "1.0")
        self.assertGreaterEqual(len(chk_data["models"]), 3)
        for m in chk_data["models"]:
            self.assertEqual(len(m["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
