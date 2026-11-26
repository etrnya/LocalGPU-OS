"""M1-D D2 test suite: RapidOCR offline inference (Image + Scanned PDF, CPU & CUDA audit)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from localgpu_os import ComputeBackend, ErrorCode, LocalGpuClient, SensorStatus
from localgpu_os.domain.hardware import HardwareProfile, HardwareSnapshot

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKERS_DIR = _REPO_ROOT / "workers"
_OCR_PDF_VENV_PY = (
    _WORKERS_DIR / "ocr_pdf" / ".venv" / "Scripts" / "python.exe"
    if os.name == "nt"
    else _WORKERS_DIR / "ocr_pdf" / ".venv" / "bin" / "python"
)


def _create_rendered_text_image(target_path: Path, lines: list[str]) -> None:
    """Render clean high-contrast text into a PNG image (zero embedded text layer)."""
    script = (
        "import fitz, json, sys\n"
        "out_png = sys.argv[1]\n"
        "lines = json.loads(sys.argv[2])\n"
        "doc = fitz.open()\n"
        "page = doc.new_page(width=800, height=300)\n"
        "y = 80\n"
        "for line in lines:\n"
        "    page.insert_text((50, y), line, fontsize=28)\n"
        "    y += 55\n"
        "pix = page.get_pixmap(dpi=200)\n"
        "pix.save(out_png)\n"
        "doc.close()\n"
    )
    subprocess.run(
        [str(_OCR_PDF_VENV_PY), "-c", script, str(target_path), json.dumps(lines)],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )


def _create_image_only_scanned_pdf(target_pdf: Path, pages_lines: list[list[str]]) -> None:
    """Create a multi-page scanned PDF containing ONLY rasterized images (zero native text)."""
    script = (
        "import fitz, json, sys\n"
        "out_pdf = sys.argv[1]\n"
        "pages_lines = json.loads(sys.argv[2])\n"
        "out_doc = fitz.open()\n"
        "for lines in pages_lines:\n"
        "    tmp = fitz.open()\n"
        "    tp = tmp.new_page(width=800, height=300)\n"
        "    y = 80\n"
        "    for line in lines:\n"
        "        tp.insert_text((50, y), line, fontsize=28)\n"
        "        y += 55\n"
        "    pix = tp.get_pixmap(dpi=180)\n"
        "    png_bytes = pix.tobytes('png')\n"
        "    tmp.close()\n"
        "    p = out_doc.new_page(width=800, height=300)\n"
        "    p.insert_image(p.rect, stream=png_bytes)\n"
        "out_doc.save(out_pdf)\n"
        "out_doc.close()\n"
    )
    subprocess.run(
        [str(_OCR_PDF_VENV_PY), "-c", script, str(target_pdf), json.dumps(pages_lines)],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )


@unittest.skipUnless(
    _OCR_PDF_VENV_PY.exists(),
    "workers/ocr_pdf/.venv is not installed; skipping M1-D D2 RapidOCR tests.",
)
class TestM1DD2RapidOcrWorker(unittest.TestCase):
    """Validates D2.1 (CPU), D2.2 (CUDA if available), D2.3 (fallback), and D2.4 (schema)."""

    def test_d2_1_image_ocr_and_full_output_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "inputs"
            input_dir.mkdir()
            img_path = input_dir / "label.png"
            _create_rendered_text_image(
                img_path,
                ["LOCALGPU OS 2026", "BATCH 4060 VERIFIED"],
            )

            output_dir = tmp_path / "output"
            state_dir = tmp_path / "state"
            work_dir = tmp_path / "work"

            client = LocalGpuClient(workers_dir=_WORKERS_DIR, state_dir=state_dir)
            result = client.run_worker(
                "ocr_pdf",
                extra_args=["--output-dir", str(output_dir), "--job-id", "job_img"],
                input_manifest={"files": [{"path": "label.png", "role": "input"}]},
                allowed_root=input_dir,
                work_dir=work_dir,
                allow_cpu_fallback=True,
            )

            self.assertTrue(result.succeeded, f"stderr={result.stderr}, output={result.output_data}")
            job = result.output_data["jobs"][0]
            self.assertEqual(job["extraction_method"], "rapidocr")
            self.assertTrue(job["ocr_invoked"])
            self.assertEqual(job["page_count"], 1)

            job_dir = output_dir / "job_img"
            doc_data = json.loads((job_dir / "document.json").read_text(encoding="utf-8"))
            meta_data = json.loads((job_dir / "metadata.json").read_text(encoding="utf-8"))
            md_text = (job_dir / "document.md").read_text(encoding="utf-8")

            # Verify recognized text, bounding boxes, and confidence scores
            self.assertIn("2026", md_text)
            self.assertIn("4060", md_text)
            self.assertEqual(doc_data["extraction_method"], "rapidocr")
            self.assertTrue(doc_data["ocr_invoked"])
            blocks = doc_data["pages"][0]["blocks"]
            self.assertGreaterEqual(len(blocks), 1)
            for blk in blocks:
                self.assertIn("text", blk)
                self.assertIn("confidence", blk)
                self.assertGreater(blk["confidence"], 0.5)
                self.assertEqual(len(blk["bbox"]), 4)

            # Verify offline model manifest with SHA-256 and backend audit
            audit = meta_data["ocr_backend_audit"]
            self.assertGreaterEqual(len(audit["model_manifest"]), 3)
            for m in audit["model_manifest"]:
                self.assertEqual(len(m["sha256"]), 64)
            self.assertIn("det", audit["active_providers"])
            self.assertIn("rec", audit["active_providers"])
            self.assertEqual(meta_data["network_policy"], "RUNTIME_OFFLINE")
            self.assertEqual(len(client.vram_gate.read_registry_snapshot().active_leases()), 0)

    def test_d2_2_scanned_pdf_multipage_ocr_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "inputs"
            input_dir.mkdir()
            scanned_pdf = input_dir / "scanned_coa.pdf"
            _create_image_only_scanned_pdf(
                scanned_pdf,
                [
                    ["SCANNED PAGE ONE", "LOT NUMBER 9981"],
                    ["SCANNED PAGE TWO", "PURITY 99.5 PERCENT"],
                ],
            )

            output_dir = tmp_path / "output"
            state_dir = tmp_path / "state"
            work_dir = tmp_path / "work"

            client = LocalGpuClient(workers_dir=_WORKERS_DIR, state_dir=state_dir)
            result = client.run_worker(
                "ocr_pdf",
                extra_args=["--output-dir", str(output_dir), "--job-id", "job_scan_pdf"],
                input_manifest={"files": [{"path": "scanned_coa.pdf", "role": "input"}]},
                allowed_root=input_dir,
                work_dir=work_dir,
                allow_cpu_fallback=True,
            )

            self.assertTrue(result.succeeded, f"stderr={result.stderr}, output={result.output_data}")
            job = result.output_data["jobs"][0]
            self.assertEqual(job["extraction_method"], "rapidocr")
            self.assertTrue(job["ocr_invoked"])
            self.assertEqual(job["page_count"], 2)

            job_dir = output_dir / "job_scan_pdf"
            doc_data = json.loads((job_dir / "document.json").read_text(encoding="utf-8"))
            meta_data = json.loads((job_dir / "metadata.json").read_text(encoding="utf-8"))

            self.assertEqual(meta_data["classification"]["kind"], "SCANNED_PDF")
            self.assertEqual(len(doc_data["pages"]), 2)
            self.assertIn("9981", doc_data["pages"][0]["text"])
            self.assertIn("99.5", doc_data["pages"][1]["text"])
            self.assertEqual(len(client.vram_gate.read_registry_snapshot().active_leases()), 0)

    def test_d2_3_real_cuda_provider_verified_on_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "inputs"
            input_dir.mkdir()
            img_path = input_dir / "cuda_check.png"
            _create_rendered_text_image(img_path, ["RTX 4060 CUDA LIVE", "VERIFIED 2026"])

            output_dir = tmp_path / "output"
            state_dir = tmp_path / "state"
            work_dir = tmp_path / "work"

            client = LocalGpuClient(workers_dir=_WORKERS_DIR, state_dir=state_dir)
            doctor = client.doctor()
            if not doctor.decision.gpu_admission_allowed:
                self.skipTest("Local CUDA GPU admission unavailable on this machine.")

            result = client.run_worker(
                "ocr_pdf",
                extra_args=["--output-dir", str(output_dir), "--job-id", "job_cuda", "--strict-backend"],
                input_manifest={"files": [{"path": "cuda_check.png", "role": "input"}]},
                allowed_root=input_dir,
                work_dir=work_dir,
                allow_cpu_fallback=False,
            )
            self.assertTrue(result.succeeded, f"stderr={result.stderr}, output={result.output_data}")
            self.assertFalse(result.used_cpu_fallback)

            meta_data = json.loads((output_dir / "job_cuda" / "metadata.json").read_text(encoding="utf-8"))
            audit = meta_data["ocr_backend_audit"]
            self.assertEqual(audit["requested_backend"], "cuda")
            self.assertEqual(audit["actual_backend"], "cuda")
            self.assertTrue(audit["cuda_verified"])
            self.assertIn("CUDAExecutionProvider", audit["active_providers"]["det"])
            self.assertIn("CUDAExecutionProvider", audit["active_providers"]["rec"])
            self.assertEqual(len(client.vram_gate.read_registry_snapshot().active_leases()), 0)

    def test_d2_4_low_vram_denies_gpu_and_safely_falls_back_to_cpu_ocr(self) -> None:
        class _LowVramClock:
            def utc_now_iso(self) -> str:
                return "2026-11-26T12:00:00Z"

            def monotonic(self) -> float:
                return 100.0

        class _LowVramDetector:
            def probe(self, device_index: int = 0) -> tuple[HardwareProfile, HardwareSnapshot]:
                profile = HardwareProfile(
                    device_id=f"cuda:{device_index}",
                    vendor="NVIDIA",
                    device_name="NVIDIA GeForce RTX 4060 Laptop GPU",
                    architecture="Ada Lovelace",
                    compute_capability="8.9",
                    total_vram_mib=8188,
                    nvenc_supported=True,
                    temperature_support=True,
                    backend_support=(ComputeBackend.CUDA, ComputeBackend.CPU_INT8),
                )
                snapshot = HardwareSnapshot(
                    device_id=profile.device_id,
                    observed_at="2026-11-26T12:00:00Z",
                    observed_monotonic=100.0,
                    freshness_ttl_seconds=10.0,
                    available_vram_mib=500,  # Far below safety + 2048 MiB
                    external_vram_mib=7688,
                    temperature_c=55,
                    utilization_percent=90,
                    sensor_status=SensorStatus.OK,
                )
                return profile, snapshot

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "inputs"
            input_dir.mkdir()
            img_path = input_dir / "fallback_doc.png"
            _create_rendered_text_image(img_path, ["CPU FALLBACK OK", "NO GPU LEASE"])

            output_dir = tmp_path / "output"
            state_dir = tmp_path / "state"
            work_dir = tmp_path / "work"

            client = LocalGpuClient(
                workers_dir=_WORKERS_DIR,
                state_dir=state_dir,
                detector=_LowVramDetector(),
                clock=_LowVramClock(),
            )

            # Without allow_cpu_fallback -> REJECTED with VRAM_INSUFFICIENT
            rejected = client.run_worker(
                "ocr_pdf",
                extra_args=["--output-dir", str(output_dir), "--job-id", "job_rej"],
                input_manifest={"files": [{"path": "fallback_doc.png", "role": "input"}]},
                allowed_root=input_dir,
                work_dir=work_dir / "rej",
                allow_cpu_fallback=False,
            )
            self.assertEqual(rejected.status, "REJECTED")
            self.assertIn(ErrorCode.VRAM_INSUFFICIENT, rejected.error_codes)
            self.assertFalse((output_dir / "job_rej").exists())

            # With allow_cpu_fallback -> cleanly executes OCR on CPU backend
            degraded = client.run_worker(
                "ocr_pdf",
                extra_args=["--output-dir", str(output_dir), "--job-id", "job_cpu_fb"],
                input_manifest={"files": [{"path": "fallback_doc.png", "role": "input"}]},
                allowed_root=input_dir,
                work_dir=work_dir / "fb",
                allow_cpu_fallback=True,
            )
            self.assertTrue(degraded.succeeded, f"stderr={degraded.stderr}, output={degraded.output_data}")
            self.assertTrue(degraded.used_cpu_fallback)

            meta_data = json.loads((output_dir / "job_cpu_fb" / "metadata.json").read_text(encoding="utf-8"))
            audit = meta_data["ocr_backend_audit"]
            self.assertEqual(audit["actual_backend"], "cpu")
            self.assertFalse(audit["cuda_verified"])
            self.assertEqual(audit["active_providers"]["det"], ["CPUExecutionProvider"])
            self.assertEqual(audit["active_providers"]["rec"], ["CPUExecutionProvider"])
            md_text = (output_dir / "job_cpu_fb" / "document.md").read_text(encoding="utf-8")
            self.assertIn("FALLBACK", md_text)
            self.assertEqual(len(client.vram_gate.read_registry_snapshot().active_leases()), 0)


if __name__ == "__main__":
    unittest.main()
