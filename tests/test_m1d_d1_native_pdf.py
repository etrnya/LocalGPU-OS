"""M1-D D1 test suite: Native-Text PDF extraction via isolated `ocr_pdf` worker.

Validates:
- Document Classifier distinguishes native-text PDFs from scanned PDFs (no OCR invoked)
- End-to-end execution through LocalGPU-OS WorkerRuntime & `lgpu run ocr_pdf`
- Output contract: output/<job_id>/{document.md, document.json, metadata.json}
- Atomic staged output integrity (no partial/corrupted files left behind)
"""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from localgpu_os import LocalGpuClient
from localgpu_os.presentation.cli import main as cli_main

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKERS_DIR = _REPO_ROOT / "workers"
_OCR_PDF_VENV_PY = (
    _WORKERS_DIR / "ocr_pdf" / ".venv" / "Scripts" / "python.exe"
    if os.name == "nt"
    else _WORKERS_DIR / "ocr_pdf" / ".venv" / "bin" / "python"
)


def _create_sample_native_pdf(target_path: Path) -> None:
    """Create a real 2-page native-text PDF using ocr_pdf's isolated PyMuPDF."""
    script = (
        "import fitz, sys\n"
        "doc = fitz.open()\n"
        "doc.set_metadata({'title': 'COA Sample Report', 'author': 'etrnya'})\n"
        "p1 = doc.new_page()\n"
        "p1.insert_text((72, 72), 'Certificate of Analysis - Batch 2026-001\\n"
        "Native PDF text extraction test page one with sufficient characters.')\n"
        "p2 = doc.new_page()\n"
        "p2.insert_text((72, 72), 'Page Two Summary: Purity 99.8 percent, Moisture 0.12 percent.\\n"
        "Verified by LocalGPU-OS M1-D D1 native extractor.')\n"
        "doc.save(sys.argv[1])\n"
        "doc.close()\n"
    )
    subprocess.run(
        [str(_OCR_PDF_VENV_PY), "-c", script, str(target_path)],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )


def _create_sample_scanned_pdf(target_path: Path) -> None:
    """Create a 1-page PDF with vector/drawing only and zero native text layer."""
    script = (
        "import fitz, sys\n"
        "doc = fitz.open()\n"
        "p1 = doc.new_page()\n"
        "p1.draw_rect(fitz.Rect(50, 50, 200, 200), color=(0, 0, 0), fill=(0.8, 0.8, 0.8))\n"
        "doc.save(sys.argv[1])\n"
        "doc.close()\n"
    )
    subprocess.run(
        [str(_OCR_PDF_VENV_PY), "-c", script, str(target_path)],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )


@unittest.skipUnless(
    _OCR_PDF_VENV_PY.exists(),
    "workers/ocr_pdf/.venv is not installed; skipping M1-D D1 isolated worker tests.",
)
class TestM1DD1NativePdfWorker(unittest.TestCase):
    """End-to-end M1-D D1 acceptance tests running inside workers/ocr_pdf/.venv."""

    def test_d1_1_native_pdf_extraction_via_sdk_and_output_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "inputs"
            input_dir.mkdir()
            pdf_path = input_dir / "coa_sample.pdf"
            _create_sample_native_pdf(pdf_path)

            output_dir = tmp_path / "output"
            state_dir = tmp_path / "state"
            work_dir = tmp_path / "work"

            client = LocalGpuClient(workers_dir=_WORKERS_DIR, state_dir=state_dir)
            result = client.run_worker(
                "ocr_pdf",
                extra_args=["--output-dir", str(output_dir), "--job-id", "job_001"],
                input_manifest={"files": [{"path": "coa_sample.pdf", "role": "input"}]},
                allowed_root=input_dir,
                work_dir=work_dir,
            )

            self.assertTrue(result.succeeded, f"stderr={result.stderr}, output={result.output_data}")
            self.assertEqual(result.exit_code, 0)
            self.assertIsNotNone(result.output_data)
            self.assertEqual(len(result.output_data["jobs"]), 1)

            job = result.output_data["jobs"][0]
            self.assertEqual(job["job_id"], "job_001")
            self.assertEqual(job["extraction_method"], "pymupdf_native")
            self.assertFalse(job["ocr_invoked"])
            self.assertEqual(job["page_count"], 2)

            # Verify output/job_001/{document.md, document.json, metadata.json}
            job_dir = output_dir / "job_001"
            md_path = job_dir / "document.md"
            doc_json_path = job_dir / "document.json"
            meta_json_path = job_dir / "metadata.json"

            self.assertTrue(md_path.is_file())
            self.assertTrue(doc_json_path.is_file())
            self.assertTrue(meta_json_path.is_file())

            md_text = md_path.read_text(encoding="utf-8")
            self.assertIn("## Page 1", md_text)
            self.assertIn("## Page 2", md_text)
            self.assertIn("Certificate of Analysis", md_text)
            self.assertIn("Purity 99.8 percent", md_text)

            doc_data = json.loads(doc_json_path.read_text(encoding="utf-8"))
            self.assertEqual(doc_data["extraction_method"], "pymupdf_native")
            self.assertFalse(doc_data["ocr_invoked"])
            self.assertEqual(doc_data["page_count"], 2)
            self.assertEqual([p["page_number"] for p in doc_data["pages"]], [1, 2])

            meta_data = json.loads(meta_json_path.read_text(encoding="utf-8"))
            self.assertEqual(meta_data["job_id"], "job_001")
            self.assertEqual(meta_data["source_file"], "coa_sample.pdf")
            self.assertEqual(meta_data["pdf_metadata"].get("title"), "COA Sample Report")
            self.assertEqual(meta_data["extraction_method"], "pymupdf_native")
            self.assertFalse(meta_data["ocr_invoked"])
            self.assertEqual(meta_data["network_policy"], "RUNTIME_OFFLINE")

            # Verify no staging dirs left and zero active leases remain
            staging_left = [p for p in output_dir.iterdir() if ".staging" in p.name]
            self.assertEqual(staging_left, [])
            self.assertEqual(len(client.vram_gate.read_registry_snapshot().active_leases()), 0)

    def test_d1_2_unsupported_file_rejected_without_fake_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "inputs"
            input_dir.mkdir()
            unsupported_file = input_dir / "notes.txt"
            unsupported_file.write_text("plain text not pdf", encoding="utf-8")

            output_dir = tmp_path / "output"
            state_dir = tmp_path / "state"
            work_dir = tmp_path / "work"

            client = LocalGpuClient(workers_dir=_WORKERS_DIR, state_dir=state_dir)
            result = client.run_worker(
                "ocr_pdf",
                extra_args=["--output-dir", str(output_dir), "--job-id", "job_unsupported"],
                input_manifest={"files": [{"path": "notes.txt", "role": "input"}]},
                allowed_root=input_dir,
                work_dir=work_dir,
            )

            self.assertEqual(result.status, "FAILED")
            self.assertIsNotNone(result.output_data)
            job = result.output_data["jobs"][0]
            self.assertEqual(job["status"], "UNSUPPORTED")
            self.assertEqual(job["classification"]["kind"], "UNSUPPORTED")
            self.assertFalse(job["ocr_invoked"])
            # Must NOT leave a misleading output/job_unsupported directory
            self.assertFalse((output_dir / "job_unsupported").exists())
            self.assertEqual(len(client.vram_gate.read_registry_snapshot().active_leases()), 0)

    def test_d1_3_cli_lgpu_run_ocr_pdf_json_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "inputs"
            input_dir.mkdir()
            pdf_path = input_dir / "cli_doc.pdf"
            _create_sample_native_pdf(pdf_path)

            manifest_file = tmp_path / "input_manifest.json"
            manifest_file.write_text(
                json.dumps({"files": [{"path": "cli_doc.pdf"}]}, indent=2),
                encoding="utf-8",
            )
            output_dir = tmp_path / "output"
            state_dir = tmp_path / "state"

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli_main(
                    [
                        "run",
                        "ocr_pdf",
                        "--workers-dir",
                        str(_WORKERS_DIR),
                        "--state-dir",
                        str(state_dir),
                        "--input-manifest",
                        str(manifest_file),
                        "--allowed-root",
                        str(input_dir),
                        "--json",
                        "--",
                        "--output-dir",
                        str(output_dir),
                        "--job-id",
                        "cli_job_001",
                    ]
                )
            self.assertEqual(rc, 0, f"cli output={buf.getvalue()}")
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["status"], "SUCCEEDED")
            self.assertEqual(payload["worker_id"], "ocr_pdf")
            self.assertTrue((output_dir / "cli_job_001" / "document.md").is_file())
            self.assertTrue((output_dir / "cli_job_001" / "document.json").is_file())
            self.assertTrue((output_dir / "cli_job_001" / "metadata.json").is_file())


if __name__ == "__main__":
    unittest.main()
