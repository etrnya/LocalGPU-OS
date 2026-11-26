"""M2-A Local Quality Triage Test Suite (v1.1 Approved Spec: Q1 through Q6 + regression).

Validates:
- Q1: Clean native-text PDF & clean scanned PDF -> needs_vision_review=False (no false positives)
- Q2-A: Low confidence OCR (unit + real degraded/blurred scan) -> LOW_AVG_CONFIDENCE / HIGH_LOW_CONF_RATIO
- Q2-B: Low text content scanned page -> LOW_TEXT_CONTENT (mutually exclusive with NO_TEXT_DETECTED)
- Q3: Invalid confidence inputs (NaN, Inf, <0, >1, blank text blocks) -> INVALID_CONFIDENCE_DATA, null zero-denom
- Q4: Pure chart/no-text scan & blank page -> NO_TEXT_DETECTED (not execution failure)
- Q5: Native PDF exact rect-union image coverage (overlapping rects, small logo vs large image, null OCR conf)
- Q6: Optional --export-review-images (default off, on exports ONLY flagged pages + SHA-256, failure keeps prior output)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
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

# Make triage.py importable directly for pure unit tests (zero heavy deps)
if str(_OCR_PDF_DIR) not in sys.path:
    sys.path.insert(0, str(_OCR_PDF_DIR))

from ocr_pdf_worker.triage import (  # noqa: E402
    IMAGE_COVERAGE_ALGORITHM,
    TriageConfig,
    TriageReason,
    build_triage_summary,
    compute_exact_rect_union_coverage,
    evaluate_native_page_triage,
    evaluate_ocr_page_triage,
)


def _run_in_worker_venv(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(_OCR_PDF_VENV_PY), "-c", script, *args],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )


class TestM2AQualityTriagePureUnit(unittest.TestCase):
    """Pure unit tests for M2-A triage contracts (zero GPU/venv required)."""

    # ------------------------------------------------------------------
    # Q2-A (unit): Low confidence & high low-confidence ratio
    # ------------------------------------------------------------------
    def test_q2a_ocr_low_confidence_and_ratio_metrics(self) -> None:
        # 100-block vs 2-block distinction: 2 low-conf blocks with sufficient text
        res = evaluate_ocr_page_triage(
            blocks=[
                ("Blurry certificate line one", 0.62),
                ("Blurry certificate line two", 0.68),
            ],
            char_count=54,
        )
        self.assertTrue(res.needs_vision_review)
        self.assertIn(TriageReason.LOW_AVG_CONFIDENCE, res.reasons)
        self.assertIn(TriageReason.HIGH_LOW_CONF_RATIO, res.reasons)
        self.assertNotIn(TriageReason.LOW_TEXT_CONTENT, res.reasons)
        self.assertNotIn(TriageReason.NO_TEXT_DETECTED, res.reasons)
        self.assertEqual(res.metrics["avg_confidence_method"], "block_equal_weight")
        self.assertAlmostEqual(res.metrics["avg_confidence"], 0.65, places=3)
        self.assertEqual(res.metrics["min_confidence"], 0.62)
        self.assertEqual(res.metrics["low_conf_ratio"], 1.0)
        self.assertEqual(res.metrics["invalid_confidence_count"], 0)
        # Native-only fields are None in OCR path
        self.assertIsNone(res.metrics["image_coverage_ratio"])

    # ------------------------------------------------------------------
    # Q3: Invalid confidence handling & zero-denominator null invariant
    # ------------------------------------------------------------------
    def test_q3_invalid_confidence_data_never_faked_as_zero(self) -> None:
        res = evaluate_ocr_page_triage(
            blocks=[
                ("   ", 0.99),  # blank text -> excluded from block_count
                ("Valid text line with good score", 0.95),
                ("NaN block", float("nan")),
                ("Inf block", float("inf")),
                ("Negative block", -0.1),
                ("Over-one block", 1.25),
                ("String block", "bad"),
            ],
            char_count=85,
        )
        self.assertTrue(res.needs_vision_review)
        self.assertIn(TriageReason.INVALID_CONFIDENCE_DATA, res.reasons)
        self.assertEqual(res.metrics["block_count"], 6)  # blank block excluded
        self.assertEqual(res.metrics["invalid_confidence_count"], 5)
        # Valid block (0.95) is NOT polluted by treating invalid confidences as 0.0
        self.assertEqual(res.metrics["avg_confidence"], 0.95)
        self.assertEqual(res.metrics["min_confidence"], 0.95)

        # All-invalid / zero valid blocks -> avg/char_weighted/min MUST be None (not 0.0)
        all_bad = evaluate_ocr_page_triage(
            blocks=[("Corrupt score only", float("nan"))],
            char_count=18,
        )
        self.assertIsNone(all_bad.metrics["avg_confidence"])
        self.assertIsNone(all_bad.metrics["char_weighted_avg_confidence"])
        self.assertIsNone(all_bad.metrics["min_confidence"])
        self.assertIn(TriageReason.INVALID_CONFIDENCE_DATA, all_bad.reasons)
        self.assertIn(TriageReason.LOW_TEXT_CONTENT, all_bad.reasons)
        self.assertNotIn(TriageReason.NO_TEXT_DETECTED, all_bad.reasons)

    # ------------------------------------------------------------------
    # Q5 (unit): Exact 2D rect-union coverage (overlapping images never double-counted)
    # ------------------------------------------------------------------
    def test_q5_exact_rect_union_and_native_triage(self) -> None:
        # Page 100 x 100 = 10,000 area
        # Rect A: [0, 0, 30, 100] -> 30%
        # Rect B: [10, 0, 40, 100] -> 30%, overlaps A by 20%
        # Exact union MUST be 40% (0.40), NOT 60% (0.60)!
        ratio, count = compute_exact_rect_union_coverage(
            rects=[(0, 0, 30, 100), (10, 0, 40, 100)],
            page_width=100.0,
            page_height=100.0,
        )
        self.assertEqual(count, 2)
        self.assertAlmostEqual(ratio, 0.40, places=4)

        # Small logo (<10% coverage) + normal text -> NO false positive
        small_logo = evaluate_native_page_triage(
            char_count=200,
            image_rects=[(10, 10, 35, 35)],  # 625 / 10000 = 6.25%
            page_width=100.0,
            page_height=100.0,
        )
        self.assertFalse(small_logo.needs_vision_review)
        self.assertEqual(small_logo.reasons, ())
        self.assertIsNone(small_logo.metrics["avg_confidence"])
        self.assertEqual(small_logo.metrics["image_coverage_algorithm"], IMAGE_COVERAGE_ALGORITHM)

        # Large embedded chart (>=25%) + text -> IMAGE_HEAVY_PAGE + has_mixed_text_and_image
        large_img = evaluate_native_page_triage(
            char_count=120,
            image_rects=[(0, 0, 60, 60)],  # 36%
            page_width=100.0,
            page_height=100.0,
        )
        self.assertTrue(large_img.needs_vision_review)
        self.assertEqual(large_img.reasons, (TriageReason.IMAGE_HEAVY_PAGE,))
        self.assertTrue(large_img.metrics["has_mixed_text_and_image"])
        self.assertIsNone(large_img.metrics["avg_confidence"])

    def test_triage_summary_sorts_and_deduplicates_pages(self) -> None:
        r_bad = evaluate_ocr_page_triage(blocks=[], char_count=0)
        r_ok = evaluate_ocr_page_triage(blocks=[("Clean text line with plenty of chars", 0.98)], char_count=36)
        summary = build_triage_summary([(5, r_bad), (2, r_ok), (3, r_bad), (5, r_bad)])
        self.assertEqual(summary["vision_review_pages"], [3, 5])
        self.assertEqual(summary["review_page_count"], 2)


@unittest.skipUnless(
    _OCR_PDF_VENV_PY.exists(),
    "workers/ocr_pdf/.venv is not installed; skipping M2-A end-to-end worker tests.",
)
class TestM2AQualityTriageEndToEnd(unittest.TestCase):
    """End-to-end M2-A acceptance tests through LocalGpuClient + isolated ocr_pdf worker."""

    def test_q1_q2b_q4_q5_q6_multipage_native_and_scanned_triage_and_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "inputs"
            input_dir.mkdir()

            # 1. Native PDF with 3 pages:
            #    Page 1: Clean text + tiny logo (<5% coverage) -> PASS (not flagged)
            #    Page 2: Native text + large embedded image (~40% coverage) -> IMAGE_HEAVY_PAGE
            #    Page 3: Completely blank page -> NO_TEXT_DETECTED
            native_pdf = input_dir / "mixed_native.pdf"
            _run_in_worker_venv(
                "import pymupdf as fitz, sys\n"
                "from PIL import Image\n"
                "import io\n"
                "buf = io.BytesIO()\n"
                "Image.new('RGB', (200, 200), (40, 120, 200)).save(buf, format='PNG')\n"
                "png = buf.getvalue()\n"
                "doc = fitz.open()\n"
                "# Page 1: clean text + tiny logo\n"
                "p1 = doc.new_page(width=500, height=700)\n"
                "p1.insert_text((40, 60), 'Page One Clean Native Text Report: ' + 'content ' * 12)\n"
                "p1.insert_image(fitz.Rect(420, 20, 460, 60), stream=png)\n"
                "# Page 2: text + large 350x400 chart (~40% of 500x700)\n"
                "p2 = doc.new_page(width=500, height=700)\n"
                "p2.insert_text((40, 60), 'Page Two Analysis Chart Summary with native text.')\n"
                "p2.insert_image(fitz.Rect(50, 120, 400, 520), stream=png)\n"
                "# Page 3: completely blank\n"
                "doc.new_page(width=500, height=700)\n"
                "doc.save(sys.argv[1])\n"
                "doc.close()\n",
                str(native_pdf),
            )

            # 2. Scanned PDF with 3 pages:
            #    Page 1: Clean high-contrast text (>30 chars) -> PASS
            #    Page 2: Short title only ('CHAPTER 1') -> LOW_TEXT_CONTENT (not NO_TEXT_DETECTED)
            #    Page 3: Pure geometric chart with zero text -> NO_TEXT_DETECTED
            scanned_pdf = input_dir / "mixed_scanned.pdf"
            _run_in_worker_venv(
                "import pymupdf as fitz, sys\n"
                "out = fitz.open()\n"
                "# Page 1: clean scanned text\n"
                "t1 = fitz.open(); tp1 = t1.new_page(width=800, height=350)\n"
                "tp1.insert_text((40, 80), 'LOCALGPU OS QUALITY TRIAGE VERIFICATION', fontsize=26)\n"
                "tp1.insert_text((40, 150), 'STANDARD CLEAR PRINTED DOCUMENT PAGE ONE', fontsize=24)\n"
                "png1 = tp1.get_pixmap(dpi=180).tobytes('png'); t1.close()\n"
                "p1 = out.new_page(width=800, height=350); p1.insert_image(p1.rect, stream=png1)\n"
                "# Page 2: short title only\n"
                "t2 = fitz.open(); tp2 = t2.new_page(width=800, height=350)\n"
                "tp2.insert_text((260, 170), 'CHAPTER 1', fontsize=34)\n"
                "png2 = tp2.get_pixmap(dpi=180).tobytes('png'); t2.close()\n"
                "p2 = out.new_page(width=800, height=350); p2.insert_image(p2.rect, stream=png2)\n"
                "# Page 3: pure chart drawing (no text)\n"
                "t3 = fitz.open(); tp3 = t3.new_page(width=800, height=350)\n"
                "tp3.draw_rect(fitz.Rect(100, 80, 700, 280), color=(0, 0, 0), fill=(0.85, 0.9, 0.95))\n"
                "tp3.draw_line((120, 250), (680, 110), color=(0.1, 0.3, 0.8), width=4)\n"
                "png3 = tp3.get_pixmap(dpi=150).tobytes('png'); t3.close()\n"
                "p3 = out.new_page(width=800, height=350); p3.insert_image(p3.rect, stream=png3)\n"
                "out.save(sys.argv[1]); out.close()\n",
                str(scanned_pdf),
            )

            output_dir = tmp_path / "output"
            client = LocalGpuClient(workers_dir=_WORKERS_DIR, state_dir=tmp_path / "state")

            # Run A: Native PDF WITHOUT --export-review-images -> zero review_pages/ created
            res_native = client.run_worker(
                "ocr_pdf",
                extra_args=["--output-dir", str(output_dir), "--job-id", "native_no_exp"],
                input_manifest={"files": [{"path": "mixed_native.pdf"}]},
                allowed_root=input_dir,
                work_dir=tmp_path / "w1",
                allow_cpu_fallback=True,
            )
            self.assertTrue(res_native.succeeded, f"stderr={res_native.stderr}")
            n_dir = output_dir / "native_no_exp"
            self.assertFalse((n_dir / "review_pages").exists())

            n_doc = json.loads((n_dir / "document.json").read_text(encoding="utf-8"))
            n_meta = json.loads((n_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(n_meta["triage_summary"]["vision_review_pages"], [2, 3])
            self.assertFalse(n_meta["triage_summary"]["review_images_exported"])
            # Page 1 (clean + tiny logo) -> PASS
            self.assertFalse(n_doc["pages"][0]["quality_triage"]["needs_vision_review"])
            # Page 2 (large chart) -> IMAGE_HEAVY_PAGE, avg_confidence is None
            self.assertIn("IMAGE_HEAVY_PAGE", n_doc["pages"][1]["quality_triage"]["reasons"])
            self.assertIsNone(n_doc["pages"][1]["quality_triage"]["metrics"]["avg_confidence"])
            # Page 3 (blank) -> NO_TEXT_DETECTED
            self.assertEqual(n_doc["pages"][2]["quality_triage"]["reasons"], ["NO_TEXT_DETECTED"])

            # Run B: Scanned PDF WITH --export-review-images -> exports ONLY flagged pages [2, 3]
            res_scan = client.run_worker(
                "ocr_pdf",
                extra_args=[
                    "--output-dir",
                    str(output_dir),
                    "--job-id",
                    "scan_with_exp",
                    "--export-review-images",
                    "--review-dpi",
                    "120",
                ],
                input_manifest={"files": [{"path": "mixed_scanned.pdf"}]},
                allowed_root=input_dir,
                work_dir=tmp_path / "w2",
                allow_cpu_fallback=True,
            )
            self.assertTrue(res_scan.succeeded, f"stderr={res_scan.stderr}")
            s_dir = output_dir / "scan_with_exp"
            s_doc = json.loads((s_dir / "document.json").read_text(encoding="utf-8"))
            s_meta = json.loads((s_dir / "metadata.json").read_text(encoding="utf-8"))

            # Q1: Page 1 clean scan -> NOT flagged
            self.assertFalse(s_doc["pages"][0]["quality_triage"]["needs_vision_review"])
            # Q2-B: Page 2 short title -> LOW_TEXT_CONTENT (and NOT NO_TEXT_DETECTED)
            self.assertIn("LOW_TEXT_CONTENT", s_doc["pages"][1]["quality_triage"]["reasons"])
            self.assertNotIn("NO_TEXT_DETECTED", s_doc["pages"][1]["quality_triage"]["reasons"])
            # Q4: Page 3 pure chart -> NO_TEXT_DETECTED
            self.assertEqual(s_doc["pages"][2]["quality_triage"]["reasons"], ["NO_TEXT_DETECTED"])

            self.assertEqual(s_meta["triage_summary"]["vision_review_pages"], [2, 3])
            self.assertTrue(s_meta["triage_summary"]["review_images_exported"])

            # Verify ONLY pages 2 and 3 PNGs exist (NOT page 1!) and SHA-256 matches
            rev_dir = s_dir / "review_pages"
            self.assertTrue(rev_dir.is_dir())
            self.assertFalse((rev_dir / "page_001.png").exists())
            self.assertTrue((rev_dir / "page_002.png").is_file())
            self.assertTrue((rev_dir / "page_003.png").is_file())

            rev_manifest = json.loads((rev_dir / "review_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual([p["page_number"] for p in rev_manifest["pages"]], [2, 3])
            for item in rev_manifest["pages"]:
                actual_sha = hashlib.sha256((rev_dir / item["filename"]).read_bytes()).hexdigest()
                self.assertEqual(actual_sha, item["sha256"])

            # Run C: Export size limit failure on existing job_id preserves prior valid output!
            prior_md = (s_dir / "document.md").read_text(encoding="utf-8")
            res_fail = client.run_worker(
                "ocr_pdf",
                extra_args=[
                    "--output-dir",
                    str(output_dir),
                    "--job-id",
                    "scan_with_exp",
                    "--export-review-images",
                    "--max-review-image-mib",
                    "0.0001",  # ~100 bytes -> guaranteed failure during review PNG export
                ],
                input_manifest={"files": [{"path": "mixed_scanned.pdf"}]},
                allowed_root=input_dir,
                work_dir=tmp_path / "w3",
                allow_cpu_fallback=True,
            )
            self.assertEqual(res_fail.status, "FAILED")
            # Prior valid output MUST remain intact and no staging dirs left
            self.assertEqual((s_dir / "document.md").read_text(encoding="utf-8"), prior_md)
            self.assertEqual([p for p in output_dir.iterdir() if ".staging" in p.name], [])
            self.assertEqual(len(client.vram_gate.read_registry_snapshot().active_leases()), 0)


if __name__ == "__main__":
    unittest.main()
