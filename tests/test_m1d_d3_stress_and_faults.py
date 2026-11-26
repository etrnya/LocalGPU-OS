"""M1-D D3 Stress & Fault-Injection Test Suite (VRAM, Thermal, Cancel, Force-Kill, Batch).

Validates LocalGPU-OS + `ocr_pdf` worker under unfavorable / fault conditions:
- D3.1: VRAM pressure — Admission rejection vs. in-flight inference OOM recovery
- D3.2: Thermal limits — COOLING (with CPU fallback), CRITICAL, SENSOR_DEGRADED
- D3.3: Multi-page cancel — mid-OCR cooperative cancellation leaves zero partial outputs
- D3.4: Forced kill — Step 7 kill preserves prior valid output, reprobes VRAM, Fail-Closed
- D3.5: Multi-file mixed batch + multi-run real RTX 4060 CUDA VRAM reclamation check
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest

from localgpu_os import (
    ComputeBackend,
    ErrorCode,
    LocalGpuClient,
    SensorStatus,
    ThermalState,
    VramGate,
    WorkerRegistry,
    WorkerRuntime,
)
from localgpu_os.domain.hardware import HardwareProfile, HardwareSnapshot
from localgpu_os.infrastructure.process_inspector import OsProcessInspector

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKERS_DIR = _REPO_ROOT / "workers"
_OCR_PDF_VENV_PY = (
    _WORKERS_DIR / "ocr_pdf" / ".venv" / "Scripts" / "python.exe"
    if os.name == "nt"
    else _WORKERS_DIR / "ocr_pdf" / ".venv" / "bin" / "python"
)


class _MutableClock:
    def __init__(self, mono: float = 100.0) -> None:
        self._mono = mono

    def utc_now_iso(self) -> str:
        return "2026-11-26T13:00:00Z"

    def monotonic(self) -> float:
        return self._mono


class _ConfigurableDetector:
    def __init__(
        self,
        clock: _MutableClock,
        available_vram_mib: int | None = 6000,
        temperature_c: int | None = 55,
        sensor_status: SensorStatus = SensorStatus.OK,
        temp_support: bool = True,
    ) -> None:
        self.clock = clock
        self.available_vram_mib = available_vram_mib
        self.temperature_c = temperature_c
        self.sensor_status = sensor_status
        self.temp_support = temp_support
        self.probe_count = 0

    def probe(self, device_index: int = 0) -> tuple[HardwareProfile, HardwareSnapshot]:
        self.probe_count += 1
        profile = HardwareProfile(
            device_id=f"cuda:{device_index}",
            vendor="NVIDIA",
            device_name="NVIDIA GeForce RTX 4060 Laptop GPU",
            architecture="Ada Lovelace",
            compute_capability="8.9",
            total_vram_mib=8188,
            nvenc_supported=True,
            temperature_support=self.temp_support,
            backend_support=(ComputeBackend.CUDA, ComputeBackend.CPU_INT8),
        )
        snapshot = HardwareSnapshot(
            device_id=profile.device_id,
            observed_at=self.clock.utc_now_iso(),
            observed_monotonic=self.clock.monotonic(),
            freshness_ttl_seconds=10.0,
            available_vram_mib=self.available_vram_mib,
            external_vram_mib=0,
            temperature_c=self.temperature_c,
            utilization_percent=15,
            sensor_status=self.sensor_status,
        )
        return profile, snapshot


def _create_native_pdf(target: Path, text: str) -> None:
    script = (
        "import pymupdf as fitz, sys\n"
        "doc = fitz.open()\n"
        "p = doc.new_page()\n"
        "p.insert_text((72, 72), sys.argv[2])\n"
        "doc.save(sys.argv[1])\n"
        "doc.close()\n"
    )
    subprocess.run(
        [str(_OCR_PDF_VENV_PY), "-c", script, str(target), text],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )


def _create_scanned_pdf(target: Path, page_count: int, prefix: str = "STRESS") -> None:
    script = (
        "import pymupdf as fitz, sys\n"
        "out_pdf, n_pages, prefix = sys.argv[1], int(sys.argv[2]), sys.argv[3]\n"
        "out_doc = fitz.open()\n"
        "for i in range(1, n_pages + 1):\n"
        "    tmp = fitz.open()\n"
        "    tp = tmp.new_page(width=800, height=400)\n"
        "    tp.insert_text((50, 100), f'{prefix} PAGE {i} OF {n_pages}', fontsize=28)\n"
        "    tp.insert_text((50, 180), f'BATCH RECORD 2026-{i:03d}', fontsize=24)\n"
        "    pix = tp.get_pixmap(dpi=160)\n"
        "    png = pix.tobytes('png')\n"
        "    tmp.close()\n"
        "    p = out_doc.new_page(width=800, height=400)\n"
        "    p.insert_image(p.rect, stream=png)\n"
        "out_doc.save(out_pdf)\n"
        "out_doc.close()\n"
    )
    subprocess.run(
        [str(_OCR_PDF_VENV_PY), "-c", script, str(target), str(page_count), prefix],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )


def _query_nvidia_vram_used_mib() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5.0,
        ).strip()
        first = out.splitlines()[0].strip()
        return int(first)
    except Exception:
        return None


@unittest.skipUnless(
    _OCR_PDF_VENV_PY.exists(),
    "workers/ocr_pdf/.venv is not installed; skipping M1-D D3 stress tests.",
)
class TestM1DD3StressAndFaults(unittest.TestCase):
    """M1-D D3 stress & fault-injection suite."""

    # ------------------------------------------------------------------
    # D3.1: VRAM pressure — (A) Admission rejection vs (B) In-flight OOM
    # ------------------------------------------------------------------
    def test_d3_1_vram_admission_rejection_and_inflight_oom_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "inputs"
            input_dir.mkdir()
            pdf = input_dir / "scan_1p.pdf"
            _create_scanned_pdf(pdf, page_count=1, prefix="OOMTEST")
            output_dir = tmp_path / "output"

            # (A) Boundary VRAM admission rejection:
            # safety (1536) + requested (2048) = 3584 MiB; provide 3500 MiB -> must REJECT
            clock = _MutableClock()
            tight_detector = _ConfigurableDetector(clock, available_vram_mib=3500)
            client_tight = LocalGpuClient(
                workers_dir=_WORKERS_DIR,
                state_dir=tmp_path / "state_tight",
                detector=tight_detector,
                clock=clock,
            )
            rej = client_tight.run_worker(
                "ocr_pdf",
                extra_args=["--output-dir", str(output_dir), "--job-id", "job_vram_rej"],
                input_manifest={"files": [{"path": "scan_1p.pdf"}]},
                allowed_root=input_dir,
                work_dir=tmp_path / "work_rej",
                allow_cpu_fallback=False,
            )
            self.assertEqual(rej.status, "REJECTED")
            self.assertIn(ErrorCode.VRAM_INSUFFICIENT, rej.error_codes)
            self.assertIsNone(rej.lease_id)
            self.assertFalse((output_dir / "job_vram_rej").exists())
            self.assertEqual(len(client_tight.vram_gate.read_registry_snapshot().active_leases()), 0)

            # (B) In-flight OOM after GPU lease is granted:
            # Inject a tiny sitecustomize hook via PYTHONPATH in a scratch dir that simulates
            # CUDA out-of-memory during RapidOcrEngine.extract_scanned_pdf
            inject_dir = tmp_path / "oom_inject"
            inject_dir.mkdir()
            (inject_dir / "sitecustomize.py").write_text(
                "import os\n"
                "if os.environ.get('TEST_SIMULATE_INFLIGHT_OOM') == '1':\n"
                "    import ocr_pdf_worker.ocr_engine as _oe\n"
                "    def _boom(self, *a, **kw):\n"
                "        raise RuntimeError('CUDA out of memory: failed to allocate 2048 MiB')\n"
                "    _oe.RapidOcrEngine.extract_scanned_pdf = _boom\n",
                encoding="utf-8",
            )

            ok_detector = _ConfigurableDetector(clock, available_vram_mib=6000)
            client_oom = LocalGpuClient(
                workers_dir=_WORKERS_DIR,
                state_dir=tmp_path / "state_oom",
                detector=ok_detector,
                clock=clock,
            )
            oom_res = client_oom.run_worker(
                "ocr_pdf",
                extra_args=["--output-dir", str(output_dir), "--job-id", "job_inflight_oom"],
                input_manifest={"files": [{"path": "scan_1p.pdf"}]},
                allowed_root=input_dir,
                work_dir=tmp_path / "work_oom",
                allow_cpu_fallback=False,
                extra_env={
                    "PYTHONPATH": f"{inject_dir}{os.pathsep}{_WORKERS_DIR / 'ocr_pdf'}",
                    "TEST_SIMULATE_INFLIGHT_OOM": "1",
                },
            )
            self.assertEqual(oom_res.status, "FAILED")
            self.assertIn(ErrorCode.WORKER_EXECUTION_FAILED, oom_res.error_codes)
            self.assertIsNotNone(oom_res.lease_id)  # Lease WAS granted before OOM
            self.assertIsNotNone(oom_res.output_data)
            self.assertEqual(oom_res.output_data["jobs"][0]["error_kind"], "OOM")
            # No partial output directory & lease MUST be cleanly released
            self.assertFalse((output_dir / "job_inflight_oom").exists())
            self.assertEqual(len(client_oom.vram_gate.read_registry_snapshot().active_leases()), 0)

            # Subsequent task on the same VramGate succeeds cleanly (no poisoned lock/lease)
            native_pdf = input_dir / "recovery_native.pdf"
            _create_native_pdf(native_pdf, "Post-OOM recovery verified with sufficient native text.")
            rec = client_oom.run_worker(
                "ocr_pdf",
                extra_args=["--output-dir", str(output_dir), "--job-id", "job_post_oom"],
                input_manifest={"files": [{"path": "recovery_native.pdf"}]},
                allowed_root=input_dir,
                work_dir=tmp_path / "work_rec",
            )
            self.assertTrue(rec.succeeded, f"stderr={rec.stderr}")
            self.assertTrue((output_dir / "job_post_oom" / "document.md").is_file())

    # ------------------------------------------------------------------
    # D3.2: Thermal limits — COOLING (with fallback), CRITICAL, SENSOR_DEGRADED
    # ------------------------------------------------------------------
    def test_d3_2_thermal_cooling_critical_and_sensor_degraded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "inputs"
            input_dir.mkdir()
            pdf = input_dir / "thermal_native.pdf"
            _create_native_pdf(pdf, "Thermal guard verification document with native text layer.")
            output_dir = tmp_path / "output"

            # 1. COOLING (85 C): denies GPU; allows CPU fallback when enabled
            clock1 = _MutableClock()
            cool_det = _ConfigurableDetector(clock1, available_vram_mib=6000, temperature_c=85)
            client_cool = LocalGpuClient(
                workers_dir=_WORKERS_DIR,
                state_dir=tmp_path / "state_cool",
                detector=cool_det,
                clock=clock1,
            )
            rej_cool = client_cool.run_worker(
                "ocr_pdf",
                extra_args=["--output-dir", str(output_dir), "--job-id", "job_cool_rej"],
                input_manifest={"files": [{"path": "thermal_native.pdf"}]},
                allowed_root=input_dir,
                work_dir=tmp_path / "work_cool_rej",
                allow_cpu_fallback=False,
            )
            self.assertEqual(rej_cool.status, "REJECTED")
            self.assertIn(ErrorCode.THERMAL_COOLING_WAIT, rej_cool.error_codes)

            fb_cool = client_cool.run_worker(
                "ocr_pdf",
                extra_args=["--output-dir", str(output_dir), "--job-id", "job_cool_fb"],
                input_manifest={"files": [{"path": "thermal_native.pdf"}]},
                allowed_root=input_dir,
                work_dir=tmp_path / "work_cool_fb",
                allow_cpu_fallback=True,
            )
            self.assertTrue(fb_cool.succeeded, f"stderr={fb_cool.stderr}")
            self.assertTrue(fb_cool.used_cpu_fallback)

            # 2. CRITICAL (90 C): denies GPU with THERMAL_CRITICAL_ABORT
            clock2 = _MutableClock()
            crit_det = _ConfigurableDetector(clock2, available_vram_mib=6000, temperature_c=90)
            client_crit = LocalGpuClient(
                workers_dir=_WORKERS_DIR,
                state_dir=tmp_path / "state_crit",
                detector=crit_det,
                clock=clock2,
            )
            rej_crit = client_crit.run_worker(
                "ocr_pdf",
                extra_args=["--output-dir", str(output_dir), "--job-id", "job_crit"],
                input_manifest={"files": [{"path": "thermal_native.pdf"}]},
                allowed_root=input_dir,
                work_dir=tmp_path / "work_crit",
                allow_cpu_fallback=False,
            )
            self.assertEqual(rej_crit.status, "REJECTED")
            self.assertIn(ErrorCode.THERMAL_CRITICAL_ABORT, rej_crit.error_codes)
            self.assertEqual(len(client_crit.vram_gate.read_registry_snapshot().active_leases()), 0)

            # 3. SENSOR_DEGRADED (temperature_c=None): denies GPU with THERMAL_SENSOR_DEGRADED
            clock3 = _MutableClock()
            deg_det = _ConfigurableDetector(
                clock3,
                available_vram_mib=6000,
                temperature_c=None,
                sensor_status=SensorStatus.DEGRADED,
                temp_support=False,
            )
            client_deg = LocalGpuClient(
                workers_dir=_WORKERS_DIR,
                state_dir=tmp_path / "state_deg",
                detector=deg_det,
                clock=clock3,
            )
            rej_deg = client_deg.run_worker(
                "ocr_pdf",
                extra_args=["--output-dir", str(output_dir), "--job-id", "job_deg"],
                input_manifest={"files": [{"path": "thermal_native.pdf"}]},
                allowed_root=input_dir,
                work_dir=tmp_path / "work_deg",
                allow_cpu_fallback=False,
            )
            self.assertEqual(rej_deg.status, "REJECTED")
            self.assertIn(ErrorCode.THERMAL_SENSOR_DEGRADED, rej_deg.error_codes)

    # ------------------------------------------------------------------
    # D3.3: Multi-page OCR cooperative cancel leaves ZERO partial outputs
    # ------------------------------------------------------------------
    def test_d3_3_multipage_ocr_cancel_clean_exit_and_no_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "inputs"
            input_dir.mkdir()
            pdf = input_dir / "multi_6p_scan.pdf"
            _create_scanned_pdf(pdf, page_count=6, prefix="CANCEL")
            output_dir = tmp_path / "output"
            work_dir = tmp_path / "work_cancel"

            # Use CPU fallback so 6 pages take several seconds -> reliable mid-flight cancel window
            clock = _MutableClock()
            low_vram_det = _ConfigurableDetector(clock, available_vram_mib=500)
            client = LocalGpuClient(
                workers_dir=_WORKERS_DIR,
                state_dir=tmp_path / "state_cancel",
                detector=low_vram_det,
                clock=clock,
            )
            cancel_event = threading.Event()

            def _cancel_after_start() -> None:
                # Wait until work_dir/input_manifest.json exists, then wait briefly and cancel
                deadline = time.monotonic() + 10.0
                while not (work_dir / "input_manifest.json").exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                time.sleep(1.0)
                cancel_event.set()

            t = threading.Thread(target=_cancel_after_start, daemon=True)
            t.start()
            res = client.run_worker(
                "ocr_pdf",
                extra_args=["--output-dir", str(output_dir), "--job-id", "job_cancel_6p"],
                input_manifest={"files": [{"path": "multi_6p_scan.pdf"}]},
                allowed_root=input_dir,
                work_dir=work_dir,
                allow_cpu_fallback=True,
                cancel_event=cancel_event,
            )
            t.join(timeout=5.0)

            self.assertEqual(res.status, "CANCELLED", f"warnings={res.warnings}, stderr={res.stderr}")
            self.assertIn(ErrorCode.WORKER_CANCELLED, res.error_codes)
            self.assertFalse(res.forced_kill)
            # Zero partial output directory and zero staging leftovers
            self.assertFalse((output_dir / "job_cancel_6p").exists())
            if output_dir.exists():
                self.assertEqual([p for p in output_dir.iterdir() if ".staging" in p.name], [])
            self.assertEqual(len(client.vram_gate.read_registry_snapshot().active_leases()), 0)

    # ------------------------------------------------------------------
    # D3.4: Forced kill preserves prior valid output, verifies dead PID & VRAM reprobe
    # ------------------------------------------------------------------
    def test_d3_4_forced_kill_preserves_prior_output_and_reprobes_vram(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "inputs"
            input_dir.mkdir()
            pdf = input_dir / "doc.pdf"
            _create_native_pdf(pdf, "Original valid document content preserved across forced kill.")
            output_dir = tmp_path / "output"

            # First produce a valid committed output for job_preserve
            clock = _MutableClock()
            det = _ConfigurableDetector(clock, available_vram_mib=6000)
            state_dir = tmp_path / "state_fk"
            client = LocalGpuClient(
                workers_dir=_WORKERS_DIR,
                state_dir=state_dir,
                detector=det,
                clock=clock,
            )
            first = client.run_worker(
                "ocr_pdf",
                extra_args=["--output-dir", str(output_dir), "--job-id", "job_preserve"],
                input_manifest={"files": [{"path": "doc.pdf"}]},
                allowed_root=input_dir,
                work_dir=tmp_path / "work_first",
            )
            self.assertTrue(first.succeeded, f"stderr={first.stderr}")
            prior_md = (output_dir / "job_preserve" / "document.md").read_text(encoding="utf-8")
            self.assertIn("Original valid document", prior_md)

            # Now run again with a hang injected BEFORE atomic commit, and short cancel grace
            hang_inject = tmp_path / "hang_inject"
            hang_inject.mkdir()
            pid_file = tmp_path / "hung_worker.pid"
            (hang_inject / "sitecustomize.py").write_text(
                "import os, time\n"
                "from pathlib import Path\n"
                "if os.environ.get('TEST_HANG_IN_WORKER') == '1':\n"
                "    import ocr_pdf_worker.native_extractor as _ne\n"
                "    def _hang(*a, **kw):\n"
                f"        Path({str(pid_file)!r}).write_text(str(os.getpid()), encoding='utf-8')\n"
                "        time.sleep(30)\n"
                "    _ne.extract_native_pdf = _hang\n",
                encoding="utf-8",
            )

            # Use a temporary copy of worker.json with short cancel_grace_seconds=0.4
            short_workers = tmp_path / "workers_short"
            wdir = short_workers / "ocr_pdf"
            wdir.mkdir(parents=True)
            manifest_data = json.loads((_WORKERS_DIR / "ocr_pdf" / "worker.json").read_text(encoding="utf-8"))
            manifest_data["cancel_grace_seconds"] = 0.4
            (wdir / "worker.json").write_text(json.dumps(manifest_data), encoding="utf-8")
            # Junction to real ocr_pdf/.venv and ocr_pdf_worker package
            if os.name == "nt":
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(wdir / ".venv"), str(_WORKERS_DIR / "ocr_pdf" / ".venv")],
                    check=True,
                    capture_output=True,
                )
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(wdir / "ocr_pdf_worker"), str(_WORKERS_DIR / "ocr_pdf" / "ocr_pdf_worker")],
                    check=True,
                    capture_output=True,
                )
            else:
                os.symlink(_WORKERS_DIR / "ocr_pdf" / ".venv", wdir / ".venv")
                os.symlink(_WORKERS_DIR / "ocr_pdf" / "ocr_pdf_worker", wdir / "ocr_pdf_worker")

            probes_before = det.probe_count
            client_fk = LocalGpuClient(
                workers_dir=short_workers,
                state_dir=state_dir,
                detector=det,
                clock=clock,
            )
            cancel_ev = threading.Event()

            def _trigger_when_hung() -> None:
                deadline = time.monotonic() + 8.0
                while not pid_file.exists() and time.monotonic() < deadline:
                    time.sleep(0.05)
                cancel_ev.set()

            t = threading.Thread(target=_trigger_when_hung, daemon=True)
            t.start()
            fk_res = client_fk.run_worker(
                "ocr_pdf",
                extra_args=["--output-dir", str(output_dir), "--job-id", "job_preserve"],
                input_manifest={"files": [{"path": "doc.pdf"}]},
                allowed_root=input_dir,
                work_dir=tmp_path / "work_fk",
                cancel_event=cancel_ev,
                extra_env={
                    "PYTHONPATH": f"{hang_inject}{os.pathsep}{wdir}",
                    "TEST_HANG_IN_WORKER": "1",
                },
            )
            t.join(timeout=5.0)

            self.assertEqual(fk_res.status, "CANCELLED")
            self.assertTrue(fk_res.forced_kill)
            self.assertTrue(fk_res.post_kill_vram_reprobed)
            self.assertGreaterEqual(det.probe_count, probes_before + 2)

            # Confirm hung worker PID is genuinely dead
            hung_pid = int(pid_file.read_text(encoding="utf-8").strip())
            self.assertFalse(OsProcessInspector().is_alive(hung_pid))

            # Prior valid output MUST remain intact & no staging dirs left
            after_md = (output_dir / "job_preserve" / "document.md").read_text(encoding="utf-8")
            self.assertEqual(after_md, prior_md)
            self.assertEqual([p for p in output_dir.iterdir() if ".staging" in p.name], [])
            self.assertEqual(len(client_fk.vram_gate.read_registry_snapshot().active_leases()), 0)

    # ------------------------------------------------------------------
    # D3.5: Mixed multi-file batch error isolation + multi-run RTX 4060 VRAM stability
    # ------------------------------------------------------------------
    def test_d3_5_mixed_batch_isolation_and_multi_run_vram_reclamation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_dir = tmp_path / "inputs"
            input_dir.mkdir()

            native_pdf = input_dir / "01_native.pdf"
            scan_pdf_a = input_dir / "02_scan_a.pdf"
            corrupt_pdf = input_dir / "03_corrupt.pdf"
            scan_pdf_b = input_dir / "04_scan_b.pdf"

            _create_native_pdf(native_pdf, "Batch Item One: Native text PDF with clean extraction.")
            _create_scanned_pdf(scan_pdf_a, page_count=1, prefix="BATCHA")
            corrupt_pdf.write_bytes(b"NOT A VALID PDF HEADER %%%% CORRUPT")
            _create_scanned_pdf(scan_pdf_b, page_count=1, prefix="BATCHB")

            output_dir = tmp_path / "output_batch"
            state_dir = tmp_path / "state_batch"
            client = LocalGpuClient(workers_dir=_WORKERS_DIR, state_dir=state_dir)

            # Part A: Single mixed batch with 1 corrupt file -> 3 valid succeed, 1 isolated failure
            batch_res = client.run_worker(
                "ocr_pdf",
                extra_args=["--output-dir", str(output_dir), "--job-id", "batch"],
                input_manifest={
                    "files": [
                        {"path": "01_native.pdf"},
                        {"path": "02_scan_a.pdf"},
                        {"path": "03_corrupt.pdf"},
                        {"path": "04_scan_b.pdf"},
                    ]
                },
                allowed_root=input_dir,
                work_dir=tmp_path / "work_batch",
                allow_cpu_fallback=True,
            )
            self.assertEqual(batch_res.status, "FAILED")  # Partial failure surfaces non-zero exit
            jobs = batch_res.output_data["jobs"]
            self.assertEqual(len(jobs), 4)
            self.assertEqual(jobs[0]["status"], "SUCCEEDED")
            self.assertEqual(jobs[0]["extraction_method"], "pymupdf_native")
            self.assertEqual(jobs[1]["status"], "SUCCEEDED")
            self.assertEqual(jobs[1]["extraction_method"], "rapidocr")
            self.assertEqual(jobs[2]["status"], "FAILED")
            self.assertEqual(jobs[3]["status"], "SUCCEEDED")
            self.assertEqual(jobs[3]["extraction_method"], "rapidocr")

            # Verify output isolation: valid jobs have full bundles, corrupt job has NO output dir
            for good_id in ("batch_001", "batch_002", "batch_004"):
                self.assertTrue((output_dir / good_id / "document.md").is_file())
                self.assertTrue((output_dir / good_id / "document.json").is_file())
                self.assertTrue((output_dir / good_id / "metadata.json").is_file())
            self.assertFalse((output_dir / "batch_003").exists())
            self.assertEqual(len(client.vram_gate.read_registry_snapshot().active_leases()), 0)

            # Part B: 3 consecutive real GPU OCR runs -> verify VRAM returns to baseline (no leak)
            vram_baseline = _query_nvidia_vram_used_mib()
            doctor = client.doctor()
            if vram_baseline is not None and doctor.decision.gpu_admission_allowed:
                for run_idx in range(1, 4):
                    r = client.run_worker(
                        "ocr_pdf",
                        extra_args=["--output-dir", str(output_dir), "--job-id", f"seq_{run_idx}"],
                        input_manifest={"files": [{"path": "02_scan_a.pdf"}]},
                        allowed_root=input_dir,
                        work_dir=tmp_path / f"work_seq_{run_idx}",
                        allow_cpu_fallback=False,
                    )
                    self.assertTrue(r.succeeded, f"seq {run_idx} stderr={r.stderr}")
                    self.assertEqual(len(client.vram_gate.read_registry_snapshot().active_leases()), 0)

                time.sleep(0.5)
                vram_after = _query_nvidia_vram_used_mib()
                self.assertIsNotNone(vram_after)
                # After isolated worker subprocess exits, VRAM delta vs baseline must stay minimal
                self.assertLess(
                    abs(vram_after - vram_baseline),
                    256,
                    f"Potential VRAM leak across runs: baseline={vram_baseline} MiB, after={vram_after} MiB",
                )


if __name__ == "__main__":
    unittest.main()
