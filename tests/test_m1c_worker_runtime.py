"""M1-C Worker Runtime test suite (PRD v1.1.2 Sections 2.5, 2.6, 3).

Uses Python standard library `unittest` (zero external dependencies, pytest-compatible).
"""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import venv

from localgpu_os import (
    ComputeBackend,
    ErrorCode,
    LocalGpuClient,
    NetworkPolicy,
    ResourceClass,
    SensorStatus,
    VramGate,
    WorkerManifest,
    WorkerRegistry,
    WorkerRuntime,
)
from localgpu_os.domain.hardware import HardwareProfile, HardwareSnapshot
from localgpu_os.domain.worker import (
    WorkerEnvNotInstalledError,
    WorkerFileLimits,
    WorkerManifestInvalidError,
    validate_json_file_manifest,
)
from localgpu_os.presentation.cli import main as cli_main

_SRC_DIR = str(Path(__file__).resolve().parents[1] / "src")


class FakeClock:
    def __init__(self, monotonic_val: float = 100.0, utc_iso: str = "2026-11-26T12:00:00Z") -> None:
        self._mono = monotonic_val
        self._utc = utc_iso

    def utc_now_iso(self) -> str:
        return self._utc

    def monotonic(self) -> float:
        return self._mono


class FakeDetector:
    def __init__(
        self,
        clock: FakeClock,
        available_vram_mib: int | None = 6000,
        temperature_c: int | None = 52,
        has_cuda: bool = True,
    ) -> None:
        self.clock = clock
        self.available_vram_mib = available_vram_mib
        self.temperature_c = temperature_c
        self.has_cuda = has_cuda
        self.probe_count = 0

    def probe(self, device_index: int = 0) -> tuple[HardwareProfile, HardwareSnapshot]:
        self.probe_count += 1
        backends = (
            (ComputeBackend.CUDA, ComputeBackend.CPU_INT8)
            if self.has_cuda
            else (ComputeBackend.CPU_INT8,)
        )
        profile = HardwareProfile(
            device_id=f"cuda:{device_index}" if self.has_cuda else "cpu",
            vendor="NVIDIA" if self.has_cuda else "CPU",
            device_name="NVIDIA GeForce RTX 4060 Laptop GPU" if self.has_cuda else "Generic CPU",
            architecture="Ada Lovelace" if self.has_cuda else None,
            compute_capability="8.9" if self.has_cuda else None,
            total_vram_mib=8188 if self.has_cuda else None,
            nvenc_supported=True if self.has_cuda else False,
            temperature_support=True if self.has_cuda else False,
            backend_support=backends,
        )
        snapshot = HardwareSnapshot(
            device_id=profile.device_id,
            observed_at=self.clock.utc_now_iso(),
            observed_monotonic=self.clock.monotonic(),
            freshness_ttl_seconds=10.0,
            available_vram_mib=self.available_vram_mib if self.has_cuda else None,
            external_vram_mib=500 if self.has_cuda else None,
            temperature_c=self.temperature_c if self.has_cuda else None,
            utilization_percent=10 if self.has_cuda else None,
            sensor_status=SensorStatus.OK if self.has_cuda else SensorStatus.UNAVAILABLE,
        )
        return profile, snapshot


class FakeDiskChecker:
    def __init__(self, free_mib: int = 4096) -> None:
        self.free_mib = free_mib

    def free_disk_mib(self, path: Path) -> int:
        return self.free_mib


def _base_manifest_dict(
    worker_id: str = "demo_worker",
    module: str = "worker_main",
    args: list[str] | None = None,
    use_gpu: bool = True,
    cpu_supported: bool = True,
    cancel_grace_seconds: float = 3.0,
    min_free_disk_mib: int = 128,
) -> dict:
    resources = (
        [
            {"resource_class": "GPU_COMPUTE", "amount": 1, "unit": "slot", "exclusive": True},
            {"resource_class": "GPU_MEMORY", "amount": 1024, "unit": "MiB"},
        ]
        if use_gpu
        else [
            {"resource_class": "CPU_COMPUTE", "amount": 2, "unit": "threads"},
        ]
    )
    return {
        "schema_version": "1.0",
        "worker_id": worker_id,
        "name": "Demo Isolated Worker",
        "version": "1.0.0",
        "entrypoint": {
            "module": module,
            "args": args or [],
        },
        "resources": resources,
        "fallback_policy": {
            "cpu_supported": cpu_supported,
            "cpu_backend": "cpu_int8",
            "max_cpu_threads": 2,
            "max_system_memory_mib": 1024,
        },
        "network_policy": "RUNTIME_OFFLINE",
        "cancel_grace_seconds": cancel_grace_seconds,
        "timeout_seconds": 30.0,
        "poll_interval_seconds": 0.05,
        "concurrency_limit": 1,
        "file_limits": {
            "max_file_size_bytes": 1024 * 1024,
            "min_free_disk_mib": min_free_disk_mib,
            "allow_base64_payload": False,
        },
    }


class TestM1CWorkerRuntime(unittest.TestCase):
    """Comprehensive M1-C verification using real isolated .venv subprocesses."""

    _shared_tmp: tempfile.TemporaryDirectory[str]
    _shared_venv_dir: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls._shared_tmp = tempfile.TemporaryDirectory()
        cls._shared_venv_dir = Path(cls._shared_tmp.name) / "shared_venv"
        venv.EnvBuilder(with_pip=False, symlinks=(os.name != "nt")).create(cls._shared_venv_dir)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._shared_tmp.cleanup()

    def _provision_worker(
        self,
        workers_dir: Path,
        worker_id: str,
        script_body: str,
        manifest_overrides: dict | None = None,
        install_venv: bool = True,
    ) -> Path:
        wdir = workers_dir / worker_id
        wdir.mkdir(parents=True, exist_ok=True)
        manifest = _base_manifest_dict(worker_id=worker_id)
        if manifest_overrides:
            manifest.update(manifest_overrides)
        (wdir / "worker.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (wdir / "worker_main.py").write_text(script_body, encoding="utf-8")

        if install_venv:
            # Link or copy the lightweight pre-built .venv structure into worker/.venv
            target_venv = wdir / ".venv"
            if os.name == "nt":
                # Windows directory junction is instant and requires no elevation
                import subprocess

                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(target_venv), str(self._shared_venv_dir)],
                    check=True,
                    capture_output=True,
                )
            else:
                os.symlink(self._shared_venv_dir, target_venv)
        return wdir

    def _worker_env_with_src(self) -> dict[str, str]:
        existing = os.environ.get("PYTHONPATH", "")
        combined = f"{_SRC_DIR}{os.pathsep}{existing}" if existing else _SRC_DIR
        return {"PYTHONPATH": combined}

    # ------------------------------------------------------------------
    # T1: WorkerManifest Contract v1.0 validation
    # ------------------------------------------------------------------
    def test_t1_worker_manifest_v1_validation(self) -> None:
        valid = WorkerManifest.from_dict(_base_manifest_dict())
        self.assertEqual(valid.schema_version, "1.0")
        self.assertEqual(valid.network_policy, NetworkPolicy.RUNTIME_OFFLINE)
        self.assertTrue(valid.uses_gpu())

        # Reject shell/command fields
        bad_shell = _base_manifest_dict()
        bad_shell["command"] = "python -c 'print(1)'"
        with self.assertRaises(WorkerManifestInvalidError) as ctx:
            WorkerManifest.from_dict(bad_shell)
        self.assertEqual(ctx.exception.error_code, ErrorCode.WORKER_MANIFEST_INVALID)

        # Reject poll_interval_seconds > 1.0 (PRD 2.5 <= 1s)
        bad_poll = _base_manifest_dict()
        bad_poll["poll_interval_seconds"] = 1.5
        with self.assertRaises(WorkerManifestInvalidError):
            WorkerManifest.from_dict(bad_poll)

        # Reject path traversal worker_id
        bad_id = _base_manifest_dict(worker_id="../evil")
        with self.assertRaises(WorkerManifestInvalidError):
            WorkerManifest.from_dict(bad_id)

        # Reject allow_base64_payload = True
        bad_b64 = _base_manifest_dict()
        bad_b64["file_limits"]["allow_base64_payload"] = True
        with self.assertRaises(WorkerManifestInvalidError):
            WorkerManifest.from_dict(bad_b64)

    # ------------------------------------------------------------------
    # T2: Strict .venv Python resolution (Windows & POSIX, never PATH)
    # ------------------------------------------------------------------
    def test_t2_strict_venv_resolution_and_missing_env_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workers_dir = Path(tmp) / "workers"
            self._provision_worker(
                workers_dir,
                "no_venv_worker",
                "print('hi')\n",
                install_venv=False,
            )
            reg_nt = WorkerRegistry(workers_dir=workers_dir, platform_name="nt")
            self.assertTrue(
                str(reg_nt.expected_venv_python_path("no_venv_worker")).endswith(
                    os.path.join(".venv", "Scripts", "python.exe")
                )
            )
            reg_posix = WorkerRegistry(workers_dir=workers_dir, platform_name="posix")
            self.assertTrue(
                str(reg_posix.expected_venv_python_path("no_venv_worker")).endswith(
                    os.path.join(".venv", "bin", "python")
                )
            )

            with self.assertRaises(WorkerEnvNotInstalledError) as ctx:
                WorkerRegistry(workers_dir=workers_dir).resolve_venv_python("no_venv_worker")
            self.assertEqual(ctx.exception.error_code, ErrorCode.WORKER_ENV_NOT_INSTALLED)

            # Runtime returns REJECTED with WORKER_ENV_NOT_INSTALLED
            clock = FakeClock()
            gate = VramGate(state_dir=Path(tmp) / "state", detector=FakeDetector(clock), clock=clock)
            runtime = WorkerRuntime(
                vram_gate=gate,
                registry=WorkerRegistry(workers_dir=workers_dir),
                disk_checker=FakeDiskChecker(),
                clock=clock,
            )
            res = runtime.run_worker("no_venv_worker")
            self.assertEqual(res.status, "REJECTED")
            self.assertIn(ErrorCode.WORKER_ENV_NOT_INSTALLED, res.error_codes)

    # ------------------------------------------------------------------
    # T3: json_file_manifest Base64 ban, root jail, path traversal, size limit
    # ------------------------------------------------------------------
    def test_t3_json_file_manifest_security_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "allowed"
            root.mkdir()
            outside = Path(tmp) / "secret.txt"
            outside.write_text("top-secret", encoding="utf-8")

            ok_file = root / "doc.txt"
            ok_file.write_text("hello world", encoding="utf-8")

            big_file = root / "big.bin"
            big_file.write_bytes(b"x" * 2048)

            limits = WorkerFileLimits(max_file_size_bytes=1024, min_free_disk_mib=64)

            # Valid file passes
            entries = validate_json_file_manifest(
                {"files": [{"path": "doc.txt", "role": "input"}]},
                allowed_root=root,
                file_limits=limits,
            )
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].path, ok_file.resolve())

            # Base64 key forbidden
            with self.assertRaises(WorkerManifestInvalidError):
                validate_json_file_manifest(
                    {"files": [{"path": "doc.txt", "content_base64": "aGVsbG8="}]},
                    allowed_root=root,
                    file_limits=limits,
                )

            # Data URI Base64 forbidden
            with self.assertRaises(WorkerManifestInvalidError):
                validate_json_file_manifest(
                    {"files": [{"path": "doc.txt", "note": "data:application/pdf;base64,JVBERi0="}]},
                    allowed_root=root,
                    file_limits=limits,
                )

            # Path traversal blocked (relative ..)
            with self.assertRaises(WorkerManifestInvalidError):
                validate_json_file_manifest(
                    {"files": [{"path": "../secret.txt"}]},
                    allowed_root=root,
                    file_limits=limits,
                )

            # Path traversal blocked (absolute outside root)
            with self.assertRaises(WorkerManifestInvalidError):
                validate_json_file_manifest(
                    {"files": [{"path": str(outside)}]},
                    allowed_root=root,
                    file_limits=limits,
                )

            # Oversized file blocked
            with self.assertRaises(WorkerManifestInvalidError):
                validate_json_file_manifest(
                    {"files": [{"path": "big.bin"}]},
                    allowed_root=root,
                    file_limits=limits,
                )

    # ------------------------------------------------------------------
    # T4: DISK_SPACE_INSUFFICIENT guard
    # ------------------------------------------------------------------
    def test_t4_disk_space_insufficient_rejects_before_lease(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workers_dir = Path(tmp) / "workers"
            self._provision_worker(workers_dir, "disk_worker", "print('ok')\n")
            clock = FakeClock()
            gate = VramGate(state_dir=Path(tmp) / "state", detector=FakeDetector(clock), clock=clock)
            runtime = WorkerRuntime(
                vram_gate=gate,
                registry=WorkerRegistry(workers_dir=workers_dir),
                disk_checker=FakeDiskChecker(free_mib=32),  # < 128 MiB required
                clock=clock,
            )
            res = runtime.run_worker("disk_worker")
            self.assertEqual(res.status, "REJECTED")
            self.assertIn(ErrorCode.DISK_SPACE_INSUFFICIENT, res.error_codes)
            self.assertIsNone(res.lease_id)
            self.assertEqual(len(gate.read_registry_snapshot().active_leases()), 0)

    # ------------------------------------------------------------------
    # T5: Real isolated .venv execution (success + non-zero failure)
    # ------------------------------------------------------------------
    def test_t5_real_isolated_execution_success_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workers_dir = Path(tmp) / "workers"
            ok_script = (
                "import sys\n"
                "from localgpu_os.sdk import WorkerContext\n"
                "ctx = WorkerContext.from_env()\n"
                "manifest = ctx.load_input_manifest()\n"
                "ctx.write_result({\n"
                "    'ok': True,\n"
                "    'mode': ctx.execution_mode,\n"
                "    'network_policy': ctx.network_policy,\n"
                "    'args': sys.argv[1:],\n"
                "    'file_count': len(manifest['files']) if manifest else 0,\n"
                "})\n"
            )
            fail_script = "import sys\nprint('boom', file=sys.stderr)\nsys.exit(3)\n"

            self._provision_worker(workers_dir, "ok_worker", ok_script)
            self._provision_worker(workers_dir, "fail_worker", fail_script)

            input_root = Path(tmp) / "inputs"
            input_root.mkdir()
            sample = input_root / "page1.txt"
            sample.write_text("sample content", encoding="utf-8")

            clock = FakeClock()
            gate = VramGate(state_dir=Path(tmp) / "state", detector=FakeDetector(clock), clock=clock)
            runtime = WorkerRuntime(
                vram_gate=gate,
                registry=WorkerRegistry(workers_dir=workers_dir),
                disk_checker=FakeDiskChecker(),
                clock=clock,
            )

            res = runtime.run_worker(
                "ok_worker",
                extra_args=["--flag", "42"],
                input_manifest={"files": [{"path": "page1.txt"}]},
                allowed_root=input_root,
                extra_env=self._worker_env_with_src(),
            )
            self.assertTrue(res.succeeded, f"stderr={res.stderr}, warnings={res.warnings}")
            self.assertEqual(res.exit_code, 0)
            self.assertEqual(res.output_data["ok"], True)
            self.assertEqual(res.output_data["network_policy"], "RUNTIME_OFFLINE")
            self.assertEqual(res.output_data["args"], ["--flag", "42"])
            self.assertEqual(res.output_data["file_count"], 1)
            self.assertEqual(len(gate.read_registry_snapshot().active_leases()), 0)

            # Non-zero exit code -> WORKER_EXECUTION_FAILED & lease cleaned up
            fail_res = runtime.run_worker("fail_worker", extra_env=self._worker_env_with_src())
            self.assertEqual(fail_res.status, "FAILED")
            self.assertEqual(fail_res.exit_code, 3)
            self.assertIn(ErrorCode.WORKER_EXECUTION_FAILED, fail_res.error_codes)
            self.assertIn("boom", fail_res.stderr)
            self.assertEqual(len(gate.read_registry_snapshot().active_leases()), 0)

    # ------------------------------------------------------------------
    # T6: 7-Step Cancel Protocol (graceful cooperative path Steps 1-6)
    # ------------------------------------------------------------------
    def test_t6_seven_step_cancel_protocol_graceful_exit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workers_dir = Path(tmp) / "workers"
            ready_marker = Path(tmp) / "worker_ready.flag"
            worker_script = (
                "from pathlib import Path\n"
                "import time\n"
                "from localgpu_os.sdk import WorkerContext\n"
                "ctx = WorkerContext.from_env()\n"
                "tmp_file = ctx.temp_dir / 'scratch.tmp'\n"
                "tmp_file.write_text('working', encoding='utf-8')\n"
                f"Path({str(ready_marker)!r}).write_text('ready', encoding='utf-8')\n"
                "while True:\n"
                "    sig = ctx.check_cancelled()\n"
                "    if sig is not None:\n"
                "        ctx.cleanup_temp_dir()\n"
                "        released = ctx.release_lease()\n"
                "        ctx.write_cancelled({'released_by_worker': released, 'reason': sig.reason})\n"
                "        break\n"
                "    time.sleep(0.05)\n"
            )
            self._provision_worker(
                workers_dir,
                "coop_worker",
                worker_script,
                manifest_overrides={"cancel_grace_seconds": 5.0},
            )

            clock = FakeClock()
            gate = VramGate(state_dir=Path(tmp) / "state", detector=FakeDetector(clock), clock=clock)
            runtime = WorkerRuntime(
                vram_gate=gate,
                registry=WorkerRegistry(workers_dir=workers_dir),
                disk_checker=FakeDiskChecker(),
                clock=clock,
            )

            cancel_event = threading.Event()
            work_dir = Path(tmp) / "run_work"

            def _trigger_cancel() -> None:
                deadline = time.monotonic() + 5.0
                while not ready_marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                cancel_event.set()

            t = threading.Thread(target=_trigger_cancel, daemon=True)
            t.start()
            res = runtime.run_worker(
                "coop_worker",
                work_dir=work_dir,
                cancel_event=cancel_event,
                extra_env=self._worker_env_with_src(),
            )
            t.join(timeout=5.0)

            self.assertEqual(res.status, "CANCELLED", f"stderr={res.stderr}")
            self.assertIn(ErrorCode.WORKER_CANCELLED, res.error_codes)
            self.assertFalse(res.forced_kill)
            self.assertIsNotNone(res.output_data)
            self.assertEqual(res.output_data["status"], "WORKER_CANCELLED")
            self.assertTrue(res.output_data["details"]["released_by_worker"])
            self.assertFalse((work_dir / "tmp" / "scratch.tmp").exists())
            self.assertEqual(len(gate.read_registry_snapshot().active_leases()), 0)

    # ------------------------------------------------------------------
    # T7: 7-Step Cancel Protocol Step 7 (forced kill + cleanup + VRAM re-probe)
    # ------------------------------------------------------------------
    def test_t7_seven_step_cancel_step7_forced_kill_and_vram_reprobe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workers_dir = Path(tmp) / "workers"
            ready_marker = Path(tmp) / "stubborn_ready.flag"
            stubborn_script = (
                "from pathlib import Path\n"
                "import os, time\n"
                "tmp_dir = Path(os.environ['LOCALGPU_TEMP_DIR'])\n"
                "(tmp_dir / 'leak.tmp').write_text('data', encoding='utf-8')\n"
                f"Path({str(ready_marker)!r}).write_text('ready', encoding='utf-8')\n"
                "time.sleep(30)\n"
            )
            self._provision_worker(
                workers_dir,
                "stubborn_worker",
                stubborn_script,
                manifest_overrides={"cancel_grace_seconds": 0.35},
            )

            clock = FakeClock()
            detector = FakeDetector(clock)
            gate = VramGate(state_dir=Path(tmp) / "state", detector=detector, clock=clock)
            runtime = WorkerRuntime(
                vram_gate=gate,
                registry=WorkerRegistry(workers_dir=workers_dir),
                disk_checker=FakeDiskChecker(),
                clock=clock,
            )

            cancel_event = threading.Event()
            work_dir = Path(tmp) / "stubborn_work"

            def _trigger_cancel() -> None:
                deadline = time.monotonic() + 5.0
                while not ready_marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                cancel_event.set()

            t = threading.Thread(target=_trigger_cancel, daemon=True)
            t.start()
            res = runtime.run_worker(
                "stubborn_worker",
                work_dir=work_dir,
                cancel_event=cancel_event,
                extra_env=self._worker_env_with_src(),
            )
            t.join(timeout=5.0)

            self.assertEqual(res.status, "CANCELLED")
            self.assertIn(ErrorCode.WORKER_CANCELLED, res.error_codes)
            self.assertTrue(res.forced_kill)
            self.assertTrue(res.post_kill_vram_reprobed)
            self.assertGreaterEqual(detector.probe_count, 2)
            self.assertFalse((work_dir / "tmp").exists())
            self.assertEqual(len(gate.read_registry_snapshot().active_leases()), 0)

    # ------------------------------------------------------------------
    # T8: Safe CPU fallback when GPU admission is denied
    # ------------------------------------------------------------------
    def test_t8_gpu_denied_safely_falls_back_to_cpu_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workers_dir = Path(tmp) / "workers"
            script = (
                "from localgpu_os.sdk import WorkerContext\n"
                "ctx = WorkerContext.from_env()\n"
                "ctx.write_result({'execution_mode': ctx.execution_mode})\n"
            )
            self._provision_worker(workers_dir, "fallback_worker", script)

            clock = FakeClock()
            # Low VRAM (500 MiB < safety 1536 MiB) -> GPU admission denied
            low_vram_detector = FakeDetector(clock, available_vram_mib=500)
            gate = VramGate(state_dir=Path(tmp) / "state", detector=low_vram_detector, clock=clock)
            runtime = WorkerRuntime(
                vram_gate=gate,
                registry=WorkerRegistry(workers_dir=workers_dir),
                disk_checker=FakeDiskChecker(),
                clock=clock,
            )

            # Without allow_cpu_fallback -> REJECTED with VRAM_INSUFFICIENT
            denied = runtime.run_worker(
                "fallback_worker",
                allow_cpu_fallback=False,
                extra_env=self._worker_env_with_src(),
            )
            self.assertEqual(denied.status, "REJECTED")
            self.assertIn(ErrorCode.VRAM_INSUFFICIENT, denied.error_codes)

            # With allow_cpu_fallback=True -> degrades cleanly to CPU_COMPUTE
            degraded = runtime.run_worker(
                "fallback_worker",
                allow_cpu_fallback=True,
                extra_env=self._worker_env_with_src(),
            )
            self.assertTrue(degraded.succeeded, f"stderr={degraded.stderr}")
            self.assertTrue(degraded.used_cpu_fallback)
            self.assertEqual(degraded.output_data["execution_mode"], "cpu_fallback")
            self.assertEqual(len(gate.read_registry_snapshot().active_leases()), 0)

    # ------------------------------------------------------------------
    # T9: Python SDK (LocalGpuClient) & `lgpu run --json` CLI contract
    # ------------------------------------------------------------------
    def test_t9_python_sdk_and_cli_run_json_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workers_dir = Path(tmp) / "workers"
            state_dir = Path(tmp) / "state"
            script = (
                "from localgpu_os.sdk import WorkerContext\n"
                "ctx = WorkerContext.from_env()\n"
                "ctx.write_result({'hello': 'sdk'})\n"
            )
            self._provision_worker(
                workers_dir,
                "cpu_sdk_worker",
                script,
                manifest_overrides={
                    "resources": [{"resource_class": "CPU_COMPUTE", "amount": 1, "unit": "threads"}]
                },
            )

            client = LocalGpuClient(
                workers_dir=workers_dir,
                state_dir=state_dir,
                disk_checker=FakeDiskChecker(),
            )
            listed = client.list_workers()
            self.assertEqual([w.worker_id for w in listed], ["cpu_sdk_worker"])

            sdk_res = client.run_worker("cpu_sdk_worker", extra_env=self._worker_env_with_src())
            self.assertTrue(sdk_res.succeeded, f"stderr={sdk_res.stderr}")
            self.assertEqual(sdk_res.output_data, {"hello": "sdk"})

            # CLI `lgpu run cpu_sdk_worker --json`
            buf = io.StringIO()
            with patch.dict(os.environ, self._worker_env_with_src()):
                with redirect_stdout(buf):
                    rc = cli_main(
                        [
                            "run",
                            "cpu_sdk_worker",
                            "--workers-dir",
                            str(workers_dir),
                            "--state-dir",
                            str(state_dir),
                            "--json",
                        ]
                    )
            self.assertEqual(rc, 0, f"cli output={buf.getvalue()}")
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["schema_version"], "1.0")
            self.assertEqual(payload["status"], "SUCCEEDED")
            self.assertEqual(payload["worker_id"], "cpu_sdk_worker")
            self.assertEqual(payload["output_data"], {"hello": "sdk"})

    # ------------------------------------------------------------------
    # T10 (Check 1): Lease released when subprocess.Popen spawn fails
    # ------------------------------------------------------------------
    def test_t10_lease_released_when_process_spawn_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workers_dir = Path(tmp) / "workers"
            self._provision_worker(workers_dir, "spawn_fail_worker", "print('never')\n")
            clock = FakeClock()
            gate = VramGate(state_dir=Path(tmp) / "state", detector=FakeDetector(clock), clock=clock)
            runtime = WorkerRuntime(
                vram_gate=gate,
                registry=WorkerRegistry(workers_dir=workers_dir),
                disk_checker=FakeDiskChecker(),
                clock=clock,
            )
            with patch("localgpu_os.application.worker_runtime.subprocess.Popen", side_effect=OSError("spawn blocked")):
                res = runtime.run_worker("spawn_fail_worker")

            self.assertEqual(res.status, "FAILED")
            self.assertIn(ErrorCode.WORKER_EXECUTION_FAILED, res.error_codes)
            self.assertIsNotNone(res.lease_id)
            self.assertEqual(len(gate.read_registry_snapshot().active_leases()), 0)

    # ------------------------------------------------------------------
    # T11 (Check 2): CPU fallback holds ZERO GPU lease & enforces CPU quota + backend
    # ------------------------------------------------------------------
    def test_t11_cpu_fallback_holds_no_gpu_lease_and_uses_cpu_backend(self) -> None:
        from localgpu_os.domain.compatibility import ResourceCapacity

        with tempfile.TemporaryDirectory() as tmp:
            workers_dir = Path(tmp) / "workers"
            state_dir = Path(tmp) / "state"
            inspect_script = (
                "import json\n"
                "from localgpu_os.sdk import WorkerContext\n"
                "ctx = WorkerContext.from_env()\n"
                "reg = json.loads((ctx.state_dir / 'leases.json').read_text(encoding='utf-8'))\n"
                "ctx.write_result({\n"
                "    'mode': ctx.execution_mode,\n"
                "    'backend': ctx.compute_backend,\n"
                "    'active_leases': reg['leases'],\n"
                "})\n"
            )
            self._provision_worker(workers_dir, "fb_check_worker", inspect_script)

            clock = FakeClock()
            low_vram_detector = FakeDetector(clock, available_vram_mib=400)
            gate = VramGate(
                state_dir=state_dir,
                detector=low_vram_detector,
                clock=clock,
                capacity=ResourceCapacity(
                    cpu_threads_total=4,
                    system_memory_mib_available=4096,
                    gpu_compute_slots=1,
                ),
            )
            runtime = WorkerRuntime(
                vram_gate=gate,
                registry=WorkerRegistry(workers_dir=workers_dir),
                disk_checker=FakeDiskChecker(),
                clock=clock,
            )
            res = runtime.run_worker(
                "fb_check_worker",
                allow_cpu_fallback=True,
                extra_env=self._worker_env_with_src(),
            )
            self.assertTrue(res.succeeded, f"stderr={res.stderr}")
            self.assertTrue(res.used_cpu_fallback)
            self.assertEqual(res.output_data["mode"], "cpu_fallback")
            self.assertEqual(res.output_data["backend"], "cpu_int8")
            active = res.output_data["active_leases"]
            self.assertEqual(len(active), 1)
            self.assertEqual(active[0]["device_id"], "cpu")
            self.assertEqual(active[0]["reserved_vram_mib"], 0)
            req_classes = {r["resource_class"] for r in active[0]["requests"]}
            self.assertEqual(req_classes, {"CPU_COMPUTE", "SYSTEM_MEMORY"})
            self.assertEqual(len(gate.read_registry_snapshot().active_leases()), 0)

            # When CPU quota is also insufficient -> rejected with zero leases left
            tight_gate = VramGate(
                state_dir=Path(tmp) / "state_tight",
                detector=low_vram_detector,
                clock=clock,
                capacity=ResourceCapacity(
                    cpu_threads_total=1,  # worker fallback asks for 2 threads
                    system_memory_mib_available=4096,
                    gpu_compute_slots=1,
                ),
            )
            tight_rt = WorkerRuntime(
                vram_gate=tight_gate,
                registry=WorkerRegistry(workers_dir=workers_dir),
                disk_checker=FakeDiskChecker(),
                clock=clock,
            )
            denied = tight_rt.run_worker("fb_check_worker", allow_cpu_fallback=True)
            self.assertEqual(denied.status, "REJECTED")
            self.assertEqual(len(tight_gate.read_registry_snapshot().active_leases()), 0)

    # ------------------------------------------------------------------
    # T12 (Check 3): Symlink / junction escape blocked by resolved root check
    # ------------------------------------------------------------------
    def test_t12_symlink_or_junction_escape_blocked(self) -> None:
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "allowed"
            root.mkdir()
            outside_dir = Path(tmp) / "outside"
            outside_dir.mkdir()
            secret = outside_dir / "secret.pdf"
            secret.write_text("confidential", encoding="utf-8")

            link_dir = root / "linked_outside"
            if os.name == "nt":
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(link_dir), str(outside_dir)],
                    check=True,
                    capture_output=True,
                )
            else:
                os.symlink(outside_dir, link_dir)

            limits = WorkerFileLimits(max_file_size_bytes=1024 * 1024, min_free_disk_mib=64)
            with self.assertRaises(WorkerManifestInvalidError) as ctx:
                validate_json_file_manifest(
                    {"files": [{"path": "linked_outside/secret.pdf"}]},
                    allowed_root=root,
                    file_limits=limits,
                )
            self.assertEqual(ctx.exception.error_code, ErrorCode.WORKER_MANIFEST_INVALID)

    # ------------------------------------------------------------------
    # T13 (Check 4): Cancel vs normal completion race preserves SUCCEEDED
    # ------------------------------------------------------------------
    def test_t13_cancel_race_with_normal_completion_preserves_succeeded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workers_dir = Path(tmp) / "workers"
            # Worker ignores cancel, writes normal result, releases lease cleanly, exits 0
            fast_script = (
                "import time\n"
                "from localgpu_os.sdk import WorkerContext\n"
                "ctx = WorkerContext.from_env()\n"
                "time.sleep(0.12)\n"
                "ctx.write_result({'status': 'OK', 'done': True})\n"
                "ctx.release_lease()\n"
            )
            self._provision_worker(
                workers_dir,
                "race_worker",
                fast_script,
                manifest_overrides={"cancel_grace_seconds": 3.0},
            )
            clock = FakeClock()
            gate = VramGate(state_dir=Path(tmp) / "state", detector=FakeDetector(clock), clock=clock)
            runtime = WorkerRuntime(
                vram_gate=gate,
                registry=WorkerRegistry(workers_dir=workers_dir),
                disk_checker=FakeDiskChecker(),
                clock=clock,
            )
            cancel_event = threading.Event()
            cancel_event.set()  # Trigger cancel immediately as worker starts

            res = runtime.run_worker(
                "race_worker",
                cancel_event=cancel_event,
                extra_env=self._worker_env_with_src(),
            )
            self.assertEqual(res.status, "SUCCEEDED", f"warnings={res.warnings}")
            self.assertEqual(res.output_data, {"status": "OK", "done": True})
            self.assertFalse(res.forced_kill)
            self.assertEqual(len(gate.read_registry_snapshot().active_leases()), 0)

    # ------------------------------------------------------------------
    # T14 (Check 5): Post-kill VRAM re-probe failure stays Fail-Closed
    # ------------------------------------------------------------------
    def test_t14_forced_kill_reprobe_failure_stays_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workers_dir = Path(tmp) / "workers"
            ready_marker = Path(tmp) / "reprobe_ready.flag"
            stubborn_script = (
                "from pathlib import Path\n"
                "import time\n"
                f"Path({str(ready_marker)!r}).write_text('ready', encoding='utf-8')\n"
                "time.sleep(30)\n"
            )
            self._provision_worker(
                workers_dir,
                "reprobe_fail_worker",
                stubborn_script,
                manifest_overrides={"cancel_grace_seconds": 0.3},
            )
            clock = FakeClock()
            detector = FakeDetector(clock, available_vram_mib=6000)
            gate = VramGate(state_dir=Path(tmp) / "state", detector=detector, clock=clock)
            runtime = WorkerRuntime(
                vram_gate=gate,
                registry=WorkerRegistry(workers_dir=workers_dir),
                disk_checker=FakeDiskChecker(),
                clock=clock,
            )
            cancel_event = threading.Event()

            def _trigger_and_break_probe() -> None:
                deadline = time.monotonic() + 5.0
                while not ready_marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                # Break VRAM probe before forced kill triggers Step 7 re-probe
                detector.available_vram_mib = None
                cancel_event.set()

            t = threading.Thread(target=_trigger_and_break_probe, daemon=True)
            t.start()
            res = runtime.run_worker(
                "reprobe_fail_worker",
                cancel_event=cancel_event,
                extra_env=self._worker_env_with_src(),
            )
            t.join(timeout=5.0)

            self.assertEqual(res.status, "CANCELLED")
            self.assertTrue(res.forced_kill)
            self.assertFalse(res.post_kill_vram_reprobed)
            self.assertIn(ErrorCode.HARDWARE_UNVERIFIED, res.error_codes)

    # ------------------------------------------------------------------
    # T15 (Check 6): Orphan worker recovery after Core OS crash & restart
    # ------------------------------------------------------------------
    def test_t15_orphan_worker_recovery_blocks_reclaim_until_stopped(self) -> None:
        import subprocess
        from localgpu_os.domain.lease import ResourceRequest

        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "state"
            clock1 = FakeClock(monotonic_val=500.0)
            gate1 = VramGate(state_dir=state_dir, detector=FakeDetector(clock1), clock=clock1)

            # Simulate Core OS acquiring lease & spawning a real worker, then crashing
            decision = gate1.acquire_lease(
                owner_id="orphan_worker",
                requests=[
                    ResourceRequest(ResourceClass.GPU_COMPUTE, 1, exclusive=True),
                    ResourceRequest(ResourceClass.GPU_MEMORY, 1024),
                ],
                ttl_seconds=5.0,
            )
            self.assertTrue(decision.admitted and decision.lease is not None)
            orphan = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(15)"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
            )
            try:
                ctime = gate1.process_inspector.get_process_create_time(orphan.pid)
                gate1.attach_process(
                    decision.lease.lease_id,
                    decision.lease.lease_generation,
                    pid=orphan.pid,
                    pid_create_time=ctime,
                )

                # Simulate Core OS crash & restart (fresh VramGate, TTL expired)
                clock2 = FakeClock(monotonic_val=600.0)
                gate2 = VramGate(state_dir=state_dir, detector=FakeDetector(clock2), clock=clock2)

                # While orphan is alive, conflicting GPU admission MUST be blocked
                blocked = gate2.acquire_lease(
                    owner_id="new_worker",
                    requests=[
                        ResourceRequest(ResourceClass.GPU_COMPUTE, 1, exclusive=True),
                        ResourceRequest(ResourceClass.GPU_MEMORY, 1024),
                    ],
                )
                self.assertFalse(blocked.admitted)
                self.assertIn(ErrorCode.LEASE_RECLAIM_BLOCKED, blocked.error_codes)
            finally:
                orphan.kill()
                orphan.wait(timeout=5.0)

            # Once orphan has stopped, restarted Core OS safely reclaims & admits new work
            recovered = gate2.acquire_lease(
                owner_id="new_worker",
                requests=[
                    ResourceRequest(ResourceClass.GPU_COMPUTE, 1, exclusive=True),
                    ResourceRequest(ResourceClass.GPU_MEMORY, 1024),
                ],
            )
            self.assertTrue(recovered.admitted)
            gate2.release_lease(recovered.lease.lease_id, recovered.lease.lease_generation)


if __name__ == "__main__":
    unittest.main()
