"""Public Python SDK for LocalGPU-OS (Core Client & Worker-side 7-Step Cancel Helper)."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import threading
from typing import Any, Sequence

from localgpu_os.application.doctor_service import DoctorService
from localgpu_os.application.status_service import StatusReport, StatusService
from localgpu_os.application.vram_gate import VramGate
from localgpu_os.application.worker_runtime import WorkerRuntime
from localgpu_os.domain.hardware import DoctorReport
from localgpu_os.domain.lease import FencingTokenMismatchError
from localgpu_os.domain.worker import CancelSignal, WorkerManifest, WorkerRunResult
from localgpu_os.infrastructure.cancel_protocol import (
    read_cancel_signal,
    write_json_atomic,
    write_worker_cancelled_result,
)
from localgpu_os.infrastructure.clock import ClockProtocol
from localgpu_os.infrastructure.disk_space import DiskSpaceCheckerProtocol
from localgpu_os.infrastructure.eviction_provider import EvictionProvider
from localgpu_os.infrastructure.nvidia_detector import HardwareDetectorProtocol
from localgpu_os.infrastructure.worker_registry import WorkerRegistry


class LocalGpuClient:
    """High-level Python SDK facade for LocalGPU-OS hardware, arbitration, and worker runtime."""

    def __init__(
        self,
        workers_dir: Path | None = None,
        state_dir: Path | None = None,
        work_root: Path | None = None,
        detector: HardwareDetectorProtocol | None = None,
        clock: ClockProtocol | None = None,
        disk_checker: DiskSpaceCheckerProtocol | None = None,
        eviction_provider: EvictionProvider | None = None,
    ) -> None:
        self._detector = detector
        self._clock = clock
        self._vram_gate = VramGate(
            state_dir=state_dir,
            detector=detector,
            clock=clock,
            eviction_provider=eviction_provider,
        )
        self._registry = WorkerRegistry(workers_dir=workers_dir)
        self._runtime = WorkerRuntime(
            vram_gate=self._vram_gate,
            registry=self._registry,
            disk_checker=disk_checker,
            clock=clock,
            work_root=work_root,
        )

    @property
    def vram_gate(self) -> VramGate:
        return self._vram_gate

    @property
    def registry(self) -> WorkerRegistry:
        return self._registry

    @property
    def runtime(self) -> WorkerRuntime:
        return self._runtime

    def doctor(self, device_index: int = 0) -> DoctorReport:
        return DoctorService(detector=self._detector, clock=self._clock).run_doctor(
            device_index=device_index
        )

    def status(self, device_index: int = 0) -> StatusReport:
        return StatusService(
            state_dir=self._vram_gate.state_dir,
            detector=self._detector,
            clock=self._clock,
        ).run_status(device_index=device_index)

    def list_workers(self) -> tuple[WorkerManifest, ...]:
        return self._registry.list_workers()

    def run_worker(
        self,
        worker_id: str,
        extra_args: Sequence[str] = (),
        input_manifest: dict[str, Any] | Path | None = None,
        allowed_root: Path | None = None,
        work_dir: Path | None = None,
        device_index: int = 0,
        allow_eviction: bool = False,
        allow_cpu_fallback: bool = False,
        ttl_seconds: float = 60.0,
        timeout_seconds: float | None = None,
        cancel_event: threading.Event | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> WorkerRunResult:
        return self._runtime.run_worker(
            worker_id=worker_id,
            extra_args=extra_args,
            input_manifest=input_manifest,
            allowed_root=allowed_root,
            work_dir=work_dir,
            device_index=device_index,
            allow_eviction=allow_eviction,
            allow_cpu_fallback=allow_cpu_fallback,
            ttl_seconds=ttl_seconds,
            timeout_seconds=timeout_seconds,
            cancel_event=cancel_event,
            extra_env=extra_env,
        )


@dataclass(frozen=True)
class WorkerContext:
    """Worker-side runtime context for cooperative 7-step cancel protocol & result emission."""

    run_id: str
    worker_id: str
    lease_id: str
    lease_generation: int
    state_dir: Path
    work_dir: Path
    temp_dir: Path
    cancel_file: Path
    result_file: Path
    network_policy: str
    execution_mode: str
    compute_backend: str
    poll_interval_seconds: float
    input_manifest_file: Path | None = None

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> WorkerContext:
        source = env if env is not None else os.environ
        input_manifest_raw = source.get("LOCALGPU_INPUT_MANIFEST")
        return cls(
            run_id=source["LOCALGPU_RUN_ID"],
            worker_id=source["LOCALGPU_WORKER_ID"],
            lease_id=source["LOCALGPU_LEASE_ID"],
            lease_generation=int(source["LOCALGPU_LEASE_GENERATION"]),
            state_dir=Path(source["LOCALGPU_STATE_DIR"]),
            work_dir=Path(source["LOCALGPU_WORK_DIR"]),
            temp_dir=Path(source["LOCALGPU_TEMP_DIR"]),
            cancel_file=Path(source["LOCALGPU_CANCEL_FILE"]),
            result_file=Path(source["LOCALGPU_RESULT_FILE"]),
            network_policy=source.get("LOCALGPU_NETWORK_POLICY", "RUNTIME_OFFLINE"),
            execution_mode=source.get("LOCALGPU_EXECUTION_MODE", "primary"),
            compute_backend=source.get("LOCALGPU_COMPUTE_BACKEND", "cpu_int8"),
            poll_interval_seconds=float(source.get("LOCALGPU_POLL_INTERVAL_SECONDS", "0.25")),
            input_manifest_file=Path(input_manifest_raw) if input_manifest_raw else None,
        )

    def check_cancelled(self) -> CancelSignal | None:
        """Step 2: Poll cancel control file without blocking."""
        return read_cancel_signal(self.cancel_file)

    def cleanup_temp_dir(self) -> None:
        """Step 3: Remove worker temporary files inside temp_dir."""
        if self.temp_dir.exists():
            for child in self.temp_dir.iterdir():
                try:
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink(missing_ok=True)
                except OSError:
                    pass

    def release_lease(self) -> bool:
        """Step 4: Release VramLease carrying lease_generation fencing token."""
        gate = VramGate(state_dir=self.state_dir)
        try:
            gate.release_lease(self.lease_id, self.lease_generation)
            return True
        except FencingTokenMismatchError:
            return False

    def write_cancelled(self, details: dict[str, Any] | None = None) -> None:
        """Step 5: Persist canonical WORKER_CANCELLED status to result_file."""
        write_worker_cancelled_result(
            result_file=self.result_file,
            run_id=self.run_id,
            worker_id=self.worker_id,
            lease_id=self.lease_id,
            lease_generation=self.lease_generation,
            details=details,
        )

    def write_result(self, payload: dict[str, Any]) -> None:
        """Atomically write structured JSON result for Core OS consumption."""
        write_json_atomic(self.result_file, payload)

    def load_input_manifest(self) -> dict[str, Any] | None:
        """Read validated json_file_manifest provided by Core OS."""
        if self.input_manifest_file is None or not self.input_manifest_file.exists():
            return None
        return json.loads(self.input_manifest_file.read_text(encoding="utf-8"))
