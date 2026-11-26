"""WorkerRuntime: isolated .venv execution, disk/manifest guards, and 7-step cancel protocol."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Sequence
import uuid

from localgpu_os.application.vram_gate import VramGate
from localgpu_os.domain.enums import ErrorCode, NetworkPolicy
from localgpu_os.domain.lease import FencingTokenMismatchError, VramLease
from localgpu_os.domain.worker import (
    CancelSignal,
    DiskSpaceInsufficientError,
    WorkerEnvNotInstalledError,
    WorkerManifest,
    WorkerManifestInvalidError,
    WorkerRunResult,
    validate_json_file_manifest,
)
from localgpu_os.infrastructure.cancel_protocol import (
    write_cancel_signal_atomic,
    write_json_atomic,
)
from localgpu_os.infrastructure.clock import ClockProtocol, SystemClock
from localgpu_os.infrastructure.disk_space import (
    DiskSpaceCheckerProtocol,
    ShutilDiskSpaceChecker,
)
from localgpu_os.infrastructure.worker_registry import WorkerRegistry


class WorkerRuntime:
    """Executes isolated workers under VramGate arbitration and 7-step cancel protocol."""

    def __init__(
        self,
        vram_gate: VramGate | None = None,
        registry: WorkerRegistry | None = None,
        disk_checker: DiskSpaceCheckerProtocol | None = None,
        clock: ClockProtocol | None = None,
        work_root: Path | None = None,
    ) -> None:
        self._clock = clock or SystemClock()
        self._vram_gate = vram_gate or VramGate(clock=self._clock)
        self._registry = registry or WorkerRegistry()
        self._disk_checker = disk_checker or ShutilDiskSpaceChecker()
        self._work_root = (
            Path(work_root)
            if work_root is not None
            else (self._vram_gate.state_dir / "runs")
        )

    @property
    def vram_gate(self) -> VramGate:
        return self._vram_gate

    @property
    def registry(self) -> WorkerRegistry:
        return self._registry

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
        """Run a registered worker inside its isolated .venv with full arbitration & cleanup."""
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        started_at = self._clock.utc_now_iso()

        # 1. Load & validate WorkerManifest v1.0
        try:
            manifest = self._registry.load_manifest(worker_id)
        except WorkerManifestInvalidError as exc:
            return WorkerRunResult(
                run_id=run_id,
                worker_id=str(worker_id),
                status="REJECTED",
                exit_code=None,
                lease_id=None,
                lease_generation=None,
                network_policy=NetworkPolicy.RUNTIME_OFFLINE,
                error_codes=(exc.error_code,),
                warnings=(str(exc),),
                started_at=started_at,
                finished_at=self._clock.utc_now_iso(),
            )

        # 2. Strictly resolve isolated .venv Python executable (never PATH)
        try:
            python_exe = self._registry.resolve_venv_python(manifest.worker_id)
        except WorkerEnvNotInstalledError as exc:
            return WorkerRunResult(
                run_id=run_id,
                worker_id=manifest.worker_id,
                status="REJECTED",
                exit_code=None,
                lease_id=None,
                lease_generation=None,
                network_policy=manifest.network_policy,
                error_codes=(exc.error_code,),
                warnings=(str(exc),),
                started_at=started_at,
                finished_at=self._clock.utc_now_iso(),
            )

        # 3. Validate optional json_file_manifest (Base64 ban, root jail, size limits)
        validated_manifest_payload: dict[str, Any] | None = None
        if input_manifest is not None:
            try:
                raw_payload = self._load_input_manifest_payload(input_manifest)
                root_dir = (
                    Path(allowed_root)
                    if allowed_root is not None
                    else (
                        input_manifest.parent
                        if isinstance(input_manifest, Path)
                        else Path.cwd()
                    )
                )
                entries = validate_json_file_manifest(
                    payload=raw_payload,
                    allowed_root=root_dir,
                    file_limits=manifest.file_limits,
                )
                validated_manifest_payload = {
                    "files": [e.to_dict() for e in entries],
                }
            except WorkerManifestInvalidError as exc:
                return WorkerRunResult(
                    run_id=run_id,
                    worker_id=manifest.worker_id,
                    status="REJECTED",
                    exit_code=None,
                    lease_id=None,
                    lease_generation=None,
                    network_policy=manifest.network_policy,
                    error_codes=(exc.error_code,),
                    warnings=(str(exc),),
                    started_at=started_at,
                    finished_at=self._clock.utc_now_iso(),
                )

        # 4. Check free disk space in work_dir & temp area (DISK_SPACE_INSUFFICIENT)
        run_dir = (Path(work_dir) if work_dir is not None else (self._work_root / run_id))
        try:
            self._check_disk_space(run_dir, manifest.file_limits.min_free_disk_mib)
            self._check_disk_space(
                Path(tempfile.gettempdir()), manifest.file_limits.min_free_disk_mib
            )
        except DiskSpaceInsufficientError as exc:
            return WorkerRunResult(
                run_id=run_id,
                worker_id=manifest.worker_id,
                status="REJECTED",
                exit_code=None,
                lease_id=None,
                lease_generation=None,
                network_policy=manifest.network_policy,
                error_codes=(exc.error_code,),
                warnings=(str(exc),),
                started_at=started_at,
                finished_at=self._clock.utc_now_iso(),
            )

        # 5. Acquire cooperative VramLease (all-or-nothing, with optional CPU fallback)
        used_cpu_fallback = False
        decision = self._vram_gate.acquire_lease(
            owner_id=f"{manifest.worker_id}:{run_id}",
            requests=manifest.resources,
            device_index=device_index,
            ttl_seconds=ttl_seconds,
            allow_eviction=allow_eviction,
        )
        warnings: list[str] = list(decision.warnings)

        if (
            not decision.admitted
            and allow_cpu_fallback
            and manifest.uses_gpu()
            and manifest.fallback_policy.cpu_supported
        ):
            fallback_decision = self._vram_gate.acquire_lease(
                owner_id=f"{manifest.worker_id}:{run_id}:cpu",
                requests=manifest.cpu_fallback_requests(),
                device_index=device_index,
                ttl_seconds=ttl_seconds,
                allow_eviction=False,
            )
            if fallback_decision.admitted:
                used_cpu_fallback = True
                warnings.append(
                    "GPU admission denied; safely degraded to CPU fallback "
                    f"({manifest.fallback_policy.cpu_backend.value}, "
                    f"threads={manifest.fallback_policy.max_cpu_threads})."
                )
                decision = fallback_decision
            else:
                warnings.extend(fallback_decision.warnings)

        if not decision.admitted or decision.lease is None:
            return WorkerRunResult(
                run_id=run_id,
                worker_id=manifest.worker_id,
                status="REJECTED",
                exit_code=None,
                lease_id=None,
                lease_generation=None,
                network_policy=manifest.network_policy,
                used_cpu_fallback=used_cpu_fallback,
                error_codes=decision.error_codes,
                warnings=tuple(warnings),
                started_at=started_at,
                finished_at=self._clock.utc_now_iso(),
            )

        lease: VramLease = decision.lease
        run_dir.mkdir(parents=True, exist_ok=True)
        cancel_file = run_dir / "cancel.json"
        result_file = run_dir / "result.json"
        temp_dir = run_dir / "tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        input_manifest_file: Path | None = None
        if validated_manifest_payload is not None:
            input_manifest_file = run_dir / "input_manifest.json"
            write_json_atomic(input_manifest_file, validated_manifest_payload)

        # 6. Build command & environment (ZERO shell=True)
        cmd = [
            str(python_exe),
            "-m",
            manifest.module,
            *manifest.args,
            *[str(a) for a in extra_args],
        ]
        env = self._build_worker_env(
            manifest=manifest,
            run_id=run_id,
            lease=lease,
            run_dir=run_dir,
            temp_dir=temp_dir,
            cancel_file=cancel_file,
            result_file=result_file,
            input_manifest_file=input_manifest_file,
            used_cpu_fallback=used_cpu_fallback,
            extra_env=extra_env,
        )

        # 7. Spawn isolated worker process and bind PID + pid_create_time to lease
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(self._registry.worker_dir(manifest.worker_id)),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
            )
        except OSError as exc:
            self._safe_release_lease(lease)
            return WorkerRunResult(
                run_id=run_id,
                worker_id=manifest.worker_id,
                status="FAILED",
                exit_code=None,
                lease_id=lease.lease_id,
                lease_generation=lease.lease_generation,
                network_policy=manifest.network_policy,
                used_cpu_fallback=used_cpu_fallback,
                error_codes=(ErrorCode.WORKER_EXECUTION_FAILED,),
                warnings=(*warnings, f"Failed to spawn worker process: {exc}"),
                started_at=started_at,
                finished_at=self._clock.utc_now_iso(),
            )

        pid_ctime = self._vram_gate.process_inspector.get_process_create_time(proc.pid)
        try:
            lease = self._vram_gate.attach_process(
                lease_id=lease.lease_id,
                lease_generation=lease.lease_generation,
                pid=proc.pid,
                pid_create_time=pid_ctime,
                transition_to_starting=True,
            )
        except FencingTokenMismatchError:
            pass

        # 8. Supervise execution, heartbeat, timeout, and 7-step cancel protocol
        effective_timeout = (
            timeout_seconds if timeout_seconds is not None else manifest.timeout_seconds
        )
        poll_interval = min(max(manifest.poll_interval_seconds, 0.02), 1.0)
        start_mono = time.monotonic()
        last_hb_mono = start_mono
        hb_interval = max(0.5, ttl_seconds * 0.4)

        cancel_requested = False
        cancel_reason = ""
        cancel_sent_mono: float | None = None
        forced_kill = False
        post_kill_vram_reprobed = False
        timed_out = False

        while True:
            ret = proc.poll()
            if ret is not None:
                break

            now_mono = time.monotonic()

            # Heartbeat renewal while running
            if not cancel_requested and (now_mono - last_hb_mono) >= hb_interval:
                try:
                    self._vram_gate.heartbeat(lease.lease_id, lease.lease_generation)
                except FencingTokenMismatchError:
                    pass
                last_hb_mono = now_mono

            # Check external cancellation trigger or timeout
            if not cancel_requested:
                if cancel_event is not None and cancel_event.is_set():
                    cancel_requested = True
                    cancel_reason = "external_cancel_requested"
                elif (now_mono - start_mono) >= effective_timeout:
                    cancel_requested = True
                    timed_out = True
                    cancel_reason = f"timeout_exceeded_{effective_timeout}s"

                if cancel_requested:
                    # Step 1: Write atomic cancel control file
                    signal = CancelSignal(
                        run_id=run_id,
                        worker_id=manifest.worker_id,
                        lease_id=lease.lease_id,
                        lease_generation=lease.lease_generation,
                        reason=cancel_reason,
                        requested_at=self._clock.utc_now_iso(),
                        grace_seconds=manifest.cancel_grace_seconds,
                    )
                    write_cancel_signal_atomic(cancel_file, signal)
                    cancel_sent_mono = now_mono

            # Step 7: Enforce cancel_grace_seconds -> force kill + cleanup + VRAM re-probe
            if cancel_requested and cancel_sent_mono is not None:
                if (now_mono - cancel_sent_mono) >= manifest.cancel_grace_seconds:
                    forced_kill = True
                    try:
                        proc.kill()
                    except OSError:
                        pass
                    try:
                        proc.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        pass
                    break

            time.sleep(poll_interval)

        # Collect stdout/stderr safely
        try:
            stdout_text, stderr_text = proc.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass
            stdout_text, stderr_text = proc.communicate()

        exit_code = proc.returncode

        # Step 7 post-kill cleanup & mandatory VRAM re-probe before releasing GPU lease
        extra_errors: list[ErrorCode] = []
        process_confirmed_stopped = proc.poll() is not None
        if forced_kill:
            self._cleanup_temp_dir(temp_dir)
            if not process_confirmed_stopped:
                extra_errors.append(ErrorCode.LEASE_RECLAIM_BLOCKED)
                warnings.append(
                    "Worker process could not be confirmed stopped after kill; "
                    "retaining lease to prevent conflicting GPU admission."
                )
            elif lease.uses_gpu():
                try:
                    _, reprobe_snap = self._vram_gate.reprobe_hardware(device_index=device_index)
                    if (
                        reprobe_snap.is_fresh(self._clock.monotonic())
                        and reprobe_snap.available_vram_mib is not None
                        and reprobe_snap.available_vram_mib >= 0
                    ):
                        post_kill_vram_reprobed = True
                    else:
                        extra_errors.append(ErrorCode.HARDWARE_UNVERIFIED)
                        warnings.append(
                            "Post-kill VRAM re-probe returned stale or unavailable VRAM; "
                            "maintaining Fail-Closed error state."
                        )
                except Exception as exc:  # pragma: no cover - defensive guard
                    extra_errors.append(ErrorCode.HARDWARE_UNVERIFIED)
                    warnings.append(f"Post-kill VRAM re-probe error: {exc}")

        # Release lease only when process is confirmed stopped (tolerant if worker already released)
        if process_confirmed_stopped:
            self._safe_release_lease(lease)

        output_data = self._read_result_file(result_file)
        finished_at = self._clock.utc_now_iso()

        # Determine canonical status & ErrorCode (handle cancel vs normal completion race)
        worker_reported_cancelled = (
            isinstance(output_data, dict)
            and output_data.get("status") == ErrorCode.WORKER_CANCELLED.value
        )
        raced_with_normal_completion = (
            cancel_requested
            and not timed_out
            and not forced_kill
            and exit_code == 0
            and not worker_reported_cancelled
            and isinstance(output_data, dict)
        )

        if timed_out:
            warnings.append(f"Worker exceeded timeout of {effective_timeout}s.")
            err_codes = tuple(dict.fromkeys([ErrorCode.WORKER_EXECUTION_FAILED, *extra_errors]))
            return WorkerRunResult(
                run_id=run_id,
                worker_id=manifest.worker_id,
                status="FAILED",
                exit_code=exit_code,
                lease_id=lease.lease_id,
                lease_generation=lease.lease_generation,
                network_policy=manifest.network_policy,
                used_cpu_fallback=used_cpu_fallback,
                forced_kill=forced_kill,
                post_kill_vram_reprobed=post_kill_vram_reprobed,
                output_data=output_data,
                stdout=stdout_text,
                stderr=stderr_text,
                error_codes=err_codes,
                warnings=tuple(warnings),
                started_at=started_at,
                finished_at=finished_at,
            )

        if raced_with_normal_completion:
            warnings.append(
                "Cancel signal raced with normal worker completion; preserving SUCCEEDED result."
            )
            return WorkerRunResult(
                run_id=run_id,
                worker_id=manifest.worker_id,
                status="SUCCEEDED",
                exit_code=exit_code,
                lease_id=lease.lease_id,
                lease_generation=lease.lease_generation,
                network_policy=manifest.network_policy,
                used_cpu_fallback=used_cpu_fallback,
                forced_kill=False,
                post_kill_vram_reprobed=False,
                output_data=output_data,
                stdout=stdout_text,
                stderr=stderr_text,
                error_codes=tuple(extra_errors),
                warnings=tuple(warnings),
                started_at=started_at,
                finished_at=finished_at,
            )

        if cancel_requested or worker_reported_cancelled:
            if forced_kill:
                warnings.append(
                    f"Worker did not exit within cancel_grace_seconds ({manifest.cancel_grace_seconds}s) "
                    "and was forcefully terminated."
                )
            err_codes = tuple(dict.fromkeys([ErrorCode.WORKER_CANCELLED, *extra_errors]))
            return WorkerRunResult(
                run_id=run_id,
                worker_id=manifest.worker_id,
                status="CANCELLED",
                exit_code=exit_code,
                lease_id=lease.lease_id,
                lease_generation=lease.lease_generation,
                network_policy=manifest.network_policy,
                used_cpu_fallback=used_cpu_fallback,
                forced_kill=forced_kill,
                post_kill_vram_reprobed=post_kill_vram_reprobed,
                output_data=output_data,
                stdout=stdout_text,
                stderr=stderr_text,
                error_codes=err_codes,
                warnings=tuple(warnings),
                started_at=started_at,
                finished_at=finished_at,
            )

        if exit_code == 0:
            return WorkerRunResult(
                run_id=run_id,
                worker_id=manifest.worker_id,
                status="SUCCEEDED",
                exit_code=exit_code,
                lease_id=lease.lease_id,
                lease_generation=lease.lease_generation,
                network_policy=manifest.network_policy,
                used_cpu_fallback=used_cpu_fallback,
                forced_kill=False,
                post_kill_vram_reprobed=False,
                output_data=output_data,
                stdout=stdout_text,
                stderr=stderr_text,
                error_codes=(),
                warnings=tuple(warnings),
                started_at=started_at,
                finished_at=finished_at,
            )

        return WorkerRunResult(
            run_id=run_id,
            worker_id=manifest.worker_id,
            status="FAILED",
            exit_code=exit_code,
            lease_id=lease.lease_id,
            lease_generation=lease.lease_generation,
            network_policy=manifest.network_policy,
            used_cpu_fallback=used_cpu_fallback,
            forced_kill=False,
            post_kill_vram_reprobed=False,
            output_data=output_data,
            stdout=stdout_text,
            stderr=stderr_text,
            error_codes=(ErrorCode.WORKER_EXECUTION_FAILED,),
            warnings=tuple(warnings),
            started_at=started_at,
            finished_at=finished_at,
        )

    def _check_disk_space(self, target_path: Path, min_free_disk_mib: int) -> None:
        if min_free_disk_mib <= 0:
            return
        free_mib = self._disk_checker.free_disk_mib(target_path)
        if free_mib < min_free_disk_mib:
            raise DiskSpaceInsufficientError(
                f"Insufficient disk space at {target_path}: "
                f"{free_mib} MiB free < required {min_free_disk_mib} MiB."
            )

    @staticmethod
    def _load_input_manifest_payload(input_manifest: dict[str, Any] | Path) -> Any:
        if isinstance(input_manifest, dict):
            return input_manifest
        path = Path(input_manifest)
        if not path.exists() or not path.is_file():
            raise WorkerManifestInvalidError(f"Input manifest file not found: {path}")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise WorkerManifestInvalidError(
                f"Failed to parse input manifest {path}: {exc}"
            ) from exc

    def _build_worker_env(
        self,
        manifest: WorkerManifest,
        run_id: str,
        lease: VramLease,
        run_dir: Path,
        temp_dir: Path,
        cancel_file: Path,
        result_file: Path,
        input_manifest_file: Path | None,
        used_cpu_fallback: bool,
        extra_env: dict[str, str] | None,
    ) -> dict[str, str]:
        env = os.environ.copy()
        compute_backend = (
            manifest.fallback_policy.cpu_backend.value
            if (used_cpu_fallback or not lease.uses_gpu())
            else "cuda"
        )
        env.update(
            {
                "LOCALGPU_RUN_ID": run_id,
                "LOCALGPU_WORKER_ID": manifest.worker_id,
                "LOCALGPU_LEASE_ID": lease.lease_id,
                "LOCALGPU_LEASE_GENERATION": str(lease.lease_generation),
                "LOCALGPU_STATE_DIR": str(self._vram_gate.state_dir),
                "LOCALGPU_WORK_DIR": str(run_dir),
                "LOCALGPU_TEMP_DIR": str(temp_dir),
                "LOCALGPU_CANCEL_FILE": str(cancel_file),
                "LOCALGPU_RESULT_FILE": str(result_file),
                "LOCALGPU_NETWORK_POLICY": manifest.network_policy.value,
                "LOCALGPU_EXECUTION_MODE": "cpu_fallback" if used_cpu_fallback else "primary",
                "LOCALGPU_COMPUTE_BACKEND": compute_backend,
                "LOCALGPU_MAX_CPU_THREADS": str(manifest.fallback_policy.max_cpu_threads),
                "LOCALGPU_MAX_SYSTEM_MEMORY_MIB": str(
                    manifest.fallback_policy.max_system_memory_mib
                ),
                "LOCALGPU_POLL_INTERVAL_SECONDS": str(manifest.poll_interval_seconds),
            }
        )
        if input_manifest_file is not None:
            env["LOCALGPU_INPUT_MANIFEST"] = str(input_manifest_file)
        if extra_env:
            for k, v in extra_env.items():
                env[str(k)] = str(v)
        return env

    def _safe_release_lease(self, lease: VramLease) -> None:
        try:
            self._vram_gate.release_lease(lease.lease_id, lease.lease_generation)
        except FencingTokenMismatchError:
            # Worker already released the lease cleanly with its fencing token.
            pass

    @staticmethod
    def _cleanup_temp_dir(temp_dir: Path) -> None:
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        except OSError:
            pass

    @staticmethod
    def _read_result_file(result_file: Path) -> dict[str, Any] | None:
        if not result_file.exists() or not result_file.is_file():
            return None
        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
