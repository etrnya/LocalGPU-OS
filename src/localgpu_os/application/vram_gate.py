"""VramGate: Three-Layer VRAM-Gate & All-or-Nothing Resource Arbitration (PRD 2.2-2.6)."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import os
from pathlib import Path
import uuid

from localgpu_os.domain.compatibility import ResourceCapacity, check_all_or_nothing
from localgpu_os.domain.enums import (
    ComputeBackend,
    ErrorCode,
    ReservationState,
    ResourceClass,
    ThermalState,
)
from localgpu_os.domain.hardware import (
    HardwareProfile,
    HardwareSnapshot,
    SafetyMarginConfig,
)
from localgpu_os.domain.lease import (
    AdmissionDecision,
    FencingTokenMismatchError,
    LeaseRegistryState,
    ResourceRequest,
    VramLease,
)
from localgpu_os.domain.thermal import ThermalGuard
from localgpu_os.infrastructure.clock import ClockProtocol, SystemClock
from localgpu_os.infrastructure.eviction_provider import EvictionProvider
from localgpu_os.infrastructure.lease_store import FileLeaseStore
from localgpu_os.infrastructure.nvidia_detector import (
    HardwareDetectorProtocol,
    NvidiaSmiDetector,
)
from localgpu_os.infrastructure.process_inspector import (
    OsProcessInspector,
    ProcessInspectorProtocol,
)
from localgpu_os.infrastructure.registry_lock import RegistryLock
from localgpu_os.infrastructure.system_memory import (
    get_available_system_memory_mib,
    get_host_cpu_threads,
)

_GPU_CLASSES = frozenset(
    {ResourceClass.GPU_COMPUTE, ResourceClass.GPU_MEMORY, ResourceClass.GPU_ENCODE}
)


def default_state_dir() -> Path:
    """Return default per-user runtime state directory for Lease Registry & Lock."""
    env_override = os.environ.get("LOCALGPU_OS_STATE_DIR")
    if env_override:
        return Path(env_override)
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home())
        return Path(base) / "LocalGPU-OS" / "state"
    xdg = os.environ.get("XDG_STATE_HOME")
    if xdg:
        return Path(xdg) / "localgpu-os"
    return Path.home() / ".local" / "state" / "localgpu-os"


def _parse_utc_iso(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        normalized = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


class VramGate:
    """Single-writer cooperative resource arbitration engine.

    Every Read-Modify-Write sequence on the Lease Registry executes inside
    a single RegistryLock acquisition.
    """

    def __init__(
        self,
        state_dir: Path | None = None,
        detector: HardwareDetectorProtocol | None = None,
        clock: ClockProtocol | None = None,
        safety_config: SafetyMarginConfig | None = None,
        thermal_guard: ThermalGuard | None = None,
        process_inspector: ProcessInspectorProtocol | None = None,
        eviction_provider: EvictionProvider | None = None,
        capacity: ResourceCapacity | None = None,
        lock_timeout_seconds: float = 5.0,
        vram_confirmation_ratio: float = 0.5,
    ) -> None:
        self._state_dir = Path(state_dir) if state_dir is not None else default_state_dir()
        self._clock = clock or SystemClock()
        self._detector = detector or NvidiaSmiDetector(clock=self._clock)
        self._safety = safety_config or SafetyMarginConfig()
        self._thermal_guard = thermal_guard or ThermalGuard()
        self._inspector = process_inspector or OsProcessInspector()
        self._eviction_provider = eviction_provider
        self._explicit_capacity = capacity
        self._lock_timeout = lock_timeout_seconds
        self._vram_confirmation_ratio = max(0.0, min(1.0, float(vram_confirmation_ratio)))

        self._lock_path = self._state_dir / "registry.lock"
        self._store = FileLeaseStore(self._state_dir / "leases.json")

    @property
    def state_dir(self) -> Path:
        return self._state_dir

    @property
    def store(self) -> FileLeaseStore:
        return self._store

    @property
    def process_inspector(self) -> ProcessInspectorProtocol:
        return self._inspector

    def reprobe_hardware(self, device_index: int = 0) -> tuple[HardwareProfile, HardwareSnapshot]:
        """Re-probe live GPU hardware state (used after forced kill in 7-step cancel Step 7)."""
        return self._detector.probe(device_index=device_index)

    def _new_lock(self) -> RegistryLock:
        return RegistryLock(self._lock_path, timeout_seconds=self._lock_timeout)

    def _resolve_capacity(self) -> ResourceCapacity:
        if self._explicit_capacity is not None:
            return self._explicit_capacity
        return ResourceCapacity(
            cpu_threads_total=get_host_cpu_threads(),
            system_memory_mib_available=get_available_system_memory_mib(),
            gpu_compute_slots=1,
            gpu_encode_sessions=0,
        )

    def _is_lease_expired(self, lease: VramLease, now_mono: float, now_utc_iso: str) -> bool:
        age_mono = now_mono - lease.heartbeat_monotonic
        if age_mono >= 0:
            return age_mono > lease.ttl_seconds
        # Monotonic clock reset (e.g., process/OS restart): fall back to UTC comparison
        hb_dt = _parse_utc_iso(lease.heartbeat_at)
        now_dt = _parse_utc_iso(now_utc_iso)
        if hb_dt is None or now_dt is None:
            # Cannot verify expiry reliably -> treat as expired candidate so inspector checks PID
            return True
        return (now_dt - hb_dt).total_seconds() > lease.ttl_seconds

    def _reconcile_expired_locked(
        self,
        state: LeaseRegistryState,
        now_mono: float,
        now_utc_iso: str,
    ) -> tuple[LeaseRegistryState, tuple[VramLease, ...], tuple[VramLease, ...]]:
        """Safely reclaim ONLY leases whose owner process is confirmed stopped.

        Returns:
            (updated_state, reclaimed_leases, reclaim_blocked_leases)
        """
        kept: list[VramLease] = []
        reclaimed: list[VramLease] = []
        blocked: list[VramLease] = []

        for lease in state.leases:
            if lease.reservation_state == ReservationState.RELEASED:
                continue

            if not self._is_lease_expired(lease, now_mono, now_utc_iso):
                kept.append(lease)
                continue

            # Heartbeat expired != worker dead! Cross-verify PID & create_time.
            if lease.pid is None:
                # No PID recorded -> cannot confirm worker stopped -> block reclaim!
                kept.append(lease)
                blocked.append(lease)
                continue

            alive = self._inspector.is_alive(lease.pid, lease.pid_create_time)
            if alive is False:
                reclaimed.append(
                    replace(lease, reservation_state=ReservationState.RELEASED)
                )
            else:
                # alive is True (still running) or None (unverified) -> NEVER reclaim!
                kept.append(lease)
                blocked.append(lease)

        if reclaimed:
            new_state = LeaseRegistryState(
                schema_version=state.schema_version,
                generation=state.generation,
                updated_at=now_utc_iso,
                leases=tuple(kept),
            )
            self._store.save(new_state)
            return new_state, tuple(reclaimed), tuple(blocked)

        return state, (), tuple(blocked)

    def reclaim_expired_leases(self) -> tuple[tuple[VramLease, ...], tuple[VramLease, ...]]:
        """Explicitly run safe stale-lease reconciliation under RegistryLock."""
        with self._new_lock():
            state = self._store.load()
            now_mono = self._clock.monotonic()
            now_utc = self._clock.utc_now_iso()
            _, reclaimed, blocked = self._reconcile_expired_locked(state, now_mono, now_utc)
            return reclaimed, blocked

    def acquire_lease(
        self,
        owner_id: str,
        requests: tuple[ResourceRequest, ...] | list[ResourceRequest],
        device_index: int = 0,
        ttl_seconds: float = 60.0,
        pid: int | None = None,
        pid_create_time: float | None = None,
        allow_eviction: bool = False,
    ) -> AdmissionDecision:
        """Atomically evaluate and grant (or reject) a multi-resource VramLease."""
        req_tuple = tuple(requests)
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive.")

        needs_gpu = any(r.resource_class in _GPU_CLASSES for r in req_tuple)
        req_gpu_mem_mib = sum(
            r.amount for r in req_tuple if r.resource_class == ResourceClass.GPU_MEMORY
        )
        safety_mib = self._safety.total_safety_mib

        with self._new_lock():
            state = self._store.load()
            now_mono = self._clock.monotonic()
            now_utc = self._clock.utc_now_iso()
            state, _, blocked_leases = self._reconcile_expired_locked(state, now_mono, now_utc)

            active_leases = state.active_leases()
            capacity = self._resolve_capacity()

            # Path A: Pure CPU / System Memory task (independent Fail-Safe degradation)
            if not needs_gpu:
                vram_pending = state.total_pending_vram_mib()
                compat = check_all_or_nothing(
                    requests=req_tuple,
                    active_leases=active_leases,
                    vram_admission_mib=0,
                    capacity=capacity,
                    device_id="cpu",
                )
                if not compat.compatible:
                    err_codes = list(compat.error_codes)
                    if blocked_leases and ErrorCode.LEASE_RECLAIM_BLOCKED not in err_codes:
                        err_codes.append(ErrorCode.LEASE_RECLAIM_BLOCKED)
                    return AdmissionDecision(
                        admitted=False,
                        lease=None,
                        vram_admission_mib=0,
                        vram_pending_mib=vram_pending,
                        vram_safety_mib=safety_mib,
                        thermal_state=self._thermal_guard.state,
                        error_codes=tuple(err_codes),
                        warnings=compat.reasons,
                    )

                lease = self._commit_new_lease_locked(
                    state=state,
                    owner_id=owner_id,
                    device_id="cpu",
                    requests=req_tuple,
                    reserved_vram_mib=0,
                    baseline_free_vram_mib=None,
                    now_utc=now_utc,
                    now_mono=now_mono,
                    ttl_seconds=ttl_seconds,
                    pid=pid,
                    pid_create_time=pid_create_time,
                )
                return AdmissionDecision(
                    admitted=True,
                    lease=lease,
                    vram_admission_mib=0,
                    vram_pending_mib=vram_pending,
                    vram_safety_mib=safety_mib,
                    thermal_state=self._thermal_guard.state,
                )

            # Path B: GPU task (Fail-Closed on hardware, freshness, thermal, VRAM, or conflicts)
            profile, snapshot = self._detector.probe(device_index=device_index)
            now_mono = self._clock.monotonic()
            vram_pending = state.total_pending_vram_mib(device_id=profile.device_id)

            # 1. Verify CUDA backend presence
            if ComputeBackend.CUDA not in profile.backend_support:
                return AdmissionDecision(
                    admitted=False,
                    lease=None,
                    vram_admission_mib=0,
                    vram_pending_mib=vram_pending,
                    vram_safety_mib=safety_mib,
                    thermal_state=ThermalState.SENSOR_DEGRADED,
                    error_codes=(ErrorCode.HARDWARE_UNVERIFIED,),
                    warnings=("No CUDA GPU available for GPU resource request.",),
                )

            # 2. Verify snapshot freshness (monotonic clock)
            if not snapshot.is_fresh(now_mono):
                return AdmissionDecision(
                    admitted=False,
                    lease=None,
                    vram_admission_mib=0,
                    vram_pending_mib=vram_pending,
                    vram_safety_mib=safety_mib,
                    thermal_state=ThermalState.SENSOR_DEGRADED,
                    error_codes=(ErrorCode.STALE_HARDWARE_SNAPSHOT,),
                    warnings=("HardwareSnapshot is stale; refusing GPU admission.",),
                )

            # 3. Evaluate ThermalGuard 4-state machine
            thermal_eval = self._thermal_guard.evaluate(profile, snapshot, now_mono)
            if not thermal_eval.gpu_admission_allowed:
                return AdmissionDecision(
                    admitted=False,
                    lease=None,
                    vram_admission_mib=0,
                    vram_pending_mib=vram_pending,
                    vram_safety_mib=safety_mib,
                    thermal_state=thermal_eval.state,
                    error_codes=thermal_eval.error_codes,
                    warnings=thermal_eval.warnings,
                )

            # 4. Verify device-level free VRAM is observable
            if snapshot.available_vram_mib is None or snapshot.available_vram_mib < 0:
                return AdmissionDecision(
                    admitted=False,
                    lease=None,
                    vram_admission_mib=0,
                    vram_pending_mib=vram_pending,
                    vram_safety_mib=safety_mib,
                    thermal_state=thermal_eval.state,
                    error_codes=(ErrorCode.HARDWARE_UNVERIFIED,),
                    warnings=("Device-level VRAM_free,observed is unavailable; Fail-Closed.",),
                )

            # 5. Dynamic admission formula (NEVER double-subtract external_vram_mib)
            vram_admission = max(0, snapshot.available_vram_mib - safety_mib - vram_pending)

            # 6. Optional external service eviction + mandatory post-eviction VRAM re-probe
            eviction_attempted = False
            extra_errors: list[ErrorCode] = []
            extra_warnings: list[str] = []

            if (
                req_gpu_mem_mib > vram_admission
                and allow_eviction
                and self._eviction_provider is not None
            ):
                eviction_attempted = True
                evict_res = self._eviction_provider.evict()
                # Mandatory re-probe: NEVER assume VRAM was freed without re-measuring
                profile, snapshot = self._detector.probe(device_index=device_index)
                now_mono = self._clock.monotonic()

                if (
                    not snapshot.is_fresh(now_mono)
                    or snapshot.available_vram_mib is None
                    or snapshot.available_vram_mib < 0
                ):
                    return AdmissionDecision(
                        admitted=False,
                        lease=None,
                        vram_admission_mib=0,
                        vram_pending_mib=vram_pending,
                        vram_safety_mib=safety_mib,
                        thermal_state=thermal_eval.state,
                        error_codes=(ErrorCode.EVICTION_FAILED, ErrorCode.HARDWARE_UNVERIFIED),
                        warnings=("Post-eviction VRAM re-probe failed or stale.",),
                        eviction_attempted=True,
                    )

                vram_admission = max(0, snapshot.available_vram_mib - safety_mib - vram_pending)
                if not evict_res.success or req_gpu_mem_mib > vram_admission:
                    extra_errors.append(ErrorCode.EVICTION_FAILED)
                    extra_warnings.append(
                        evict_res.message
                        or f"Post-eviction VRAM_admission ({vram_admission} MiB) still below "
                        f"requested ({req_gpu_mem_mib} MiB)."
                    )

            # 7. All-or-nothing multi-resource compatibility check
            compat = check_all_or_nothing(
                requests=req_tuple,
                active_leases=active_leases,
                vram_admission_mib=vram_admission,
                capacity=capacity,
                device_id=profile.device_id,
            )
            if not compat.compatible or extra_errors:
                combined_errors = list(dict.fromkeys([*extra_errors, *compat.error_codes]))
                # If conflict is caused by an expired-but-alive (reclaim-blocked) GPU lease,
                # surface LEASE_RECLAIM_BLOCKED explicitly (PRD 2.2 D & 2.6).
                if blocked_leases and any(
                    bl.device_id == profile.device_id for bl in blocked_leases
                ):
                    if ErrorCode.LEASE_RECLAIM_BLOCKED not in combined_errors:
                        combined_errors.insert(0, ErrorCode.LEASE_RECLAIM_BLOCKED)
                return AdmissionDecision(
                    admitted=False,
                    lease=None,
                    vram_admission_mib=vram_admission,
                    vram_pending_mib=vram_pending,
                    vram_safety_mib=safety_mib,
                    thermal_state=thermal_eval.state,
                    error_codes=tuple(combined_errors),
                    warnings=tuple([*extra_warnings, *compat.reasons]),
                    eviction_attempted=eviction_attempted,
                )

            # 8. Commit new lease atomically with incremented lease_generation
            lease = self._commit_new_lease_locked(
                state=state,
                owner_id=owner_id,
                device_id=profile.device_id,
                requests=req_tuple,
                reserved_vram_mib=req_gpu_mem_mib,
                baseline_free_vram_mib=snapshot.available_vram_mib,
                now_utc=now_utc,
                now_mono=now_mono,
                ttl_seconds=ttl_seconds,
                pid=pid,
                pid_create_time=pid_create_time,
            )
            return AdmissionDecision(
                admitted=True,
                lease=lease,
                vram_admission_mib=vram_admission,
                vram_pending_mib=vram_pending + req_gpu_mem_mib,
                vram_safety_mib=safety_mib,
                thermal_state=thermal_eval.state,
                eviction_attempted=eviction_attempted,
            )

    def _commit_new_lease_locked(
        self,
        state: LeaseRegistryState,
        owner_id: str,
        device_id: str,
        requests: tuple[ResourceRequest, ...],
        reserved_vram_mib: int,
        baseline_free_vram_mib: int | None,
        now_utc: str,
        now_mono: float,
        ttl_seconds: float,
        pid: int | None,
        pid_create_time: float | None,
    ) -> VramLease:
        new_generation = state.generation + 1
        lease = VramLease(
            lease_id=f"lease-{uuid.uuid4().hex[:12]}",
            lease_generation=new_generation,
            owner_id=owner_id,
            device_id=device_id,
            requests=requests,
            reserved_vram_mib=reserved_vram_mib,
            reservation_state=ReservationState.RESERVED,
            vram_observed_confirmed=False,
            baseline_free_vram_mib=baseline_free_vram_mib,
            created_at=now_utc,
            heartbeat_at=now_utc,
            heartbeat_monotonic=now_mono,
            ttl_seconds=ttl_seconds,
            pid=pid,
            pid_create_time=pid_create_time,
        )
        new_state = LeaseRegistryState(
            schema_version=state.schema_version,
            generation=new_generation,
            updated_at=now_utc,
            leases=(*state.active_leases(), lease),
        )
        self._store.save(new_state)
        return lease

    def transition_state(
        self,
        lease_id: str,
        lease_generation: int,
        new_state: ReservationState,
        device_index: int = 0,
    ) -> VramLease:
        """Transition a lease's ReservationState with fencing token & VRAM observation check."""
        with self._new_lock():
            state = self._store.load()
            target = self._find_and_verify_lease(state, lease_id, lease_generation)
            now_utc = self._clock.utc_now_iso()
            now_mono = self._clock.monotonic()

            if new_state == ReservationState.RELEASED:
                updated = replace(
                    target,
                    reservation_state=ReservationState.RELEASED,
                    heartbeat_at=now_utc,
                    heartbeat_monotonic=now_mono,
                )
                remaining = tuple(
                    l for l in state.active_leases() if l.lease_id != lease_id
                )
                self._store.save(
                    LeaseRegistryState(
                        schema_version=state.schema_version,
                        generation=state.generation,
                        updated_at=now_utc,
                        leases=remaining,
                    )
                )
                return updated

            confirmed = target.vram_observed_confirmed
            if new_state == ReservationState.ACTIVE:
                confirmed = self._confirm_vram_observation(target, device_index=device_index)

            updated = replace(
                target,
                reservation_state=new_state,
                vram_observed_confirmed=confirmed,
                heartbeat_at=now_utc,
                heartbeat_monotonic=now_mono,
            )
            new_leases = tuple(
                updated if l.lease_id == lease_id else l for l in state.active_leases()
            )
            self._store.save(
                LeaseRegistryState(
                    schema_version=state.schema_version,
                    generation=state.generation,
                    updated_at=now_utc,
                    leases=new_leases,
                )
            )
            return updated

    def heartbeat(self, lease_id: str, lease_generation: int) -> VramLease:
        """Renew lease heartbeat after verifying fencing token (lease_generation)."""
        with self._new_lock():
            state = self._store.load()
            target = self._find_and_verify_lease(state, lease_id, lease_generation)
            now_utc = self._clock.utc_now_iso()
            now_mono = self._clock.monotonic()
            updated = replace(
                target,
                heartbeat_at=now_utc,
                heartbeat_monotonic=now_mono,
            )
            new_leases = tuple(
                updated if l.lease_id == lease_id else l for l in state.active_leases()
            )
            self._store.save(
                LeaseRegistryState(
                    schema_version=state.schema_version,
                    generation=state.generation,
                    updated_at=now_utc,
                    leases=new_leases,
                )
            )
            return updated

    def attach_process(
        self,
        lease_id: str,
        lease_generation: int,
        pid: int,
        pid_create_time: float | None = None,
        transition_to_starting: bool = True,
    ) -> VramLease:
        """Bind spawned worker PID & pid_create_time to an active lease under RegistryLock."""
        with self._new_lock():
            state = self._store.load()
            target = self._find_and_verify_lease(state, lease_id, lease_generation)
            now_utc = self._clock.utc_now_iso()
            now_mono = self._clock.monotonic()
            next_state = (
                ReservationState.STARTING
                if transition_to_starting and target.reservation_state == ReservationState.RESERVED
                else target.reservation_state
            )
            updated = replace(
                target,
                pid=pid,
                pid_create_time=pid_create_time,
                reservation_state=next_state,
                heartbeat_at=now_utc,
                heartbeat_monotonic=now_mono,
            )
            new_leases = tuple(
                updated if l.lease_id == lease_id else l for l in state.active_leases()
            )
            self._store.save(
                LeaseRegistryState(
                    schema_version=state.schema_version,
                    generation=state.generation,
                    updated_at=now_utc,
                    leases=new_leases,
                )
            )
            return updated

    def release_lease(self, lease_id: str, lease_generation: int) -> VramLease:
        """Release a lease after verifying its fencing token."""
        return self.transition_state(
            lease_id=lease_id,
            lease_generation=lease_generation,
            new_state=ReservationState.RELEASED,
        )

    def _confirm_vram_observation(self, lease: VramLease, device_index: int = 0) -> bool:
        """Verify device-level free VRAM reflects the reservation before leaving VRAM_pending."""
        if lease.reserved_vram_mib <= 0:
            return True
        if lease.baseline_free_vram_mib is None:
            return False
        _, snapshot = self._detector.probe(device_index=device_index)
        if not snapshot.is_fresh(self._clock.monotonic()):
            return False
        if snapshot.available_vram_mib is None:
            return False
        observed_drop = lease.baseline_free_vram_mib - snapshot.available_vram_mib
        threshold = int(lease.reserved_vram_mib * self._vram_confirmation_ratio)
        return observed_drop >= threshold

    @staticmethod
    def _find_and_verify_lease(
        state: LeaseRegistryState,
        lease_id: str,
        lease_generation: int,
    ) -> VramLease:
        for lease in state.active_leases():
            if lease.lease_id == lease_id:
                if lease.lease_generation != lease_generation:
                    raise FencingTokenMismatchError(
                        f"Lease {lease_id} expected generation {lease.lease_generation}, "
                        f"got {lease_generation}."
                    )
                return lease
        raise FencingTokenMismatchError(
            f"Lease {lease_id} (generation {lease_generation}) is not active or has been released."
        )

    def read_registry_snapshot(self) -> LeaseRegistryState:
        """Read current LeaseRegistryState safely under RegistryLock."""
        with self._new_lock():
            return self._store.load()
