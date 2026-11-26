"""Multi-resource all-or-nothing compatibility matrix (PRD v1.1.2 Section 2.3).

Core OS Policy Supremacy: Worker Manifests may NEVER relax Core OS exclusivity
or concurrency limits.
"""

from __future__ import annotations

from dataclasses import dataclass

from localgpu_os.domain.enums import ErrorCode, ResourceClass
from localgpu_os.domain.lease import ResourceRequest, VramLease


@dataclass(frozen=True)
class ResourceCapacity:
    """Host resource limits enforced by Core OS."""

    cpu_threads_total: int = 4
    system_memory_mib_available: int | None = None
    gpu_compute_slots: int = 1  # M1 conservative exclusivity: 1 slot
    gpu_encode_sessions: int = 0  # Reserved for M3 (NVENC)


@dataclass(frozen=True)
class CompatibilityCheckResult:
    compatible: bool
    error_codes: tuple[ErrorCode, ...] = ()
    reasons: tuple[str, ...] = ()


def check_all_or_nothing(
    requests: tuple[ResourceRequest, ...],
    active_leases: tuple[VramLease, ...],
    vram_admission_mib: int,
    capacity: ResourceCapacity,
    device_id: str,
) -> CompatibilityCheckResult:
    """Validate an entire ResourceRequest bundle atomically against Core OS policy."""
    if not requests:
        return CompatibilityCheckResult(
            compatible=False,
            error_codes=(ErrorCode.WORKER_MANIFEST_INVALID,),
            reasons=("ResourceRequest list must not be empty.",),
        )

    errors: list[ErrorCode] = []
    reasons: list[str] = []

    # Aggregate requested amounts per resource class within this atomic bundle
    req_gpu_compute = 0
    req_gpu_memory = 0
    req_cpu_threads = 0
    req_sys_memory = 0
    req_gpu_encode = 0

    for req in requests:
        if req.resource_class == ResourceClass.GPU_COMPUTE:
            req_gpu_compute += req.amount
        elif req.resource_class == ResourceClass.GPU_MEMORY:
            req_gpu_memory += req.amount
        elif req.resource_class == ResourceClass.CPU_COMPUTE:
            req_cpu_threads += req.amount
        elif req.resource_class == ResourceClass.SYSTEM_MEMORY:
            req_sys_memory += req.amount
        elif req.resource_class == ResourceClass.GPU_ENCODE:
            req_gpu_encode += req.amount

    # Aggregate existing active allocations
    used_gpu_compute = 0
    used_cpu_threads = 0
    used_sys_memory = 0
    used_gpu_encode = 0

    for lease in active_leases:
        for existing in lease.requests:
            if existing.resource_class == ResourceClass.GPU_COMPUTE and lease.device_id == device_id:
                used_gpu_compute += existing.amount
            elif existing.resource_class == ResourceClass.CPU_COMPUTE:
                used_cpu_threads += existing.amount
            elif existing.resource_class == ResourceClass.SYSTEM_MEMORY:
                used_sys_memory += existing.amount
            elif existing.resource_class == ResourceClass.GPU_ENCODE and lease.device_id == device_id:
                used_gpu_encode += existing.amount

    # 1. GPU_COMPUTE: M1 conservative exclusivity (Core OS supremacy: ignores exclusive=False)
    if req_gpu_compute > 0:
        if used_gpu_compute + req_gpu_compute > capacity.gpu_compute_slots:
            errors.append(ErrorCode.VRAM_LOCK_TIMEOUT)
            reasons.append(
                f"GPU_COMPUTE slots exhausted on {device_id} "
                f"(used={used_gpu_compute}, requested={req_gpu_compute}, max={capacity.gpu_compute_slots})."
            )

    # 2. GPU_MEMORY: checked against live VRAM_admission (already reflects safety & VRAM_pending)
    if req_gpu_memory > 0:
        if req_gpu_memory > vram_admission_mib:
            errors.append(ErrorCode.VRAM_INSUFFICIENT)
            reasons.append(
                f"Requested GPU_MEMORY ({req_gpu_memory} MiB) exceeds "
                f"VRAM_admission ({vram_admission_mib} MiB)."
            )

    # 3. GPU_ENCODE: interface reserved for M3
    if req_gpu_encode > 0:
        if used_gpu_encode + req_gpu_encode > capacity.gpu_encode_sessions:
            errors.append(ErrorCode.HARDWARE_UNVERIFIED)
            reasons.append(
                f"GPU_ENCODE sessions unavailable in M1 "
                f"(used={used_gpu_encode}, requested={req_gpu_encode}, max={capacity.gpu_encode_sessions})."
            )

    # 4. CPU_COMPUTE: bounded by host CPU thread budget (parallel with GPU allowed)
    if req_cpu_threads > 0:
        if used_cpu_threads + req_cpu_threads > capacity.cpu_threads_total:
            errors.append(ErrorCode.VRAM_LOCK_TIMEOUT)
            reasons.append(
                f"CPU_COMPUTE threads exhausted "
                f"(used={used_cpu_threads}, requested={req_cpu_threads}, max={capacity.cpu_threads_total})."
            )

    # 5. SYSTEM_MEMORY: basic RAM budget check when host RAM is observable
    if req_sys_memory > 0 and capacity.system_memory_mib_available is not None:
        if used_sys_memory + req_sys_memory > capacity.system_memory_mib_available:
            errors.append(ErrorCode.VRAM_INSUFFICIENT)
            reasons.append(
                f"SYSTEM_MEMORY request ({req_sys_memory} MiB, active={used_sys_memory} MiB) "
                f"exceeds available RAM ({capacity.system_memory_mib_available} MiB)."
            )

    if errors:
        # Deduplicate preserving order
        unique_errors = tuple(dict.fromkeys(errors))
        return CompatibilityCheckResult(
            compatible=False,
            error_codes=unique_errors,
            reasons=tuple(reasons),
        )

    return CompatibilityCheckResult(compatible=True)
