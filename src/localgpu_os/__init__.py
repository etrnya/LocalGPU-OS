"""LocalGPU-OS: Local Compute Resource Arbitration Layer."""

from localgpu_os.application.vram_gate import VramGate
from localgpu_os.application.worker_runtime import WorkerRuntime
from localgpu_os.domain.enums import (
    ComputeBackend,
    ErrorCode,
    ExecutionPolicyTier,
    NetworkPolicy,
    ReservationState,
    ResourceClass,
    SensorStatus,
    ThermalState,
)
from localgpu_os.domain.lease import AdmissionDecision, ResourceRequest, VramLease
from localgpu_os.domain.worker import (
    CancelSignal,
    WorkerFallbackPolicy,
    WorkerFileLimits,
    WorkerManifest,
    WorkerRunResult,
)
from localgpu_os.infrastructure.worker_registry import WorkerRegistry
from localgpu_os.sdk import LocalGpuClient, WorkerContext

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AdmissionDecision",
    "CancelSignal",
    "ComputeBackend",
    "ErrorCode",
    "ExecutionPolicyTier",
    "LocalGpuClient",
    "NetworkPolicy",
    "ReservationState",
    "ResourceClass",
    "ResourceRequest",
    "SensorStatus",
    "ThermalState",
    "VramGate",
    "VramLease",
    "WorkerContext",
    "WorkerFallbackPolicy",
    "WorkerFileLimits",
    "WorkerManifest",
    "WorkerRegistry",
    "WorkerRunResult",
    "WorkerRuntime",
]
