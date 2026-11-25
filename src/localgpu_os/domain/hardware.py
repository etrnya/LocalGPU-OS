"""Hardware domain models separating static capabilities from dynamic observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from localgpu_os.domain.enums import (
    ComputeBackend,
    ErrorCode,
    ExecutionPolicyTier,
    SensorStatus,
    ThermalState,
)


@dataclass(frozen=True)
class HardwareProfile:
    """Static physical hardware capabilities. Unsupported fields MUST be None, never 0."""

    device_id: str
    vendor: str
    device_name: str
    architecture: str | None = None
    compute_capability: str | None = None
    driver_version: str | None = None
    total_vram_mib: int | None = None
    memory_bandwidth_gbps: float | None = None
    power_limit_w: float | None = None
    nvenc_supported: bool | None = None
    temperature_support: bool = False
    backend_support: tuple[ComputeBackend, ...] = (ComputeBackend.CPU_INT8,)


@dataclass(frozen=True)
class HardwareSnapshot:
    """Point-in-time dynamic hardware observation with monotonic freshness semantics."""

    device_id: str
    observed_at: str
    observed_monotonic: float
    freshness_ttl_seconds: float = 10.0
    available_vram_mib: int | None = None
    external_vram_mib: int | None = None
    temperature_c: int | None = None
    utilization_percent: int | None = None
    sensor_status: SensorStatus = SensorStatus.UNAVAILABLE

    def is_fresh(self, now_monotonic: float) -> bool:
        """Return True only if monotonic age is non-negative and within TTL."""
        if self.freshness_ttl_seconds <= 0:
            return False
        age = now_monotonic - self.observed_monotonic
        if age < 0:
            return False
        return age <= self.freshness_ttl_seconds


@dataclass(frozen=True)
class SafetyMarginConfig:
    """VRAM safety margins in MiB (aligned with PRD v1.1.2 Section 2.2)."""

    system_reserve_mib: int = 768
    runtime_overhead_mib: int = 512
    uncertainty_margin_mib: int = 256

    @property
    def total_safety_mib(self) -> int:
        return (
            self.system_reserve_mib
            + self.runtime_overhead_mib
            + self.uncertainty_margin_mib
        )


@dataclass(frozen=True)
class ThermalThresholds:
    """Configurable per-GPU thermal thresholds in Celsius."""

    cooling_temp_c: int = 82
    resume_temp_c: int = 72
    critical_temp_c: int = 88


@dataclass(frozen=True)
class ExecutionPolicyDecision:
    """Result of PolicyEngine evaluation over a profile and live snapshot."""

    tier: ExecutionPolicyTier
    recommended_backend: ComputeBackend
    gpu_admission_allowed: bool
    thermal_state: ThermalState
    vram_admission_mib: int
    vram_safety_mib: int
    error_codes: tuple[ErrorCode, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DoctorReport:
    """M1-A canonical hardware snapshot report (schema_version 1.0)."""

    schema_version: str
    profile: HardwareProfile
    snapshot: HardwareSnapshot
    decision: ExecutionPolicyDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "device": {
                "device_id": self.profile.device_id,
                "vendor": self.profile.vendor,
                "name": self.profile.device_name,
                "architecture": self.profile.architecture,
                "compute_capability": self.profile.compute_capability,
                "driver_version": self.profile.driver_version,
                "total_vram_mib": self.profile.total_vram_mib,
                "memory_bandwidth_gbps": self.profile.memory_bandwidth_gbps,
                "power_limit_w": self.profile.power_limit_w,
            },
            "snapshot": {
                "observed_at": self.snapshot.observed_at,
                "freshness_ttl_seconds": self.snapshot.freshness_ttl_seconds,
                "available_vram_mib": self.snapshot.available_vram_mib,
                "external_vram_mib": self.snapshot.external_vram_mib,
                "temperature_c": self.snapshot.temperature_c,
                "utilization_percent": self.snapshot.utilization_percent,
                "sensor_status": self.snapshot.sensor_status.value,
            },
            "capabilities": {
                "cuda": ComputeBackend.CUDA in self.profile.backend_support,
                "nvenc": self.profile.nvenc_supported,
                "temperature_support": self.profile.temperature_support,
                "backends": [b.value for b in self.profile.backend_support],
            },
            "execution_policy": {
                "tier": self.decision.tier.value,
                "recommended_backend": self.decision.recommended_backend.value,
                "gpu_admission_allowed": self.decision.gpu_admission_allowed,
                "thermal_state": self.decision.thermal_state.value,
                "vram_admission_mib": self.decision.vram_admission_mib,
                "vram_safety_mib": self.decision.vram_safety_mib,
                "error_codes": [e.value for e in self.decision.error_codes],
                "warnings": list(self.decision.warnings),
            },
        }
