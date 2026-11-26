"""Application service producing canonical M1-B `lgpu status` reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from localgpu_os.application.vram_gate import VramGate
from localgpu_os.domain.enums import ErrorCode, ThermalState
from localgpu_os.domain.hardware import HardwareProfile, HardwareSnapshot, SafetyMarginConfig
from localgpu_os.domain.lease import LeaseRegistryState, LeaseStateCorruptedError
from localgpu_os.domain.thermal import ThermalGuard
from localgpu_os.infrastructure.clock import ClockProtocol, SystemClock
from localgpu_os.infrastructure.nvidia_detector import (
    HardwareDetectorProtocol,
    NvidiaSmiDetector,
)

STATUS_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class StatusReport:
    schema_version: str
    profile: HardwareProfile
    snapshot: HardwareSnapshot
    thermal_state: ThermalState
    vram_safety_mib: int
    vram_pending_mib: int
    vram_admission_mib: int
    registry_generation: int
    active_leases: tuple[dict[str, Any], ...]
    error_codes: tuple[ErrorCode, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "device": {
                "device_id": self.profile.device_id,
                "vendor": self.profile.vendor,
                "name": self.profile.device_name,
                "total_vram_mib": self.profile.total_vram_mib,
            },
            "snapshot": {
                "observed_at": self.snapshot.observed_at,
                "available_vram_mib": self.snapshot.available_vram_mib,
                "external_vram_mib": self.snapshot.external_vram_mib,
                "temperature_c": self.snapshot.temperature_c,
                "utilization_percent": self.snapshot.utilization_percent,
                "sensor_status": self.snapshot.sensor_status.value,
            },
            "arbitration": {
                "thermal_state": self.thermal_state.value,
                "vram_safety_mib": self.vram_safety_mib,
                "vram_pending_mib": self.vram_pending_mib,
                "vram_admission_mib": self.vram_admission_mib,
                "registry_generation": self.registry_generation,
                "active_lease_count": len(self.active_leases),
                "active_leases": list(self.active_leases),
                "error_codes": [e.value for e in self.error_codes],
                "warnings": list(self.warnings),
            },
        }


class StatusService:
    """Orchestrates live hardware snapshot + Lease Registry inspection for `lgpu status`."""

    def __init__(
        self,
        state_dir: Path | None = None,
        detector: HardwareDetectorProtocol | None = None,
        clock: ClockProtocol | None = None,
        safety_config: SafetyMarginConfig | None = None,
        thermal_guard: ThermalGuard | None = None,
    ) -> None:
        self._clock = clock or SystemClock()
        self._detector = detector or NvidiaSmiDetector(clock=self._clock)
        self._safety = safety_config or SafetyMarginConfig()
        self._thermal = thermal_guard or ThermalGuard()
        self._gate = VramGate(
            state_dir=state_dir,
            detector=self._detector,
            clock=self._clock,
            safety_config=self._safety,
            thermal_guard=self._thermal,
        )

    def run_status(self, device_index: int = 0) -> StatusReport:
        profile, snapshot = self._detector.probe(device_index=device_index)
        now_mono = self._clock.monotonic()
        thermal_eval = self._thermal.evaluate(profile, snapshot, now_mono)

        errors: list[ErrorCode] = list(thermal_eval.error_codes)
        warnings: list[str] = list(thermal_eval.warnings)
        safety_mib = self._safety.total_safety_mib

        try:
            reg_state: LeaseRegistryState = self._gate.read_registry_snapshot()
        except LeaseStateCorruptedError as exc:
            errors.append(ErrorCode.LEASE_STATE_CORRUPTED)
            warnings.append(str(exc))
            return StatusReport(
                schema_version=STATUS_SCHEMA_VERSION,
                profile=profile,
                snapshot=snapshot,
                thermal_state=thermal_eval.state,
                vram_safety_mib=safety_mib,
                vram_pending_mib=0,
                vram_admission_mib=0,
                registry_generation=0,
                active_leases=(),
                error_codes=tuple(dict.fromkeys(errors)),
                warnings=tuple(warnings),
            )

        vram_pending = reg_state.total_pending_vram_mib(device_id=profile.device_id)
        if snapshot.available_vram_mib is None or snapshot.available_vram_mib < 0:
            vram_admission = 0
            if ErrorCode.HARDWARE_UNVERIFIED not in errors:
                errors.append(ErrorCode.HARDWARE_UNVERIFIED)
        else:
            vram_admission = max(0, snapshot.available_vram_mib - safety_mib - vram_pending)

        return StatusReport(
            schema_version=STATUS_SCHEMA_VERSION,
            profile=profile,
            snapshot=snapshot,
            thermal_state=thermal_eval.state,
            vram_safety_mib=safety_mib,
            vram_pending_mib=vram_pending,
            vram_admission_mib=vram_admission,
            registry_generation=reg_state.generation,
            active_leases=tuple(lease.to_dict() for lease in reg_state.active_leases()),
            error_codes=tuple(dict.fromkeys(errors)),
            warnings=tuple(warnings),
        )
