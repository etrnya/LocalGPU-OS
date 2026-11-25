"""Pure PolicyEngine for dynamic execution tiering and fail-safe GPU checks."""

from __future__ import annotations

from localgpu_os.domain.enums import (
    ComputeBackend,
    ErrorCode,
    ExecutionPolicyTier,
    SensorStatus,
    ThermalState,
)
from localgpu_os.domain.hardware import (
    ExecutionPolicyDecision,
    HardwareProfile,
    HardwareSnapshot,
    SafetyMarginConfig,
    ThermalThresholds,
)

# Descriptive dynamic tier thresholds based on live admission VRAM (MiB)
_TIER_STANDARD_MIN_MIB = 4608
_TIER_PRO_MIN_MIB = 8192
_TIER_ULTRA_MIN_MIB = 16384


class PolicyEngine:
    """Stateless policy evaluator. Decouples physical GPU model from runtime strategy."""

    def __init__(
        self,
        safety_config: SafetyMarginConfig | None = None,
        thermal_thresholds: ThermalThresholds | None = None,
    ) -> None:
        self._safety = safety_config or SafetyMarginConfig()
        self._thermal = thermal_thresholds or ThermalThresholds()

    def evaluate(
        self,
        profile: HardwareProfile,
        snapshot: HardwareSnapshot,
        now_monotonic: float,
    ) -> ExecutionPolicyDecision:
        errors: list[ErrorCode] = []
        warnings: list[str] = []
        safety_mib = self._safety.total_safety_mib

        # 1. CPU-only host (no CUDA backend declared)
        if ComputeBackend.CUDA not in profile.backend_support:
            return ExecutionPolicyDecision(
                tier=ExecutionPolicyTier.TIER_CPU_ONLY,
                recommended_backend=ComputeBackend.CPU_INT8,
                gpu_admission_allowed=False,
                thermal_state=ThermalState.SENSOR_DEGRADED,
                vram_admission_mib=0,
                vram_safety_mib=safety_mib,
                error_codes=(),
                warnings=("No CUDA GPU detected; degrading gracefully to CPU_INT8.",),
            )

        # 2. Snapshot freshness gate (monotonic clock, Fail-Closed)
        if not snapshot.is_fresh(now_monotonic):
            errors.append(ErrorCode.STALE_HARDWARE_SNAPSHOT)
            warnings.append("Hardware snapshot is stale; refusing GPU admission.")
            return ExecutionPolicyDecision(
                tier=ExecutionPolicyTier.TIER_CPU_ONLY,
                recommended_backend=ComputeBackend.CPU_INT8,
                gpu_admission_allowed=False,
                thermal_state=ThermalState.SENSOR_DEGRADED,
                vram_admission_mib=0,
                vram_safety_mib=safety_mib,
                error_codes=tuple(errors),
                warnings=tuple(warnings),
            )

        # 3. Thermal state evaluation (4-state machine: NORMAL/COOLING/CRITICAL/SENSOR_DEGRADED)
        thermal_state = self._evaluate_thermal(profile, snapshot, errors, warnings)

        # 4. Live device-level free VRAM verification (Fail-Closed on probe failure)
        if snapshot.available_vram_mib is None or snapshot.available_vram_mib < 0:
            errors.append(ErrorCode.HARDWARE_UNVERIFIED)
            warnings.append("Device-level free VRAM unavailable; Fail-Closed for GPU tasks.")
            return ExecutionPolicyDecision(
                tier=ExecutionPolicyTier.TIER_CPU_ONLY,
                recommended_backend=ComputeBackend.CPU_INT8,
                gpu_admission_allowed=False,
                thermal_state=thermal_state,
                vram_admission_mib=0,
                vram_safety_mib=safety_mib,
                error_codes=tuple(errors),
                warnings=tuple(warnings),
            )

        # 5. Dynamic admission VRAM (clamped >= 0, never double-subtract external_vram_mib)
        vram_admission_mib = max(0, snapshot.available_vram_mib - safety_mib)

        if vram_admission_mib == 0:
            errors.append(ErrorCode.VRAM_INSUFFICIENT)
            warnings.append(
                f"Available VRAM ({snapshot.available_vram_mib} MiB) does not exceed "
                f"safety margin ({safety_mib} MiB); GPU admission denied."
            )
            return ExecutionPolicyDecision(
                tier=ExecutionPolicyTier.TIER_CPU_ONLY,
                recommended_backend=ComputeBackend.CPU_INT8,
                gpu_admission_allowed=False,
                thermal_state=thermal_state,
                vram_admission_mib=0,
                vram_safety_mib=safety_mib,
                error_codes=tuple(errors),
                warnings=tuple(warnings),
            )

        # 6. Dynamic execution policy tier derived from live admission VRAM
        tier = self._classify_tier(vram_admission_mib)

        # 7. Critical thermal blocks GPU admission even when VRAM is plentiful
        if thermal_state == ThermalState.CRITICAL:
            return ExecutionPolicyDecision(
                tier=tier,
                recommended_backend=ComputeBackend.CPU_INT8,
                gpu_admission_allowed=False,
                thermal_state=thermal_state,
                vram_admission_mib=vram_admission_mib,
                vram_safety_mib=safety_mib,
                error_codes=tuple(errors),
                warnings=tuple(warnings),
            )

        return ExecutionPolicyDecision(
            tier=tier,
            recommended_backend=ComputeBackend.CUDA,
            gpu_admission_allowed=True,
            thermal_state=thermal_state,
            vram_admission_mib=vram_admission_mib,
            vram_safety_mib=safety_mib,
            error_codes=tuple(errors),
            warnings=tuple(warnings),
        )

    def _evaluate_thermal(
        self,
        profile: HardwareProfile,
        snapshot: HardwareSnapshot,
        errors: list[ErrorCode],
        warnings: list[str],
    ) -> ThermalState:
        if (
            not profile.temperature_support
            or snapshot.sensor_status != SensorStatus.OK
            or snapshot.temperature_c is None
        ):
            errors.append(ErrorCode.THERMAL_SENSOR_DEGRADED)
            warnings.append(
                "GPU temperature sensor unavailable or degraded; conservative mode active."
            )
            return ThermalState.SENSOR_DEGRADED

        if snapshot.temperature_c >= self._thermal.critical_temp_c:
            errors.append(ErrorCode.THERMAL_CRITICAL_ABORT)
            warnings.append(
                f"GPU temperature ({snapshot.temperature_c}C) reached CRITICAL threshold "
                f"({self._thermal.critical_temp_c}C)."
            )
            return ThermalState.CRITICAL

        if snapshot.temperature_c >= self._thermal.cooling_temp_c:
            errors.append(ErrorCode.THERMAL_COOLING_WAIT)
            warnings.append(
                f"GPU temperature ({snapshot.temperature_c}C) exceeded COOLING threshold "
                f"({self._thermal.cooling_temp_c}C)."
            )
            return ThermalState.COOLING

        return ThermalState.NORMAL

    @staticmethod
    def _classify_tier(vram_admission_mib: int) -> ExecutionPolicyTier:
        if vram_admission_mib >= _TIER_ULTRA_MIN_MIB:
            return ExecutionPolicyTier.TIER_ULTRA
        if vram_admission_mib >= _TIER_PRO_MIN_MIB:
            return ExecutionPolicyTier.TIER_PRO
        if vram_admission_mib >= _TIER_STANDARD_MIN_MIB:
            return ExecutionPolicyTier.TIER_STANDARD
        return ExecutionPolicyTier.TIER_LITE
