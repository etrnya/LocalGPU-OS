"""Stateful 4-state ThermalGuard with hysteresis and sensor degradation recovery (PRD 2.4)."""

from __future__ import annotations

from dataclasses import dataclass

from localgpu_os.domain.enums import ErrorCode, SensorStatus, ThermalState
from localgpu_os.domain.hardware import HardwareProfile, HardwareSnapshot


@dataclass(frozen=True)
class ThermalGuardConfig:
    """Configurable thresholds and hysteresis durations for ThermalGuard."""

    cooling_temp_c: int = 82
    resume_temp_c: int = 72
    critical_temp_c: int = 88
    cooling_hysteresis_seconds: float = 15.0
    critical_hysteresis_seconds: float = 60.0
    sensor_recovery_samples: int = 2


@dataclass(frozen=True)
class ThermalEvaluation:
    state: ThermalState
    gpu_admission_allowed: bool
    error_codes: tuple[ErrorCode, ...] = ()
    warnings: tuple[str, ...] = ()


class ThermalGuard:
    """4-state cooperative thermal state machine: NORMAL / COOLING / CRITICAL / SENSOR_DEGRADED."""

    def __init__(self, config: ThermalGuardConfig | None = None) -> None:
        self._config = config or ThermalGuardConfig()
        self._state: ThermalState = ThermalState.NORMAL
        self._below_resume_since_mono: float | None = None
        self._consecutive_valid_samples: int = 0

    @property
    def state(self) -> ThermalState:
        return self._state

    @property
    def config(self) -> ThermalGuardConfig:
        return self._config

    def evaluate(
        self,
        profile: HardwareProfile,
        snapshot: HardwareSnapshot,
        now_monotonic: float,
    ) -> ThermalEvaluation:
        # Sensor unavailable, degraded, missing temperature, or stale snapshot -> SENSOR_DEGRADED
        sensor_valid = (
            profile.temperature_support
            and snapshot.sensor_status == SensorStatus.OK
            and snapshot.temperature_c is not None
            and snapshot.is_fresh(now_monotonic)
        )

        if not sensor_valid:
            self._state = ThermalState.SENSOR_DEGRADED
            self._below_resume_since_mono = None
            self._consecutive_valid_samples = 0
            return ThermalEvaluation(
                state=ThermalState.SENSOR_DEGRADED,
                gpu_admission_allowed=False,
                error_codes=(ErrorCode.THERMAL_SENSOR_DEGRADED,),
                warnings=(
                    "GPU temperature sensor unavailable, degraded, or snapshot stale; "
                    "conservative mode active (high-load GPU tasks blocked).",
                ),
            )

        assert snapshot.temperature_c is not None
        temp = snapshot.temperature_c

        # Recovery from SENSOR_DEGRADED requires consecutive valid fresh samples
        if self._state == ThermalState.SENSOR_DEGRADED:
            self._consecutive_valid_samples += 1
            if self._consecutive_valid_samples < self._config.sensor_recovery_samples:
                return ThermalEvaluation(
                    state=ThermalState.SENSOR_DEGRADED,
                    gpu_admission_allowed=False,
                    error_codes=(ErrorCode.THERMAL_SENSOR_DEGRADED,),
                    warnings=(
                        f"Sensor recovering ({self._consecutive_valid_samples}/"
                        f"{self._config.sensor_recovery_samples} valid samples).",
                    ),
                )
            # Enough consecutive valid samples -> re-enter evaluation from NORMAL baseline
            self._state = ThermalState.NORMAL
            self._below_resume_since_mono = None

        # Escalations always take immediate effect
        if temp >= self._config.critical_temp_c:
            self._state = ThermalState.CRITICAL
            self._below_resume_since_mono = None
            return ThermalEvaluation(
                state=ThermalState.CRITICAL,
                gpu_admission_allowed=False,
                error_codes=(ErrorCode.THERMAL_CRITICAL_ABORT,),
                warnings=(
                    f"GPU temperature ({temp}C) reached CRITICAL threshold "
                    f"({self._config.critical_temp_c}C).",
                ),
            )

        if temp >= self._config.cooling_temp_c:
            if self._state != ThermalState.CRITICAL:
                self._state = ThermalState.COOLING
            self._below_resume_since_mono = None
            if self._state == ThermalState.CRITICAL:
                return ThermalEvaluation(
                    state=ThermalState.CRITICAL,
                    gpu_admission_allowed=False,
                    error_codes=(ErrorCode.THERMAL_CRITICAL_ABORT,),
                    warnings=(
                        f"GPU remains in CRITICAL cooldown (temp {temp}C > resume "
                        f"{self._config.resume_temp_c}C).",
                    ),
                )
            return ThermalEvaluation(
                state=ThermalState.COOLING,
                gpu_admission_allowed=False,
                error_codes=(ErrorCode.THERMAL_COOLING_WAIT,),
                warnings=(
                    f"GPU temperature ({temp}C) exceeded COOLING threshold "
                    f"({self._config.cooling_temp_c}C).",
                ),
            )

        # Hysteresis recovery for CRITICAL and COOLING
        if self._state in (ThermalState.CRITICAL, ThermalState.COOLING):
            required_hold = (
                self._config.critical_hysteresis_seconds
                if self._state == ThermalState.CRITICAL
                else self._config.cooling_hysteresis_seconds
            )
            if temp <= self._config.resume_temp_c:
                if self._below_resume_since_mono is None:
                    self._below_resume_since_mono = now_monotonic
                elapsed = now_monotonic - self._below_resume_since_mono
                if elapsed >= required_hold:
                    self._state = ThermalState.NORMAL
                    self._below_resume_since_mono = None
                    return ThermalEvaluation(
                        state=ThermalState.NORMAL,
                        gpu_admission_allowed=True,
                    )
            else:
                self._below_resume_since_mono = None

            err = (
                ErrorCode.THERMAL_CRITICAL_ABORT
                if self._state == ThermalState.CRITICAL
                else ErrorCode.THERMAL_COOLING_WAIT
            )
            return ThermalEvaluation(
                state=self._state,
                gpu_admission_allowed=False,
                error_codes=(err,),
                warnings=(
                    f"GPU in {self._state.value} hysteresis cooldown "
                    f"(temp {temp}C, resume <= {self._config.resume_temp_c}C for {required_hold:.0f}s).",
                ),
            )

        # NORMAL state
        return ThermalEvaluation(
            state=ThermalState.NORMAL,
            gpu_admission_allowed=True,
        )
