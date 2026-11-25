"""Application service producing canonical M1-A hardware doctor reports."""

from __future__ import annotations

from localgpu_os.domain.hardware import DoctorReport
from localgpu_os.domain.policy import PolicyEngine
from localgpu_os.infrastructure.clock import ClockProtocol, SystemClock
from localgpu_os.infrastructure.nvidia_detector import (
    HardwareDetectorProtocol,
    NvidiaSmiDetector,
)

SCHEMA_VERSION = "1.0"


class DoctorService:
    """Orchestrates hardware probing and policy evaluation for `lgpu doctor`."""

    def __init__(
        self,
        detector: HardwareDetectorProtocol | None = None,
        policy_engine: PolicyEngine | None = None,
        clock: ClockProtocol | None = None,
    ) -> None:
        self._clock = clock or SystemClock()
        self._detector = detector or NvidiaSmiDetector(clock=self._clock)
        self._policy = policy_engine or PolicyEngine()

    def run_doctor(self, device_index: int = 0) -> DoctorReport:
        profile, snapshot = self._detector.probe(device_index=device_index)
        decision = self._policy.evaluate(
            profile=profile,
            snapshot=snapshot,
            now_monotonic=self._clock.monotonic(),
        )
        return DoctorReport(
            schema_version=SCHEMA_VERSION,
            profile=profile,
            snapshot=snapshot,
            decision=decision,
        )
