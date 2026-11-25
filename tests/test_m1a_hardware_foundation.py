"""M1-A Hardware Foundation test suite (Scenarios S1-S6 + Contract Units U1-U6).

Uses Python standard library `unittest` so tests run with zero external dependencies
(and remains 100% compatible with `pytest`).
"""

from __future__ import annotations

import ast
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from localgpu_os.application.doctor_service import DoctorService
from localgpu_os.domain.enums import (
    ComputeBackend,
    ErrorCode,
    ExecutionPolicyTier,
    SensorStatus,
    ThermalState,
)
from localgpu_os.domain.hardware import HardwareProfile, HardwareSnapshot
from localgpu_os.domain.policy import PolicyEngine
from localgpu_os.infrastructure.nvidia_detector import (
    NvidiaSmiDetector,
    parse_nvidia_smi_csv,
)
from localgpu_os.presentation.cli import main as cli_main


class FakeClock:
    def __init__(self, monotonic_val: float = 100.0, utc_iso: str = "2026-11-25T14:00:00Z") -> None:
        self._mono = monotonic_val
        self._utc = utc_iso

    def utc_now_iso(self) -> str:
        return self._utc

    def monotonic(self) -> float:
        return self._mono


def _nvidia_profile(
    total_vram_mib: int,
    name: str = "NVIDIA GeForce RTX 4060 Laptop GPU",
    temp_support: bool = True,
) -> HardwareProfile:
    return HardwareProfile(
        device_id="cuda:0",
        vendor="NVIDIA",
        device_name=name,
        architecture="Ada Lovelace",
        compute_capability="8.9",
        driver_version="621.14",
        total_vram_mib=total_vram_mib,
        nvenc_supported=True,
        temperature_support=temp_support,
        backend_support=(ComputeBackend.CUDA, ComputeBackend.CPU_INT8),
    )


def _snapshot(
    available_vram_mib: int | None,
    external_vram_mib: int | None = 0,
    temperature_c: int | None = 55,
    sensor_status: SensorStatus = SensorStatus.OK,
    observed_monotonic: float = 100.0,
    ttl: float = 10.0,
) -> HardwareSnapshot:
    return HardwareSnapshot(
        device_id="cuda:0",
        observed_at="2026-11-25T14:00:00Z",
        observed_monotonic=observed_monotonic,
        freshness_ttl_seconds=ttl,
        available_vram_mib=available_vram_mib,
        external_vram_mib=external_vram_mib,
        temperature_c=temperature_c,
        utilization_percent=10,
        sensor_status=sensor_status,
    )


class TestM1AHardwareFoundation(unittest.TestCase):
    # --- Required Boundary Scenarios (S1 - S6: 100% Scenario Coverage) ---

    def test_s1_8gb_gpu_idle_is_standard(self) -> None:
        engine = PolicyEngine()
        decision = engine.evaluate(
            _nvidia_profile(8188),
            _snapshot(available_vram_mib=8188, external_vram_mib=0, temperature_c=55),
            now_monotonic=102.0,
        )
        self.assertEqual(decision.tier, ExecutionPolicyTier.TIER_STANDARD)
        self.assertEqual(decision.recommended_backend, ComputeBackend.CUDA)
        self.assertTrue(decision.gpu_admission_allowed)
        self.assertEqual(decision.thermal_state, ThermalState.NORMAL)
        self.assertEqual(decision.vram_admission_mib, 8188 - 1536)
        self.assertEqual(decision.error_codes, ())

    def test_s2_8gb_gpu_4gb_external_degrades_to_lite(self) -> None:
        engine = PolicyEngine()
        decision = engine.evaluate(
            _nvidia_profile(8188),
            _snapshot(available_vram_mib=4092, external_vram_mib=4096, temperature_c=62),
            now_monotonic=101.0,
        )
        # Must NOT double-subtract external_vram_mib!
        self.assertEqual(decision.vram_admission_mib, 4092 - 1536)
        self.assertEqual(decision.tier, ExecutionPolicyTier.TIER_LITE)
        self.assertTrue(decision.gpu_admission_allowed)

    def test_s3_8gb_gpu_7gb_external_denies_gpu_admission(self) -> None:
        engine = PolicyEngine()
        decision = engine.evaluate(
            _nvidia_profile(8188),
            _snapshot(available_vram_mib=1020, external_vram_mib=7168, temperature_c=65),
            now_monotonic=101.0,
        )
        self.assertEqual(decision.vram_admission_mib, 0)
        self.assertFalse(decision.gpu_admission_allowed)
        self.assertEqual(decision.recommended_backend, ComputeBackend.CPU_INT8)
        self.assertIn(ErrorCode.VRAM_INSUFFICIENT, decision.error_codes)

    def test_s4_16gb_gpu_12gb_external_dynamically_degrades_to_lite(self) -> None:
        engine = PolicyEngine()
        decision = engine.evaluate(
            _nvidia_profile(16384, name="NVIDIA GeForce RTX 4080"),
            _snapshot(available_vram_mib=4096, external_vram_mib=12288, temperature_c=68),
            now_monotonic=101.0,
        )
        self.assertEqual(decision.tier, ExecutionPolicyTier.TIER_LITE)
        self.assertTrue(decision.gpu_admission_allowed)
        self.assertEqual(decision.vram_admission_mib, 4096 - 1536)

    def test_s5_24gb_gpu_vram_probe_failure_fails_closed(self) -> None:
        engine = PolicyEngine()
        decision = engine.evaluate(
            _nvidia_profile(24576, name="NVIDIA GeForce RTX 4090"),
            _snapshot(available_vram_mib=None, external_vram_mib=None, temperature_c=50),
            now_monotonic=101.0,
        )
        self.assertFalse(decision.gpu_admission_allowed)
        self.assertEqual(decision.recommended_backend, ComputeBackend.CPU_INT8)
        self.assertEqual(decision.tier, ExecutionPolicyTier.TIER_CPU_ONLY)
        self.assertIn(ErrorCode.HARDWARE_UNVERIFIED, decision.error_codes)

    def test_s6_missing_temperature_sensor_enters_sensor_degraded(self) -> None:
        engine = PolicyEngine()
        decision = engine.evaluate(
            _nvidia_profile(8188, temp_support=False),
            _snapshot(
                available_vram_mib=7000,
                temperature_c=None,
                sensor_status=SensorStatus.DEGRADED,
            ),
            now_monotonic=101.0,
        )
        self.assertEqual(decision.thermal_state, ThermalState.SENSOR_DEGRADED)
        self.assertIn(ErrorCode.THERMAL_SENSOR_DEGRADED, decision.error_codes)

    # --- Core Contract & Defense Unit Tests (U1 - U6) ---

    def test_u1_snapshot_freshness_monotonic_clock_and_stale_rejection(self) -> None:
        engine = PolicyEngine()
        snap = _snapshot(available_vram_mib=7000, observed_monotonic=100.0, ttl=10.0)
        self.assertTrue(snap.is_fresh(110.0))
        self.assertFalse(snap.is_fresh(110.1))
        self.assertFalse(snap.is_fresh(99.0))  # monotonic regression -> stale

        stale_decision = engine.evaluate(_nvidia_profile(8188), snap, now_monotonic=115.0)
        self.assertFalse(stale_decision.gpu_admission_allowed)
        self.assertIn(ErrorCode.STALE_HARDWARE_SNAPSHOT, stale_decision.error_codes)

    def test_u2_thermal_cooling_and_critical_branches(self) -> None:
        engine = PolicyEngine()
        cooling = engine.evaluate(
            _nvidia_profile(8188),
            _snapshot(available_vram_mib=7000, temperature_c=84),
            now_monotonic=101.0,
        )
        self.assertEqual(cooling.thermal_state, ThermalState.COOLING)
        self.assertIn(ErrorCode.THERMAL_COOLING_WAIT, cooling.error_codes)

        critical = engine.evaluate(
            _nvidia_profile(8188),
            _snapshot(available_vram_mib=7000, temperature_c=89),
            now_monotonic=101.0,
        )
        self.assertEqual(critical.thermal_state, ThermalState.CRITICAL)
        self.assertFalse(critical.gpu_admission_allowed)
        self.assertIn(ErrorCode.THERMAL_CRITICAL_ABORT, critical.error_codes)

    def test_u3_no_nvidia_smi_gracefully_degrades_to_cpu_only(self) -> None:
        detector = NvidiaSmiDetector(clock=FakeClock())
        with patch("subprocess.run", side_effect=FileNotFoundError("nvidia-smi not found")):
            report = DoctorService(detector=detector, clock=FakeClock()).run_doctor()
        self.assertEqual(report.profile.vendor, "CPU")
        self.assertEqual(report.decision.tier, ExecutionPolicyTier.TIER_CPU_ONLY)
        self.assertEqual(report.decision.recommended_backend, ComputeBackend.CPU_INT8)
        self.assertFalse(report.decision.gpu_admission_allowed)

    def test_u4_nvidia_smi_csv_parser_handles_na_and_multi_gpu_selection(self) -> None:
        csv_text = (
            "0, NVIDIA GeForce RTX 4060 Laptop GPU, 8188, 7000, 1188, [N/A], 12, 621.14, 8.9, [Not Supported]\n"
            "1, NVIDIA GeForce RTX 4090, 24576, 23000, 1576, 48, 5, 621.14, 8.9, 450.00\n"
        )
        parsed0 = parse_nvidia_smi_csv(
            csv_text, device_index=0, observed_at="2026-11-25T14:00:00Z", observed_monotonic=10.0
        )
        self.assertIsNotNone(parsed0)
        assert parsed0 is not None
        profile0, snap0 = parsed0
        self.assertEqual(profile0.device_name, "NVIDIA GeForce RTX 4060 Laptop GPU")
        self.assertIsNone(profile0.power_limit_w)  # NEVER 0
        self.assertFalse(profile0.temperature_support)
        self.assertIsNone(snap0.temperature_c)
        self.assertEqual(snap0.sensor_status, SensorStatus.DEGRADED)

        parsed1 = parse_nvidia_smi_csv(
            csv_text, device_index=1, observed_at="2026-11-25T14:00:00Z", observed_monotonic=10.0
        )
        self.assertIsNotNone(parsed1)
        assert parsed1 is not None
        profile1, snap1 = parsed1
        self.assertEqual(profile1.device_id, "cuda:1")
        self.assertEqual(profile1.power_limit_w, 450.0)
        self.assertEqual(snap1.temperature_c, 48)

    def test_u5_cli_doctor_json_matches_schema_v1(self) -> None:
        csv_text = "0, NVIDIA GeForce RTX 4060 Laptop GPU, 8188, 7200, 988, 52, 8, 621.14, 8.9, 115.00\n"

        class FakeProc:
            returncode = 0
            stdout = csv_text

        buf = io.StringIO()
        with patch("subprocess.run", return_value=FakeProc()), redirect_stdout(buf):
            rc = cli_main(["doctor", "--json"])
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(
            set(payload.keys()),
            {"schema_version", "device", "snapshot", "capabilities", "execution_policy"},
        )
        self.assertEqual(payload["execution_policy"]["tier"], "TIER_STANDARD")

    def test_u6_ast_scan_zero_shell_true_across_source(self) -> None:
        src_root = Path(__file__).resolve().parents[1] / "src" / "localgpu_os"
        violations: list[str] = []
        for py_file in src_root.rglob("*.py"):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    for kw in node.keywords:
                        if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                            violations.append(f"{py_file}:{node.lineno}")
        self.assertEqual(violations, [], f"Found forbidden shell=True: {violations}")


if __name__ == "__main__":
    unittest.main()
