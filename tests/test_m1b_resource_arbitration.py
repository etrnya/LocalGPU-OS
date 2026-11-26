"""M1-B Resource Arbitration test suite (PRD v1.1.2 Sections 2.2, 2.3, 2.4, 2.6).

Uses Python standard library `unittest` (zero external dependencies, pytest-compatible).
"""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from localgpu_os.application.vram_gate import VramGate
from localgpu_os.domain.compatibility import ResourceCapacity
from localgpu_os.domain.enums import (
    ComputeBackend,
    ErrorCode,
    ReservationState,
    ResourceClass,
    SensorStatus,
    ThermalState,
)
from localgpu_os.domain.hardware import HardwareProfile, HardwareSnapshot
from localgpu_os.domain.lease import (
    FencingTokenMismatchError,
    LeaseStateCorruptedError,
    ResourceRequest,
    VramLockTimeoutError,
)
from localgpu_os.domain.thermal import ThermalGuard, ThermalGuardConfig
from localgpu_os.infrastructure.eviction_provider import (
    EvictionResult,
    OllamaEvictionProvider,
)
from localgpu_os.infrastructure.process_inspector import OsProcessInspector
from localgpu_os.infrastructure.registry_lock import RegistryLock
from localgpu_os.presentation.cli import main as cli_main


class MutableFakeClock:
    def __init__(self, monotonic_val: float = 100.0, utc_iso: str = "2026-11-26T10:00:00Z") -> None:
        self._mono = monotonic_val
        self._utc = utc_iso

    def advance(self, seconds: float) -> None:
        self._mono += seconds

    def set_monotonic(self, value: float) -> None:
        self._mono = value

    def utc_now_iso(self) -> str:
        return self._utc

    def monotonic(self) -> float:
        return self._mono


class FakeDetector:
    def __init__(
        self,
        clock: MutableFakeClock,
        available_vram_mib: int | None = 7000,
        external_vram_mib: int | None = 1188,
        temperature_c: int | None = 50,
        sensor_status: SensorStatus = SensorStatus.OK,
        temp_support: bool = True,
        has_cuda: bool = True,
        snapshot_age_offset: float = 0.0,
    ) -> None:
        self.clock = clock
        self.available_vram_mib = available_vram_mib
        self.external_vram_mib = external_vram_mib
        self.temperature_c = temperature_c
        self.sensor_status = sensor_status
        self.temp_support = temp_support
        self.has_cuda = has_cuda
        self.snapshot_age_offset = snapshot_age_offset
        self.probe_count = 0

    def probe(self, device_index: int = 0) -> tuple[HardwareProfile, HardwareSnapshot]:
        self.probe_count += 1
        backends = (
            (ComputeBackend.CUDA, ComputeBackend.CPU_INT8)
            if self.has_cuda
            else (ComputeBackend.CPU_INT8,)
        )
        profile = HardwareProfile(
            device_id=f"cuda:{device_index}" if self.has_cuda else "cpu",
            vendor="NVIDIA" if self.has_cuda else "CPU",
            device_name="NVIDIA GeForce RTX 4060 Laptop GPU" if self.has_cuda else "Generic CPU",
            architecture="Ada Lovelace" if self.has_cuda else None,
            compute_capability="8.9" if self.has_cuda else None,
            total_vram_mib=8188 if self.has_cuda else None,
            nvenc_supported=True if self.has_cuda else False,
            temperature_support=self.temp_support if self.has_cuda else False,
            backend_support=backends,
        )
        snapshot = HardwareSnapshot(
            device_id=profile.device_id,
            observed_at=self.clock.utc_now_iso(),
            observed_monotonic=self.clock.monotonic() - self.snapshot_age_offset,
            freshness_ttl_seconds=10.0,
            available_vram_mib=self.available_vram_mib if self.has_cuda else None,
            external_vram_mib=self.external_vram_mib if self.has_cuda else None,
            temperature_c=self.temperature_c if self.has_cuda else None,
            utilization_percent=10 if self.has_cuda else None,
            sensor_status=self.sensor_status if self.has_cuda else SensorStatus.UNAVAILABLE,
        )
        return profile, snapshot


class FakeProcessInspector:
    def __init__(self, alive_map: dict[int, bool | None] | None = None) -> None:
        self.alive_map: dict[int, bool | None] = alive_map or {}

    def is_alive(self, pid: int, expected_create_time: float | None = None) -> bool | None:
        return self.alive_map.get(pid, False)

    def current_process_create_time(self) -> float | None:
        return 1700000000.0


class FakeEvictionProvider:
    def __init__(
        self,
        detector: FakeDetector,
        freed_vram_mib: int = 0,
        succeed: bool = True,
    ) -> None:
        self._detector = detector
        self._freed = freed_vram_mib
        self._succeed = succeed
        self.evict_calls = 0

    @property
    def name(self) -> str:
        return "fake-ollama"

    def is_available(self, timeout_seconds: float = 2.0) -> bool:
        return True

    def evict(self, timeout_seconds: float = 10.0) -> EvictionResult:
        self.evict_calls += 1
        if self._succeed and self._detector.available_vram_mib is not None:
            self._detector.available_vram_mib += self._freed
        return EvictionResult(
            provider_name=self.name,
            attempted=True,
            success=self._succeed,
            evicted_models=("qwen2.5:7b",) if self._succeed else (),
            error_code=None if self._succeed else ErrorCode.EVICTION_FAILED,
            message="ok" if self._succeed else "connection refused",
        )


class TestM1BResourceArbitration(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.state_dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # T1: Real cross-OS-process RegistryLock mutual exclusion & timeout
    def test_t1_registry_lock_cross_process_mutual_exclusion(self) -> None:
        lock_path = self.state_dir / "registry.lock"
        src_dir = str(Path(__file__).resolve().parents[1] / "src")
        holder_script = (
            "import sys, time\n"
            "from pathlib import Path\n"
            "from localgpu_os.infrastructure.registry_lock import RegistryLock\n"
            "lock = RegistryLock(Path(sys.argv[1]), timeout_seconds=5.0)\n"
            "with lock:\n"
            "    print('LOCKED', flush=True)\n"
            "    sys.stdin.readline()\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = src_dir + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        with subprocess.Popen(
            [sys.executable, "-c", holder_script, str(lock_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        ) as proc:
            assert proc.stdout is not None and proc.stdin is not None
            ready = proc.stdout.readline().strip()
            self.assertEqual(ready, "LOCKED")

            # Competing acquisition must time out with VRAM_LOCK_TIMEOUT
            contender = RegistryLock(lock_path, timeout_seconds=0.15, poll_interval_seconds=0.02)
            with self.assertRaises(VramLockTimeoutError) as ctx:
                contender.acquire()
            self.assertEqual(ctx.exception.error_code, ErrorCode.VRAM_LOCK_TIMEOUT)

            # Release child lock and verify parent can now acquire cleanly
            proc.stdin.write("go\n")
            proc.stdin.flush()
            proc.wait(timeout=5.0)

        with RegistryLock(lock_path, timeout_seconds=1.0):
            self.assertTrue(lock_path.exists())
        # Persistent lockfile invariant: never unlinked on release
        self.assertTrue(lock_path.exists())

    # T2: Corrupted lease registry fails closed with LEASE_STATE_CORRUPTED
    def test_t2_corrupted_lease_registry_fails_closed(self) -> None:
        (self.state_dir / "leases.json").write_text("{corrupted json", encoding="utf-8")
        clock = MutableFakeClock()
        gate = VramGate(state_dir=self.state_dir, detector=FakeDetector(clock), clock=clock)

        with self.assertRaises(LeaseStateCorruptedError) as ctx:
            gate.acquire_lease(
                owner_id="worker-1",
                requests=[ResourceRequest(ResourceClass.GPU_MEMORY, 1024)],
            )
        self.assertEqual(ctx.exception.error_code, ErrorCode.LEASE_STATE_CORRUPTED)

    # T3: VRAM_pending observation-confirmed transition & no double subtraction
    def test_t3_vram_pending_observation_confirmed_transition(self) -> None:
        clock = MutableFakeClock()
        # 7000 free - 1536 safety = 5464 admission
        detector = FakeDetector(clock, available_vram_mib=7000, external_vram_mib=1188)
        gate = VramGate(state_dir=self.state_dir, detector=detector, clock=clock)

        dec1 = gate.acquire_lease(
            owner_id="worker-a",
            requests=[ResourceRequest(ResourceClass.GPU_MEMORY, 3000)],
        )
        self.assertTrue(dec1.admitted)
        assert dec1.lease is not None
        self.assertEqual(dec1.vram_admission_mib, 7000 - 1536)
        self.assertEqual(dec1.vram_pending_mib, 3000)

        # Transition to ACTIVE BEFORE GPU actually allocates memory -> stays in VRAM_pending!
        active_unconfirmed = gate.transition_state(
            dec1.lease.lease_id, dec1.lease.lease_generation, ReservationState.ACTIVE
        )
        self.assertFalse(active_unconfirmed.vram_observed_confirmed)
        self.assertTrue(active_unconfirmed.counts_toward_pending())
        self.assertEqual(gate.read_registry_snapshot().total_pending_vram_mib("cuda:0"), 3000)

        # Now GPU actually allocates ~3000 MiB (free drops from 7000 -> 4000)
        detector.available_vram_mib = 4000
        active_confirmed = gate.transition_state(
            dec1.lease.lease_id, dec1.lease.lease_generation, ReservationState.ACTIVE
        )
        self.assertTrue(active_confirmed.vram_observed_confirmed)
        self.assertFalse(active_confirmed.counts_toward_pending())
        # Pending drops to 0 so a second GPU_MEMORY-only lease sees 4000 - 1536 = 2464 (no double subtraction!)
        self.assertEqual(gate.read_registry_snapshot().total_pending_vram_mib("cuda:0"), 0)

        dec2 = gate.acquire_lease(
            owner_id="worker-b",
            requests=[ResourceRequest(ResourceClass.GPU_MEMORY, 2000)],
        )
        self.assertTrue(dec2.admitted)
        self.assertEqual(dec2.vram_admission_mib, 4000 - 1536)

    # T4: Fencing token monotonic increment & stale generation rejection
    def test_t4_fencing_token_monotonic_and_stale_rejection(self) -> None:
        clock = MutableFakeClock()
        gate = VramGate(state_dir=self.state_dir, detector=FakeDetector(clock), clock=clock)

        d1 = gate.acquire_lease("w1", [ResourceRequest(ResourceClass.GPU_COMPUTE, 1)])
        assert d1.lease is not None
        self.assertEqual(d1.lease.lease_generation, 1)

        with self.assertRaises(FencingTokenMismatchError) as ctx:
            gate.heartbeat(d1.lease.lease_id, lease_generation=999)
        self.assertEqual(ctx.exception.error_code, ErrorCode.FENCING_TOKEN_MISMATCH)

        gate.release_lease(d1.lease.lease_id, d1.lease.lease_generation)

        # Second lease must have strictly incremented generation
        d2 = gate.acquire_lease("w2", [ResourceRequest(ResourceClass.GPU_COMPUTE, 1)])
        assert d2.lease is not None
        self.assertEqual(d2.lease.lease_generation, 2)

        # Old worker attempting to release with stale lease_id/generation fails
        with self.assertRaises(FencingTokenMismatchError):
            gate.release_lease(d1.lease.lease_id, d1.lease.lease_generation)

    # T5: Expired heartbeat with alive/unverified process -> LEASE_RECLAIM_BLOCKED; dead -> reclaimed
    def test_t5_reclaim_blocked_when_alive_and_reclaimed_when_dead(self) -> None:
        clock = MutableFakeClock(monotonic_val=100.0)
        inspector = FakeProcessInspector(alive_map={4242: True})
        gate = VramGate(
            state_dir=self.state_dir,
            detector=FakeDetector(clock),
            clock=clock,
            process_inspector=inspector,
        )

        d1 = gate.acquire_lease(
            owner_id="worker-hung",
            requests=[
                ResourceRequest(ResourceClass.GPU_COMPUTE, 1),
                ResourceRequest(ResourceClass.GPU_MEMORY, 2000),
            ],
            ttl_seconds=10.0,
            pid=4242,
            pid_create_time=1700000000.0,
        )
        self.assertTrue(d1.admitted)

        # Advance past TTL while PID 4242 is STILL alive -> MUST NOT reclaim
        clock.advance(30.0)
        d2 = gate.acquire_lease(
            owner_id="worker-new",
            requests=[ResourceRequest(ResourceClass.GPU_COMPUTE, 1)],
        )
        self.assertFalse(d2.admitted)
        self.assertIn(ErrorCode.LEASE_RECLAIM_BLOCKED, d2.error_codes)

        # Unverified (None) must ALSO block reclaim
        inspector.alive_map[4242] = None
        d3 = gate.acquire_lease(
            owner_id="worker-new",
            requests=[ResourceRequest(ResourceClass.GPU_COMPUTE, 1)],
        )
        self.assertFalse(d3.admitted)
        self.assertIn(ErrorCode.LEASE_RECLAIM_BLOCKED, d3.error_codes)

        # Once confirmed dead (False) -> safely reclaimed and new lease admitted!
        inspector.alive_map[4242] = False
        d4 = gate.acquire_lease(
            owner_id="worker-new",
            requests=[ResourceRequest(ResourceClass.GPU_COMPUTE, 1)],
        )
        self.assertTrue(d4.admitted)
        assert d4.lease is not None
        self.assertEqual(d4.lease.owner_id, "worker-new")

    # T6: All-or-nothing multi-resource atomicity & Core OS policy supremacy
    def test_t6_all_or_nothing_atomicity_and_policy_supremacy(self) -> None:
        clock = MutableFakeClock()
        detector = FakeDetector(clock, available_vram_mib=4000)  # admission = 2464 MiB
        gate = VramGate(
            state_dir=self.state_dir,
            detector=detector,
            clock=clock,
            capacity=ResourceCapacity(cpu_threads_total=8, system_memory_mib_available=16384),
        )

        # Bundle asks for valid CPU (2 threads) + excessive GPU_MEMORY (5000 MiB) -> 0 partial leases!
        rejected = gate.acquire_lease(
            owner_id="bundle-fail",
            requests=[
                ResourceRequest(ResourceClass.CPU_COMPUTE, 2),
                ResourceRequest(ResourceClass.GPU_COMPUTE, 1),
                ResourceRequest(ResourceClass.GPU_MEMORY, 5000),
            ],
        )
        self.assertFalse(rejected.admitted)
        self.assertIn(ErrorCode.VRAM_INSUFFICIENT, rejected.error_codes)
        self.assertEqual(len(gate.read_registry_snapshot().active_leases()), 0)

        # Valid GPU_COMPUTE lease admitted
        g1 = gate.acquire_lease(
            owner_id="gpu-1",
            requests=[ResourceRequest(ResourceClass.GPU_COMPUTE, 1, exclusive=False)],
        )
        self.assertTrue(g1.admitted)

        # Second GPU_COMPUTE lease declaring exclusive=False MUST still be blocked (Core OS supremacy)
        g2 = gate.acquire_lease(
            owner_id="gpu-2",
            requests=[ResourceRequest(ResourceClass.GPU_COMPUTE, 1, exclusive=False)],
        )
        self.assertFalse(g2.admitted)
        self.assertIn(ErrorCode.VRAM_LOCK_TIMEOUT, g2.error_codes)

        # Pure CPU_COMPUTE lease CAN run in parallel with GPU_COMPUTE
        c1 = gate.acquire_lease(
            owner_id="cpu-1",
            requests=[ResourceRequest(ResourceClass.CPU_COMPUTE, 4)],
        )
        self.assertTrue(c1.admitted)

    # T7: Pure CPU task succeeds even when GPU probe fails (Fail-Safe matrix)
    def test_t7_cpu_task_degrades_independently_when_gpu_unverified(self) -> None:
        clock = MutableFakeClock()
        detector = FakeDetector(clock, available_vram_mib=None, sensor_status=SensorStatus.UNAVAILABLE)
        gate = VramGate(
            state_dir=self.state_dir,
            detector=detector,
            clock=clock,
            capacity=ResourceCapacity(cpu_threads_total=4, system_memory_mib_available=8192),
        )

        gpu_dec = gate.acquire_lease("gpu-job", [ResourceRequest(ResourceClass.GPU_COMPUTE, 1)])
        self.assertFalse(gpu_dec.admitted)

        cpu_dec = gate.acquire_lease("cpu-job", [ResourceRequest(ResourceClass.CPU_COMPUTE, 2)])
        self.assertTrue(cpu_dec.admitted)
        assert cpu_dec.lease is not None
        self.assertEqual(cpu_dec.lease.device_id, "cpu")

    # T8: ThermalGuard 4-state transitions, hysteresis, and sensor recovery
    def test_t8_thermal_guard_four_states_and_hysteresis(self) -> None:
        clock = MutableFakeClock(monotonic_val=100.0)
        detector = FakeDetector(clock, temperature_c=55)
        guard = ThermalGuard(
            ThermalGuardConfig(
                cooling_temp_c=82,
                resume_temp_c=72,
                critical_temp_c=88,
                cooling_hysteresis_seconds=15.0,
                critical_hysteresis_seconds=60.0,
                sensor_recovery_samples=2,
            )
        )

        # NORMAL -> COOLING at 83C
        detector.temperature_c = 83
        p, s = detector.probe()
        ev = guard.evaluate(p, s, clock.monotonic())
        self.assertEqual(ev.state, ThermalState.COOLING)
        self.assertFalse(ev.gpu_admission_allowed)
        self.assertIn(ErrorCode.THERMAL_COOLING_WAIT, ev.error_codes)

        # Drops to 70C, holds 10s (< 15s) -> still COOLING
        detector.temperature_c = 70
        p, s = detector.probe()
        self.assertEqual(guard.evaluate(p, s, clock.monotonic()).state, ThermalState.COOLING)
        clock.advance(10.0)
        p, s = detector.probe()
        self.assertEqual(guard.evaluate(p, s, clock.monotonic()).state, ThermalState.COOLING)

        # Holds another 6s (total 16s >= 15s) -> recovers to NORMAL
        clock.advance(6.0)
        p, s = detector.probe()
        ev_ok = guard.evaluate(p, s, clock.monotonic())
        self.assertEqual(ev_ok.state, ThermalState.NORMAL)
        self.assertTrue(ev_ok.gpu_admission_allowed)

        # Escalate to CRITICAL at 89C -> requires 60s <= 72C
        detector.temperature_c = 89
        p, s = detector.probe()
        ev_crit = guard.evaluate(p, s, clock.monotonic())
        self.assertEqual(ev_crit.state, ThermalState.CRITICAL)
        self.assertIn(ErrorCode.THERMAL_CRITICAL_ABORT, ev_crit.error_codes)

        detector.temperature_c = 68
        p, s = detector.probe()
        guard.evaluate(p, s, clock.monotonic())
        clock.advance(30.0)
        p, s = detector.probe()
        self.assertEqual(guard.evaluate(p, s, clock.monotonic()).state, ThermalState.CRITICAL)
        clock.advance(31.0)
        p, s = detector.probe()
        self.assertEqual(guard.evaluate(p, s, clock.monotonic()).state, ThermalState.NORMAL)

        # Sensor failure -> SENSOR_DEGRADED -> requires 2 consecutive valid samples
        detector.sensor_status = SensorStatus.DEGRADED
        detector.temperature_c = None
        p, s = detector.probe()
        ev_deg = guard.evaluate(p, s, clock.monotonic())
        self.assertEqual(ev_deg.state, ThermalState.SENSOR_DEGRADED)
        self.assertIn(ErrorCode.THERMAL_SENSOR_DEGRADED, ev_deg.error_codes)

        detector.sensor_status = SensorStatus.OK
        detector.temperature_c = 50
        p, s = detector.probe()
        self.assertEqual(guard.evaluate(p, s, clock.monotonic()).state, ThermalState.SENSOR_DEGRADED)
        p, s = detector.probe()
        self.assertEqual(guard.evaluate(p, s, clock.monotonic()).state, ThermalState.NORMAL)

    # T9: EvictionProvider + mandatory post-eviction VRAM re-probe
    def test_t9_eviction_provider_reprobe_success_and_failure(self) -> None:
        clock = MutableFakeClock()
        # Initial free 3000 - 1536 safety = 1464 admission; request needs 3500 MiB
        detector = FakeDetector(clock, available_vram_mib=3000)

        # Case A: eviction claims success AND re-probe shows +3000 MiB freed -> admitted
        evictor_ok = FakeEvictionProvider(detector, freed_vram_mib=3000, succeed=True)
        gate_ok = VramGate(
            state_dir=self.state_dir,
            detector=detector,
            clock=clock,
            eviction_provider=evictor_ok,
        )
        dec_ok = gate_ok.acquire_lease(
            owner_id="ocr-worker",
            requests=[ResourceRequest(ResourceClass.GPU_MEMORY, 3500)],
            allow_eviction=True,
        )
        self.assertTrue(dec_ok.admitted)
        self.assertTrue(dec_ok.eviction_attempted)
        self.assertEqual(evictor_ok.evict_calls, 1)
        self.assertGreaterEqual(detector.probe_count, 2)

        # Case B: eviction returns success=True but VRAM did NOT actually free -> EVICTION_FAILED!
        detector2 = FakeDetector(clock, available_vram_mib=3000)
        evictor_noop = FakeEvictionProvider(detector2, freed_vram_mib=0, succeed=True)
        gate_fail = VramGate(
            state_dir=self.state_dir / "case_b",
            detector=detector2,
            clock=clock,
            eviction_provider=evictor_noop,
        )
        dec_fail = gate_fail.acquire_lease(
            owner_id="ocr-worker",
            requests=[ResourceRequest(ResourceClass.GPU_MEMORY, 3500)],
            allow_eviction=True,
        )
        self.assertFalse(dec_fail.admitted)
        self.assertTrue(dec_fail.eviction_attempted)
        self.assertIn(ErrorCode.EVICTION_FAILED, dec_fail.error_codes)
        self.assertIn(ErrorCode.VRAM_INSUFFICIENT, dec_fail.error_codes)

    # T10: CLI `lgpu status --json` contract
    def test_t10_cli_status_json_contract(self) -> None:
        csv_text = "0, NVIDIA GeForce RTX 4060 Laptop GPU, 8188, 7200, 988, 48, 5, 621.14, 8.9, 115.00\n"

        class FakeProc:
            returncode = 0
            stdout = csv_text

        buf = io.StringIO()
        with patch("subprocess.run", return_value=FakeProc()), redirect_stdout(buf):
            rc = cli_main(["status", "--json", "--state-dir", str(self.state_dir)])
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["schema_version"], "1.0")
        self.assertEqual(set(payload.keys()), {"schema_version", "device", "snapshot", "arbitration"})
        self.assertEqual(payload["arbitration"]["thermal_state"], "NORMAL")
        self.assertEqual(payload["arbitration"]["vram_admission_mib"], 7200 - 1536)
        self.assertEqual(payload["arbitration"]["active_lease_count"], 0)

    # T11: Real OsProcessInspector verification (current PID, exited child, create_time mismatch)
    def test_t11_os_process_inspector_real_process_checks(self) -> None:
        inspector = OsProcessInspector()
        my_pid = os.getpid()
        my_ctime = inspector.current_process_create_time()
        self.assertTrue(inspector.is_alive(my_pid, expected_create_time=my_ctime))

        # On Windows where create_time is available, mismatched create_time signals PID reuse -> False
        if my_ctime is not None:
            self.assertFalse(inspector.is_alive(my_pid, expected_create_time=my_ctime - 3600.0))

        # Spawn a short-lived child process, wait for exit, then confirm is_alive == False
        with subprocess.Popen([sys.executable, "-c", "pass"]) as child:
            child_pid = child.pid
            child.wait(timeout=5.0)
        self.assertFalse(inspector.is_alive(child_pid))

    # T12: Core OS restart recovery (monotonic clock reset) still blocks reclaim of alive worker
    def test_t12_core_os_crash_restart_monotonic_reset_blocks_live_reclaim(self) -> None:
        clock1 = MutableFakeClock(monotonic_val=5000.0, utc_iso="2026-11-26T10:00:00Z")
        inspector = FakeProcessInspector(alive_map={7777: True})
        gate1 = VramGate(
            state_dir=self.state_dir,
            detector=FakeDetector(clock1),
            clock=clock1,
            process_inspector=inspector,
        )
        d1 = gate1.acquire_lease(
            owner_id="surviving-worker",
            requests=[ResourceRequest(ResourceClass.GPU_COMPUTE, 1)],
            ttl_seconds=10.0,
            pid=7777,
            pid_create_time=1700000000.0,
        )
        self.assertTrue(d1.admitted)

        # Simulate Core OS crash & restart: fresh VramGate, monotonic clock reset to 5.0 (< 5000.0),
        # UTC moved 5 minutes ahead -> expired candidate, but PID 7777 is still alive!
        clock2 = MutableFakeClock(monotonic_val=5.0, utc_iso="2026-11-26T10:05:00Z")
        gate2 = VramGate(
            state_dir=self.state_dir,
            detector=FakeDetector(clock2),
            clock=clock2,
            process_inspector=inspector,
        )
        d2 = gate2.acquire_lease(
            owner_id="new-worker",
            requests=[ResourceRequest(ResourceClass.GPU_COMPUTE, 1)],
        )
        self.assertFalse(d2.admitted)
        self.assertIn(ErrorCode.LEASE_RECLAIM_BLOCKED, d2.error_codes)

    # T13: OllamaEvictionProvider HTTP flow (/api/ps -> /api/generate keep_alive=0)
    def test_t13_ollama_eviction_provider_http_flow(self) -> None:
        class _FakeResp:
            def __init__(self, body: bytes, status: int = 200) -> None:
                self._body = body
                self.status = status

            def read(self) -> bytes:
                return self._body

            def __enter__(self) -> _FakeResp:
                return self

            def __exit__(self, *args: object) -> None:
                return None

        called_urls: list[str] = []

        def _fake_urlopen(req: object, timeout: float = 10.0) -> _FakeResp:
            url = getattr(req, "full_url", "")
            called_urls.append(url)
            if url.endswith("/api/ps"):
                return _FakeResp(json.dumps({"models": [{"name": "llama3:8b"}]}).encode("utf-8"))
            return _FakeResp(b'{"done":true}')

        provider = OllamaEvictionProvider()
        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            res = provider.evict(timeout_seconds=3.0)
        self.assertTrue(res.success)
        self.assertEqual(res.evicted_models, ("llama3:8b",))
        self.assertEqual(len(called_urls), 2)
        self.assertTrue(called_urls[0].endswith("/api/ps"))
        self.assertTrue(called_urls[1].endswith("/api/generate"))


if __name__ == "__main__":
    unittest.main()
