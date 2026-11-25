"""Single-GPU NVIDIA hardware detector via nvidia-smi (zero heavy deps, zero shell=True)."""

from __future__ import annotations

import platform
import subprocess
from typing import Protocol

from localgpu_os.domain.enums import ComputeBackend, SensorStatus
from localgpu_os.domain.hardware import HardwareProfile, HardwareSnapshot
from localgpu_os.infrastructure.clock import ClockProtocol, SystemClock

_QUERY_FIELDS = (
    "index",
    "name",
    "memory.total",
    "memory.free",
    "memory.used",
    "temperature.gpu",
    "utilization.gpu",
    "driver_version",
    "compute_cap",
    "power.limit",
)

_ARCH_BY_CC_MAJOR_MINOR: dict[str, str] = {
    "7.5": "Turing",
    "8.0": "Ampere",
    "8.6": "Ampere",
    "8.9": "Ada Lovelace",
    "9.0": "Hopper",
    "10.0": "Blackwell",
    "12.0": "Blackwell",
}


class HardwareDetectorProtocol(Protocol):
    def probe(self, device_index: int = 0) -> tuple[HardwareProfile, HardwareSnapshot]: ...


def _parse_optional_str(raw: str) -> str | None:
    val = raw.strip()
    if not val or val.startswith("[") or val.upper() in {"N/A", "UNKNOWN", "NOT SUPPORTED"}:
        return None
    return val


def _parse_optional_int(raw: str) -> int | None:
    cleaned = _parse_optional_str(raw)
    if cleaned is None:
        return None
    try:
        return int(round(float(cleaned)))
    except ValueError:
        return None


def _parse_optional_float(raw: str) -> float | None:
    cleaned = _parse_optional_str(raw)
    if cleaned is None:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _infer_architecture(compute_cap: str | None) -> str | None:
    if not compute_cap:
        return None
    return _ARCH_BY_CC_MAJOR_MINOR.get(compute_cap.strip())


def _infer_nvenc_support(device_name: str, compute_cap: str | None) -> bool | None:
    upper = device_name.upper()
    if any(tag in upper for tag in ("RTX", "GTX", "QUADRO", "L4", "L40", "T4", "A10")):
        return True
    if compute_cap:
        try:
            major = int(compute_cap.split(".")[0])
            if major >= 6:
                return True
        except ValueError:
            return None
    return None


def parse_nvidia_smi_csv(
    csv_output: str,
    device_index: int,
    observed_at: str,
    observed_monotonic: float,
    freshness_ttl_seconds: float = 10.0,
) -> tuple[HardwareProfile, HardwareSnapshot] | None:
    """Parse nvidia-smi CSV lines and select the target single GPU by index."""
    selected: list[str] | None = None
    for line in csv_output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = [p.strip() for p in stripped.split(",")]
        if len(parts) < len(_QUERY_FIELDS):
            continue
        idx = _parse_optional_int(parts[0])
        if idx == device_index:
            selected = parts
            break

    if selected is None:
        return None

    name = _parse_optional_str(selected[1]) or "Unknown NVIDIA GPU"
    total_vram = _parse_optional_int(selected[2])
    free_vram = _parse_optional_int(selected[3])
    used_vram = _parse_optional_int(selected[4])
    temp_c = _parse_optional_int(selected[5])
    util_pct = _parse_optional_int(selected[6])
    driver_ver = _parse_optional_str(selected[7])
    compute_cap = _parse_optional_str(selected[8])
    power_limit = _parse_optional_float(selected[9])

    temp_supported = temp_c is not None
    sensor_status = SensorStatus.OK if temp_supported else SensorStatus.DEGRADED

    profile = HardwareProfile(
        device_id=f"cuda:{device_index}",
        vendor="NVIDIA",
        device_name=name,
        architecture=_infer_architecture(compute_cap),
        compute_capability=compute_cap,
        driver_version=driver_ver,
        total_vram_mib=total_vram,
        memory_bandwidth_gbps=None,
        power_limit_w=power_limit,
        nvenc_supported=_infer_nvenc_support(name, compute_cap),
        temperature_support=temp_supported,
        backend_support=(ComputeBackend.CUDA, ComputeBackend.CPU_INT8),
    )

    snapshot = HardwareSnapshot(
        device_id=f"cuda:{device_index}",
        observed_at=observed_at,
        observed_monotonic=observed_monotonic,
        freshness_ttl_seconds=freshness_ttl_seconds,
        available_vram_mib=free_vram,
        external_vram_mib=used_vram,
        temperature_c=temp_c,
        utilization_percent=util_pct,
        sensor_status=sensor_status,
    )
    return profile, snapshot


class NvidiaSmiDetector:
    """Probes a single selected NVIDIA GPU via nvidia-smi subprocess (shell=False)."""

    def __init__(
        self,
        clock: ClockProtocol | None = None,
        freshness_ttl_seconds: float = 10.0,
        timeout_seconds: float = 5.0,
    ) -> None:
        self._clock = clock or SystemClock()
        self._freshness_ttl = freshness_ttl_seconds
        self._timeout = timeout_seconds

    def probe(self, device_index: int = 0) -> tuple[HardwareProfile, HardwareSnapshot]:
        observed_at = self._clock.utc_now_iso()
        observed_monotonic = self._clock.monotonic()

        cmd = [
            "nvidia-smi",
            f"--query-gpu={','.join(_QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                shell=False,
                check=False,
            )
        except FileNotFoundError:
            return self._cpu_fallback(observed_at, observed_monotonic)
        except (subprocess.TimeoutExpired, OSError):
            return self._unverified_gpu(device_index, observed_at, observed_monotonic)

        if proc.returncode != 0:
            return self._unverified_gpu(device_index, observed_at, observed_monotonic)

        parsed = parse_nvidia_smi_csv(
            proc.stdout,
            device_index=device_index,
            observed_at=observed_at,
            observed_monotonic=observed_monotonic,
            freshness_ttl_seconds=self._freshness_ttl,
        )
        if parsed is None:
            return self._unverified_gpu(device_index, observed_at, observed_monotonic)
        return parsed

    def _cpu_fallback(
        self, observed_at: str, observed_monotonic: float
    ) -> tuple[HardwareProfile, HardwareSnapshot]:
        cpu_name = platform.processor() or "Generic CPU"
        profile = HardwareProfile(
            device_id="cpu",
            vendor="CPU",
            device_name=cpu_name,
            nvenc_supported=False,
            temperature_support=False,
            backend_support=(ComputeBackend.CPU_INT8,),
        )
        snapshot = HardwareSnapshot(
            device_id="cpu",
            observed_at=observed_at,
            observed_monotonic=observed_monotonic,
            freshness_ttl_seconds=self._freshness_ttl,
            sensor_status=SensorStatus.UNAVAILABLE,
        )
        return profile, snapshot

    def _unverified_gpu(
        self, device_index: int, observed_at: str, observed_monotonic: float
    ) -> tuple[HardwareProfile, HardwareSnapshot]:
        profile = HardwareProfile(
            device_id=f"cuda:{device_index}",
            vendor="NVIDIA",
            device_name="Unverified NVIDIA GPU",
            temperature_support=False,
            backend_support=(ComputeBackend.CUDA, ComputeBackend.CPU_INT8),
        )
        snapshot = HardwareSnapshot(
            device_id=f"cuda:{device_index}",
            observed_at=observed_at,
            observed_monotonic=observed_monotonic,
            freshness_ttl_seconds=self._freshness_ttl,
            available_vram_mib=None,
            external_vram_mib=None,
            temperature_c=None,
            utilization_percent=None,
            sensor_status=SensorStatus.UNAVAILABLE,
        )
        return profile, snapshot
