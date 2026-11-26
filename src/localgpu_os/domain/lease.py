"""Lease, ResourceRequest, and Arbitration domain models (GLOSSARY.md v1.1.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from localgpu_os.domain.enums import (
    ErrorCode,
    ReservationState,
    ResourceClass,
    ThermalState,
)

LEASE_SCHEMA_VERSION = "1.0"

_DEFAULT_UNITS: dict[ResourceClass, str] = {
    ResourceClass.GPU_COMPUTE: "slot",
    ResourceClass.GPU_MEMORY: "MiB",
    ResourceClass.CPU_COMPUTE: "threads",
    ResourceClass.GPU_ENCODE: "session",
    ResourceClass.SYSTEM_MEMORY: "MiB",
}


class ArbitrationError(Exception):
    """Base domain exception carrying a canonical GLOSSARY ErrorCode."""

    def __init__(self, error_code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


class VramLockTimeoutError(ArbitrationError):
    def __init__(self, message: str = "Timed out waiting for RegistryLock.") -> None:
        super().__init__(ErrorCode.VRAM_LOCK_TIMEOUT, message)


class LeaseStateCorruptedError(ArbitrationError):
    def __init__(self, message: str = "Lease registry state is corrupted.") -> None:
        super().__init__(ErrorCode.LEASE_STATE_CORRUPTED, message)


class FencingTokenMismatchError(ArbitrationError):
    def __init__(self, message: str = "Fencing token (lease_generation) mismatch.") -> None:
        super().__init__(ErrorCode.FENCING_TOKEN_MISMATCH, message)


class LeaseReclaimBlockedError(ArbitrationError):
    def __init__(
        self, message: str = "Cannot reclaim lease while original owner stop is unverified."
    ) -> None:
        super().__init__(ErrorCode.LEASE_RECLAIM_BLOCKED, message)


@dataclass(frozen=True)
class ResourceRequest:
    """Single resource demand within an all-or-nothing request bundle."""

    resource_class: ResourceClass
    amount: int
    unit: str = ""
    exclusive: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.amount, int) or isinstance(self.amount, bool) or self.amount <= 0:
            raise ValueError(f"ResourceRequest amount must be a positive int, got {self.amount!r}")
        if not self.unit:
            object.__setattr__(self, "unit", _DEFAULT_UNITS[self.resource_class])

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_class": self.resource_class.value,
            "amount": self.amount,
            "unit": self.unit,
            "exclusive": self.exclusive,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResourceRequest:
        if not isinstance(data, dict):
            raise LeaseStateCorruptedError("ResourceRequest entry must be a JSON object.")
        try:
            rc = ResourceClass(data["resource_class"])
            amount = data["amount"]
            unit = str(data.get("unit", _DEFAULT_UNITS[rc]))
            exclusive = bool(data.get("exclusive", False))
            return cls(resource_class=rc, amount=amount, unit=unit, exclusive=exclusive)
        except (KeyError, ValueError, TypeError) as exc:
            raise LeaseStateCorruptedError(f"Invalid ResourceRequest: {exc}") from exc


@dataclass(frozen=True)
class VramLease:
    """Cooperative resource lease with fencing token and pending-VRAM observation state."""

    lease_id: str
    lease_generation: int
    owner_id: str
    device_id: str
    requests: tuple[ResourceRequest, ...]
    reserved_vram_mib: int
    reservation_state: ReservationState
    vram_observed_confirmed: bool
    baseline_free_vram_mib: int | None
    created_at: str
    heartbeat_at: str
    heartbeat_monotonic: float
    ttl_seconds: float
    pid: int | None = None
    pid_create_time: float | None = None

    def counts_toward_pending(self) -> bool:
        """Whether reserved_vram_mib must still be counted in VRAM_pending (PRD 2.2 B)."""
        if self.reserved_vram_mib <= 0:
            return False
        if self.reservation_state in (ReservationState.RESERVED, ReservationState.STARTING):
            return True
        if self.reservation_state in (ReservationState.ACTIVE, ReservationState.RELEASING):
            return not self.vram_observed_confirmed
        return False

    def uses_gpu(self) -> bool:
        return any(
            req.resource_class
            in (ResourceClass.GPU_COMPUTE, ResourceClass.GPU_MEMORY, ResourceClass.GPU_ENCODE)
            for req in self.requests
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "lease_generation": self.lease_generation,
            "owner_id": self.owner_id,
            "device_id": self.device_id,
            "requests": [r.to_dict() for r in self.requests],
            "reserved_vram_mib": self.reserved_vram_mib,
            "reservation_state": self.reservation_state.value,
            "vram_observed_confirmed": self.vram_observed_confirmed,
            "baseline_free_vram_mib": self.baseline_free_vram_mib,
            "created_at": self.created_at,
            "heartbeat_at": self.heartbeat_at,
            "heartbeat_monotonic": self.heartbeat_monotonic,
            "ttl_seconds": self.ttl_seconds,
            "pid": self.pid,
            "pid_create_time": self.pid_create_time,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VramLease:
        if not isinstance(data, dict):
            raise LeaseStateCorruptedError("Lease entry must be a JSON object.")
        try:
            raw_reqs = data["requests"]
            if not isinstance(raw_reqs, list):
                raise LeaseStateCorruptedError("Lease requests must be a list.")
            requests = tuple(ResourceRequest.from_dict(item) for item in raw_reqs)
            gen = data["lease_generation"]
            if not isinstance(gen, int) or isinstance(gen, bool) or gen < 1:
                raise LeaseStateCorruptedError(f"Invalid lease_generation: {gen!r}")
            reserved = data["reserved_vram_mib"]
            if not isinstance(reserved, int) or isinstance(reserved, bool) or reserved < 0:
                raise LeaseStateCorruptedError(f"Invalid reserved_vram_mib: {reserved!r}")
            ttl = float(data["ttl_seconds"])
            if ttl <= 0:
                raise LeaseStateCorruptedError(f"Invalid ttl_seconds: {ttl!r}")
            baseline = data.get("baseline_free_vram_mib")
            if baseline is not None and (not isinstance(baseline, int) or isinstance(baseline, bool)):
                raise LeaseStateCorruptedError(f"Invalid baseline_free_vram_mib: {baseline!r}")
            pid = data.get("pid")
            if pid is not None and (not isinstance(pid, int) or isinstance(pid, bool)):
                raise LeaseStateCorruptedError(f"Invalid pid: {pid!r}")
            pid_ctime = data.get("pid_create_time")
            if pid_ctime is not None:
                pid_ctime = float(pid_ctime)
            return cls(
                lease_id=str(data["lease_id"]),
                lease_generation=gen,
                owner_id=str(data["owner_id"]),
                device_id=str(data["device_id"]),
                requests=requests,
                reserved_vram_mib=reserved,
                reservation_state=ReservationState(data["reservation_state"]),
                vram_observed_confirmed=bool(data["vram_observed_confirmed"]),
                baseline_free_vram_mib=baseline,
                created_at=str(data["created_at"]),
                heartbeat_at=str(data["heartbeat_at"]),
                heartbeat_monotonic=float(data["heartbeat_monotonic"]),
                ttl_seconds=ttl,
                pid=pid,
                pid_create_time=pid_ctime,
            )
        except LeaseStateCorruptedError:
            raise
        except (KeyError, ValueError, TypeError) as exc:
            raise LeaseStateCorruptedError(f"Corrupted VramLease record: {exc}") from exc


@dataclass(frozen=True)
class LeaseRegistryState:
    """Persistent state of the Lease Registry protected by RegistryLock."""

    schema_version: str = LEASE_SCHEMA_VERSION
    generation: int = 0
    updated_at: str = ""
    leases: tuple[VramLease, ...] = ()

    def active_leases(self) -> tuple[VramLease, ...]:
        return tuple(
            lease for lease in self.leases if lease.reservation_state != ReservationState.RELEASED
        )

    def total_pending_vram_mib(self, device_id: str | None = None) -> int:
        total = 0
        for lease in self.active_leases():
            if device_id is not None and lease.device_id != device_id:
                continue
            if lease.counts_toward_pending():
                total += lease.reserved_vram_mib
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generation": self.generation,
            "updated_at": self.updated_at,
            "leases": [lease.to_dict() for lease in self.leases],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LeaseRegistryState:
        if not isinstance(data, dict):
            raise LeaseStateCorruptedError("Lease registry root must be a JSON object.")
        schema_version = data.get("schema_version")
        if schema_version != LEASE_SCHEMA_VERSION:
            raise LeaseStateCorruptedError(
                f"Unsupported lease registry schema_version: {schema_version!r}"
            )
        generation = data.get("generation")
        if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
            raise LeaseStateCorruptedError(f"Invalid registry generation: {generation!r}")
        raw_leases = data.get("leases")
        if not isinstance(raw_leases, list):
            raise LeaseStateCorruptedError("Registry leases field must be a list.")
        leases = tuple(VramLease.from_dict(item) for item in raw_leases)
        return cls(
            schema_version=schema_version,
            generation=generation,
            updated_at=str(data.get("updated_at", "")),
            leases=leases,
        )


@dataclass(frozen=True)
class AdmissionDecision:
    """Outcome of VramGate all-or-nothing multi-resource admission control."""

    admitted: bool
    lease: VramLease | None
    vram_admission_mib: int
    vram_pending_mib: int
    vram_safety_mib: int
    thermal_state: ThermalState
    error_codes: tuple[ErrorCode, ...] = ()
    warnings: tuple[str, ...] = ()
    eviction_attempted: bool = False
