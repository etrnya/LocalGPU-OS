"""Worker Manifest Contract v1.0, JsonFileManifest, CancelSignal, and Runtime result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

from localgpu_os.domain.enums import (
    ComputeBackend,
    ErrorCode,
    NetworkPolicy,
    ResourceClass,
)
from localgpu_os.domain.lease import ArbitrationError, ResourceRequest

WORKER_MANIFEST_SCHEMA_VERSION = "1.0"
CANCEL_SIGNAL_SCHEMA_VERSION = "1.0"
WORKER_RUN_RESULT_SCHEMA_VERSION = "1.0"

_WORKER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_MODULE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
_FORBIDDEN_BASE64_KEYS = frozenset(
    {
        "base64",
        "content_base64",
        "data_base64",
        "file_base64",
        "file_data",
        "data_uri",
        "inline_bytes",
        "inline_content",
    }
)
_DATA_URI_BASE64_RE = re.compile(r"^data:[^;]+;base64,", re.IGNORECASE)


class WorkerRuntimeError(ArbitrationError):
    """Base exception for M1-C Worker Runtime errors."""


class WorkerManifestInvalidError(WorkerRuntimeError):
    def __init__(self, message: str = "Worker manifest violates Contract v1.0.") -> None:
        super().__init__(ErrorCode.WORKER_MANIFEST_INVALID, message)


class WorkerEnvNotInstalledError(WorkerRuntimeError):
    def __init__(self, message: str = "Worker isolated .venv Python executable not found.") -> None:
        super().__init__(ErrorCode.WORKER_ENV_NOT_INSTALLED, message)


class DiskSpaceInsufficientError(WorkerRuntimeError):
    def __init__(self, message: str = "Insufficient free disk space for worker execution.") -> None:
        super().__init__(ErrorCode.DISK_SPACE_INSUFFICIENT, message)


class WorkerCancelledError(WorkerRuntimeError):
    def __init__(self, message: str = "Worker was cancelled.") -> None:
        super().__init__(ErrorCode.WORKER_CANCELLED, message)


class WorkerExecutionFailedError(WorkerRuntimeError):
    def __init__(self, message: str = "Worker execution failed.") -> None:
        super().__init__(ErrorCode.WORKER_EXECUTION_FAILED, message)


def validate_worker_id(worker_id: str) -> str:
    """Validate worker_id to strictly prevent path traversal and invalid identifiers."""
    if not isinstance(worker_id, str) or not _WORKER_ID_RE.fullmatch(worker_id):
        raise WorkerManifestInvalidError(
            f"Invalid worker_id {worker_id!r}: must match {_WORKER_ID_RE.pattern}."
        )
    return worker_id


@dataclass(frozen=True)
class WorkerFallbackPolicy:
    """Explicit CPU fallback declaration inside Worker Manifest v1.0 (PRD 2.5)."""

    cpu_supported: bool = False
    cpu_backend: ComputeBackend = ComputeBackend.CPU_INT8
    max_cpu_threads: int = 2
    max_system_memory_mib: int = 2048

    def __post_init__(self) -> None:
        if not isinstance(self.cpu_supported, bool):
            raise WorkerManifestInvalidError("fallback_policy.cpu_supported must be a bool.")
        if (
            not isinstance(self.max_cpu_threads, int)
            or isinstance(self.max_cpu_threads, bool)
            or self.max_cpu_threads <= 0
        ):
            raise WorkerManifestInvalidError(
                f"fallback_policy.max_cpu_threads must be a positive int, got {self.max_cpu_threads!r}."
            )
        if (
            not isinstance(self.max_system_memory_mib, int)
            or isinstance(self.max_system_memory_mib, bool)
            or self.max_system_memory_mib <= 0
        ):
            raise WorkerManifestInvalidError(
                "fallback_policy.max_system_memory_mib must be a positive int, "
                f"got {self.max_system_memory_mib!r}."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_supported": self.cpu_supported,
            "cpu_backend": self.cpu_backend.value,
            "max_cpu_threads": self.max_cpu_threads,
            "max_system_memory_mib": self.max_system_memory_mib,
        }

    @classmethod
    def from_dict(cls, data: Any) -> WorkerFallbackPolicy:
        if not isinstance(data, dict):
            raise WorkerManifestInvalidError("fallback_policy must be a JSON object.")
        if "cpu_supported" not in data or not isinstance(data["cpu_supported"], bool):
            raise WorkerManifestInvalidError("fallback_policy.cpu_supported must be a boolean.")
        try:
            backend = ComputeBackend(data.get("cpu_backend", ComputeBackend.CPU_INT8.value))
        except ValueError as exc:
            raise WorkerManifestInvalidError(f"Invalid fallback_policy.cpu_backend: {exc}") from exc
        if backend != ComputeBackend.CPU_INT8:
            raise WorkerManifestInvalidError(
                f"fallback_policy.cpu_backend must be 'cpu_int8', got {backend.value!r}."
            )
        max_threads = data.get("max_cpu_threads")
        max_mem = data.get("max_system_memory_mib")
        if max_threads is None or max_mem is None:
            raise WorkerManifestInvalidError(
                "fallback_policy must declare max_cpu_threads and max_system_memory_mib."
            )
        return cls(
            cpu_supported=data["cpu_supported"],
            cpu_backend=backend,
            max_cpu_threads=max_threads,
            max_system_memory_mib=max_mem,
        )


@dataclass(frozen=True)
class WorkerFileLimits:
    """Large-file and disk boundary limits for json_file_manifest (PRD 2.5)."""

    max_file_size_bytes: int = 100 * 1024 * 1024  # 100 MiB default
    min_free_disk_mib: int = 512
    allow_base64_payload: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_file_size_bytes, int)
            or isinstance(self.max_file_size_bytes, bool)
            or self.max_file_size_bytes <= 0
        ):
            raise WorkerManifestInvalidError(
                f"file_limits.max_file_size_bytes must be a positive int, got {self.max_file_size_bytes!r}."
            )
        if (
            not isinstance(self.min_free_disk_mib, int)
            or isinstance(self.min_free_disk_mib, bool)
            or self.min_free_disk_mib < 0
        ):
            raise WorkerManifestInvalidError(
                f"file_limits.min_free_disk_mib must be a non-negative int, got {self.min_free_disk_mib!r}."
            )
        if self.allow_base64_payload is not False:
            raise WorkerManifestInvalidError(
                "file_limits.allow_base64_payload MUST be false (Base64 file payloads are forbidden)."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_file_size_bytes": self.max_file_size_bytes,
            "min_free_disk_mib": self.min_free_disk_mib,
            "allow_base64_payload": self.allow_base64_payload,
        }

    @classmethod
    def from_dict(cls, data: Any) -> WorkerFileLimits:
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise WorkerManifestInvalidError("file_limits must be a JSON object.")
        return cls(
            max_file_size_bytes=data.get("max_file_size_bytes", 100 * 1024 * 1024),
            min_free_disk_mib=data.get("min_free_disk_mib", 512),
            allow_base64_payload=data.get("allow_base64_payload", False),
        )


@dataclass(frozen=True)
class WorkerManifest:
    """Validated Worker Manifest Contract v1.0 (PRD 2.5).

    Note: Declares `module` and `args` only (never PATH executables or shell commands).
    Any `concurrency_limit` declared here cannot relax Core OS Resource Policy.
    """

    schema_version: str
    worker_id: str
    name: str
    version: str
    module: str
    args: tuple[str, ...]
    resources: tuple[ResourceRequest, ...]
    fallback_policy: WorkerFallbackPolicy
    network_policy: NetworkPolicy = NetworkPolicy.RUNTIME_OFFLINE
    cancel_grace_seconds: float = 5.0
    timeout_seconds: float = 300.0
    poll_interval_seconds: float = 0.25
    concurrency_limit: int = 1
    file_limits: WorkerFileLimits = field(default_factory=WorkerFileLimits)

    def __post_init__(self) -> None:
        if self.schema_version != WORKER_MANIFEST_SCHEMA_VERSION:
            raise WorkerManifestInvalidError(
                f"Unsupported worker manifest schema_version: {self.schema_version!r} "
                f"(expected {WORKER_MANIFEST_SCHEMA_VERSION!r})."
            )
        validate_worker_id(self.worker_id)
        if not isinstance(self.name, str) or not self.name.strip():
            raise WorkerManifestInvalidError("Worker manifest 'name' must be a non-empty string.")
        if not isinstance(self.version, str) or not self.version.strip():
            raise WorkerManifestInvalidError("Worker manifest 'version' must be a non-empty string.")
        if not isinstance(self.module, str) or not _MODULE_NAME_RE.fullmatch(self.module):
            raise WorkerManifestInvalidError(
                f"Invalid Python module path {self.module!r}: must be a dotted identifier."
            )
        if not all(isinstance(a, str) for a in self.args):
            raise WorkerManifestInvalidError("Worker manifest 'args' must be a list of strings.")
        if not self.resources:
            raise WorkerManifestInvalidError(
                "Worker manifest 'resources' must declare at least one ResourceRequest."
            )
        if not isinstance(self.cancel_grace_seconds, (int, float)) or isinstance(
            self.cancel_grace_seconds, bool
        ) or self.cancel_grace_seconds <= 0:
            raise WorkerManifestInvalidError("cancel_grace_seconds must be > 0.")
        if not isinstance(self.timeout_seconds, (int, float)) or isinstance(
            self.timeout_seconds, bool
        ) or self.timeout_seconds <= 0:
            raise WorkerManifestInvalidError("timeout_seconds must be > 0.")
        if (
            not isinstance(self.poll_interval_seconds, (int, float))
            or isinstance(self.poll_interval_seconds, bool)
            or self.poll_interval_seconds <= 0
            or self.poll_interval_seconds > 1.0
        ):
            raise WorkerManifestInvalidError(
                "poll_interval_seconds must satisfy 0 < poll_interval_seconds <= 1.0 (PRD 2.5)."
            )
        if (
            not isinstance(self.concurrency_limit, int)
            or isinstance(self.concurrency_limit, bool)
            or self.concurrency_limit <= 0
        ):
            raise WorkerManifestInvalidError("concurrency_limit must be a positive int.")

    def uses_gpu(self) -> bool:
        return any(
            r.resource_class
            in (ResourceClass.GPU_COMPUTE, ResourceClass.GPU_MEMORY, ResourceClass.GPU_ENCODE)
            for r in self.resources
        )

    def cpu_fallback_requests(self) -> tuple[ResourceRequest, ...]:
        """Build conservative CPU_COMPUTE + SYSTEM_MEMORY requests for CPU fallback."""
        return (
            ResourceRequest(
                resource_class=ResourceClass.CPU_COMPUTE,
                amount=self.fallback_policy.max_cpu_threads,
            ),
            ResourceRequest(
                resource_class=ResourceClass.SYSTEM_MEMORY,
                amount=self.fallback_policy.max_system_memory_mib,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "worker_id": self.worker_id,
            "name": self.name,
            "version": self.version,
            "entrypoint": {
                "module": self.module,
                "args": list(self.args),
            },
            "resources": [r.to_dict() for r in self.resources],
            "fallback_policy": self.fallback_policy.to_dict(),
            "network_policy": self.network_policy.value,
            "cancel_grace_seconds": self.cancel_grace_seconds,
            "timeout_seconds": self.timeout_seconds,
            "poll_interval_seconds": self.poll_interval_seconds,
            "concurrency_limit": self.concurrency_limit,
            "file_limits": self.file_limits.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Any) -> WorkerManifest:
        if not isinstance(data, dict):
            raise WorkerManifestInvalidError("Worker manifest root must be a JSON object.")

        # Reject forbidden shell / PATH executable fields
        for forbidden_key in ("command", "shell", "executable", "cmd"):
            if forbidden_key in data:
                raise WorkerManifestInvalidError(
                    f"Forbidden field {forbidden_key!r}: manifest must declare 'entrypoint.module' "
                    "and 'entrypoint.args' only (zero PATH/shell dependency)."
                )

        entrypoint = data.get("entrypoint")
        if not isinstance(entrypoint, dict):
            raise WorkerManifestInvalidError(
                "Worker manifest must contain an 'entrypoint' object with 'module' and 'args'."
            )
        for forbidden_ep in ("command", "shell", "executable", "cmd"):
            if forbidden_ep in entrypoint:
                raise WorkerManifestInvalidError(
                    f"Forbidden entrypoint field {forbidden_ep!r}; only 'module' and 'args' are allowed."
                )

        raw_args = entrypoint.get("args", [])
        if not isinstance(raw_args, list) or not all(isinstance(x, str) for x in raw_args):
            raise WorkerManifestInvalidError("entrypoint.args must be a list of strings.")

        raw_resources = data.get("resources")
        if not isinstance(raw_resources, list) or not raw_resources:
            raise WorkerManifestInvalidError("resources must be a non-empty list.")
        parsed_resources: list[ResourceRequest] = []
        for idx, item in enumerate(raw_resources):
            if not isinstance(item, dict):
                raise WorkerManifestInvalidError(f"resources[{idx}] must be a JSON object.")
            try:
                rc = ResourceClass(item["resource_class"])
                amount = item["amount"]
                unit = str(item.get("unit", ""))
                exclusive = bool(item.get("exclusive", False))
                parsed_resources.append(
                    ResourceRequest(
                        resource_class=rc,
                        amount=amount,
                        unit=unit,
                        exclusive=exclusive,
                    )
                )
            except (KeyError, ValueError, TypeError) as exc:
                raise WorkerManifestInvalidError(f"Invalid resources[{idx}]: {exc}") from exc

        if "fallback_policy" not in data:
            raise WorkerManifestInvalidError("Worker manifest must declare 'fallback_policy'.")
        fallback = WorkerFallbackPolicy.from_dict(data["fallback_policy"])

        raw_net = data.get("network_policy", NetworkPolicy.RUNTIME_OFFLINE.value)
        try:
            network_policy = NetworkPolicy(raw_net)
        except ValueError as exc:
            raise WorkerManifestInvalidError(f"Invalid network_policy: {exc}") from exc

        file_limits = WorkerFileLimits.from_dict(data.get("file_limits"))

        try:
            return cls(
                schema_version=str(data.get("schema_version", "")),
                worker_id=data.get("worker_id", ""),
                name=data.get("name", ""),
                version=data.get("version", ""),
                module=entrypoint.get("module", ""),
                args=tuple(raw_args),
                resources=tuple(parsed_resources),
                fallback_policy=fallback,
                network_policy=network_policy,
                cancel_grace_seconds=float(data.get("cancel_grace_seconds", 5.0)),
                timeout_seconds=float(data.get("timeout_seconds", 300.0)),
                poll_interval_seconds=float(data.get("poll_interval_seconds", 0.25)),
                concurrency_limit=data.get("concurrency_limit", 1),
                file_limits=file_limits,
            )
        except WorkerManifestInvalidError:
            raise
        except (TypeError, ValueError) as exc:
            raise WorkerManifestInvalidError(f"Invalid worker manifest field: {exc}") from exc


@dataclass(frozen=True)
class JsonFileManifestEntry:
    """Validated file reference inside a json_file_manifest (no inline Base64)."""

    path: Path
    size_bytes: int
    role: str = "input"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "role": self.role,
        }


def _scan_for_base64_payload(obj: Any, path_prefix: str = "$") -> None:
    """Recursively reject inline Base64 file payloads inside json_file_manifest."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_lower = str(key).lower()
            if key_lower in _FORBIDDEN_BASE64_KEYS:
                raise WorkerManifestInvalidError(
                    f"Inline Base64/binary field {path_prefix}.{key} is forbidden in json_file_manifest."
                )
            _scan_for_base64_payload(value, f"{path_prefix}.{key}")
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            _scan_for_base64_payload(item, f"{path_prefix}[{idx}]")
    elif isinstance(obj, str):
        if _DATA_URI_BASE64_RE.match(obj):
            raise WorkerManifestInvalidError(
                f"Data URI Base64 payload at {path_prefix} is forbidden in json_file_manifest."
            )


def validate_json_file_manifest(
    payload: Any,
    allowed_root: Path,
    file_limits: WorkerFileLimits,
) -> tuple[JsonFileManifestEntry, ...]:
    """Validate json_file_manifest against Base64 ban, root jail, path traversal, and size limits."""
    if not isinstance(payload, dict):
        raise WorkerManifestInvalidError("json_file_manifest must be a JSON object.")

    _scan_for_base64_payload(payload)

    files = payload.get("files")
    if not isinstance(files, list):
        raise WorkerManifestInvalidError("json_file_manifest must contain a 'files' list.")

    resolved_root = Path(allowed_root).resolve()
    validated: list[JsonFileManifestEntry] = []

    for idx, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise WorkerManifestInvalidError(f"json_file_manifest.files[{idx}] must be an object.")
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise WorkerManifestInvalidError(
                f"json_file_manifest.files[{idx}].path must be a non-empty string."
            )

        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = resolved_root / candidate

        try:
            resolved_path = candidate.resolve()
        except OSError as exc:
            raise WorkerManifestInvalidError(
                f"Cannot resolve path {raw_path!r}: {exc}"
            ) from exc

        # Strict root containment check (blocks '..' and escaping symlinks)
        try:
            resolved_path.relative_to(resolved_root)
        except ValueError as exc:
            raise WorkerManifestInvalidError(
                f"Path traversal blocked: {raw_path!r} escapes allowed_root {str(resolved_root)!r}."
            ) from exc

        if not resolved_path.exists() or not resolved_path.is_file():
            raise WorkerManifestInvalidError(
                f"Manifest input file does not exist or is not a regular file: {resolved_path}"
            )

        size_bytes = resolved_path.stat().st_size
        if size_bytes > file_limits.max_file_size_bytes:
            raise WorkerManifestInvalidError(
                f"File {resolved_path.name} ({size_bytes} bytes) exceeds "
                f"max_file_size_bytes ({file_limits.max_file_size_bytes} bytes)."
            )

        role = str(entry.get("role", "input"))
        validated.append(
            JsonFileManifestEntry(path=resolved_path, size_bytes=size_bytes, role=role)
        )

    return tuple(validated)


@dataclass(frozen=True)
class CancelSignal:
    """Atomic cross-platform cancel control file payload (PRD 2.5 Step 1)."""

    run_id: str
    worker_id: str
    lease_id: str
    lease_generation: int
    reason: str
    requested_at: str
    grace_seconds: float
    schema_version: str = CANCEL_SIGNAL_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "worker_id": self.worker_id,
            "lease_id": self.lease_id,
            "lease_generation": self.lease_generation,
            "reason": self.reason,
            "requested_at": self.requested_at,
            "grace_seconds": self.grace_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CancelSignal:
        return cls(
            schema_version=str(data.get("schema_version", CANCEL_SIGNAL_SCHEMA_VERSION)),
            run_id=str(data["run_id"]),
            worker_id=str(data["worker_id"]),
            lease_id=str(data["lease_id"]),
            lease_generation=int(data["lease_generation"]),
            reason=str(data.get("reason", "cancelled")),
            requested_at=str(data.get("requested_at", "")),
            grace_seconds=float(data.get("grace_seconds", 5.0)),
        )


@dataclass(frozen=True)
class WorkerRunResult:
    """Structured outcome of a WorkerRuntime.run_worker execution."""

    run_id: str
    worker_id: str
    status: str  # "SUCCEEDED" | "CANCELLED" | "FAILED" | "REJECTED"
    exit_code: int | None
    lease_id: str | None
    lease_generation: int | None
    network_policy: NetworkPolicy
    used_cpu_fallback: bool = False
    forced_kill: bool = False
    post_kill_vram_reprobed: bool = False
    output_data: dict[str, Any] | None = None
    stdout: str = ""
    stderr: str = ""
    error_codes: tuple[ErrorCode, ...] = ()
    warnings: tuple[str, ...] = ()
    started_at: str = ""
    finished_at: str = ""
    schema_version: str = WORKER_RUN_RESULT_SCHEMA_VERSION

    @property
    def succeeded(self) -> bool:
        return self.status == "SUCCEEDED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "worker_id": self.worker_id,
            "status": self.status,
            "exit_code": self.exit_code,
            "lease_id": self.lease_id,
            "lease_generation": self.lease_generation,
            "network_policy": self.network_policy.value,
            "used_cpu_fallback": self.used_cpu_fallback,
            "forced_kill": self.forced_kill,
            "post_kill_vram_reprobed": self.post_kill_vram_reprobed,
            "output_data": self.output_data,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error_codes": [e.value for e in self.error_codes],
            "warnings": list(self.warnings),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }
