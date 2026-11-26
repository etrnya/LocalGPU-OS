"""WorkerRegistry: manifest loading and strict isolated .venv Python resolution (PRD 2.5)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from localgpu_os.domain.worker import (
    WorkerEnvNotInstalledError,
    WorkerManifest,
    WorkerManifestInvalidError,
    validate_worker_id,
)


def default_workers_dir() -> Path:
    """Return default workers directory (LOCALGPU_OS_WORKERS_DIR or ./workers)."""
    env_override = os.environ.get("LOCALGPU_OS_WORKERS_DIR")
    if env_override:
        return Path(env_override)
    return Path.cwd() / "workers"


class WorkerRegistry:
    """Discovers workers, validates Worker Manifest Contract v1.0, and resolves .venv Python."""

    def __init__(self, workers_dir: Path | None = None, platform_name: str | None = None) -> None:
        self._workers_dir = Path(workers_dir) if workers_dir is not None else default_workers_dir()
        self._platform_name = platform_name or os.name

    @property
    def workers_dir(self) -> Path:
        return self._workers_dir

    def worker_dir(self, worker_id: str) -> Path:
        safe_id = validate_worker_id(worker_id)
        root = self._workers_dir.resolve()
        candidate = (root / safe_id).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise WorkerManifestInvalidError(
                f"Worker directory for {worker_id!r} escapes workers_dir."
            ) from exc
        return candidate

    def manifest_path(self, worker_id: str) -> Path:
        return self.worker_dir(worker_id) / "worker.json"

    def load_manifest(self, worker_id: str) -> WorkerManifest:
        safe_id = validate_worker_id(worker_id)
        path = self.manifest_path(safe_id)
        if not path.exists() or not path.is_file():
            raise WorkerManifestInvalidError(
                f"Worker manifest not found at {path} for worker_id={safe_id!r}."
            )
        try:
            raw = path.read_text(encoding="utf-8")
            parsed = json.loads(raw)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise WorkerManifestInvalidError(
                f"Failed to read/parse worker manifest {path}: {exc}"
            ) from exc

        manifest = WorkerManifest.from_dict(parsed)
        if manifest.worker_id != safe_id:
            raise WorkerManifestInvalidError(
                f"Manifest worker_id {manifest.worker_id!r} does not match directory name {safe_id!r}."
            )
        return manifest

    def expected_venv_python_path(self, worker_id: str) -> Path:
        wdir = self.worker_dir(worker_id)
        if self._platform_name == "nt":
            return wdir / ".venv" / "Scripts" / "python.exe"
        return wdir / ".venv" / "bin" / "python"

    def resolve_venv_python(self, worker_id: str) -> Path:
        """Strictly resolve isolated worker .venv Python executable (never falls back to PATH)."""
        python_exe = self.expected_venv_python_path(worker_id)
        if not python_exe.exists() or not python_exe.is_file():
            raise WorkerEnvNotInstalledError(
                f"Worker {worker_id!r} isolated Python environment not installed: "
                f"expected executable at {python_exe}."
            )
        return python_exe

    def list_workers(self) -> tuple[WorkerManifest, ...]:
        """List all valid workers currently installed under workers_dir."""
        if not self._workers_dir.exists() or not self._workers_dir.is_dir():
            return ()
        manifests: list[WorkerManifest] = []
        for entry in sorted(self._workers_dir.iterdir(), key=lambda p: p.name):
            if not entry.is_dir():
                continue
            manifest_file = entry / "worker.json"
            if not manifest_file.exists():
                continue
            manifests.append(self.load_manifest(entry.name))
        return tuple(manifests)
