"""Atomic same-filesystem JSON LeaseStore (fsync + os.replace, Fail-Closed on corruption)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading

from localgpu_os.domain.lease import LeaseRegistryState, LeaseStateCorruptedError


class FileLeaseStore:
    """Persists LeaseRegistryState atomically within the same directory/filesystem.

    Callers MUST hold RegistryLock across the full Read-Modify-Write sequence.
    """

    def __init__(self, state_path: Path) -> None:
        self._state_path = Path(state_path)

    @property
    def state_path(self) -> Path:
        return self._state_path

    def load(self) -> LeaseRegistryState:
        if not self._state_path.exists():
            return LeaseRegistryState()
        try:
            raw = self._state_path.read_text(encoding="utf-8")
            parsed = json.loads(raw)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise LeaseStateCorruptedError(
                f"Failed to read/parse lease registry at {self._state_path}: {exc}"
            ) from exc
        return LeaseRegistryState.from_dict(parsed)

    def save(self, state: LeaseRegistryState) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        tid = threading.get_ident()
        tmp_path = self._state_path.parent / f".{self._state_path.name}.{pid}.{tid}.tmp"
        payload = json.dumps(state.to_dict(), indent=2, ensure_ascii=False)
        try:
            with open(tmp_path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, self._state_path)
        except BaseException:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            raise
