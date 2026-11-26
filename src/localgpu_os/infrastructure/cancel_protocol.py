"""Atomic cross-platform 7-step cancel control file helpers (PRD 2.5)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from typing import Any

from localgpu_os.domain.enums import ErrorCode
from localgpu_os.domain.worker import CancelSignal


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically persist a JSON file on the same filesystem via fsync + os.replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    tid = threading.get_ident()
    tmp_path = target.parent / f".{target.name}.{pid}.{tid}.tmp"
    content = json.dumps(payload, indent=2, ensure_ascii=False)
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


def write_cancel_signal_atomic(cancel_file: Path, signal: CancelSignal) -> None:
    """Step 1 of 7-step cancel protocol: atomically write cancel control file."""
    write_json_atomic(cancel_file, signal.to_dict())


def read_cancel_signal(cancel_file: Path) -> CancelSignal | None:
    """Race-safe reader for the cancel control file. Returns None if not yet present."""
    path = Path(cancel_file)
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
        return CancelSignal.from_dict(data)
    except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def write_worker_cancelled_result(
    result_file: Path,
    run_id: str,
    worker_id: str,
    lease_id: str | None = None,
    lease_generation: int | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """Step 5 of 7-step cancel protocol: worker writes WORKER_CANCELLED status atomically."""
    payload: dict[str, Any] = {
        "status": ErrorCode.WORKER_CANCELLED.value,
        "error_code": ErrorCode.WORKER_CANCELLED.value,
        "run_id": run_id,
        "worker_id": worker_id,
        "lease_id": lease_id,
        "lease_generation": lease_generation,
    }
    if details:
        payload["details"] = details
    write_json_atomic(result_file, payload)
