"""Disk space availability checker for M1-C DISK_SPACE_INSUFFICIENT guard."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Protocol


class DiskSpaceCheckerProtocol(Protocol):
    """Returns available free disk space in MiB for the filesystem containing `path`."""

    def free_disk_mib(self, path: Path) -> int: ...


class ShutilDiskSpaceChecker:
    """Cross-platform zero-dependency disk space checker backed by shutil.disk_usage."""

    def free_disk_mib(self, path: Path) -> int:
        target = Path(path).resolve()
        while not target.exists() and target.parent != target:
            target = target.parent
        usage = shutil.disk_usage(target)
        return int(usage.free // (1024 * 1024))
