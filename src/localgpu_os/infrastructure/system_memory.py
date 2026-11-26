"""Host CPU thread and available system RAM probe (zero external dependencies)."""

from __future__ import annotations

import os
from pathlib import Path

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]


def get_host_cpu_threads() -> int:
    return max(1, os.cpu_count() or 1)


def get_available_system_memory_mib() -> int | None:
    """Return currently available physical RAM in MiB, or None if unavailable."""
    if os.name == "nt":
        try:
            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPhys // (1024 * 1024))
        except (OSError, AttributeError):
            return None
        return None

    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) // 1024
        except (OSError, ValueError):
            return None
    return None
