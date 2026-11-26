"""OS process liveness & creation-time verification (zero external deps, Fail-Closed)."""

from __future__ import annotations

import os
from typing import Protocol

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _STILL_ACTIVE = 259
    _ERROR_ACCESS_DENIED = 5
    _ERROR_INVALID_PARAMETER = 87

    class _FILETIME(ctypes.Structure):
        _fields_ = [
            ("dwLowDateTime", wintypes.DWORD),
            ("dwHighDateTime", wintypes.DWORD),
        ]

    def _filetime_to_unix(ft: _FILETIME) -> float:
        # Windows FILETIME is 100-ns intervals since 1601-01-01 UTC
        ticks = (ft.dwHighDateTime << 32) | ft.dwLowDateTime
        return (ticks / 10_000_000.0) - 11_644_473_600.0


class ProcessInspectorProtocol(Protocol):
    """Checks whether a lease owner process is confirmed stopped.

    Returns:
        True  -> Process is confirmed still alive (matching pid_create_time if given).
        False -> Process is confirmed stopped (or PID reused by a different process).
        None  -> Liveness could not be verified (caller MUST treat conservatively).
    """

    def is_alive(self, pid: int, expected_create_time: float | None = None) -> bool | None: ...

    def current_process_create_time(self) -> float | None: ...


class OsProcessInspector:
    """Zero-dependency cross-platform process inspector."""

    def is_alive(self, pid: int, expected_create_time: float | None = None) -> bool | None:
        if pid <= 0:
            return False
        if os.name == "nt":
            return self._is_alive_windows(pid, expected_create_time)
        return self._is_alive_posix(pid)

    def current_process_create_time(self) -> float | None:
        if os.name == "nt":
            return self._get_create_time_windows(os.getpid())
        return None

    def _is_alive_windows(
        self, pid: int, expected_create_time: float | None
    ) -> bool | None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            err = ctypes.get_last_error()
            if err == _ERROR_INVALID_PARAMETER:
                return False
            if err == _ERROR_ACCESS_DENIED:
                # Process exists but cannot be inspected -> conservatively alive / unverified
                return True
            return None

        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return None
            if exit_code.value != _STILL_ACTIVE:
                return False

            if expected_create_time is not None:
                actual_ctime = self._read_handle_create_time(kernel32, handle)
                if actual_ctime is None:
                    return None
                # Allow 2s tolerance for clock/rounding differences; beyond that PID was reused
                if abs(actual_ctime - expected_create_time) > 2.0:
                    return False
            return True
        finally:
            kernel32.CloseHandle(handle)

    def _get_create_time_windows(self, pid: int) -> float | None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return None
        try:
            return self._read_handle_create_time(kernel32, handle)
        finally:
            kernel32.CloseHandle(handle)

    @staticmethod
    def _read_handle_create_time(kernel32: ctypes.WinDLL, handle: int) -> float | None:
        creation = _FILETIME()
        exit_t = _FILETIME()
        kernel_t = _FILETIME()
        user_t = _FILETIME()
        ok = kernel32.GetProcessTimes(
            handle,
            ctypes.byref(creation),
            ctypes.byref(exit_t),
            ctypes.byref(kernel_t),
            ctypes.byref(user_t),
        )
        if not ok:
            return None
        return _filetime_to_unix(creation)

    @staticmethod
    def _is_alive_posix(pid: int) -> bool | None:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return None
