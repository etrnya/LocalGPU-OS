"""Cross-platform single-writer RegistryLock (Windows msvcrt / POSIX fcntl).

Follows the persistent sentinel lockfile invariant:
- Ensure lockfile has at least 1 byte before Windows msvcrt.locking().
- Never unlink the lockfile on release to eliminate inode TOCTOU races.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
from types import TracebackType

from localgpu_os.domain.lease import VramLockTimeoutError

if os.name == "nt":
    import msvcrt
else:
    import fcntl


class RegistryLock:
    """Single-writer cross-process lock protecting Lease Registry RMW transactions."""

    def __init__(
        self,
        lock_path: Path,
        timeout_seconds: float = 5.0,
        poll_interval_seconds: float = 0.02,
    ) -> None:
        self._lock_path = Path(lock_path)
        self._timeout = max(0.0, float(timeout_seconds))
        self._poll = max(0.005, float(poll_interval_seconds))
        self._fd: int | None = None

    @property
    def lock_path(self) -> Path:
        return self._lock_path

    def acquire(self) -> None:
        if self._fd is not None:
            raise RuntimeError("RegistryLock is already held by this instance.")

        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self._lock_path), os.O_CREAT | os.O_RDWR)

        try:
            if os.name == "nt":
                # Ensure >= 1 sentinel byte exists so msvcrt.locking locks byte 0 reliably
                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"0")
                    os.lseek(fd, 0, os.SEEK_SET)

            deadline = time.monotonic() + self._timeout
            while True:
                try:
                    self._try_lock_fd(fd)
                    self._fd = fd
                    return
                except OSError:
                    if time.monotonic() >= deadline:
                        raise VramLockTimeoutError(
                            f"Timed out after {self._timeout:.2f}s waiting for RegistryLock "
                            f"at {self._lock_path}."
                        )
                    time.sleep(self._poll)
        except BaseException:
            os.close(fd)
            raise

    def release(self) -> None:
        if self._fd is None:
            return
        fd = self._fd
        self._fd = None
        try:
            self._unlock_fd(fd)
        finally:
            os.close(fd)

    def __enter__(self) -> RegistryLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.release()

    @staticmethod
    def _try_lock_fd(fd: int) -> None:
        if os.name == "nt":
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _unlock_fd(fd: int) -> None:
        if os.name == "nt":
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
