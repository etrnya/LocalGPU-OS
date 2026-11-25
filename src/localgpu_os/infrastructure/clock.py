"""Clock abstraction providing UTC timestamps and monotonic age measurement."""

from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Protocol


class ClockProtocol(Protocol):
    def utc_now_iso(self) -> str: ...
    def monotonic(self) -> float: ...


class SystemClock:
    """Default system clock using UTC ISO-8601 and time.monotonic()."""

    def utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def monotonic(self) -> float:
        return time.monotonic()
