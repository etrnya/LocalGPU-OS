"""Abstract EvictionProvider and OllamaEvictionProvider (PRD v1.1.2 Section 2.2 E)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Protocol
import urllib.error
import urllib.parse
import urllib.request

from localgpu_os.domain.enums import ErrorCode

_ALLOWED_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


@dataclass(frozen=True)
class EvictionResult:
    provider_name: str
    attempted: bool
    success: bool
    evicted_models: tuple[str, ...] = ()
    error_code: ErrorCode | None = None
    message: str = ""


class EvictionProvider(Protocol):
    """Abstract interface for evicting external resident GPU services (e.g., Ollama)."""

    @property
    def name(self) -> str: ...

    def is_available(self, timeout_seconds: float = 2.0) -> bool: ...

    def evict(self, timeout_seconds: float = 10.0) -> EvictionResult: ...


class OllamaEvictionProvider:
    """First EvictionProvider implementation targeting local Ollama (127.0.0.1:11434).

    Never assumes VRAM is freed merely because HTTP 200 returned; VramGate MUST
    re-probe device-level VRAM_free,observed after eviction.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:11434") -> None:
        parsed = urllib.parse.urlparse(base_url)
        host = (parsed.hostname or "").lower()
        if (
            parsed.scheme != "http"
            or host not in _ALLOWED_LOOPBACK_HOSTS
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError(
                "OllamaEvictionProvider only permits loopback http:// URLs "
                f"(127.0.0.1, localhost, [::1]), got {base_url!r}."
            )
        self._base_url = base_url.rstrip("/")

    @property
    def name(self) -> str:
        return "ollama"

    def is_available(self, timeout_seconds: float = 2.0) -> bool:
        try:
            req = urllib.request.Request(f"{self._base_url}/api/ps", method="GET")
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                return resp.status == 200
        except (urllib.error.URLError, TimeoutError, OSError):
            return False

    def evict(self, timeout_seconds: float = 10.0) -> EvictionResult:
        try:
            ps_req = urllib.request.Request(f"{self._base_url}/api/ps", method="GET")
            with urllib.request.urlopen(ps_req, timeout=timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            return EvictionResult(
                provider_name=self.name,
                attempted=True,
                success=False,
                error_code=ErrorCode.EVICTION_FAILED,
                message=f"Failed to query Ollama /api/ps: {exc}",
            )

        models = [
            str(item["name"])
            for item in data.get("models", [])
            if isinstance(item, dict) and item.get("name")
        ]
        if not models:
            return EvictionResult(
                provider_name=self.name,
                attempted=True,
                success=True,
                evicted_models=(),
                message="No resident Ollama models loaded.",
            )

        evicted: list[str] = []
        for model_name in models:
            payload = json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8")
            unload_req = urllib.request.Request(
                f"{self._base_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(unload_req, timeout=timeout_seconds) as resp:
                    if resp.status == 200:
                        evicted.append(model_name)
                    else:
                        return EvictionResult(
                            provider_name=self.name,
                            attempted=True,
                            success=False,
                            evicted_models=tuple(evicted),
                            error_code=ErrorCode.EVICTION_FAILED,
                            message=f"Ollama unload for {model_name} returned HTTP {resp.status}.",
                        )
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                return EvictionResult(
                    provider_name=self.name,
                    attempted=True,
                    success=False,
                    evicted_models=tuple(evicted),
                    error_code=ErrorCode.EVICTION_FAILED,
                    message=f"Failed to unload Ollama model {model_name}: {exc}",
                )

        return EvictionResult(
            provider_name=self.name,
            attempted=True,
            success=True,
            evicted_models=tuple(evicted),
            message=f"Requested unload for Ollama models: {', '.join(evicted)}.",
        )
