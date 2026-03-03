from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class TargetResponse:
    ok: bool
    latency_ms: float
    ttft_ms: float | None
    error: str
    output_text: str | None = None
    raw: dict[str, Any] | None = None


class BenchTarget(Protocol):
    async def infer(self, payload: dict[str, Any], *, deadline_ms: int) -> TargetResponse:
        ...
