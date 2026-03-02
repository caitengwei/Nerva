from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError

from scripts.bench.targets.base import TargetResponse

JSONSender = Callable[[str, dict[str, Any], int], Awaitable[dict[str, Any]]]


class TritonInferTarget:
    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        sender: JSONSender | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._sender = sender or self._default_sender

    async def infer(self, payload: dict[str, Any], *, deadline_ms: int) -> TargetResponse:
        start_ns = time.perf_counter_ns()
        prompt = str(payload.get("prompt", ""))
        request_body = {
            "inputs": [
                {
                    "name": "PROMPT",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [prompt],
                }
            ]
        }
        url = f"{self._base_url}/v2/models/{self._model_name}/infer"

        try:
            data = await self._sender(url, request_body, deadline_ms)
            output_text = _extract_triton_text(data)
            if output_text is None:
                return TargetResponse(
                    ok=False,
                    latency_ms=_latency_ms(start_ns),
                    ttft_ms=None,
                    error="missing outputs in Triton response",
                    raw=data,
                )
            return TargetResponse(
                ok=True,
                latency_ms=_latency_ms(start_ns),
                ttft_ms=None,
                error="",
                output_text=output_text,
                raw=data,
            )
        except Exception as exc:  # pragma: no cover - network/transport failures
            return TargetResponse(
                ok=False,
                latency_ms=_latency_ms(start_ns),
                ttft_ms=None,
                error=str(exc),
            )

    async def _default_sender(
        self,
        url: str,
        payload: dict[str, Any],
        timeout_ms: int,
    ) -> dict[str, Any]:
        timeout_s = max(timeout_ms / 1000.0, 0.001)

        def _send() -> dict[str, Any]:
            body = json.dumps(payload).encode("utf-8")
            req = request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("Triton response must be a JSON object")
                return data

        try:
            return await asyncio.to_thread(_send)
        except (HTTPError, URLError) as exc:
            raise RuntimeError(str(exc)) from exc


def _extract_triton_text(data: dict[str, Any]) -> str | None:
    outputs = data.get("outputs")
    if not isinstance(outputs, list):
        return None

    for output in outputs:
        if not isinstance(output, dict):
            continue
        values = output.get("data")
        if not isinstance(values, list) or not values:
            continue
        first = values[0]
        if isinstance(first, str):
            return first

    return None


def _latency_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0
