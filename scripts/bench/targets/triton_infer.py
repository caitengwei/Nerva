from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
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
        self._client: httpx.AsyncClient | None = None
        self._sender = sender or self._default_sender

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def infer(self, payload: dict[str, Any], *, deadline_ms: int) -> TargetResponse:
        start_ns = time.perf_counter_ns()
        request_body = {
            "inputs": _build_triton_inputs(payload, deadline_ms=deadline_ms),
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
        if self._client is None:
            self._client = httpx.AsyncClient()

        try:
            response = await self._client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout_s,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(str(exc)) from exc

        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Triton response must be a JSON object")
        return data


def _extract_triton_text(data: dict[str, Any]) -> str | None:
    outputs = data.get("outputs")
    if not isinstance(outputs, list):
        return None

    # Prefer semantically explicit names from phase7 full-e2e ensemble.
    for wanted_name in ("OUTPUT_TEXT", "TEXT", "output_text", "text", "RAW", "raw"):
        for output in outputs:
            if not isinstance(output, dict):
                continue
            if str(output.get("name", "")).upper() != wanted_name.upper():
                continue
            values = output.get("data")
            if not isinstance(values, list) or not values:
                continue
            first = values[0]
            if isinstance(first, str):
                return first

    # Fallback for generic Triton schemas that do not set stable output names.
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


def _build_triton_inputs(payload: dict[str, Any], *, deadline_ms: int) -> list[dict[str, Any]]:
    if "text" in payload or "image_bytes" in payload or "image_size" in payload:
        text = str(payload.get("text", ""))
        image_size = _image_size_from_payload(payload)
        max_tokens = _int_payload(payload.get("max_tokens"), default=256, minimum=1)
        temperature = _float_payload(payload.get("temperature"), default=1.0, minimum=0.0)
        top_p = _float_payload(payload.get("top_p"), default=1.0, minimum=1e-6)
        per_request_deadline_ms = _int_payload(
            payload.get("deadline_ms"),
            default=deadline_ms,
            minimum=1,
        )
        return [
            {
                "name": "TEXT",
                "shape": [1],
                "datatype": "BYTES",
                "data": [text],
            },
            {
                "name": "IMAGE_SIZE",
                "shape": [1],
                "datatype": "INT32",
                "data": [image_size],
            },
            {
                "name": "MAX_TOKENS",
                "shape": [1],
                "datatype": "INT32",
                "data": [max_tokens],
            },
            {
                "name": "TEMPERATURE",
                "shape": [1],
                "datatype": "FP32",
                "data": [temperature],
            },
            {
                "name": "TOP_P",
                "shape": [1],
                "datatype": "FP32",
                "data": [top_p],
            },
            {
                "name": "DEADLINE_MS",
                "shape": [1],
                "datatype": "INT32",
                "data": [per_request_deadline_ms],
            },
        ]

    prompt = str(payload.get("prompt", ""))
    return [
        {
            "name": "PROMPT",
            "shape": [1],
            "datatype": "BYTES",
            "data": [prompt],
        }
    ]


def _image_size_from_payload(payload: dict[str, Any]) -> int:
    image_bytes = payload.get("image_bytes")
    if isinstance(image_bytes, bytes | bytearray | memoryview):
        return len(image_bytes)

    image_size = payload.get("image_size", 0)
    try:
        parsed = int(image_size)
    except Exception:
        return 0
    return max(parsed, 0)


def _int_payload(value: object, *, default: int, minimum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return default
    return max(parsed, minimum)


def _float_payload(value: object, *, default: float, minimum: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        return default
    return max(parsed, minimum)
