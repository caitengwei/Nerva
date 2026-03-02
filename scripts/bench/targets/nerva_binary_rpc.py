from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import msgpack
from scripts.bench.targets.base import TargetResponse

from nerva.server.protocol import Frame, FrameType, ProtocolError, decode_frame, encode_frame

BinarySender = Callable[[str, bytes, dict[str, str], int], Awaitable[bytes]]


class NervaBinaryRPCTarget:
    def __init__(
        self,
        *,
        base_url: str,
        pipeline_name: str,
        sender: BinarySender | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._pipeline_name = pipeline_name
        self._sender = sender or self._default_sender

    async def infer(self, payload: dict[str, Any], *, deadline_ms: int) -> TargetResponse:
        start_ns = time.perf_counter_ns()
        request_id = int(time.time_ns() & ((1 << 63) - 1))
        body = _build_request_body(self._pipeline_name, payload, request_id)

        headers = {
            "content-type": "application/x-nerva-rpc",
            "x-nerva-deadline-ms": str(int(time.time() * 1000) + deadline_ms),
            "x-nerva-stream": "0",
        }
        url = f"{self._base_url}/rpc/{self._pipeline_name}"

        try:
            raw = await self._sender(url, body, headers, deadline_ms)
            ok, output_text, raw_output, error = _parse_response_frames(raw)
        except Exception as exc:  # pragma: no cover - network/transport failures
            return TargetResponse(
                ok=False,
                latency_ms=_latency_ms(start_ns),
                ttft_ms=None,
                error=str(exc),
            )

        return TargetResponse(
            ok=ok,
            latency_ms=_latency_ms(start_ns),
            ttft_ms=None,
            error=error,
            output_text=output_text,
            raw=raw_output,
        )

    async def _default_sender(
        self,
        url: str,
        body: bytes,
        headers: dict[str, str],
        timeout_ms: int,
    ) -> bytes:
        timeout_s = max(timeout_ms / 1000.0, 0.001)
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, content=body, headers=headers, timeout=timeout_s)
            resp.raise_for_status()
            return resp.content


def _build_request_body(pipeline_name: str, payload: dict[str, Any], request_id: int) -> bytes:
    open_payload = msgpack.packb({"pipeline": pipeline_name}, use_bin_type=True)
    data_payload = msgpack.packb(payload, use_bin_type=True)
    return (
        encode_frame(Frame(FrameType.OPEN, request_id, 0, open_payload))
        + encode_frame(Frame(FrameType.DATA, request_id, 0, data_payload))
        + encode_frame(Frame(FrameType.END, request_id, 0, b""))
    )


def _parse_response_frames(raw: bytes) -> tuple[bool, str | None, dict[str, Any] | None, str]:
    offset = 0
    first_data: dict[str, Any] | None = None

    while offset < len(raw):
        frame, consumed = decode_frame(raw[offset:])
        offset += consumed

        if frame.frame_type == FrameType.ERROR:
            error_dict = _unpack_dict(frame.payload)
            code = str(error_dict.get("code", "UNKNOWN"))
            message = str(error_dict.get("message", ""))
            error = f"{code}: {message}".strip()
            return False, None, error_dict, error

        if frame.frame_type == FrameType.DATA and first_data is None:
            first_data = _unpack_dict(frame.payload)

    if first_data is None:
        return False, None, None, "no DATA frame in response"

    output_text = _extract_text(first_data)
    return True, output_text, first_data, ""


def _unpack_dict(payload: bytes) -> dict[str, Any]:
    try:
        decoded = msgpack.unpackb(payload, raw=False)
    except Exception as exc:
        raise ProtocolError(f"invalid msgpack payload: {exc}") from exc

    if not isinstance(decoded, dict):
        raise ProtocolError("decoded payload is not an object")
    return decoded


def _extract_text(data: dict[str, Any]) -> str | None:
    for key in ("text", "output_text", "raw"):
        value = data.get(key)
        if isinstance(value, str):
            return value
    return None


def _latency_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0
