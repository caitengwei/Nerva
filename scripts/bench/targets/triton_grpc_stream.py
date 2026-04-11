from __future__ import annotations

import inspect
import time
from collections.abc import AsyncIterable, AsyncIterator, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from scripts.bench.targets.base import TargetResponse
from scripts.bench.targets.triton_infer import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    _extract_triton_text,
    _float_payload,
    _int_payload,
)

if TYPE_CHECKING:
    from types import ModuleType


@dataclass(frozen=True)
class GrpcTensor:
    name: str
    datatype: str
    shape: tuple[int, ...]
    value: str | bytes | int | float | bool


@dataclass(frozen=True)
class StreamingChunk:
    text: str | None
    raw: dict[str, Any] | None = None


StreamSender = Callable[
    [str, str, list[GrpcTensor], int],
    AsyncIterable[StreamingChunk],
]


class TritonGrpcStreamingTarget:
    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        stream: bool = True,
        sender: StreamSender | None = None,
    ) -> None:
        self._base_url = _normalize_grpc_url(base_url)
        self._model_name = model_name
        self._stream = stream
        self._client: Any | None = None
        self._sender = sender or self._default_sender

    async def aclose(self) -> None:
        if self._client is None:
            return

        close = getattr(self._client, "close", None)
        if close is not None:
            maybe_awaitable = close()
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable
        self._client = None

    async def infer(self, payload: dict[str, Any], *, deadline_ms: int) -> TargetResponse:
        start_ns = time.perf_counter_ns()
        ttft_ms: float | None = None
        output_text: str | None = None
        raw_chunks: list[dict[str, Any]] = []
        tensors = _build_triton_grpc_tensors(
            payload,
            deadline_ms=deadline_ms,
            stream=self._stream,
        )

        try:
            async for chunk in self._sender(
                self._base_url,
                self._model_name,
                tensors,
                deadline_ms,
            ):
                if ttft_ms is None:
                    ttft_ms = _latency_ms(start_ns)
                if chunk.raw is not None:
                    raw_chunks.append(chunk.raw)
                if chunk.text is not None:
                    output_text = chunk.text

            if output_text is None:
                return TargetResponse(
                    ok=False,
                    latency_ms=_latency_ms(start_ns),
                    ttft_ms=ttft_ms,
                    error="missing outputs in Triton gRPC stream",
                    raw={"chunks": raw_chunks} if raw_chunks else None,
                )

            return TargetResponse(
                ok=True,
                latency_ms=_latency_ms(start_ns),
                ttft_ms=ttft_ms,
                error="",
                output_text=output_text,
                raw={"chunks": raw_chunks} if raw_chunks else None,
            )
        except Exception as exc:  # pragma: no cover - network/transport failures
            return TargetResponse(
                ok=False,
                latency_ms=_latency_ms(start_ns),
                ttft_ms=ttft_ms,
                error=str(exc),
            )

    async def _default_sender(
        self,
        base_url: str,
        model_name: str,
        tensors: list[GrpcTensor],
        timeout_ms: int,
    ) -> AsyncIterator[StreamingChunk]:
        try:
            import numpy as np
            import tritonclient.grpc as grpcclient
            import tritonclient.grpc.aio as grpcclient_aio
        except ImportError as exc:
            raise ImportError(
                "Triton gRPC streaming target requires tritonclient[grpc]. "
                "Install it with: pip install 'tritonclient[grpc]'"
            ) from exc

        if self._client is None:
            self._client = grpcclient_aio.InferenceServerClient(url=base_url)

        request = {
            "model_name": model_name,
            "inputs": [_to_grpc_input(grpcclient, np, tensor) for tensor in tensors],
        }

        timeout_s = max(timeout_ms / 1000.0, 0.001)
        async for result, error in self._client.stream_infer(
            _single_request_iterator(request),
            stream_timeout=timeout_s,
        ):
            if error is not None:
                raise RuntimeError(str(error))
            if result is None:
                continue

            raw: dict[str, Any] | None = None
            try:
                maybe_raw = result.get_response(as_json=True)
                if isinstance(maybe_raw, dict):
                    raw = maybe_raw
            except Exception:
                raw = None

            yield StreamingChunk(
                text=_extract_stream_text(result, raw),
                raw=raw,
            )


def _build_triton_grpc_tensors(
    payload: dict[str, Any],
    *,
    deadline_ms: int,
    stream: bool = True,
) -> list[GrpcTensor]:
    if "text" in payload or "image_bytes" in payload:
        text = str(payload.get("text", ""))
        raw_bytes = payload.get("image_bytes", b"")
        if not isinstance(raw_bytes, (bytes, bytearray, memoryview)):
            raw_bytes = b""
        max_tokens = _int_payload(payload.get("max_tokens"), default=DEFAULT_MAX_TOKENS, minimum=1)
        temperature = _float_payload(payload.get("temperature"), default=DEFAULT_TEMPERATURE, minimum=0.0)
        top_p = _float_payload(payload.get("top_p"), default=DEFAULT_TOP_P, minimum=1e-6)
        per_request_deadline_ms = _int_payload(
            payload.get("deadline_ms"),
            default=deadline_ms,
            minimum=1,
        )
        return [
            GrpcTensor(name="TEXT", datatype="BYTES", shape=(1,), value=text),
            GrpcTensor(
                name="IMAGE_BYTES",
                datatype="BYTES",
                shape=(1,),
                value=bytes(raw_bytes),
            ),
            GrpcTensor(name="MAX_TOKENS", datatype="INT32", shape=(1,), value=max_tokens),
            GrpcTensor(name="TEMPERATURE", datatype="FP32", shape=(1,), value=temperature),
            GrpcTensor(name="TOP_P", datatype="FP32", shape=(1,), value=top_p),
            GrpcTensor(
                name="DEADLINE_MS",
                datatype="INT32",
                shape=(1,),
                value=per_request_deadline_ms,
            ),
            GrpcTensor(name="STREAM", datatype="BOOL", shape=(1,), value=stream),
        ]

    prompt = str(payload.get("prompt", ""))
    return [GrpcTensor(name="PROMPT", datatype="BYTES", shape=(1,), value=prompt)]


def _normalize_grpc_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    if parsed.scheme:
        if parsed.hostname is None:
            raise ValueError(f"invalid Triton gRPC URL: {base_url}")
        if parsed.port is None:
            return parsed.hostname
        return f"{parsed.hostname}:{parsed.port}"
    return base_url.rstrip("/")


async def _single_request_iterator(request: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    yield request


def _to_grpc_input(grpcclient: ModuleType, np: ModuleType, tensor: GrpcTensor) -> Any:
    infer_input = grpcclient.InferInput(tensor.name, list(tensor.shape), tensor.datatype)
    infer_input.set_data_from_numpy(_tensor_to_numpy(np, tensor))
    return infer_input


def _tensor_to_numpy(np: ModuleType, tensor: GrpcTensor) -> Any:
    if tensor.datatype == "BYTES":
        return np.array([tensor.value], dtype=object)
    if tensor.datatype == "INT32":
        return np.array([int(tensor.value)], dtype=np.int32)
    if tensor.datatype == "FP32":
        return np.array([float(tensor.value)], dtype=np.float32)
    if tensor.datatype == "BOOL":
        return np.array([bool(tensor.value)], dtype=np.bool_)
    raise ValueError(f"unsupported Triton datatype: {tensor.datatype}")


def _extract_stream_text(result: Any, raw: dict[str, Any] | None) -> str | None:
    for wanted_name in ("OUTPUT_TEXT", "TEXT", "output_text", "text", "RAW", "raw"):
        try:
            values = result.as_numpy(wanted_name)
        except Exception:
            continue
        if values is None or len(values) == 0:
            continue
        return _coerce_output_text(values.reshape(-1)[0])

    if raw is not None:
        return _extract_triton_text(raw)
    return None


def _coerce_output_text(value: Any) -> str | None:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    try:
        if isinstance(value.item(), bytes):
            return value.item().decode("utf-8", errors="replace")
        if isinstance(value.item(), str):
            return value.item()
    except Exception:
        return None
    return None


def _latency_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000.0
