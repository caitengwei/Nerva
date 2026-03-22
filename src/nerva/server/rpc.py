"""Binary RPC handler for Nerva inference serving."""

from __future__ import annotations

import time
import uuid
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import msgpack
import structlog
from starlette.applications import Starlette
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

if TYPE_CHECKING:
    from starlette.requests import Request

import nerva.observability.timing as _timing
from nerva.observability.metrics import NervaMetrics, get_metrics
from nerva.server.protocol import Frame, FrameType, ProtocolError, decode_frame, encode_frame

logger = structlog.get_logger(__name__)

CONTENT_TYPE = "application/x-nerva-rpc"


class ErrorCode(IntEnum):
    OK = 0
    INVALID_ARGUMENT = 3
    DEADLINE_EXCEEDED = 4
    RESOURCE_EXHAUSTED = 8
    INTERNAL = 13
    UNAVAILABLE = 14


def _error_frame(request_id: int, code: ErrorCode, message: str) -> bytes:
    """Build an ERROR frame."""
    # DEADLINE_EXCEEDED: do not retry (request already timed out).
    # RESOURCE_EXHAUSTED / UNAVAILABLE: retryable with backoff.
    retryable = code in (ErrorCode.RESOURCE_EXHAUSTED, ErrorCode.UNAVAILABLE)
    payload = msgpack.packb(
        {"code": int(code), "message": message, "retryable": retryable},
        use_bin_type=True,
    )
    return encode_frame(Frame(FrameType.ERROR, request_id, 0, payload))


def _map_exception(exc: BaseException) -> tuple[ErrorCode, str]:
    """Map an execution exception to an error code and message.

    NOTE: This is a transitional implementation using string matching.
    When modifying exception messages in worker/proxy.py or the backend
    layer, you MUST update this function to keep mappings consistent.
    Future work: replace with a typed exception hierarchy (NervaError subclasses).
    """
    msg = str(exc)
    if "UNAVAILABLE" in msg:
        return ErrorCode.UNAVAILABLE, msg
    if "RESOURCE_EXHAUSTED" in msg:
        return ErrorCode.RESOURCE_EXHAUSTED, msg
    if "DEADLINE_EXCEEDED" in msg:
        return ErrorCode.DEADLINE_EXCEEDED, msg
    return ErrorCode.INTERNAL, msg


def _parse_frames(data: bytes) -> list[Frame]:
    """Parse all frames from a byte buffer.

    Wraps *data* in a memoryview so that ``decode_frame`` returns zero-copy
    payload slices — no intermediate byte copies for the entire parse loop.
    """
    frames: list[Frame] = []
    mv = memoryview(data)
    offset = 0
    while offset < len(mv):
        frame, consumed = decode_frame(mv, offset)
        frames.append(frame)
        offset += consumed
    return frames


def _classify_frames(
    frames: list[Frame],
) -> tuple[Frame | None, Frame | None, Frame | None]:
    """Single-pass classification: return (open, data, end) first-of-each."""
    open_frame: Frame | None = None
    data_frame: Frame | None = None
    end_frame: Frame | None = None
    for f in frames:
        if open_frame is None and f.frame_type == FrameType.OPEN:
            open_frame = f
        elif data_frame is None and f.frame_type == FrameType.DATA:
            data_frame = f
        elif end_frame is None and f.frame_type == FrameType.END:
            end_frame = f
        if open_frame is not None and data_frame is not None and end_frame is not None:
            break
    return open_frame, data_frame, end_frame


class RpcHandler:
    """Handles binary RPC requests for a set of pipelines.

    Each pipeline is an object with an async execute(inputs) method
    (typically an Executor instance).
    """

    def __init__(
        self,
        pipelines: dict[str, Any],
        metrics: NervaMetrics | None = None,
    ) -> None:
        self._pipelines = pipelines
        self._metrics: NervaMetrics = metrics if metrics is not None else get_metrics()

    async def handle(self, request: Request) -> Response:
        """Process a single unary RPC request."""
        pipeline_name = request.path_params.get("pipeline_name", "")

        # Generate or extract request ID.
        request_id_str = request.headers.get("x-nerva-request-id")
        if request_id_str:
            try:
                request_id = int(request_id_str)
            except ValueError:
                request_id = uuid.uuid4().int & ((1 << 64) - 1)
                logger.warning("invalid_request_id_header", value=request_id_str)
            else:
                if request_id < 0 or request_id >= (1 << 64):
                    logger.warning("out_of_range_request_id_header", value=request_id_str)
                    request_id = uuid.uuid4().int & ((1 << 64) - 1)
        else:
            request_id = uuid.uuid4().int & ((1 << 64) - 1)

        # Validate pipeline exists.
        if pipeline_name not in self._pipelines:
            return Response(
                content=_error_frame(
                    request_id,
                    ErrorCode.INVALID_ARGUMENT,
                    f"unknown pipeline: '{pipeline_name}'",
                ),
                media_type=CONTENT_TYPE,
            )

        # Validate deadline header.
        deadline_header = request.headers.get("x-nerva-deadline-ms")
        if not deadline_header:
            return Response(
                content=_error_frame(
                    request_id,
                    ErrorCode.INVALID_ARGUMENT,
                    "missing required header: x-nerva-deadline-ms",
                ),
                media_type=CONTENT_TYPE,
            )

        # Validate stream mode header (Phase 4: unary only).
        stream_header = request.headers.get("x-nerva-stream")
        if stream_header is None:
            return Response(
                content=_error_frame(
                    request_id,
                    ErrorCode.INVALID_ARGUMENT,
                    "missing required header: x-nerva-stream",
                ),
                media_type=CONTENT_TYPE,
            )
        if stream_header not in ("0", "1", "2"):
            return Response(
                content=_error_frame(
                    request_id,
                    ErrorCode.INVALID_ARGUMENT,
                    f"unsupported stream mode: {stream_header!r} (supported: '0', '1', '2')",
                ),
                media_type=CONTENT_TYPE,
            )

        # Convert absolute epoch ms to relative TTL ms.
        try:
            deadline_epoch_ms = int(deadline_header)
        except ValueError:
            return Response(
                content=_error_frame(
                    request_id,
                    ErrorCode.INVALID_ARGUMENT,
                    f"invalid x-nerva-deadline-ms: {deadline_header!r}",
                ),
                media_type=CONTENT_TYPE,
            )
        now_ms = int(time.time() * 1000)
        deadline_ms = deadline_epoch_ms - now_ms

        if deadline_ms <= 0:
            return Response(
                content=_error_frame(
                    request_id, ErrorCode.DEADLINE_EXCEEDED, "deadline already expired"
                ),
                media_type=CONTENT_TYPE,
            )

        # Read and parse frames.
        t_body_start = time.perf_counter()
        body = await request.body()
        t_body_end = time.perf_counter()
        try:
            frames = _parse_frames(body)
        except ProtocolError as e:
            return Response(
                content=_error_frame(
                    request_id, ErrorCode.INVALID_ARGUMENT, str(e)
                ),
                media_type=CONTENT_TYPE,
            )

        # Single-pass frame classification (avoids 3 list comprehensions).
        open_frame, data_frame, end_frame = _classify_frames(frames)

        if open_frame is None:
            return Response(
                content=_error_frame(
                    request_id, ErrorCode.INVALID_ARGUMENT, "no OPEN frame received"
                ),
                media_type=CONTENT_TYPE,
            )
        try:
            open_meta = msgpack.unpackb(open_frame.payload, raw=False)
        except Exception:
            return Response(
                content=_error_frame(
                    request_id, ErrorCode.INVALID_ARGUMENT, "invalid OPEN frame payload"
                ),
                media_type=CONTENT_TYPE,
            )
        open_pipeline = open_meta.get("pipeline") if isinstance(open_meta, dict) else None
        if open_pipeline and open_pipeline != pipeline_name:
            return Response(
                content=_error_frame(
                    request_id,
                    ErrorCode.INVALID_ARGUMENT,
                    f"OPEN pipeline '{open_pipeline}' does not match URL '{pipeline_name}'",
                ),
                media_type=CONTENT_TYPE,
            )

        if end_frame is None:
            return Response(
                content=_error_frame(
                    request_id, ErrorCode.INVALID_ARGUMENT, "no END frame received"
                ),
                media_type=CONTENT_TYPE,
            )

        if data_frame is None:
            return Response(
                content=_error_frame(
                    request_id, ErrorCode.INVALID_ARGUMENT, "no DATA frame received"
                ),
                media_type=CONTENT_TYPE,
            )

        try:
            t_parse_start = time.perf_counter()
            inputs: dict[str, Any] = msgpack.unpackb(data_frame.payload, raw=False)
            rpc_parse_ms = round((time.perf_counter() - t_parse_start) * 1000, 3)
            rpc_body_read_ms = round((t_body_end - t_body_start) * 1000, 3)
        except Exception:
            return Response(
                content=_error_frame(
                    request_id, ErrorCode.INVALID_ARGUMENT, "invalid DATA frame payload"
                ),
                media_type=CONTENT_TYPE,
            )

        # Route to streaming path for x-nerva-stream=1 or x-nerva-stream=2.
        # Mode 2 (full-duplex) is handled the same as mode 1 in MVP: input is buffered
        # above via `await request.body()`, output is streamed.
        if stream_header in ("1", "2"):
            return self._handle_stream(request_id, pipeline_name, inputs, deadline_ms)

        # Execute pipeline.
        executor = self._pipelines[pipeline_name]
        structlog.contextvars.bind_contextvars(
            request_id=str(request_id), pipeline=pipeline_name
        )
        t0 = time.monotonic()
        t_exec_start = time.perf_counter()
        self._metrics.request_in_flight.labels(pipeline=pipeline_name).inc()
        try:
            result = await executor.execute(
                inputs,
                deadline_ms=deadline_ms,
                request_id=str(request_id),
            )
        except Exception as exc:
            code, message = _map_exception(exc)
            self._metrics.request_total.labels(
                pipeline=pipeline_name, status=code.name.lower()
            ).inc()
            return Response(
                content=_error_frame(request_id, code, message),
                media_type=CONTENT_TYPE,
            )
        finally:
            elapsed = time.monotonic() - t0
            self._metrics.request_duration_seconds.labels(pipeline=pipeline_name).observe(elapsed)
            self._metrics.request_in_flight.labels(pipeline=pipeline_name).dec()
            structlog.contextvars.clear_contextvars()

        rpc_execute_ms = round((time.perf_counter() - t_exec_start) * 1000, 3)

        self._metrics.request_total.labels(pipeline=pipeline_name, status="ok").inc()

        # Build response: DATA + END.
        t_ser_start = time.perf_counter()
        resp_data = encode_frame(
            Frame(FrameType.DATA, request_id, 0, msgpack.packb(result, use_bin_type=True))
        )
        resp_end = encode_frame(
            Frame(FrameType.END, request_id, 0, msgpack.packb({"status": 0}, use_bin_type=True))
        )
        rpc_serialize_ms = round((time.perf_counter() - t_ser_start) * 1000, 3)

        _timing.write({
            "event": "rpc_timing",
            "request_id": str(request_id),
            "pipeline": pipeline_name,
            "rpc_body_read_ms": rpc_body_read_ms,
            "rpc_parse_ms": rpc_parse_ms,
            "rpc_execute_ms": rpc_execute_ms,
            "rpc_serialize_ms": rpc_serialize_ms,
            "rpc_overhead_ms": round(
                rpc_body_read_ms + rpc_parse_ms + rpc_serialize_ms, 3
            ),
        })

        return Response(
            content=resp_data + resp_end,
            media_type=CONTENT_TYPE,
        )


    def _handle_stream(
        self,
        request_id: int,
        pipeline_name: str,
        inputs: dict[str, Any],
        deadline_ms: int,
    ) -> StreamingResponse:
        """Return a StreamingResponse that yields DATA frames per chunk, then END.

        Exceptions inside the generator are caught and yield an ERROR frame instead
        of propagating — StreamingResponse would close the connection on an unhandled
        exception, preventing the client from receiving a structured error response.
        """
        executor = self._pipelines[pipeline_name]

        async def generate() -> Any:
            try:
                async for chunk in executor.execute_stream(
                    inputs,
                    deadline_ms=deadline_ms,
                    request_id=str(request_id),
                ):
                    yield encode_frame(
                        Frame(
                            FrameType.DATA,
                            request_id,
                            0,
                            msgpack.packb(chunk, use_bin_type=True),
                        )
                    )
                yield encode_frame(
                    Frame(
                        FrameType.END,
                        request_id,
                        0,
                        msgpack.packb({"status": 0}, use_bin_type=True),
                    )
                )
            except Exception as exc:
                code, message = _map_exception(exc)
                yield encode_frame(
                    Frame(
                        FrameType.ERROR,
                        request_id,
                        0,
                        msgpack.packb(
                            {"code": int(code), "message": message},
                            use_bin_type=True,
                        ),
                    )
                )
                # Do not re-raise: generator returns normally so Starlette
                # closes the response cleanly (no TCP connection reset).

        return StreamingResponse(generate(), media_type=CONTENT_TYPE)


def build_rpc_app(pipelines: dict[str, Any]) -> Starlette:
    """Build a minimal Starlette app with RPC routes for testing."""
    handler = RpcHandler(pipelines)
    return Starlette(
        routes=[Route("/rpc/{pipeline_name}", handler.handle, methods=["POST"])],
    )
