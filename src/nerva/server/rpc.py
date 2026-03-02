"""Binary RPC handler for Nerva inference serving."""

from __future__ import annotations

import time
import uuid
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import msgpack
import structlog
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Route

if TYPE_CHECKING:
    from starlette.requests import Request

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


def _error_frame(request_id: int, code: ErrorCode, message: str) -> bytes:
    """Build an ERROR frame."""
    retryable = code in (ErrorCode.DEADLINE_EXCEEDED, ErrorCode.RESOURCE_EXHAUSTED)
    payload = msgpack.packb(
        {"code": int(code), "message": message, "retryable": retryable},
        use_bin_type=True,
    )
    return encode_frame(Frame(FrameType.ERROR, request_id, 0, payload))


def _map_exception(exc: BaseException) -> tuple[ErrorCode, str]:
    """Map an execution exception to an error code and message."""
    msg = str(exc)
    if "RESOURCE_EXHAUSTED" in msg:
        return ErrorCode.RESOURCE_EXHAUSTED, msg
    if "DEADLINE_EXCEEDED" in msg:
        return ErrorCode.DEADLINE_EXCEEDED, msg
    return ErrorCode.INTERNAL, msg


def _parse_frames(data: bytes) -> list[Frame]:
    """Parse all frames from a byte buffer."""
    frames: list[Frame] = []
    offset = 0
    while offset < len(data):
        frame, consumed = decode_frame(data[offset:])
        frames.append(frame)
        offset += consumed
    return frames


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
        if stream_header != "0":
            return Response(
                content=_error_frame(
                    request_id,
                    ErrorCode.INVALID_ARGUMENT,
                    f"unsupported stream mode: {stream_header!r} (only '0' supported)",
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
        body = await request.body()
        try:
            frames = _parse_frames(body)
        except ProtocolError as e:
            return Response(
                content=_error_frame(
                    request_id, ErrorCode.INVALID_ARGUMENT, str(e)
                ),
                media_type=CONTENT_TYPE,
            )

        # Validate OPEN frame exists and pipeline name matches URL path.
        open_frames = [f for f in frames if f.frame_type == FrameType.OPEN]
        if not open_frames:
            return Response(
                content=_error_frame(
                    request_id, ErrorCode.INVALID_ARGUMENT, "no OPEN frame received"
                ),
                media_type=CONTENT_TYPE,
            )
        try:
            open_meta = msgpack.unpackb(open_frames[0].payload, raw=False)
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

        # Validate END frame presence (unary protocol: OPEN + DATA + END).
        end_frames = [f for f in frames if f.frame_type == FrameType.END]
        if not end_frames:
            return Response(
                content=_error_frame(
                    request_id, ErrorCode.INVALID_ARGUMENT, "no END frame received"
                ),
                media_type=CONTENT_TYPE,
            )

        # Extract DATA payload (first DATA frame; MVP only uses one).
        data_frames = [f for f in frames if f.frame_type == FrameType.DATA]
        if not data_frames:
            return Response(
                content=_error_frame(
                    request_id, ErrorCode.INVALID_ARGUMENT, "no DATA frame received"
                ),
                media_type=CONTENT_TYPE,
            )

        try:
            inputs: dict[str, Any] = msgpack.unpackb(data_frames[0].payload, raw=False)
        except Exception:
            return Response(
                content=_error_frame(
                    request_id, ErrorCode.INVALID_ARGUMENT, "invalid DATA frame payload"
                ),
                media_type=CONTENT_TYPE,
            )

        # Execute pipeline.
        executor = self._pipelines[pipeline_name]
        structlog.contextvars.bind_contextvars(
            request_id=str(request_id), pipeline=pipeline_name
        )
        t0 = time.monotonic()
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

        self._metrics.request_total.labels(pipeline=pipeline_name, status="ok").inc()

        # Build response: DATA + END.
        resp_data = encode_frame(
            Frame(FrameType.DATA, request_id, 0, msgpack.packb(result, use_bin_type=True))
        )
        resp_end = encode_frame(
            Frame(FrameType.END, request_id, 0, msgpack.packb({"status": 0}, use_bin_type=True))
        )
        return Response(
            content=resp_data + resp_end,
            media_type=CONTENT_TYPE,
        )


def build_rpc_app(pipelines: dict[str, Any]) -> Starlette:
    """Build a minimal Starlette app with RPC routes for testing."""
    handler = RpcHandler(pipelines)
    return Starlette(
        routes=[Route("/rpc/{pipeline_name}", handler.handle, methods=["POST"])],
    )
