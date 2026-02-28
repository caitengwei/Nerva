"""Binary RPC handler for Nerva inference serving."""

from __future__ import annotations

import logging
import time
import uuid
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import msgpack
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Route

if TYPE_CHECKING:
    from starlette.requests import Request

from nerva.server.protocol import Frame, FrameType, ProtocolError, decode_frame, encode_frame

logger = logging.getLogger(__name__)

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
        {"code": int(code), "message": message, "retryable": retryable}
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

    def __init__(self, pipelines: dict[str, Any]) -> None:
        self._pipelines = pipelines

    async def handle(self, request: Request) -> Response:
        """Process a single unary RPC request."""
        pipeline_name = request.path_params.get("pipeline_name", "")

        # Generate or extract request ID.
        request_id_str = request.headers.get("x-nerva-request-id")
        request_id = (
            int(request_id_str) if request_id_str else uuid.uuid4().int & ((1 << 64) - 1)
        )

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

        # Convert absolute epoch ms to relative TTL ms.
        deadline_epoch_ms = int(deadline_header)
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

        # Extract DATA payload (first DATA frame).
        data_frames = [f for f in frames if f.frame_type == FrameType.DATA]
        if not data_frames:
            return Response(
                content=_error_frame(
                    request_id, ErrorCode.INVALID_ARGUMENT, "no DATA frame received"
                ),
                media_type=CONTENT_TYPE,
            )

        inputs: dict[str, Any] = msgpack.unpackb(data_frames[0].payload, raw=False)

        # Execute pipeline.
        executor = self._pipelines[pipeline_name]
        try:
            result = await executor.execute(
                inputs,
                deadline_ms=deadline_ms,
                request_id=str(request_id),
            )
        except Exception as exc:
            code, message = _map_exception(exc)
            return Response(
                content=_error_frame(request_id, code, message),
                media_type=CONTENT_TYPE,
            )

        # Build response: DATA + END.
        resp_data = encode_frame(
            Frame(FrameType.DATA, request_id, 0, msgpack.packb(result))
        )
        resp_end = encode_frame(
            Frame(FrameType.END, request_id, 0, msgpack.packb({"status": 0}))
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
