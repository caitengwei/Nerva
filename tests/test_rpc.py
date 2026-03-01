# tests/test_rpc.py
"""Tests for nerva.server.rpc — Binary RPC handler."""

import time
from typing import Any
from unittest.mock import AsyncMock

import msgpack
from starlette.testclient import TestClient

from nerva.server.protocol import Frame, FrameType, encode_frame
from nerva.server.rpc import ErrorCode, build_rpc_app


def _make_request_frames(
    pipeline: str, inputs: dict[str, Any], request_id: int = 1
) -> bytes:
    """Build OPEN + DATA + END frame bytes."""
    open_payload = msgpack.packb({"pipeline": pipeline})
    data_payload = msgpack.packb(inputs)
    frames = b""
    frames += encode_frame(Frame(FrameType.OPEN, request_id, 0, open_payload))
    frames += encode_frame(Frame(FrameType.DATA, request_id, 0, data_payload))
    frames += encode_frame(Frame(FrameType.END, request_id, 0, b""))
    return frames


def _decode_response_frames(body: bytes) -> list[Frame]:
    """Decode all frames from response body."""
    from nerva.server.protocol import decode_frame

    frames = []
    offset = 0
    while offset < len(body):
        frame, consumed = decode_frame(body[offset:])
        frames.append(frame)
        offset += consumed
    return frames


class TestErrorCode:
    def test_codes(self) -> None:
        assert ErrorCode.OK == 0
        assert ErrorCode.INVALID_ARGUMENT == 3
        assert ErrorCode.DEADLINE_EXCEEDED == 4
        assert ErrorCode.RESOURCE_EXHAUSTED == 8
        assert ErrorCode.INTERNAL == 13


class TestRpcHandler:
    def _make_app(
        self,
        executor_result: dict[str, Any] | None = None,
        executor_side_effect: Exception | None = None,
    ) -> TestClient:
        mock_executor = AsyncMock()
        if executor_side_effect:
            mock_executor.execute.side_effect = executor_side_effect
        else:
            mock_executor.execute.return_value = executor_result or {"out": 1}

        pipelines = {"classify": mock_executor}
        app = build_rpc_app(pipelines)
        return TestClient(app)

    def test_unary_happy_path(self) -> None:
        client = self._make_app(executor_result={"label": "cat", "score": 0.95})
        body = _make_request_frames("classify", {"text": "hello"})
        deadline = int(time.time() * 1000) + 30000
        resp = client.post(
            "/rpc/classify",
            content=body,
            headers={
                "content-type": "application/x-nerva-rpc",
                "x-nerva-deadline-ms": str(deadline),
                "x-nerva-stream": "0",
            },
        )
        assert resp.status_code == 200
        frames = _decode_response_frames(resp.content)
        assert len(frames) == 2  # DATA + END
        assert frames[0].frame_type == FrameType.DATA
        result = msgpack.unpackb(frames[0].payload)
        assert result == {"label": "cat", "score": 0.95}
        assert frames[1].frame_type == FrameType.END

    def test_unknown_pipeline(self) -> None:
        client = self._make_app()
        body = _make_request_frames("nonexistent", {"x": 1})
        deadline = int(time.time() * 1000) + 30000
        resp = client.post(
            "/rpc/nonexistent",
            content=body,
            headers={
                "content-type": "application/x-nerva-rpc",
                "x-nerva-deadline-ms": str(deadline),
                "x-nerva-stream": "0",
            },
        )
        assert resp.status_code == 200
        frames = _decode_response_frames(resp.content)
        assert len(frames) == 1
        assert frames[0].frame_type == FrameType.ERROR
        error = msgpack.unpackb(frames[0].payload)
        assert error["code"] == ErrorCode.INVALID_ARGUMENT

    def test_deadline_already_expired(self) -> None:
        client = self._make_app()
        body = _make_request_frames("classify", {"x": 1})
        deadline = int(time.time() * 1000) - 1000  # 1 second ago
        resp = client.post(
            "/rpc/classify",
            content=body,
            headers={
                "content-type": "application/x-nerva-rpc",
                "x-nerva-deadline-ms": str(deadline),
                "x-nerva-stream": "0",
            },
        )
        frames = _decode_response_frames(resp.content)
        assert frames[0].frame_type == FrameType.ERROR
        error = msgpack.unpackb(frames[0].payload)
        assert error["code"] == ErrorCode.DEADLINE_EXCEEDED

    def test_executor_exception_maps_to_internal(self) -> None:
        client = self._make_app(
            executor_side_effect=RuntimeError("unexpected")
        )
        body = _make_request_frames("classify", {"x": 1})
        deadline = int(time.time() * 1000) + 30000
        resp = client.post(
            "/rpc/classify",
            content=body,
            headers={
                "content-type": "application/x-nerva-rpc",
                "x-nerva-deadline-ms": str(deadline),
                "x-nerva-stream": "0",
            },
        )
        frames = _decode_response_frames(resp.content)
        assert frames[0].frame_type == FrameType.ERROR
        error = msgpack.unpackb(frames[0].payload)
        assert error["code"] == ErrorCode.INTERNAL

    def test_resource_exhausted_error(self) -> None:
        client = self._make_app(
            executor_side_effect=RuntimeError("RESOURCE_EXHAUSTED")
        )
        body = _make_request_frames("classify", {"x": 1})
        deadline = int(time.time() * 1000) + 30000
        resp = client.post(
            "/rpc/classify",
            content=body,
            headers={
                "content-type": "application/x-nerva-rpc",
                "x-nerva-deadline-ms": str(deadline),
                "x-nerva-stream": "0",
            },
        )
        frames = _decode_response_frames(resp.content)
        error = msgpack.unpackb(frames[0].payload)
        assert error["code"] == ErrorCode.RESOURCE_EXHAUSTED

    def test_missing_deadline_header(self) -> None:
        client = self._make_app()
        body = _make_request_frames("classify", {"x": 1})
        resp = client.post(
            "/rpc/classify",
            content=body,
            headers={
                "content-type": "application/x-nerva-rpc",
                "x-nerva-stream": "0",
            },
        )
        frames = _decode_response_frames(resp.content)
        assert frames[0].frame_type == FrameType.ERROR
        error = msgpack.unpackb(frames[0].payload)
        assert error["code"] == ErrorCode.INVALID_ARGUMENT

    def test_deadline_passed_to_executor(self) -> None:
        mock_executor = AsyncMock()
        mock_executor.execute.return_value = {"out": 1}
        app = build_rpc_app({"test": mock_executor})
        client = TestClient(app)

        body = _make_request_frames("test", {"x": 1})
        deadline = int(time.time() * 1000) + 5000
        client.post(
            "/rpc/test",
            content=body,
            headers={
                "content-type": "application/x-nerva-rpc",
                "x-nerva-deadline-ms": str(deadline),
                "x-nerva-stream": "0",
            },
        )
        call_kwargs = mock_executor.execute.call_args
        assert "deadline_ms" in call_kwargs.kwargs
        assert call_kwargs.kwargs["deadline_ms"] > 0

    def _rpc_headers(self, deadline_offset_ms: int = 30000) -> dict[str, str]:
        deadline = int(time.time() * 1000) + deadline_offset_ms
        return {
            "content-type": "application/x-nerva-rpc",
            "x-nerva-deadline-ms": str(deadline),
            "x-nerva-stream": "0",
        }

    def test_invalid_deadline_header(self) -> None:
        client = self._make_app()
        body = _make_request_frames("classify", {"x": 1})
        headers = self._rpc_headers()
        headers["x-nerva-deadline-ms"] = "not-a-number"
        resp = client.post("/rpc/classify", content=body, headers=headers)
        frames = _decode_response_frames(resp.content)
        assert frames[0].frame_type == FrameType.ERROR
        error = msgpack.unpackb(frames[0].payload)
        assert error["code"] == ErrorCode.INVALID_ARGUMENT
        assert "invalid x-nerva-deadline-ms" in error["message"]

    def test_missing_stream_header(self) -> None:
        client = self._make_app()
        body = _make_request_frames("classify", {"x": 1})
        deadline = int(time.time() * 1000) + 30000
        resp = client.post(
            "/rpc/classify",
            content=body,
            headers={
                "content-type": "application/x-nerva-rpc",
                "x-nerva-deadline-ms": str(deadline),
            },
        )
        frames = _decode_response_frames(resp.content)
        assert frames[0].frame_type == FrameType.ERROR
        error = msgpack.unpackb(frames[0].payload)
        assert error["code"] == ErrorCode.INVALID_ARGUMENT
        assert "x-nerva-stream" in error["message"]

    def test_unsupported_stream_mode(self) -> None:
        client = self._make_app()
        body = _make_request_frames("classify", {"x": 1})
        headers = self._rpc_headers()
        headers["x-nerva-stream"] = "1"
        resp = client.post("/rpc/classify", content=body, headers=headers)
        frames = _decode_response_frames(resp.content)
        assert frames[0].frame_type == FrameType.ERROR
        error = msgpack.unpackb(frames[0].payload)
        assert error["code"] == ErrorCode.INVALID_ARGUMENT
        assert "unsupported stream mode" in error["message"]

    def test_missing_open_frame(self) -> None:
        """DATA frame only, no OPEN frame."""
        client = self._make_app()
        data_payload = msgpack.packb({"x": 1})
        body = encode_frame(Frame(FrameType.DATA, 1, 0, data_payload))
        resp = client.post("/rpc/classify", content=body, headers=self._rpc_headers())
        frames = _decode_response_frames(resp.content)
        assert frames[0].frame_type == FrameType.ERROR
        error = msgpack.unpackb(frames[0].payload)
        assert error["code"] == ErrorCode.INVALID_ARGUMENT
        assert "no OPEN frame" in error["message"]

    def test_open_frame_pipeline_mismatch(self) -> None:
        """OPEN frame pipeline name differs from URL path."""
        client = self._make_app()
        open_payload = msgpack.packb({"pipeline": "wrong_name"})
        data_payload = msgpack.packb({"x": 1})
        body = b""
        body += encode_frame(Frame(FrameType.OPEN, 1, 0, open_payload))
        body += encode_frame(Frame(FrameType.DATA, 1, 0, data_payload))
        body += encode_frame(Frame(FrameType.END, 1, 0, b""))
        resp = client.post("/rpc/classify", content=body, headers=self._rpc_headers())
        frames = _decode_response_frames(resp.content)
        assert frames[0].frame_type == FrameType.ERROR
        error = msgpack.unpackb(frames[0].payload)
        assert error["code"] == ErrorCode.INVALID_ARGUMENT
        assert "does not match URL" in error["message"]

    def test_invalid_data_frame_payload(self) -> None:
        """DATA frame with invalid msgpack payload."""
        client = self._make_app()
        open_payload = msgpack.packb({"pipeline": "classify"})
        body = b""
        body += encode_frame(Frame(FrameType.OPEN, 1, 0, open_payload))
        body += encode_frame(Frame(FrameType.DATA, 1, 0, b"\xff\xfe\xfd"))
        body += encode_frame(Frame(FrameType.END, 1, 0, b""))
        resp = client.post("/rpc/classify", content=body, headers=self._rpc_headers())
        frames = _decode_response_frames(resp.content)
        assert frames[0].frame_type == FrameType.ERROR
        error = msgpack.unpackb(frames[0].payload)
        assert error["code"] == ErrorCode.INVALID_ARGUMENT
        assert "invalid DATA frame" in error["message"]

    def test_deadline_exceeded_string_match(self) -> None:
        """Exception containing DEADLINE_EXCEEDED maps correctly."""
        client = self._make_app(
            executor_side_effect=RuntimeError("DEADLINE_EXCEEDED: timeout")
        )
        body = _make_request_frames("classify", {"x": 1})
        resp = client.post("/rpc/classify", content=body, headers=self._rpc_headers())
        frames = _decode_response_frames(resp.content)
        error = msgpack.unpackb(frames[0].payload)
        assert error["code"] == ErrorCode.DEADLINE_EXCEEDED
