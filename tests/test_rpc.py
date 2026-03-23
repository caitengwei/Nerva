# tests/test_rpc.py
"""Tests for nerva.server.rpc — Binary RPC handler."""

import time
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import msgpack
from starlette.testclient import TestClient

from nerva.server.protocol import Frame, FrameType, encode_frame
from nerva.server.rpc import ErrorCode, build_rpc_app


class _MockStreamExecutor:
    """Executor mock that supports both execute() and execute_stream()."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        stream_error: Exception | None = None,
    ) -> None:
        self._chunks = chunks
        self._stream_error = stream_error

    async def execute(
        self, inputs: Any, *, deadline_ms: int = 0, request_id: str = ""
    ) -> dict[str, Any]:
        return self._chunks[-1] if self._chunks else {}

    async def execute_stream(
        self, inputs: Any, *, deadline_ms: int = 0, request_id: str = ""
    ) -> AsyncIterator[dict[str, Any]]:
        for chunk in self._chunks:
            yield chunk
        if self._stream_error is not None:
            raise self._stream_error


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
    mv = memoryview(body)
    offset = 0
    while offset < len(mv):
        frame, consumed = decode_frame(mv, offset)
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
        assert ErrorCode.UNAVAILABLE == 14


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
        result = msgpack.unpackb(frames[0].payload, raw=False)
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
        error = msgpack.unpackb(frames[0].payload, raw=False)
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
        error = msgpack.unpackb(frames[0].payload, raw=False)
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
        error = msgpack.unpackb(frames[0].payload, raw=False)
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
        error = msgpack.unpackb(frames[0].payload, raw=False)
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
        error = msgpack.unpackb(frames[0].payload, raw=False)
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
        error = msgpack.unpackb(frames[0].payload, raw=False)
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
        error = msgpack.unpackb(frames[0].payload, raw=False)
        assert error["code"] == ErrorCode.INVALID_ARGUMENT
        assert "x-nerva-stream" in error["message"]

    def test_unsupported_stream_mode(self) -> None:
        client = self._make_app()
        body = _make_request_frames("classify", {"x": 1})
        headers = self._rpc_headers()
        headers["x-nerva-stream"] = "99"  # anything beyond 0/1/2
        resp = client.post("/rpc/classify", content=body, headers=headers)
        frames = _decode_response_frames(resp.content)
        assert frames[0].frame_type == FrameType.ERROR
        error = msgpack.unpackb(frames[0].payload, raw=False)
        assert error["code"] == ErrorCode.INVALID_ARGUMENT
        assert "unsupported stream mode" in error["message"]

    def test_missing_end_frame(self) -> None:
        """OPEN + DATA but no END frame."""
        client = self._make_app()
        open_payload = msgpack.packb({"pipeline": "classify"})
        data_payload = msgpack.packb({"x": 1})
        body = b""
        body += encode_frame(Frame(FrameType.OPEN, 1, 0, open_payload))
        body += encode_frame(Frame(FrameType.DATA, 1, 0, data_payload))
        resp = client.post("/rpc/classify", content=body, headers=self._rpc_headers())
        frames = _decode_response_frames(resp.content)
        assert frames[0].frame_type == FrameType.ERROR
        error = msgpack.unpackb(frames[0].payload, raw=False)
        assert error["code"] == ErrorCode.INVALID_ARGUMENT
        assert "no END frame" in error["message"]

    def test_missing_open_frame(self) -> None:
        """DATA frame only, no OPEN frame."""
        client = self._make_app()
        data_payload = msgpack.packb({"x": 1})
        body = encode_frame(Frame(FrameType.DATA, 1, 0, data_payload))
        resp = client.post("/rpc/classify", content=body, headers=self._rpc_headers())
        frames = _decode_response_frames(resp.content)
        assert frames[0].frame_type == FrameType.ERROR
        error = msgpack.unpackb(frames[0].payload, raw=False)
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
        error = msgpack.unpackb(frames[0].payload, raw=False)
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
        error = msgpack.unpackb(frames[0].payload, raw=False)
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
        error = msgpack.unpackb(frames[0].payload, raw=False)
        assert error["code"] == ErrorCode.DEADLINE_EXCEEDED

    def test_unavailable_worker_disconnect(self) -> None:
        """WorkerProxy disconnect message maps to UNAVAILABLE (not INTERNAL)."""
        client = self._make_app(
            executor_side_effect=RuntimeError("UNAVAILABLE: worker disconnected")
        )
        body = _make_request_frames("classify", {"x": 1})
        resp = client.post("/rpc/classify", content=body, headers=self._rpc_headers())
        frames = _decode_response_frames(resp.content)
        error = msgpack.unpackb(frames[0].payload, raw=False)
        assert error["code"] == ErrorCode.UNAVAILABLE
        assert error["retryable"] is True

    def test_unavailable_error(self) -> None:
        """Exception containing UNAVAILABLE maps to UNAVAILABLE code."""
        client = self._make_app(
            executor_side_effect=RuntimeError("UNAVAILABLE: worker crashed")
        )
        body = _make_request_frames("classify", {"x": 1})
        resp = client.post("/rpc/classify", content=body, headers=self._rpc_headers())
        frames = _decode_response_frames(resp.content)
        error = msgpack.unpackb(frames[0].payload, raw=False)
        assert error["code"] == ErrorCode.UNAVAILABLE

    def test_unavailable_is_retryable(self) -> None:
        """UNAVAILABLE errors should be marked retryable."""
        client = self._make_app(
            executor_side_effect=RuntimeError("UNAVAILABLE: worker down")
        )
        body = _make_request_frames("classify", {"x": 1})
        resp = client.post("/rpc/classify", content=body, headers=self._rpc_headers())
        frames = _decode_response_frames(resp.content)
        error = msgpack.unpackb(frames[0].payload, raw=False)
        assert error["retryable"] is True

    def test_deadline_exceeded_not_retryable(self) -> None:
        """DEADLINE_EXCEEDED errors should NOT be marked retryable."""
        client = self._make_app(
            executor_side_effect=RuntimeError("DEADLINE_EXCEEDED: timeout")
        )
        body = _make_request_frames("classify", {"x": 1})
        resp = client.post("/rpc/classify", content=body, headers=self._rpc_headers())
        frames = _decode_response_frames(resp.content)
        error = msgpack.unpackb(frames[0].payload, raw=False)
        assert error["retryable"] is False

    def test_resource_exhausted_is_retryable(self) -> None:
        """RESOURCE_EXHAUSTED errors should be marked retryable."""
        client = self._make_app(
            executor_side_effect=RuntimeError("RESOURCE_EXHAUSTED")
        )
        body = _make_request_frames("classify", {"x": 1})
        resp = client.post("/rpc/classify", content=body, headers=self._rpc_headers())
        frames = _decode_response_frames(resp.content)
        error = msgpack.unpackb(frames[0].payload, raw=False)
        assert error["retryable"] is True


from prometheus_client import CollectorRegistry  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from starlette.routing import Route  # noqa: E402

from nerva.observability.metrics import NervaMetrics  # noqa: E402
from nerva.server.rpc import RpcHandler  # noqa: E402


class TestRpcHandlerMetrics:
    def _make_app_with_metrics(
        self,
        executor_result: dict[str, Any] | None = None,
        executor_side_effect: Exception | None = None,
    ) -> tuple[TestClient, NervaMetrics]:
        mock_executor = AsyncMock()
        if executor_side_effect:
            mock_executor.execute.side_effect = executor_side_effect
        else:
            mock_executor.execute.return_value = executor_result or {"out": 1}

        reg = CollectorRegistry()
        m = NervaMetrics(registry=reg)
        handler = RpcHandler({"classify": mock_executor}, metrics=m)
        from starlette.applications import Starlette
        from starlette.routing import Route
        app = Starlette(routes=[Route("/rpc/{pipeline_name}", handler.handle, methods=["POST"])])
        return TestClient(app), m

    def _rpc_headers(self, deadline_offset_ms: int = 30000) -> dict[str, str]:
        deadline = int(time.time() * 1000) + deadline_offset_ms
        return {
            "content-type": "application/x-nerva-rpc",
            "x-nerva-deadline-ms": str(deadline),
            "x-nerva-stream": "0",
        }

    def test_request_total_incremented_on_success(self) -> None:
        client, m = self._make_app_with_metrics()
        body = _make_request_frames("classify", {"x": 1})
        client.post("/rpc/classify", content=body, headers=self._rpc_headers())
        assert m.request_total.labels(pipeline="classify", status="ok")._value.get() == 1.0

    def test_request_in_flight_returns_to_zero(self) -> None:
        client, m = self._make_app_with_metrics()
        body = _make_request_frames("classify", {"x": 1})
        client.post("/rpc/classify", content=body, headers=self._rpc_headers())
        assert m.request_in_flight.labels(pipeline="classify")._value.get() == 0.0

    def test_request_total_incremented_on_error(self) -> None:
        client, m = self._make_app_with_metrics(
            executor_side_effect=RuntimeError("something went wrong")
        )
        body = _make_request_frames("classify", {"x": 1})
        client.post("/rpc/classify", content=body, headers=self._rpc_headers())
        assert m.request_total.labels(pipeline="classify", status="internal")._value.get() == 1.0


class TestRpcHandlerStreaming:
    """Tests for x-nerva-stream=1 and x-nerva-stream=2 streaming paths."""

    def _stream_headers(self, mode: str = "1") -> dict[str, str]:
        deadline = int(time.time() * 1000) + 30000
        return {
            "content-type": "application/x-nerva-rpc",
            "x-nerva-deadline-ms": str(deadline),
            "x-nerva-stream": mode,
        }

    def _make_app(
        self,
        chunks: list[dict[str, Any]],
        stream_error: Exception | None = None,
    ) -> TestClient:
        executor = _MockStreamExecutor(chunks, stream_error)
        pipelines = {"classify": executor}
        app = build_rpc_app(pipelines)
        return TestClient(app)

    def test_stream_mode1_yields_data_frames_plus_end(self) -> None:
        """x-nerva-stream=1: each chunk becomes a DATA frame, followed by END."""
        chunks = [{"tok": 0}, {"tok": 1}, {"tok": 2}]
        client = self._make_app(chunks)
        body = _make_request_frames("classify", {"prompt": "hi"})
        resp = client.post("/rpc/classify", content=body, headers=self._stream_headers("1"))

        assert resp.status_code == 200
        frames = _decode_response_frames(resp.content)
        assert len(frames) == 4  # 3 DATA + 1 END

        for i, frame in enumerate(frames[:3]):
            assert frame.frame_type == FrameType.DATA
            payload = msgpack.unpackb(frame.payload, raw=False)
            assert payload == {"tok": i}

        assert frames[-1].frame_type == FrameType.END

    def test_stream_mode2_input_buffered_same_as_mode1(self) -> None:
        """x-nerva-stream=2: input is buffered (MVP), output streaming same as mode 1."""
        chunks = [{"tok": 0}, {"tok": 1}]
        client = self._make_app(chunks)
        body = _make_request_frames("classify", {"audio": "..."})
        resp = client.post("/rpc/classify", content=body, headers=self._stream_headers("2"))

        assert resp.status_code == 200
        frames = _decode_response_frames(resp.content)
        assert len(frames) == 3  # 2 DATA + 1 END
        assert frames[0].frame_type == FrameType.DATA
        assert frames[1].frame_type == FrameType.DATA
        assert frames[2].frame_type == FrameType.END

    def test_stream_error_mid_stream_yields_error_frame(self) -> None:
        """Exception during streaming yields ERROR frame instead of crashing."""
        chunks = [{"tok": 0}]
        client = self._make_app(chunks, stream_error=RuntimeError("model exploded"))
        body = _make_request_frames("classify", {"x": 1})
        resp = client.post("/rpc/classify", content=body, headers=self._stream_headers("1"))

        assert resp.status_code == 200
        frames = _decode_response_frames(resp.content)
        # Should contain the successful chunk + an ERROR frame (no END frame).
        data_frames = [f for f in frames if f.frame_type == FrameType.DATA]
        error_frames = [f for f in frames if f.frame_type == FrameType.ERROR]
        assert len(data_frames) == 1
        assert len(error_frames) == 1
        error = msgpack.unpackb(error_frames[0].payload, raw=False)
        assert error["code"] == ErrorCode.INTERNAL

    def test_stream_mode_invalid_rejected(self) -> None:
        """x-nerva-stream=3 (unknown mode) is rejected with INVALID_ARGUMENT."""
        client = self._make_app([])
        body = _make_request_frames("classify", {"x": 1})
        resp = client.post("/rpc/classify", content=body, headers=self._stream_headers("3"))

        assert resp.status_code == 200
        frames = _decode_response_frames(resp.content)
        assert frames[0].frame_type == FrameType.ERROR
        error = msgpack.unpackb(frames[0].payload, raw=False)
        assert error["code"] == ErrorCode.INVALID_ARGUMENT

    def test_stream_error_frame_includes_retryable(self) -> None:
        """ERROR frame from streaming path must include 'retryable' field (protocol consistency)."""
        chunks: list[dict[str, Any]] = []
        client = self._make_app(chunks, stream_error=RuntimeError("fail"))
        body = _make_request_frames("classify", {"x": 1})
        resp = client.post("/rpc/classify", content=body, headers=self._stream_headers("1"))

        frames = _decode_response_frames(resp.content)
        error_frames = [f for f in frames if f.frame_type == FrameType.ERROR]
        assert len(error_frames) == 1
        error = msgpack.unpackb(error_frames[0].payload, raw=False)
        assert "retryable" in error

    def _make_stream_app_with_metrics(
        self,
        chunks: list[dict[str, Any]],
        stream_error: Exception | None = None,
    ) -> tuple[TestClient, Any]:
        executor = _MockStreamExecutor(chunks, stream_error)
        reg = CollectorRegistry()
        m = NervaMetrics(registry=reg)
        handler = RpcHandler({"classify": executor}, metrics=m)
        app = Starlette(routes=[Route("/rpc/{pipeline_name}", handler.handle, methods=["POST"])])
        return TestClient(app), m

    def test_stream_request_total_incremented_on_success(self) -> None:
        """request_total is incremented with status=ok after successful stream."""
        client, m = self._make_stream_app_with_metrics([{"tok": 0}])
        body = _make_request_frames("classify", {"x": 1})
        client.post("/rpc/classify", content=body, headers=self._stream_headers("1"))
        assert m.request_total.labels(pipeline="classify", status="ok")._value.get() == 1.0

    def test_stream_request_in_flight_returns_to_zero(self) -> None:
        """request_in_flight returns to 0 after stream completes."""
        client, m = self._make_stream_app_with_metrics([{"tok": 0}])
        body = _make_request_frames("classify", {"x": 1})
        client.post("/rpc/classify", content=body, headers=self._stream_headers("1"))
        assert m.request_in_flight.labels(pipeline="classify")._value.get() == 0.0

    def test_stream_request_total_incremented_on_error(self) -> None:
        """request_total is incremented with error status when streaming raises."""
        client, m = self._make_stream_app_with_metrics(
            [], stream_error=RuntimeError("exploded")
        )
        body = _make_request_frames("classify", {"x": 1})
        client.post("/rpc/classify", content=body, headers=self._stream_headers("1"))
        assert m.request_total.labels(pipeline="classify", status="internal")._value.get() == 1.0
