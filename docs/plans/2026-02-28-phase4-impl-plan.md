# Phase 4 — Binary RPC Serving Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose Nerva pipelines via HTTP with a binary RPC protocol (unary mode), auto-managing workers.

**Architecture:** `serve()` scans Graphs for model names, looks up ModelHandles from a global registry, auto-spawns workers via WorkerManager, assembles a Starlette ASGI app with binary RPC handler + management routes, and starts uvicorn. Binary frames use 32-byte fixed header + msgpack payload.

**Tech Stack:** Starlette, uvicorn, msgpack, httpx (test client), struct (frame header)

---

### Task 1: Frame codec — FrameType, Frame, encode/decode

**Files:**
- Create: `src/nerva/server/protocol.py`
- Test: `tests/test_protocol.py`

**Context:**
- Protocol spec: `docs/plans/protocol-v1.md`
- MVP defaults: `docs/plans/mvp-defaults.md` (`PROTOCOL_MAX_FRAME_PAYLOAD_BYTES=4MiB`)
- Frame header is 32 bytes, big-endian, layout:
  - bytes 0-1: magic `0x4E56`
  - byte 2: version `1`
  - byte 3: frame_type (enum)
  - byte 4: flags
  - bytes 5-7: reserved (zero)
  - bytes 8-15: request_id (u64)
  - bytes 16-19: stream_id (u32, fixed 1 for MVP)
  - bytes 20-23: payload_length (u32)
  - bytes 24-27: crc32 (u32, fixed 0 for MVP)
  - bytes 28-31: ext_hdr_len (u32, fixed 0 for MVP)

**Step 1: Write the failing tests**

```python
# tests/test_protocol.py
"""Tests for nerva.server.protocol — Binary frame codec."""

import pytest

from nerva.server.protocol import (
    HEADER_SIZE,
    MAGIC,
    MAX_PAYLOAD_BYTES,
    VERSION,
    Frame,
    FrameType,
    ProtocolError,
    decode_frame,
    encode_frame,
)


class TestFrameType:
    def test_all_types_defined(self) -> None:
        assert FrameType.OPEN == 1
        assert FrameType.DATA == 2
        assert FrameType.END == 3
        assert FrameType.ERROR == 4
        assert FrameType.HEARTBEAT == 5


class TestConstants:
    def test_header_size(self) -> None:
        assert HEADER_SIZE == 32

    def test_magic(self) -> None:
        assert MAGIC == 0x4E56

    def test_version(self) -> None:
        assert VERSION == 1

    def test_max_payload(self) -> None:
        assert MAX_PAYLOAD_BYTES == 4 * 1024 * 1024


class TestRoundtrip:
    def test_data_frame_roundtrip(self) -> None:
        frame = Frame(
            frame_type=FrameType.DATA,
            request_id=42,
            flags=0,
            payload=b'{"value": 1}',
        )
        raw = encode_frame(frame)
        decoded, consumed = decode_frame(raw)
        assert consumed == HEADER_SIZE + len(frame.payload)
        assert decoded.frame_type == FrameType.DATA
        assert decoded.request_id == 42
        assert decoded.flags == 0
        assert decoded.payload == b'{"value": 1}'

    def test_open_frame_roundtrip(self) -> None:
        frame = Frame(
            frame_type=FrameType.OPEN,
            request_id=100,
            flags=0,
            payload=b"\x81\xa8pipeline\xa8classify",  # msgpack
        )
        raw = encode_frame(frame)
        decoded, consumed = decode_frame(raw)
        assert decoded.frame_type == FrameType.OPEN
        assert decoded.request_id == 100
        assert decoded.payload == frame.payload

    def test_empty_payload(self) -> None:
        frame = Frame(frame_type=FrameType.END, request_id=1, flags=0, payload=b"")
        raw = encode_frame(frame)
        decoded, consumed = decode_frame(raw)
        assert decoded.frame_type == FrameType.END
        assert decoded.payload == b""
        assert consumed == HEADER_SIZE

    def test_error_frame_roundtrip(self) -> None:
        frame = Frame(
            frame_type=FrameType.ERROR,
            request_id=99,
            flags=0,
            payload=b"\x83\xa4code\x04\xa7message\xb2DEADLINE_EXCEEDED\xa9retryable\xc3",
        )
        raw = encode_frame(frame)
        decoded, _ = decode_frame(raw)
        assert decoded.frame_type == FrameType.ERROR
        assert decoded.request_id == 99


class TestDecodeValidation:
    def test_short_buffer(self) -> None:
        with pytest.raises(ProtocolError, match="incomplete header"):
            decode_frame(b"\x00" * 10)

    def test_bad_magic(self) -> None:
        raw = b"\xFF\xFF" + b"\x00" * 30
        with pytest.raises(ProtocolError, match="magic"):
            decode_frame(raw)

    def test_bad_version(self) -> None:
        import struct
        header = struct.pack(">HBB", 0x4E56, 99, 2)  # bad version
        header += b"\x00" * (32 - len(header))
        with pytest.raises(ProtocolError, match="version"):
            decode_frame(header)

    def test_payload_exceeds_max(self) -> None:
        frame = Frame(frame_type=FrameType.DATA, request_id=1, flags=0, payload=b"x")
        raw = bytearray(encode_frame(frame))
        # Overwrite payload_length to exceed max
        import struct
        struct.pack_into(">I", raw, 20, MAX_PAYLOAD_BYTES + 1)
        with pytest.raises(ProtocolError, match="payload"):
            decode_frame(bytes(raw))

    def test_incomplete_payload(self) -> None:
        frame = Frame(
            frame_type=FrameType.DATA, request_id=1, flags=0, payload=b"hello"
        )
        raw = encode_frame(frame)
        with pytest.raises(ProtocolError, match="incomplete payload"):
            decode_frame(raw[:-2])  # Truncate payload


class TestEncodeValidation:
    def test_payload_too_large(self) -> None:
        frame = Frame(
            frame_type=FrameType.DATA,
            request_id=1,
            flags=0,
            payload=b"x" * (MAX_PAYLOAD_BYTES + 1),
        )
        with pytest.raises(ProtocolError, match="payload"):
            encode_frame(frame)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_protocol.py -v`
Expected: FAIL (ImportError — module doesn't exist)

**Step 3: Write implementation**

```python
# src/nerva/server/protocol.py
"""Binary frame codec for Nerva RPC protocol.

Frame layout (32-byte fixed header + variable payload):
  bytes 0-1:   magic (0x4E56)
  byte  2:     version (1)
  byte  3:     frame_type
  byte  4:     flags
  bytes 5-7:   reserved
  bytes 8-15:  request_id (u64)
  bytes 16-19: stream_id (u32, fixed 1)
  bytes 20-23: payload_length (u32)
  bytes 24-27: crc32 (u32, fixed 0)
  bytes 28-31: ext_hdr_len (u32, fixed 0)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

MAGIC = 0x4E56
VERSION = 1
HEADER_SIZE = 32
MAX_PAYLOAD_BYTES = 4 * 1024 * 1024  # 4 MiB

# Header format: magic(H) version(B) frame_type(B) flags(B) reserved(3x) request_id(Q)
#                stream_id(I) payload_len(I) crc32(I) ext_hdr_len(I)
_HEADER_FMT = ">HBBB3xQIIII"


class FrameType(IntEnum):
    OPEN = 1
    DATA = 2
    END = 3
    ERROR = 4
    HEARTBEAT = 5


class ProtocolError(Exception):
    """Raised on protocol-level encoding/decoding errors."""


@dataclass
class Frame:
    """A single binary RPC frame."""

    frame_type: FrameType
    request_id: int
    flags: int
    payload: bytes


def encode_frame(frame: Frame) -> bytes:
    """Encode a Frame into wire bytes (header + payload)."""
    if len(frame.payload) > MAX_PAYLOAD_BYTES:
        raise ProtocolError(
            f"payload size {len(frame.payload)} exceeds max {MAX_PAYLOAD_BYTES}"
        )
    header = struct.pack(
        _HEADER_FMT,
        MAGIC,
        VERSION,
        frame.frame_type,
        frame.flags,
        frame.request_id,
        1,  # stream_id (fixed)
        len(frame.payload),
        0,  # crc32 (fixed)
        0,  # ext_hdr_len (fixed)
    )
    return header + frame.payload


def decode_frame(data: bytes) -> tuple[Frame, int]:
    """Decode a Frame from wire bytes.

    Returns:
        (frame, consumed_bytes) tuple.

    Raises:
        ProtocolError: On invalid header, bad magic/version, or size violations.
    """
    if len(data) < HEADER_SIZE:
        raise ProtocolError(
            f"incomplete header: got {len(data)} bytes, need {HEADER_SIZE}"
        )

    magic, version, frame_type, flags, request_id, stream_id, payload_len, crc32, ext_hdr_len = (
        struct.unpack_from(_HEADER_FMT, data)
    )

    if magic != MAGIC:
        raise ProtocolError(f"bad magic: expected 0x{MAGIC:04X}, got 0x{magic:04X}")
    if version != VERSION:
        raise ProtocolError(f"unsupported version: {version}")
    if payload_len > MAX_PAYLOAD_BYTES:
        raise ProtocolError(
            f"payload size {payload_len} exceeds max {MAX_PAYLOAD_BYTES}"
        )

    total = HEADER_SIZE + payload_len
    if len(data) < total:
        raise ProtocolError(
            f"incomplete payload: got {len(data)} bytes, need {total}"
        )

    payload = data[HEADER_SIZE:total]
    frame = Frame(
        frame_type=FrameType(frame_type),
        request_id=request_id,
        flags=flags,
        payload=payload,
    )
    return frame, total
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_protocol.py -v`
Expected: ALL PASS

**Step 5: Lint + type check**

Run: `uv run ruff check src/nerva/server/protocol.py tests/test_protocol.py && uv run mypy src/nerva/server/protocol.py`

**Step 6: Commit**

```bash
git add src/nerva/server/protocol.py tests/test_protocol.py
git commit -m "feat(protocol): add binary frame codec with FrameType, Frame, encode/decode"
```

---

### Task 2: Model handle registry

**Files:**
- Modify: `src/nerva/core/model.py`
- Test: `tests/test_model.py`

**Context:**
- `serve()` needs to look up `ModelHandle` by `model_name` to auto-spawn workers
- Pattern: follows `backends/registry.py` (global dict + accessor functions)
- `Graph` nodes only store `model_name` (string), not `ModelHandle`

**Step 1: Write the failing tests**

Add to `tests/test_model.py`:

```python
from nerva.core.model import get_model_handle, list_model_handles, _model_registry


class TestModelRegistry:
    def setup_method(self) -> None:
        _model_registry.clear()

    def teardown_method(self) -> None:
        _model_registry.clear()

    def test_model_registers_handle(self) -> None:
        handle = model("test_reg", DummyModel)
        assert get_model_handle("test_reg") is handle

    def test_get_missing_raises(self) -> None:
        with pytest.raises(KeyError, match="test_missing"):
            get_model_handle("test_missing")

    def test_list_handles(self) -> None:
        model("a", DummyModel)
        model("b", DummyModel)
        handles = list_model_handles()
        assert set(handles.keys()) == {"a", "b"}

    def test_duplicate_name_overwrites(self) -> None:
        model("dup", DummyModel, device="cpu")
        handle2 = model("dup", DummyModel, device="cuda:0")
        assert get_model_handle("dup") is handle2
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_model.py::TestModelRegistry -v`
Expected: FAIL (ImportError — functions don't exist)

**Step 3: Write implementation**

Add to `src/nerva/core/model.py`:

```python
# After ModelHandle class definition, before model() function:

_model_registry: dict[str, ModelHandle] = {}


def get_model_handle(name: str) -> ModelHandle:
    """Look up a registered ModelHandle by name.

    Raises:
        KeyError: If no handle with the given name has been registered.
    """
    if name not in _model_registry:
        raise KeyError(f"No model handle registered for '{name}'")
    return _model_registry[name]


def list_model_handles() -> dict[str, ModelHandle]:
    """Return a copy of all registered model handles."""
    return dict(_model_registry)
```

And in the `model()` function, after creating the handle, add:

```python
    handle = ModelHandle(name=name, ...)
    _model_registry[name] = handle
    return handle
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_model.py -v`
Expected: ALL PASS

**Step 5: Lint + type check**

Run: `uv run ruff check src/nerva/core/model.py && uv run mypy src/nerva/core/model.py`

**Step 6: Commit**

```bash
git add src/nerva/core/model.py tests/test_model.py
git commit -m "feat(model): add model handle registry for serve() auto-discovery"
```

---

### Task 3: RPC handler

**Files:**
- Create: `src/nerva/server/rpc.py`
- Test: `tests/test_rpc.py`

**Context:**
- Reads OPEN+DATA+END frames from request body, executes pipeline, returns DATA+END or ERROR
- Uses `Executor` for pipeline execution
- Converts absolute `x-nerva-deadline-ms` header to relative TTL for `InferContext`
- Error mapping: pipeline not found → `INVALID_ARGUMENT(3)`, deadline → `DEADLINE_EXCEEDED(4)`, etc.
- Uses `starlette.requests.Request` and `starlette.responses.Response`

**Step 1: Write the failing tests**

```python
# tests/test_rpc.py
"""Tests for nerva.server.rpc — Binary RPC handler."""

import time
from typing import Any
from unittest.mock import AsyncMock

import msgpack
import pytest
from starlette.testclient import TestClient

from nerva.backends.base import InferContext
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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_rpc.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

```python
# src/nerva/server/rpc.py
"""Binary RPC handler for Nerva inference serving."""

from __future__ import annotations

import logging
import time
import uuid
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import msgpack
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from nerva.server.protocol import Frame, FrameType, ProtocolError, decode_frame, encode_frame

if TYPE_CHECKING:
    pass

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
        if request_id_str:
            request_id = int(request_id_str)
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

        inputs = msgpack.unpackb(data_frames[0].payload, raw=False)

        # Execute pipeline.
        executor = self._pipelines[pipeline_name]
        try:
            result = await executor.execute(inputs)
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
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_rpc.py -v`
Expected: ALL PASS

**Step 5: Lint + type check**

Run: `uv run ruff check src/nerva/server/rpc.py tests/test_rpc.py && uv run mypy src/nerva/server/rpc.py`

**Step 6: Commit**

```bash
git add src/nerva/server/rpc.py tests/test_rpc.py
git commit -m "feat(rpc): add binary RPC handler with frame parsing and error mapping"
```

---

### Task 4: ASGI app assembly + management routes

**Files:**
- Create: `src/nerva/server/app.py`
- Test: `tests/test_app.py`

**Context:**
- `build_app()` assembles the full ASGI application
- RPC routes at `/rpc/{pipeline_name}`
- Management routes: `GET /v1/health`, `GET /v1/models`
- Takes `pipelines: dict[str, Any]` (executors) and `model_info: list[dict]` (for models endpoint)

**Step 1: Write the failing tests**

```python
# tests/test_app.py
"""Tests for nerva.server.app — ASGI app assembly + management routes."""

from unittest.mock import AsyncMock

from starlette.testclient import TestClient

from nerva.server.app import build_app


def _make_mock_executor(result: dict | None = None) -> AsyncMock:
    mock = AsyncMock()
    mock.execute.return_value = result or {"out": 1}
    return mock


class TestHealthRoute:
    def test_health_ok(self) -> None:
        app = build_app(pipelines={}, model_info=[])
        client = TestClient(app)
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestModelsRoute:
    def test_models_empty(self) -> None:
        app = build_app(pipelines={}, model_info=[])
        client = TestClient(app)
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        assert resp.json() == {"models": []}

    def test_models_lists_registered(self) -> None:
        info = [
            {"name": "sentiment", "backend": "pytorch", "device": "cpu"},
            {"name": "encoder", "backend": "pytorch", "device": "cuda:0"},
        ]
        app = build_app(pipelines={}, model_info=info)
        client = TestClient(app)
        resp = client.get("/v1/models")
        data = resp.json()
        assert len(data["models"]) == 2
        names = {m["name"] for m in data["models"]}
        assert names == {"sentiment", "encoder"}


class TestRpcRouteIntegration:
    def test_rpc_route_mounted(self) -> None:
        """Verify RPC route is accessible through the full app."""
        import time
        import msgpack
        from nerva.server.protocol import Frame, FrameType, encode_frame

        executor = _make_mock_executor({"result": "ok"})
        app = build_app(pipelines={"test": executor}, model_info=[])
        client = TestClient(app)

        open_frame = encode_frame(
            Frame(FrameType.OPEN, 1, 0, msgpack.packb({"pipeline": "test"}))
        )
        data_frame = encode_frame(
            Frame(FrameType.DATA, 1, 0, msgpack.packb({"x": 1}))
        )
        end_frame = encode_frame(Frame(FrameType.END, 1, 0, b""))

        deadline = int(time.time() * 1000) + 30000
        resp = client.post(
            "/rpc/test",
            content=open_frame + data_frame + end_frame,
            headers={
                "content-type": "application/x-nerva-rpc",
                "x-nerva-deadline-ms": str(deadline),
                "x-nerva-stream": "0",
            },
        )
        assert resp.status_code == 200
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_app.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

```python
# src/nerva/server/app.py
"""ASGI application assembly for Nerva inference server."""

from __future__ import annotations

from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from nerva.server.rpc import RpcHandler


def build_app(
    pipelines: dict[str, Any],
    model_info: list[dict[str, Any]],
) -> Starlette:
    """Assemble the full Nerva ASGI application.

    Args:
        pipelines: Mapping from pipeline name to executor (with execute() method).
        model_info: List of model info dicts for the /v1/models endpoint.

    Returns:
        A Starlette ASGI application.
    """
    rpc_handler = RpcHandler(pipelines)

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def models(request: Request) -> JSONResponse:
        return JSONResponse({"models": model_info})

    return Starlette(
        routes=[
            Route("/rpc/{pipeline_name}", rpc_handler.handle, methods=["POST"]),
            Route("/v1/health", health, methods=["GET"]),
            Route("/v1/models", models, methods=["GET"]),
        ],
    )
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_app.py -v`
Expected: ALL PASS

**Step 5: Lint + type check**

Run: `uv run ruff check src/nerva/server/app.py tests/test_app.py && uv run mypy src/nerva/server/app.py`

**Step 6: Commit**

```bash
git add src/nerva/server/app.py tests/test_app.py
git commit -m "feat(app): add ASGI app assembly with management routes"
```

---

### Task 5: serve() top-level API

**Files:**
- Create: `src/nerva/server/serve.py`
- Test: `tests/test_serve.py`

**Context:**
- `serve()` is the user-facing API: `serve({"classify": graph}, host="0.0.0.0", port=8080)`
- Internal flow:
  1. Collect all unique model names from all Graphs
  2. Look up ModelHandles from registry (`get_model_handle`)
  3. Spawn workers via WorkerManager
  4. Construct Executor for each pipeline (graph + proxies dict)
  5. Build ASGI app via `build_app()`
  6. Run `uvicorn.run(app, host, port)`
  7. On shutdown, cleanup workers
- Key dependency: `InferContext.deadline_ms` is relative TTL — the RPC handler does the conversion

**Step 1: Write the failing tests**

```python
# tests/test_serve.py
"""Tests for nerva.server.serve — serve() internals."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nerva.core.graph import Graph, Node
from nerva.core.model import Model, _model_registry, model
from nerva.server.serve import _collect_model_names, _build_pipelines


class DummyModel(Model):
    def load(self) -> None:
        pass

    async def infer(self, inputs: dict) -> dict:
        return {"ok": True}


class TestCollectModelNames:
    def test_single_graph(self) -> None:
        g = Graph()
        g.add_node(Node(id="a_1", model_name="a"))
        g.add_node(Node(id="b_1", model_name="b"))
        names = _collect_model_names({"pipe": g})
        assert names == {"a", "b"}

    def test_multiple_graphs_dedup(self) -> None:
        g1 = Graph()
        g1.add_node(Node(id="a_1", model_name="shared"))
        g2 = Graph()
        g2.add_node(Node(id="a_2", model_name="shared"))
        g2.add_node(Node(id="b_1", model_name="unique"))
        names = _collect_model_names({"p1": g1, "p2": g2})
        assert names == {"shared", "unique"}

    def test_skips_control_flow_nodes(self) -> None:
        g = Graph()
        g.add_node(Node(id="a_1", model_name="a"))
        g.add_node(Node(id="cond_1", model_name="cond", node_type="cond"))
        g.add_node(Node(id="par_1", model_name="parallel", node_type="parallel"))
        names = _collect_model_names({"pipe": g})
        assert names == {"a"}


class TestBuildPipelines:
    def setup_method(self) -> None:
        _model_registry.clear()

    def teardown_method(self) -> None:
        _model_registry.clear()

    async def test_build_pipelines(self) -> None:
        handle = model("echo", DummyModel)
        g = Graph()
        g.add_node(Node(id="echo_1", model_name="echo"))

        mock_manager = AsyncMock()
        mock_proxy = AsyncMock()
        mock_manager.start_worker.return_value = mock_proxy

        executors, model_info = await _build_pipelines(
            {"classify": g}, mock_manager
        )
        assert "classify" in executors
        assert len(model_info) == 1
        assert model_info[0]["name"] == "echo"
        mock_manager.start_worker.assert_called_once_with(handle)

    async def test_missing_handle_raises(self) -> None:
        g = Graph()
        g.add_node(Node(id="x_1", model_name="nonexistent"))
        mock_manager = AsyncMock()

        with pytest.raises(KeyError, match="nonexistent"):
            await _build_pipelines({"pipe": g}, mock_manager)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_serve.py -v`
Expected: FAIL (ImportError)

**Step 3: Write implementation**

```python
# src/nerva/server/serve.py
"""serve() — top-level API for starting Nerva inference server."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import uvicorn

from nerva.backends.base import InferContext
from nerva.core.model import get_model_handle
from nerva.engine.executor import Executor
from nerva.server.app import build_app
from nerva.worker.manager import WorkerManager

if TYPE_CHECKING:
    from nerva.core.graph import Graph
    from nerva.engine.executor import InferableProxy

logger = logging.getLogger(__name__)


def _collect_model_names(pipelines: dict[str, Graph]) -> set[str]:
    """Collect all unique model names from pipeline graphs.

    Skips control-flow nodes (cond, parallel) which are not real models.
    """
    names: set[str] = set()
    for graph in pipelines.values():
        for node in graph.nodes:
            if node.node_type == "call":
                names.add(node.model_name)
    return names


async def _build_pipelines(
    pipelines: dict[str, Graph],
    manager: WorkerManager,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Spawn workers and build executors for each pipeline.

    Returns:
        (executors, model_info) tuple.
    """
    model_names = _collect_model_names(pipelines)

    # Spawn one worker per model.
    proxies: dict[str, InferableProxy] = {}
    model_info: list[dict[str, Any]] = []
    for name in sorted(model_names):
        handle = get_model_handle(name)
        proxy = await manager.start_worker(handle)
        proxies[name] = proxy
        model_info.append({
            "name": handle.name,
            "backend": handle.backend,
            "device": handle.device,
        })

    # Build one executor per pipeline.
    # Note: Executor requires an InferContext at construction time.
    # The RPC handler creates a per-request InferContext and passes it
    # via a wrapper that delegates to executor.execute() with the real context.
    executors: dict[str, _PipelineExecutor] = {}
    for pipe_name, graph in pipelines.items():
        executors[pipe_name] = _PipelineExecutor(graph, proxies)

    return executors, model_info


class _PipelineExecutor:
    """Wraps Graph + proxies to provide execute(inputs) for the RPC handler.

    Creates a fresh Executor with a per-request InferContext on each call.
    """

    def __init__(self, graph: Graph, proxies: dict[str, InferableProxy]) -> None:
        self._graph = graph
        self._proxies = proxies

    async def execute(self, inputs: Any, *, deadline_ms: int = 30000, request_id: str = "") -> Any:
        """Execute the pipeline with a fresh InferContext."""
        if not request_id:
            import uuid
            request_id = str(uuid.uuid4())
        ctx = InferContext(request_id=request_id, deadline_ms=deadline_ms)
        executor = Executor(self._graph, self._proxies, ctx)
        return await executor.execute(inputs)


def serve(
    pipelines: dict[str, Graph],
    *,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
    """Start the Nerva inference server (blocking).

    Scans all Graphs for model declarations, auto-spawns worker processes,
    builds the ASGI application, and starts uvicorn.

    Args:
        pipelines: Mapping from pipeline name to traced Graph.
        host: Bind address.
        port: Bind port.
    """

    async def _run() -> None:
        manager = WorkerManager()
        try:
            executors, model_info = await _build_pipelines(pipelines, manager)
            app = build_app(pipelines=executors, model_info=model_info)
            config = uvicorn.Config(app, host=host, port=port, log_level="info")
            server = uvicorn.Server(config)
            await server.serve()
        finally:
            await manager.shutdown_all()

    asyncio.run(_run())
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_serve.py -v`
Expected: ALL PASS

**Step 5: Lint + type check**

Run: `uv run ruff check src/nerva/server/serve.py tests/test_serve.py && uv run mypy src/nerva/server/serve.py`

**Step 6: Commit**

```bash
git add src/nerva/server/serve.py tests/test_serve.py
git commit -m "feat(serve): add serve() top-level API with auto worker management"
```

---

### Task 6: Wire RPC handler to use _PipelineExecutor properly

**Files:**
- Modify: `src/nerva/server/rpc.py`
- Modify: `tests/test_rpc.py`

**Context:**
- The RPC handler needs to pass `deadline_ms` and `request_id` to `_PipelineExecutor.execute()`
- Currently `RpcHandler.handle()` calls `executor.execute(inputs)` but doesn't pass deadline/request_id
- Update the handler to pass these as keyword arguments

**Step 1: Update RPC handler**

In `src/nerva/server/rpc.py`, update the execute call in `RpcHandler.handle()`:

```python
        # Replace:
        result = await executor.execute(inputs)
        # With:
        result = await executor.execute(
            inputs,
            deadline_ms=deadline_ms,
            request_id=str(request_id),
        )
```

**Step 2: Add test for deadline passthrough**

Add to `tests/test_rpc.py`:

```python
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
```

**Step 3: Run tests**

Run: `uv run pytest tests/test_rpc.py -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add src/nerva/server/rpc.py tests/test_rpc.py
git commit -m "fix(rpc): pass deadline_ms and request_id to pipeline executor"
```

---

### Task 7: Public API export + __init__.py updates

**Files:**
- Modify: `src/nerva/__init__.py`
- Modify: `src/nerva/server/__init__.py`

**Context:**
- Export `serve` from `nerva` package
- Export `get_model_handle`, `list_model_handles` from `nerva` package

**Step 1: Update `src/nerva/server/__init__.py`**

```python
"""Nerva server package."""
```

**Step 2: Update `src/nerva/__init__.py`**

Add imports:
```python
from nerva.core.model import get_model_handle, list_model_handles
from nerva.server.serve import serve
```

Add to `__all__`:
```python
"get_model_handle",
"list_model_handles",
"serve",
```

**Step 3: Run full test suite**

Run: `uv run pytest tests/ -q --tb=short`
Expected: ALL PASS (159 + new tests)

**Step 4: Lint + type check**

Run: `uv run ruff check src/ tests/ && uv run mypy`

**Step 5: Commit**

```bash
git add src/nerva/__init__.py src/nerva/server/__init__.py
git commit -m "feat(api): export serve, get_model_handle, list_model_handles"
```

---

### Task 8: End-to-end integration tests

**Files:**
- Create: `tests/test_phase4_e2e.py`

**Context:**
- Uses real Worker processes + httpx/TestClient
- Tests: unary happy path, unknown pipeline, expired deadline, health + models endpoints
- Reuses `EchoModel` from `tests/helpers.py`
- Uses ASGI transport (no real port needed)

**Step 1: Write integration tests**

```python
# tests/test_phase4_e2e.py
"""Phase 4 end-to-end tests: Binary RPC with real Workers."""

from __future__ import annotations

import time
from typing import Any

import msgpack
import pytest

from nerva import model, trace
from nerva.backends.base import InferContext
from nerva.core.model import _model_registry
from nerva.server.app import build_app
from nerva.server.protocol import Frame, FrameType, decode_frame, encode_frame
from nerva.server.serve import _build_pipelines
from nerva.worker.manager import WorkerManager
from starlette.testclient import TestClient
from tests.helpers import EchoModel


def _make_frames(pipeline: str, inputs: dict[str, Any], request_id: int = 1) -> bytes:
    frames = b""
    frames += encode_frame(
        Frame(FrameType.OPEN, request_id, 0, msgpack.packb({"pipeline": pipeline}))
    )
    frames += encode_frame(
        Frame(FrameType.DATA, request_id, 0, msgpack.packb(inputs))
    )
    frames += encode_frame(Frame(FrameType.END, request_id, 0, b""))
    return frames


def _decode_frames(body: bytes) -> list[Frame]:
    frames = []
    offset = 0
    while offset < len(body):
        frame, consumed = decode_frame(body[offset:])
        frames.append(frame)
        offset += consumed
    return frames


@pytest.fixture
async def e2e_client() -> Any:
    """Build full app with real Worker, yield TestClient, cleanup."""
    _model_registry.clear()
    handle = model("echo", EchoModel, backend="pytorch", device="cpu")
    graph = trace(lambda inp: handle(inp))

    manager = WorkerManager()
    executors, model_info = await _build_pipelines({"echo": graph}, manager)
    app = build_app(pipelines=executors, model_info=model_info)
    client = TestClient(app)
    yield client
    await manager.shutdown_all()
    _model_registry.clear()


def _rpc_headers(deadline_offset_ms: int = 30000) -> dict[str, str]:
    deadline = int(time.time() * 1000) + deadline_offset_ms
    return {
        "content-type": "application/x-nerva-rpc",
        "x-nerva-deadline-ms": str(deadline),
        "x-nerva-stream": "0",
    }


async def test_e2e_unary_happy_path(e2e_client: TestClient) -> None:
    body = _make_frames("echo", {"value": "hello"})
    resp = e2e_client.post("/rpc/echo", content=body, headers=_rpc_headers())
    assert resp.status_code == 200
    frames = _decode_frames(resp.content)
    assert len(frames) == 2
    assert frames[0].frame_type == FrameType.DATA
    result = msgpack.unpackb(frames[0].payload, raw=False)
    assert result == {"echo": "hello"}
    assert frames[1].frame_type == FrameType.END


async def test_e2e_unknown_pipeline(e2e_client: TestClient) -> None:
    body = _make_frames("nonexistent", {"x": 1})
    resp = e2e_client.post("/rpc/nonexistent", content=body, headers=_rpc_headers())
    frames = _decode_frames(resp.content)
    assert frames[0].frame_type == FrameType.ERROR
    error = msgpack.unpackb(frames[0].payload, raw=False)
    assert error["code"] == 3  # INVALID_ARGUMENT


async def test_e2e_expired_deadline(e2e_client: TestClient) -> None:
    body = _make_frames("echo", {"value": "x"})
    resp = e2e_client.post(
        "/rpc/echo", content=body, headers=_rpc_headers(deadline_offset_ms=-1000)
    )
    frames = _decode_frames(resp.content)
    assert frames[0].frame_type == FrameType.ERROR
    error = msgpack.unpackb(frames[0].payload, raw=False)
    assert error["code"] == 4  # DEADLINE_EXCEEDED


async def test_e2e_health(e2e_client: TestClient) -> None:
    resp = e2e_client.get("/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_e2e_models(e2e_client: TestClient) -> None:
    resp = e2e_client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["models"]) == 1
    assert data["models"][0]["name"] == "echo"
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_phase4_e2e.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add tests/test_phase4_e2e.py
git commit -m "test(phase4): add e2e integration tests for binary RPC with real Workers"
```

---

### Task 9: Static checks + full regression + roadmap update

**Files:**
- Modify: `docs/plans/2026-02-25-mvp-roadmap.md`

**Step 1: Run full suite**

```bash
uv run pytest tests/ -v
uv run ruff check src/ tests/
uv run mypy
```

Expected: ALL PASS, 0 errors, 0 issues.

**Step 2: Update roadmap**

In `docs/plans/2026-02-25-mvp-roadmap.md`, update Phase 4 section:
- Change status from `⬜ 待设计` to `✅ 已完成 (2026-02-28)`
- Add key design decisions (Binary RPC + msgpack, serve() blocking API, auto worker management)
- Add verification results (test count, ruff, mypy)
- Add output files list
- Update timeline chart

**Step 3: Commit**

```bash
git add docs/plans/2026-02-25-mvp-roadmap.md
git commit -m "docs: mark Phase 4 complete in roadmap"
```
