# tests/test_phase4_e2e.py
"""Phase 4 end-to-end tests: Binary RPC with real Workers."""

from __future__ import annotations

import time
from typing import Any

import httpx
import msgpack
import pytest

from nerva import model, trace
from nerva.server.app import build_app
from nerva.server.protocol import Frame, FrameType, decode_frame, encode_frame
from nerva.server.serve import _build_pipelines
from nerva.worker.manager import WorkerManager
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
    """Build full app with real Worker, yield async httpx client, cleanup."""
    handle = model("echo", EchoModel, backend="pytorch", device="cpu")
    graph = trace(lambda inp: handle(inp))

    manager = WorkerManager()
    executors, model_info = await _build_pipelines({"echo": graph}, manager)
    app = build_app(pipelines=executors, model_info=model_info)

    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        try:
            yield client
        finally:
            await manager.shutdown_all()


def _rpc_headers(deadline_offset_ms: int = 30000) -> dict[str, str]:
    deadline = int(time.time() * 1000) + deadline_offset_ms
    return {
        "content-type": "application/x-nerva-rpc",
        "x-nerva-deadline-ms": str(deadline),
        "x-nerva-stream": "0",
    }


async def test_e2e_unary_happy_path(e2e_client: httpx.AsyncClient) -> None:
    body = _make_frames("echo", {"value": "hello"})
    resp = await e2e_client.post("/rpc/echo", content=body, headers=_rpc_headers())
    assert resp.status_code == 200
    frames = _decode_frames(resp.content)
    assert len(frames) == 2
    assert frames[0].frame_type == FrameType.DATA
    result = msgpack.unpackb(frames[0].payload, raw=False)
    assert result == {"echo": "hello"}
    assert frames[1].frame_type == FrameType.END


async def test_e2e_unknown_pipeline(e2e_client: httpx.AsyncClient) -> None:
    body = _make_frames("nonexistent", {"x": 1})
    resp = await e2e_client.post("/rpc/nonexistent", content=body, headers=_rpc_headers())
    frames = _decode_frames(resp.content)
    assert frames[0].frame_type == FrameType.ERROR
    error = msgpack.unpackb(frames[0].payload, raw=False)
    assert error["code"] == 3  # INVALID_ARGUMENT


async def test_e2e_expired_deadline(e2e_client: httpx.AsyncClient) -> None:
    body = _make_frames("echo", {"value": "x"})
    resp = await e2e_client.post(
        "/rpc/echo", content=body, headers=_rpc_headers(deadline_offset_ms=-1000)
    )
    frames = _decode_frames(resp.content)
    assert frames[0].frame_type == FrameType.ERROR
    error = msgpack.unpackb(frames[0].payload, raw=False)
    assert error["code"] == 4  # DEADLINE_EXCEEDED


async def test_e2e_health(e2e_client: httpx.AsyncClient) -> None:
    resp = await e2e_client.get("/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_e2e_models(e2e_client: httpx.AsyncClient) -> None:
    resp = await e2e_client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["models"]) == 1
    assert data["models"][0]["name"] == "echo"
