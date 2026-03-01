# tests/test_app.py
"""Tests for nerva.server.app — ASGI app assembly + management routes."""

import time
from typing import Any
from unittest.mock import AsyncMock

import msgpack
from starlette.testclient import TestClient

from nerva.server.app import build_app
from nerva.server.protocol import Frame, FrameType, encode_frame


def _make_mock_executor(result: dict[str, Any] | None = None) -> AsyncMock:
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
        info: list[dict[str, Any]] = [
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
