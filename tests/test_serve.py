# tests/test_serve.py
"""Tests for nerva.server.serve — serve() internals."""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from nerva.core.graph import Graph, Node
from nerva.core.model import Model, model
from nerva.core.proxy import trace
from nerva.server.serve import _build_pipelines, _collect_model_names, _NervaASGIApp


class DummyModel(Model):
    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
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

    def test_recurses_into_cond_branches(self) -> None:
        true_branch = Graph()
        true_branch.add_node(Node(id="b_1", model_name="b"))
        false_branch = Graph()
        false_branch.add_node(Node(id="c_1", model_name="c"))

        g = Graph()
        g.add_node(Node(id="a_1", model_name="a"))
        g.add_node(
            Node(
                id="cond_1",
                model_name="cond",
                node_type="cond",
                true_branch=true_branch,
                false_branch=false_branch,
            )
        )
        names = _collect_model_names({"pipe": g})
        assert names == {"a", "b", "c"}

    def test_recurses_into_parallel_branches(self) -> None:
        branch1 = Graph()
        branch1.add_node(Node(id="b_1", model_name="b"))
        branch2 = Graph()
        branch2.add_node(Node(id="c_1", model_name="c"))

        g = Graph()
        g.add_node(Node(id="a_1", model_name="a"))
        g.add_node(
            Node(
                id="par_1",
                model_name="parallel",
                node_type="parallel",
                branches=[branch1, branch2],
            )
        )
        names = _collect_model_names({"pipe": g})
        assert names == {"a", "b", "c"}

    def test_deduplicates_across_branches(self) -> None:
        """Same model used in both branches is only spawned once."""
        true_branch = Graph()
        true_branch.add_node(Node(id="shared_1", model_name="shared"))
        false_branch = Graph()
        false_branch.add_node(Node(id="shared_2", model_name="shared"))

        g = Graph()
        g.add_node(
            Node(
                id="cond_1",
                model_name="cond",
                node_type="cond",
                true_branch=true_branch,
                false_branch=false_branch,
            )
        )
        names = _collect_model_names({"pipe": g})
        assert names == {"shared"}


class TestBuildPipelines:
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


class TestNervaASGIAppLifecycle:
    async def test_manual_shutdown_after_http_request(self) -> None:
        """无 lifespan 场景下可通过 app.shutdown() 回收资源。"""
        import httpx
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        async def _ok(_request: Any) -> JSONResponse:
            return JSONResponse({"ok": True})

        on_startup = AsyncMock()
        on_shutdown = AsyncMock()
        app = _NervaASGIApp(
            starlette_app=Starlette(routes=[Route("/ok", _ok, methods=["GET"])]),
            on_startup=on_startup,
            on_shutdown=on_shutdown,
        )

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/ok")

        assert resp.status_code == 200
        on_startup.assert_awaited_once()
        on_shutdown.assert_not_awaited()

        await app.shutdown()
        await app.shutdown()
        on_shutdown.assert_awaited_once()


class TestBuildNervaApp:
    async def test_health_endpoint(self) -> None:
        """build_nerva_app() 启动后 /v1/health 返回 ok。"""
        import httpx

        from nerva.server.serve import build_nerva_app
        from tests.helpers import EchoModel

        handle = model("echo_app_health", EchoModel, backend="pytorch", device="cpu")
        graph = trace(lambda inp: handle(inp))
        app = build_nerva_app({"echo": graph})

        transport = httpx.ASGITransport(app=app)
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/v1/health")
        finally:
            await app.shutdown()
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    async def test_rpc_echo_end_to_end(self) -> None:
        """build_nerva_app() 能跑通真实 Worker 进程的推理请求。"""
        import time

        import httpx
        import msgpack

        from nerva import trace
        from nerva.server.protocol import Frame, FrameType, decode_frame, encode_frame
        from nerva.server.serve import build_nerva_app
        from tests.helpers import EchoModel

        handle = model("echo_app_rpc", EchoModel, backend="pytorch", device="cpu")
        graph = trace(lambda inp: handle(inp))
        app = build_nerva_app({"echo": graph})

        def _make_body() -> bytes:
            return (
                encode_frame(Frame(FrameType.OPEN, 1, 0, msgpack.packb({"pipeline": "echo"})))
                + encode_frame(Frame(FrameType.DATA, 1, 0, msgpack.packb({"value": "world"})))
                + encode_frame(Frame(FrameType.END, 1, 0, b""))
            )

        deadline = int(time.time() * 1000) + 30000
        headers = {
            "content-type": "application/x-nerva-rpc",
            "x-nerva-deadline-ms": str(deadline),
            "x-nerva-stream": "0",
        }

        transport = httpx.ASGITransport(app=app)
        try:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/rpc/echo", content=_make_body(), headers=headers)
        finally:
            await app.shutdown()

        assert resp.status_code == 200
        frames = []
        offset = 0
        while offset < len(resp.content):
            frame, consumed = decode_frame(resp.content[offset:])
            frames.append(frame)
            offset += consumed
        data_frame = next(f for f in frames if f.frame_type == FrameType.DATA)
        result = msgpack.unpackb(data_frame.payload, raw=False)
        assert result == {"echo": "world"}
