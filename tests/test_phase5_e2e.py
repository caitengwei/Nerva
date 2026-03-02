"""Phase 5 end-to-end tests — observability + VLLMBackend integration.

All tests run without real GPU/vllm.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import msgpack
import pytest
from prometheus_client import CollectorRegistry

from nerva.observability.logging import configure_logging
from nerva.observability.metrics import NervaMetrics


def _make_vllm_mock() -> Any:
    async def _generate(prompt: str, params: Any, req_id: str):  # type: ignore[no-untyped-def]
        out = MagicMock()
        out.outputs = [MagicMock()]
        out.outputs[0].text = f"response to: {prompt}"
        out.finished = True
        yield out

    engine = MagicMock()
    engine.generate = _generate
    vllm_mod = MagicMock()
    vllm_mod.AsyncLLMEngine.from_engine_args = MagicMock(return_value=engine)
    vllm_mod.AsyncEngineArgs = MagicMock(return_value=MagicMock())
    vllm_mod.SamplingParams = MagicMock(return_value=MagicMock())
    return vllm_mod


class TestObservabilityPublicAPI:
    def test_configure_logging_exported(self) -> None:
        import nerva
        assert hasattr(nerva, "configure_logging")

    def test_get_metrics_exported(self) -> None:
        import nerva
        assert hasattr(nerva, "get_metrics")

    def test_nerva_metrics_exported(self) -> None:
        import nerva
        assert hasattr(nerva, "NervaMetrics")

    def test_configure_logging_no_crash(self) -> None:
        configure_logging(dev=True)

    def test_two_metrics_with_separate_registries(self) -> None:
        r1 = CollectorRegistry()
        r2 = CollectorRegistry()
        m1 = NervaMetrics(registry=r1)
        m2 = NervaMetrics(registry=r2)
        m1.request_total.labels(pipeline="p1", status="ok").inc()
        val = m2.request_total.labels(pipeline="p1", status="ok")._value.get()
        assert val == 0.0


class TestVLLMBackendE2E:
    async def test_full_infer_flow(self) -> None:
        from nerva.backends.base import InferContext, ModelConfig
        from nerva.backends.vllm import VLLMBackend

        with patch.dict("sys.modules", {"vllm": _make_vllm_mock()}):
            backend = VLLMBackend()
            await backend.load_model(ModelConfig(
                model_name="e2e-llm", model_class=object,
                device="cuda:0", backend_options={"model_path": "/models/test"},
            ))
            ctx = InferContext(request_id="e2e-001", deadline_ms=5000)
            result = await backend.infer({"prompt": "What is Nerva?"}, ctx)
            assert "text" in result
            assert len(result["text"]) > 0

    async def test_infer_stream_yields_final_chunk(self) -> None:
        from nerva.backends.base import InferContext, ModelConfig
        from nerva.backends.vllm import VLLMBackend

        with patch.dict("sys.modules", {"vllm": _make_vllm_mock()}):
            backend = VLLMBackend()
            await backend.load_model(ModelConfig(
                model_name="e2e-llm", model_class=object,
                device="cuda:0", backend_options={"model_path": "/models/test"},
            ))
            ctx = InferContext(request_id="e2e-stream-001", deadline_ms=5000)
            chunks = [c async for c in backend.infer_stream({"prompt": "Hello"}, ctx)]
            assert len(chunks) >= 1
            assert chunks[-1]["finished"] is True


class TestMetricsEndpointE2E:
    def test_metrics_endpoint_returns_200(self) -> None:
        from starlette.testclient import TestClient

        from nerva.server.app import build_app
        app = build_app(pipelines={}, model_info=[])
        resp = TestClient(app).get("/metrics")
        assert resp.status_code == 200

    def test_rpc_handler_metrics_recorded(self) -> None:
        from starlette.applications import Starlette
        from starlette.routing import Route
        from starlette.testclient import TestClient

        from nerva.server.protocol import Frame, FrameType, encode_frame
        from nerva.server.rpc import RpcHandler

        reg = CollectorRegistry()
        m = NervaMetrics(registry=reg)
        executor = AsyncMock()
        executor.execute.return_value = {"out": "ok"}
        handler = RpcHandler({"pipe": executor}, metrics=m)
        app = Starlette(routes=[Route("/rpc/{pipeline_name}", handler.handle, methods=["POST"])])

        open_body = msgpack.packb({"pipeline": "pipe"})
        data_body = msgpack.packb({"x": 1})
        body = (
            encode_frame(Frame(FrameType.OPEN, 1, 0, open_body))
            + encode_frame(Frame(FrameType.DATA, 1, 0, data_body))
            + encode_frame(Frame(FrameType.END, 1, 0, b""))
        )
        deadline = int(time.time() * 1000) + 30000
        TestClient(app).post(
            "/rpc/pipe", content=body,
            headers={
                "content-type": "application/x-nerva-rpc",
                "x-nerva-deadline-ms": str(deadline),
                "x-nerva-stream": "0",
            },
        )
        assert m.request_total.labels(pipeline="pipe", status="ok")._value.get() == 1.0
        assert m.request_in_flight.labels(pipeline="pipe")._value.get() == 0.0


@pytest.mark.gpu
class TestVLLMGPUIntegration:
    """Requires real GPU + vllm. Run with: pytest -m gpu"""
    async def test_placeholder(self) -> None:
        pytest.skip("GPU tests require real hardware.")
