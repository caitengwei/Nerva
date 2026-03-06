from __future__ import annotations

from typing import Any

import msgpack
import scripts.bench.targets.nerva_binary_rpc as nerva_binary_rpc
import scripts.bench.targets.triton_infer as triton_infer
import scripts.bench.targets.vllm_openai_api as vllm_openai_api
from scripts.bench.targets.base import TargetResponse
from scripts.bench.targets.nerva_binary_rpc import NervaBinaryRPCTarget
from scripts.bench.targets.triton_infer import TritonInferTarget
from scripts.bench.targets.vllm_openai_api import VLLMOpenAIAPITarget

from nerva.server.protocol import Frame, FrameType, encode_frame


def test_target_response_schema() -> None:
    r = TargetResponse(ok=True, latency_ms=12.3, ttft_ms=None, error="")
    assert r.ok is True
    assert r.latency_ms > 0


async def test_nerva_binary_rpc_parses_data_and_error_frames() -> None:
    data_bytes = (
        encode_frame(
            Frame(
                frame_type=FrameType.DATA,
                request_id=1,
                flags=0,
                payload=msgpack.packb({"text": "hello"}, use_bin_type=True),
            )
        )
        + encode_frame(Frame(frame_type=FrameType.END, request_id=1, flags=0, payload=b""))
    )

    async def sender_data(
        url: str,
        body: bytes,
        headers: dict[str, str],
        timeout_ms: int,
    ) -> bytes:
        del url, body, headers, timeout_ms
        return data_bytes

    ok_target = NervaBinaryRPCTarget(
        base_url="http://localhost:8080",
        pipeline_name="phase7_mm_vllm",
        sender=sender_data,
    )
    ok_resp = await ok_target.infer({"prompt": "hi"}, deadline_ms=1000)
    assert ok_resp.ok is True
    assert ok_resp.output_text == "hello"

    error_bytes = encode_frame(
        Frame(
            frame_type=FrameType.ERROR,
            request_id=1,
            flags=0,
            payload=msgpack.packb({"code": "INTERNAL", "message": "boom"}, use_bin_type=True),
        )
    )

    async def sender_error(
        url: str,
        body: bytes,
        headers: dict[str, str],
        timeout_ms: int,
    ) -> bytes:
        del url, body, headers, timeout_ms
        return error_bytes

    err_target = NervaBinaryRPCTarget(
        base_url="http://localhost:8080",
        pipeline_name="phase7_mm_vllm",
        sender=sender_error,
    )
    err_resp = await err_target.infer({"prompt": "hi"}, deadline_ms=1000)
    assert err_resp.ok is False
    assert "boom" in err_resp.error


async def test_nerva_binary_rpc_rejects_data_without_text_field() -> None:
    data_bytes = (
        encode_frame(
            Frame(
                frame_type=FrameType.DATA,
                request_id=1,
                flags=0,
                payload=msgpack.packb({"foo": "bar"}, use_bin_type=True),
            )
        )
        + encode_frame(Frame(frame_type=FrameType.END, request_id=1, flags=0, payload=b""))
    )

    async def sender(
        url: str,
        body: bytes,
        headers: dict[str, str],
        timeout_ms: int,
    ) -> bytes:
        del url, body, headers, timeout_ms
        return data_bytes

    target = NervaBinaryRPCTarget(
        base_url="http://localhost:8080",
        pipeline_name="phase7_mm_vllm",
        sender=sender,
    )
    resp = await target.infer({"prompt": "hi"}, deadline_ms=1000)
    assert resp.ok is False
    assert "no text field found" in resp.error


async def test_vllm_openai_api_parses_text_field() -> None:
    captured: dict[str, Any] = {}

    async def sender(
        url: str,
        payload: dict[str, Any],
        timeout_ms: int,
    ) -> dict[str, Any]:
        del url, timeout_ms
        captured.update(payload)
        return {"choices": [{"text": "hello from vllm"}]}

    target = VLLMOpenAIAPITarget(
        base_url="http://127.0.0.1:8001",
        model_name="phase7",
        sender=sender,
    )
    resp = await target.infer({"text": "hi", "image_bytes": b"\x00" * 16}, deadline_ms=1000)
    assert resp.ok is True
    assert resp.output_text == "hello from vllm"
    assert captured["prompt"] == "[image_bytes=16]\nhi"
    assert captured["max_tokens"] == 256
    assert captured["temperature"] == 1.0


async def test_vllm_openai_api_postprocess_normalizes_output_text() -> None:
    async def sender(
        url: str,
        payload: dict[str, Any],
        timeout_ms: int,
    ) -> dict[str, Any]:
        del url, payload, timeout_ms
        return {"choices": [{"text": "  hello from vllm  "}]}

    target = VLLMOpenAIAPITarget(
        base_url="http://127.0.0.1:8001",
        model_name="phase7",
        sender=sender,
    )
    resp = await target.infer({"text": "hi", "image_bytes": b"\x00" * 16}, deadline_ms=1000)
    assert resp.ok is True
    assert resp.output_text == "hello from vllm"


async def test_vllm_openai_api_accepts_custom_sampling_params() -> None:
    captured: dict[str, Any] = {}

    async def sender(
        url: str,
        payload: dict[str, Any],
        timeout_ms: int,
    ) -> dict[str, Any]:
        del url, timeout_ms
        captured.update(payload)
        return {"choices": [{"text": "ok"}]}

    target = VLLMOpenAIAPITarget(
        base_url="http://127.0.0.1:8001",
        model_name="phase7",
        sender=sender,
    )
    resp = await target.infer(
        {
            "text": "hi",
            "image_bytes": b"\x00" * 16,
            "max_tokens": 128,
            "temperature": 0.2,
            "top_p": 0.9,
        },
        deadline_ms=1000,
    )
    assert resp.ok is True
    assert captured["max_tokens"] == 128
    assert captured["temperature"] == 0.2
    assert captured["top_p"] == 0.9


async def test_vllm_openai_api_default_sender_reuses_async_client(monkeypatch: Any) -> None:
    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": [{"text": "ok"}]}

    class _FakeClient:
        def __init__(self) -> None:
            self.post_calls = 0
            self.closed = False

        async def post(
            self,
            url: str,
            *,
            json: dict[str, Any],
            headers: dict[str, str],
            timeout: float,
        ) -> _FakeResponse:
            del url, json, headers, timeout
            self.post_calls += 1
            return _FakeResponse()

        async def aclose(self) -> None:
            self.closed = True

    created_clients: list[_FakeClient] = []

    def _make_client(*args: Any, **kwargs: Any) -> _FakeClient:
        del args, kwargs
        client = _FakeClient()
        created_clients.append(client)
        return client

    monkeypatch.setattr(vllm_openai_api.httpx, "AsyncClient", _make_client)

    target = VLLMOpenAIAPITarget(base_url="http://127.0.0.1:8001", model_name="phase7")
    first = await target.infer({"prompt": "hi-1"}, deadline_ms=1000)
    second = await target.infer({"prompt": "hi-2"}, deadline_ms=1000)
    await target.aclose()

    assert first.ok is True
    assert second.ok is True
    assert len(created_clients) == 1
    assert created_clients[0].post_calls == 2
    assert created_clients[0].closed is True


async def test_triton_infer_default_sender_reuses_async_client(monkeypatch: Any) -> None:
    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"outputs": [{"name": "text", "data": ["ok"]}]}

    class _FakeClient:
        def __init__(self) -> None:
            self.post_calls = 0
            self.closed = False

        async def post(
            self,
            url: str,
            *,
            json: dict[str, Any],
            headers: dict[str, str],
            timeout: float,
        ) -> _FakeResponse:
            del url, json, headers, timeout
            self.post_calls += 1
            return _FakeResponse()

        async def aclose(self) -> None:
            self.closed = True

    created_clients: list[_FakeClient] = []

    def _make_client(*args: Any, **kwargs: Any) -> _FakeClient:
        del args, kwargs
        client = _FakeClient()
        created_clients.append(client)
        return client

    monkeypatch.setattr(triton_infer.httpx, "AsyncClient", _make_client)

    target = TritonInferTarget(base_url="http://127.0.0.1:8002", model_name="phase7_mm_vllm")
    first = await target.infer({"prompt": "hi-1"}, deadline_ms=1000)
    second = await target.infer({"prompt": "hi-2"}, deadline_ms=1000)
    await target.aclose()

    assert first.ok is True
    assert second.ok is True
    assert len(created_clients) == 1
    assert created_clients[0].post_calls == 2
    assert created_clients[0].closed is True


async def test_nerva_binary_rpc_default_sender_reuses_async_client(monkeypatch: Any) -> None:
    data_bytes = (
        encode_frame(
            Frame(
                frame_type=FrameType.DATA,
                request_id=1,
                flags=0,
                payload=msgpack.packb({"text": "ok"}, use_bin_type=True),
            )
        )
        + encode_frame(Frame(frame_type=FrameType.END, request_id=1, flags=0, payload=b""))
    )

    class _FakeResponse:
        def __init__(self, payload: bytes) -> None:
            self.content = payload

        def raise_for_status(self) -> None:
            return None

    class _FakeClient:
        def __init__(self) -> None:
            self.post_calls = 0
            self.closed = False

        async def post(
            self,
            url: str,
            *,
            content: bytes,
            headers: dict[str, str],
            timeout: float,
        ) -> _FakeResponse:
            del url, content, headers, timeout
            self.post_calls += 1
            return _FakeResponse(data_bytes)

        async def aclose(self) -> None:
            self.closed = True

    created_clients: list[_FakeClient] = []

    def _make_client(*args: Any, **kwargs: Any) -> _FakeClient:
        del args, kwargs
        client = _FakeClient()
        created_clients.append(client)
        return client

    monkeypatch.setattr(nerva_binary_rpc.httpx, "AsyncClient", _make_client)

    target = NervaBinaryRPCTarget(base_url="http://localhost:8080", pipeline_name="phase7_mm_vllm")
    first = await target.infer({"prompt": "hi-1"}, deadline_ms=1000)
    second = await target.infer({"prompt": "hi-2"}, deadline_ms=1000)
    await target.aclose()

    assert first.ok is True
    assert first.output_text == "ok"
    assert second.ok is True
    assert len(created_clients) == 1
    assert created_clients[0].post_calls == 2
    assert created_clients[0].closed is True


async def test_triton_infer_parses_json_output() -> None:
    async def sender(
        url: str,
        payload: dict[str, Any],
        timeout_ms: int,
    ) -> dict[str, Any]:
        del url, payload, timeout_ms
        return {"outputs": [{"name": "text", "data": ["hello from triton"]}]}

    target = TritonInferTarget(
        base_url="http://127.0.0.1:8002",
        model_name="phase7_mm_vllm",
        sender=sender,
    )
    resp = await target.infer({"prompt": "hi"}, deadline_ms=1000)
    assert resp.ok is True
    assert resp.output_text == "hello from triton"


async def test_triton_infer_builds_full_e2e_inputs_for_ensemble() -> None:
    captured: dict[str, Any] = {}

    async def sender(
        url: str,
        payload: dict[str, Any],
        timeout_ms: int,
    ) -> dict[str, Any]:
        del url, timeout_ms
        captured.update(payload)
        return {"outputs": [{"name": "OUTPUT_TEXT", "data": ["hello from triton"]}]}

    target = TritonInferTarget(
        base_url="http://127.0.0.1:8002",
        model_name="phase7_mm_vllm",
        sender=sender,
    )
    resp = await target.infer(
        {
            "text": "hi",
            "image_bytes": b"\x00" * 16,
            "max_tokens": 128,
            "temperature": 0.2,
            "top_p": 0.9,
        },
        deadline_ms=1000,
    )

    assert resp.ok is True
    assert resp.output_text == "hello from triton"
    assert isinstance(captured.get("inputs"), list)
    inputs = captured["inputs"]
    names = {item["name"] for item in inputs}
    assert names == {"TEXT", "IMAGE_SIZE", "MAX_TOKENS", "TEMPERATURE", "TOP_P", "DEADLINE_MS"}
    by_name = {item["name"]: item for item in inputs}
    assert by_name["DEADLINE_MS"]["data"] == [1000]


async def test_triton_infer_uses_module_default_sampling_values() -> None:
    captured: dict[str, Any] = {}

    async def sender(
        url: str,
        payload: dict[str, Any],
        timeout_ms: int,
    ) -> dict[str, Any]:
        del url, timeout_ms
        captured.update(payload)
        return {"outputs": [{"name": "OUTPUT_TEXT", "data": ["ok"]}]}

    target = TritonInferTarget(
        base_url="http://127.0.0.1:8002",
        model_name="phase7_mm_vllm",
        sender=sender,
    )
    resp = await target.infer({"text": "hi", "image_bytes": b"\x00" * 16}, deadline_ms=1000)

    assert resp.ok is True
    inputs = captured["inputs"]
    by_name = {item["name"]: item for item in inputs}
    assert by_name["MAX_TOKENS"]["data"] == [triton_infer.DEFAULT_MAX_TOKENS]
    assert by_name["TEMPERATURE"]["data"] == [triton_infer.DEFAULT_TEMPERATURE]
    assert by_name["TOP_P"]["data"] == [triton_infer.DEFAULT_TOP_P]
