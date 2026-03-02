from __future__ import annotations

from typing import Any

import msgpack
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


async def test_vllm_openai_api_parses_text_field() -> None:
    async def sender(
        url: str,
        payload: dict[str, Any],
        timeout_ms: int,
    ) -> dict[str, Any]:
        del url, payload, timeout_ms
        return {"choices": [{"text": "hello from vllm"}]}

    target = VLLMOpenAIAPITarget(
        base_url="http://127.0.0.1:8001",
        model_name="phase7",
        sender=sender,
    )
    resp = await target.infer({"prompt": "hi"}, deadline_ms=1000)
    assert resp.ok is True
    assert resp.output_text == "hello from vllm"


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
