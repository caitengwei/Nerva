from __future__ import annotations

from typing import Any

import pytest
from examples.mm_vllm_server import MMPreprocessModel, build_mm_vllm_graph


def test_mm_vllm_graph_nodes() -> None:
    graph = build_mm_vllm_graph()
    model_names = [n.model_name for n in graph.nodes if n.node_type == "call"]
    assert model_names == ["mm_preprocess", "mm_vllm", "mm_postprocess"]


async def test_mm_preprocess_model_keeps_sampling_params() -> None:
    model = MMPreprocessModel()
    payload: dict[str, Any] = {
        "text": "hello",
        "image_bytes": b"\x00" * 16,
        "max_tokens": 128,
        "temperature": 0.2,
        "top_p": 0.9,
    }
    result = await model.infer(payload)
    assert result["prompt"] == "[image_bytes=16]\nhello"
    assert result["max_tokens"] == 128
    assert result["temperature"] == 0.2
    assert result["top_p"] == 0.9


@pytest.mark.parametrize("invalid_top_p", [0.0, -0.1, 1.1, float("inf"), float("nan")])
async def test_mm_preprocess_model_rejects_invalid_top_p(invalid_top_p: float) -> None:
    model = MMPreprocessModel()
    with pytest.raises(ValueError, match="top_p must be finite and in \\(0, 1\\]"):
        await model.infer({"text": "hello", "image_bytes": b"\x00", "top_p": invalid_top_p})


@pytest.mark.parametrize("invalid_temperature", [-0.1, float("inf"), float("nan")])
async def test_mm_preprocess_model_rejects_invalid_temperature(
    invalid_temperature: float,
) -> None:
    model = MMPreprocessModel()
    with pytest.raises(ValueError, match="temperature must be finite and >= 0"):
        await model.infer(
            {
                "text": "hello",
                "image_bytes": b"\x00",
                "temperature": invalid_temperature,
            }
        )
