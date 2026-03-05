from __future__ import annotations

from typing import Any

from examples.phase7_multimodal_vllm_server import MMPreprocessModel, build_phase7_graph


def test_phase7_graph_nodes() -> None:
    graph = build_phase7_graph()
    model_names = [n.model_name for n in graph.nodes if n.node_type == "call"]
    assert model_names == ["mm_preprocess", "mm_vllm", "mm_postprocess"]


async def test_phase7_preprocess_model_keeps_sampling_params() -> None:
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
