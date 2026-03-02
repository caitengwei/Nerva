from __future__ import annotations

from examples.phase7_multimodal_vllm_server import build_phase7_graph


def test_phase7_graph_nodes() -> None:
    graph = build_phase7_graph()
    model_names = [n.model_name for n in graph.nodes if n.node_type == "call"]
    assert model_names == ["mm_preprocess", "mm_vllm", "mm_postprocess"]
