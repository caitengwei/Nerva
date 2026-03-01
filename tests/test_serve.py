# tests/test_serve.py
"""Tests for nerva.server.serve — serve() internals."""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from nerva.core.graph import Graph, Node
from nerva.core.model import Model, model
from nerva.server.serve import _build_pipelines, _collect_model_names


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
