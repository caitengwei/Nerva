"""Tests for nerva.core.proxy — Proxy, TraceContext, trace()."""

import asyncio
from typing import Any

import pytest

from nerva import Model, model, trace
from nerva.core.proxy import Proxy, _extract_proxy_edges, get_trace_context

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class DummyModel(Model):
    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"out": 1}


# ---------------------------------------------------------------------------
# Proxy tests
# ---------------------------------------------------------------------------


class TestProxy:
    def test_getitem_single(self) -> None:
        p = Proxy(source_node_id="n1", name="out")
        child = p["features"]
        assert child._field_path == ("features",)
        assert child.source_node_id == "n1"

    def test_getitem_chained(self) -> None:
        p = Proxy(source_node_id="n1", name="out")
        child = p["a"]["b"]["c"]
        assert child._field_path == ("a", "b", "c")
        assert child.source_node_id == "n1"

    def test_getitem_non_string_raises(self) -> None:
        p = Proxy(source_node_id="n1", name="out")
        with pytest.raises(TypeError, match="must be a string"):
            p[0]  # type: ignore[index]

    def test_repr(self) -> None:
        p = Proxy(source_node_id="n1", name="out")
        assert "n1" in repr(p)


# ---------------------------------------------------------------------------
# _extract_proxy_edges tests
# ---------------------------------------------------------------------------


class TestExtractProxyEdges:
    def test_single_proxy(self) -> None:
        p = Proxy(source_node_id="src", name="out", field_path=("features",))
        edges = _extract_proxy_edges(p, "dst")
        assert len(edges) == 1
        assert edges[0].src == "src"
        assert edges[0].dst == "dst"
        assert edges[0].src_field_path == ("features",)
        assert edges[0].dst_input_key is None

    def test_dict_of_proxies(self) -> None:
        p1 = Proxy(source_node_id="s1", name="img", field_path=("features",))
        p2 = Proxy(source_node_id="s2", name="txt", field_path=("emb",))
        edges = _extract_proxy_edges({"img": p1, "txt": p2}, "dst")
        assert len(edges) == 2
        keys = {e.dst_input_key for e in edges}
        assert keys == {"img", "txt"}

    def test_none_source_ignored(self) -> None:
        p = Proxy(source_node_id=None, name="input_0")
        edges = _extract_proxy_edges(p, "dst")
        assert len(edges) == 0

    def test_dict_with_non_proxy_values(self) -> None:
        p = Proxy(source_node_id="s1", name="out")
        edges = _extract_proxy_edges({"data": p, "const": 42}, "dst")
        assert len(edges) == 1
        assert edges[0].dst_input_key == "data"


# ---------------------------------------------------------------------------
# trace() tests
# ---------------------------------------------------------------------------


class TestTrace:
    def test_trace_linear_chain(self) -> None:
        a = model("a", DummyModel)
        b = model("b", DummyModel)
        c = model("c", DummyModel)

        def pipeline(x: Any) -> Any:
            return c(b(a(x)))

        g = trace(pipeline)
        assert len(g.nodes) == 3
        assert len(g.edges) == 2
        topo = g.topological_sort()
        assert [n.model_name for n in topo] == ["a", "b", "c"]

    def test_trace_with_getitem(self) -> None:
        a = model("a", DummyModel)
        b = model("b", DummyModel)

        def pipeline(x: Any) -> Any:
            out_a = a(x)
            return b(out_a["features"])

        g = trace(pipeline)
        assert len(g.edges) == 1
        assert g.edges[0].src_field_path == ("features",)

    def test_trace_diamond_with_getitem(self) -> None:
        a = model("a", DummyModel)
        b = model("b", DummyModel)
        c = model("c", DummyModel)

        def pipeline(x: Any) -> Any:
            out = a(x)
            out_b = b(out["left"])
            out_c = c(out["right"])
            return out_b, out_c

        g = trace(pipeline)
        assert len(g.nodes) == 3
        assert len(g.edges) == 2

        # Check field paths.
        edge_map = {e.dst: e for e in g.edges}
        b_node = next(n for n in g.nodes if n.model_name == "b")
        c_node = next(n for n in g.nodes if n.model_name == "c")
        assert edge_map[b_node.id].src_field_path == ("left",)
        assert edge_map[c_node.id].src_field_path == ("right",)

    def test_trace_dict_input_mapping(self) -> None:
        a = model("enc_a", DummyModel)
        b = model("enc_b", DummyModel)
        fusion = model("fusion", DummyModel)

        def pipeline(x: Any) -> Any:
            out_a = a(x)
            out_b = b(x)
            return fusion({"img": out_a["features"], "txt": out_b["emb"]})

        g = trace(pipeline)
        assert len(g.nodes) == 3

        fusion_node = next(n for n in g.nodes if n.model_name == "fusion")
        incoming = g.incoming_edges(fusion_node.id)
        assert len(incoming) == 2
        keys = {e.dst_input_key for e in incoming}
        assert keys == {"img", "txt"}

    def test_context_none_outside_trace(self) -> None:
        assert get_trace_context() is None

    def test_model_handle_outside_trace_raises(self) -> None:
        h = model("test", DummyModel)
        with pytest.raises(RuntimeError, match="outside of a pipeline context"):
            h({"value": 1})

    async def test_contextvars_isolation(self) -> None:
        """Concurrent trace() calls in asyncio tasks don't interfere."""
        a = model("a", DummyModel)
        b = model("b", DummyModel)

        def pipeline_a(x: Any) -> Any:
            return a(x)

        def pipeline_b(x: Any) -> Any:
            return b(x)

        g1, g2 = await asyncio.gather(
            asyncio.to_thread(trace, pipeline_a),
            asyncio.to_thread(trace, pipeline_b),
        )
        assert len(g1.nodes) == 1
        assert g1.nodes[0].model_name == "a"
        assert len(g2.nodes) == 1
        assert g2.nodes[0].model_name == "b"
