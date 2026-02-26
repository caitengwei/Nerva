"""Tests for nerva.core.graph — Graph IR (Node, Edge, Graph)."""

import pytest

from nerva.core.graph import Edge, Graph, Node


class TestGraphConstruction:
    def test_add_node_and_edge(self) -> None:
        g = Graph()
        n1 = Node(id="a", model_name="model_a")
        n2 = Node(id="b", model_name="model_b")
        g.add_node(n1)
        g.add_node(n2)
        g.add_edge(Edge(src="a", dst="b"))

        assert len(g.nodes) == 2
        assert len(g.edges) == 1

    def test_node_map(self) -> None:
        g = Graph()
        g.add_node(Node(id="x", model_name="mx"))
        g.add_node(Node(id="y", model_name="my"))
        nm = g.node_map
        assert nm["x"].model_name == "mx"
        assert nm["y"].model_name == "my"

    def test_predecessors_and_successors(self) -> None:
        g = Graph()
        g.add_node(Node(id="a", model_name="a"))
        g.add_node(Node(id="b", model_name="b"))
        g.add_node(Node(id="c", model_name="c"))
        g.add_edge(Edge(src="a", dst="b"))
        g.add_edge(Edge(src="a", dst="c"))

        assert g.predecessors("b") == ["a"]
        assert set(g.successors("a")) == {"b", "c"}
        assert g.predecessors("a") == []

    def test_incoming_edges(self) -> None:
        g = Graph()
        g.add_node(Node(id="a", model_name="a"))
        g.add_node(Node(id="b", model_name="b"))
        e = Edge(src="a", dst="b", src_field_path=("features",), dst_input_key="img")
        g.add_edge(e)

        incoming = g.incoming_edges("b")
        assert len(incoming) == 1
        assert incoming[0].src_field_path == ("features",)
        assert incoming[0].dst_input_key == "img"


class TestTopologicalSort:
    def test_linear_chain(self) -> None:
        g = Graph()
        for name in ["a", "b", "c"]:
            g.add_node(Node(id=name, model_name=name))
        g.add_edge(Edge(src="a", dst="b"))
        g.add_edge(Edge(src="b", dst="c"))

        topo = g.topological_sort()
        assert [n.id for n in topo] == ["a", "b", "c"]

    def test_diamond(self) -> None:
        g = Graph()
        for name in ["a", "b", "c", "d"]:
            g.add_node(Node(id=name, model_name=name))
        g.add_edge(Edge(src="a", dst="b"))
        g.add_edge(Edge(src="a", dst="c"))
        g.add_edge(Edge(src="b", dst="d"))
        g.add_edge(Edge(src="c", dst="d"))

        topo = g.topological_sort()
        ids = [n.id for n in topo]
        assert ids[0] == "a"
        assert ids[-1] == "d"
        assert set(ids[1:3]) == {"b", "c"}

    def test_cycle_detection(self) -> None:
        g = Graph()
        g.add_node(Node(id="a", model_name="a"))
        g.add_node(Node(id="b", model_name="b"))
        g.add_edge(Edge(src="a", dst="b"))
        g.add_edge(Edge(src="b", dst="a"))

        with pytest.raises(ValueError, match="cycle"):
            g.topological_sort()


class TestEdge:
    def test_edge_is_frozen(self) -> None:
        e = Edge(src="a", dst="b")
        with pytest.raises(AttributeError):
            e.src = "c"  # type: ignore[misc]

    def test_edge_field_path(self) -> None:
        e = Edge(src="a", dst="b", src_field_path=("features", "0"))
        assert e.src_field_path == ("features", "0")
        assert e.dst_input_key is None

    def test_edge_dst_input_key(self) -> None:
        e = Edge(src="a", dst="b", dst_input_key="img")
        assert e.dst_input_key == "img"
