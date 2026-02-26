"""Tests for nerva.core.primitives — cond() and parallel()."""

from typing import Any

from nerva import Model, cond, model, parallel, trace


class DummyModel(Model):
    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"out": 1}


class TestParallel:
    def test_basic_parallel(self) -> None:
        a = model("a", DummyModel)
        b = model("b", DummyModel)
        c = model("c", DummyModel)

        def pipeline(x: Any) -> Any:
            out_a, out_b = parallel(
                lambda: a(x),
                lambda: b(x),
            )
            return c({"img": out_a, "txt": out_b})

        g = trace(pipeline)
        # Top level: parallel node + c node.
        assert len(g.nodes) == 2
        par_node = next(n for n in g.nodes if n.node_type == "parallel")
        assert len(par_node.branches) == 2
        assert par_node.branches[0].nodes[0].model_name == "a"
        assert par_node.branches[1].nodes[0].model_name == "b"

    def test_parallel_output_proxy_field_path(self) -> None:
        a = model("a", DummyModel)
        b = model("b", DummyModel)
        c = model("c", DummyModel)

        def pipeline(x: Any) -> Any:
            out_a, out_b = parallel(
                lambda: a(x),
                lambda: b(x),
            )
            # Access features from each branch output.
            return c({"img": out_a["features"], "txt": out_b["features"]})

        g = trace(pipeline)
        c_node = next(n for n in g.nodes if n.model_name == "c")
        incoming = g.incoming_edges(c_node.id)
        assert len(incoming) == 2

        # Each edge should have field_path starting with branch index.
        paths = {e.dst_input_key: e.src_field_path for e in incoming}
        assert paths["img"] == ("0", "features")
        assert paths["txt"] == ("1", "features")


class TestCond:
    def test_basic_cond(self) -> None:
        pred = model("pred", DummyModel)
        a = model("a", DummyModel)
        b = model("b", DummyModel)
        c = model("c", DummyModel)

        def pipeline(x: Any) -> Any:
            pred_out = pred(x)
            features = cond(
                pred_out,
                lambda: a(x),
                lambda: b(x),
            )
            return c(features)

        g = trace(pipeline)
        # Top-level: pred, cond, c.
        assert len(g.nodes) == 3
        cond_node = next(n for n in g.nodes if n.node_type == "cond")
        assert cond_node.true_branch is not None
        assert cond_node.false_branch is not None
        assert cond_node.true_branch.nodes[0].model_name == "a"
        assert cond_node.false_branch.nodes[0].model_name == "b"

    def test_cond_edge_from_predicate(self) -> None:
        pred = model("pred", DummyModel)
        a = model("a", DummyModel)
        b = model("b", DummyModel)

        def pipeline(x: Any) -> Any:
            pred_out = pred(x)
            return cond(pred_out, lambda: a(x), lambda: b(x))

        g = trace(pipeline)
        cond_node = next(n for n in g.nodes if n.node_type == "cond")
        pred_node = next(n for n in g.nodes if n.model_name == "pred")
        # Edge from pred → cond.
        edges_to_cond = g.incoming_edges(cond_node.id)
        assert len(edges_to_cond) == 1
        assert edges_to_cond[0].src == pred_node.id

    def test_nested_parallel_in_cond(self) -> None:
        pred = model("pred", DummyModel)
        a = model("a", DummyModel)
        b = model("b", DummyModel)
        c = model("c", DummyModel)

        def pipeline(x: Any) -> Any:
            pred_out = pred(x)
            return cond(
                pred_out,
                lambda: a(x),
                lambda: parallel(lambda: b(x), lambda: c(x)),
            )

        g = trace(pipeline)
        cond_node = next(n for n in g.nodes if n.node_type == "cond")
        assert cond_node.true_branch is not None
        assert len(cond_node.true_branch.nodes) == 1
        assert cond_node.false_branch is not None
        # False branch has a parallel node.
        assert len(cond_node.false_branch.nodes) == 1
        assert cond_node.false_branch.nodes[0].node_type == "parallel"
