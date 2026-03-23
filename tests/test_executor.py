"""Tests for nerva.engine.executor — Event-driven DAG Executor (mock WorkerProxy)."""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import pytest

from nerva.backends.base import InferContext
from nerva.core.graph import Edge, Graph, Node
from nerva.engine.executor import Executor, InferableStreamProxy, resolve_field_path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_context(request_id: str = "req-1") -> InferContext:
    return InferContext(request_id=request_id, deadline_ms=5000)


def make_mock_proxy(side_effect: Any = None, return_value: Any = None) -> AsyncMock:
    """Create a mock that satisfies InferableProxy."""
    mock = AsyncMock()
    if side_effect is not None:
        mock.infer.side_effect = side_effect
    elif return_value is not None:
        mock.infer.return_value = return_value
    else:
        mock.infer.return_value = {"out": 1}
    return mock


# ---------------------------------------------------------------------------
# resolve_field_path
# ---------------------------------------------------------------------------


class TestResolveFieldPath:
    def test_empty_path(self) -> None:
        assert resolve_field_path({"a": 1}, ()) == {"a": 1}

    def test_single_key(self) -> None:
        assert resolve_field_path({"features": [1, 2, 3]}, ("features",)) == [1, 2, 3]

    def test_nested_path(self) -> None:
        data = {"a": {"b": {"c": 42}}}
        assert resolve_field_path(data, ("a", "b", "c")) == 42

    def test_missing_key_raises(self) -> None:
        with pytest.raises(KeyError):
            resolve_field_path({"a": 1}, ("b",))


# ---------------------------------------------------------------------------
# Executor: linear chain
# ---------------------------------------------------------------------------


class TestExecutorLinear:
    async def test_linear_chain(self) -> None:
        """a -> b -> c"""
        g = Graph()
        g.add_node(Node(id="a_1", model_name="a"))
        g.add_node(Node(id="b_1", model_name="b"))
        g.add_node(Node(id="c_1", model_name="c"))
        g.add_edge(Edge(src="a_1", dst="b_1"))
        g.add_edge(Edge(src="b_1", dst="c_1"))

        proxy_a = make_mock_proxy(return_value={"x": 10})
        proxy_b = make_mock_proxy(return_value={"y": 20})
        proxy_c = make_mock_proxy(return_value={"z": 30})

        ctx = make_context()
        executor = Executor(g, {"a": proxy_a, "b": proxy_b, "c": proxy_c}, ctx)
        result = await executor.execute({"input": 1})

        assert result == {"z": 30}
        proxy_a.infer.assert_called_once()
        proxy_b.infer.assert_called_once()
        proxy_c.infer.assert_called_once()

    async def test_linear_with_field_path(self) -> None:
        """a -> b, where b receives a["features"]"""
        g = Graph()
        g.add_node(Node(id="a_1", model_name="a"))
        g.add_node(Node(id="b_1", model_name="b"))
        g.add_edge(Edge(src="a_1", dst="b_1", src_field_path=("features",)))

        proxy_a = make_mock_proxy(return_value={"features": [1, 2, 3], "meta": "x"})
        proxy_b = make_mock_proxy(return_value={"result": "ok"})

        ctx = make_context()
        executor = Executor(g, {"a": proxy_a, "b": proxy_b}, ctx)
        result = await executor.execute({"data": "in"})

        assert result == {"result": "ok"}
        # b should receive [1, 2, 3], not the full dict.
        proxy_b.infer.assert_called_once()
        call_args = proxy_b.infer.call_args
        assert call_args[0][0] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Executor: diamond
# ---------------------------------------------------------------------------


class TestExecutorDiamond:
    async def test_diamond(self) -> None:
        """a -> (b, c) -> d, d gets dict inputs"""
        g = Graph()
        g.add_node(Node(id="a_1", model_name="a"))
        g.add_node(Node(id="b_1", model_name="b"))
        g.add_node(Node(id="c_1", model_name="c"))
        g.add_node(Node(id="d_1", model_name="d"))
        g.add_edge(Edge(src="a_1", dst="b_1"))
        g.add_edge(Edge(src="a_1", dst="c_1"))
        g.add_edge(Edge(
            src="b_1", dst="d_1",
            src_field_path=("feat",), dst_input_key="img",
        ))
        g.add_edge(Edge(
            src="c_1", dst="d_1",
            src_field_path=("emb",), dst_input_key="txt",
        ))

        proxy_a = make_mock_proxy(return_value={"raw": "data"})
        proxy_b = make_mock_proxy(return_value={"feat": [1, 2]})
        proxy_c = make_mock_proxy(return_value={"emb": [3, 4]})
        proxy_d = make_mock_proxy(return_value={"final": "result"})

        ctx = make_context()
        executor = Executor(
            g,
            {"a": proxy_a, "b": proxy_b, "c": proxy_c, "d": proxy_d},
            ctx,
        )
        result = await executor.execute({"input": "x"})

        assert result == {"final": "result"}
        # d should receive assembled dict.
        proxy_d.infer.assert_called_once()
        call_args = proxy_d.infer.call_args
        assert call_args[0][0] == {"img": [1, 2], "txt": [3, 4]}


# ---------------------------------------------------------------------------
# Executor: parallel node
# ---------------------------------------------------------------------------


class TestExecutorParallel:
    async def test_parallel_node(self) -> None:
        """parallel(a, b) as a single node"""
        branch_a = Graph()
        branch_a.add_node(Node(id="a_1", model_name="a"))
        branch_b = Graph()
        branch_b.add_node(Node(id="b_1", model_name="b"))

        g = Graph()
        par_node = Node(
            id="par_1", model_name="parallel", node_type="parallel",
            branches=[branch_a, branch_b],
        )
        g.add_node(par_node)

        proxy_a = make_mock_proxy(return_value={"feat_a": 1})
        proxy_b = make_mock_proxy(return_value={"feat_b": 2})

        ctx = make_context()
        executor = Executor(g, {"a": proxy_a, "b": proxy_b}, ctx)
        result = await executor.execute({"input": "x"})

        # parallel output is {"0": result_a, "1": result_b}
        assert result == {"0": {"feat_a": 1}, "1": {"feat_b": 2}}


# ---------------------------------------------------------------------------
# Executor: cond node
# ---------------------------------------------------------------------------


class TestExecutorCond:
    async def test_cond_true_branch(self) -> None:
        true_branch = Graph()
        true_branch.add_node(Node(id="a_1", model_name="a"))
        false_branch = Graph()
        false_branch.add_node(Node(id="b_1", model_name="b"))

        g = Graph()
        cond_node = Node(
            id="cond_1", model_name="cond", node_type="cond",
            true_branch=true_branch, false_branch=false_branch,
        )
        g.add_node(cond_node)

        proxy_a = make_mock_proxy(return_value={"result": "true_path"})
        proxy_b = make_mock_proxy(return_value={"result": "false_path"})

        ctx = make_context()
        # Truthy input → true branch.
        executor = Executor(g, {"a": proxy_a, "b": proxy_b}, ctx)
        result = await executor.execute(True)
        assert result == {"result": "true_path"}

    async def test_cond_false_branch(self) -> None:
        true_branch = Graph()
        true_branch.add_node(Node(id="a_1", model_name="a"))
        false_branch = Graph()
        false_branch.add_node(Node(id="b_1", model_name="b"))

        g = Graph()
        cond_node = Node(
            id="cond_1", model_name="cond", node_type="cond",
            true_branch=true_branch, false_branch=false_branch,
        )
        g.add_node(cond_node)

        proxy_a = make_mock_proxy(return_value={"result": "true_path"})
        proxy_b = make_mock_proxy(return_value={"result": "false_path"})

        ctx = make_context()
        # Falsy input → false branch.
        executor = Executor(g, {"a": proxy_a, "b": proxy_b}, ctx)
        result = await executor.execute(False)
        assert result == {"result": "false_path"}

    async def test_cond_uses_predicate_for_branch_but_passes_payload_to_branch(self) -> None:
        true_branch = Graph()
        true_branch.add_node(Node(id="a_1", model_name="a"))
        false_branch = Graph()
        false_branch.add_node(Node(id="b_1", model_name="b"))

        g = Graph()
        g.add_node(Node(id="pred_1", model_name="pred"))
        g.add_node(Node(
            id="cond_1",
            model_name="cond",
            node_type="cond",
            true_branch=true_branch,
            false_branch=false_branch,
        ))
        g.add_edge(Edge(src="pred_1", dst="cond_1", src_field_path=("flag",)))

        proxy_pred = make_mock_proxy(return_value={"flag": True})
        proxy_a = make_mock_proxy(return_value={"result": "true_path"})
        proxy_b = make_mock_proxy(return_value={"result": "false_path"})

        payload = {"value": "hello"}
        ctx = make_context()
        executor = Executor(
            g,
            {"pred": proxy_pred, "a": proxy_a, "b": proxy_b},
            ctx,
        )
        result = await executor.execute(payload)

        assert result == {"result": "true_path"}
        proxy_pred.infer.assert_called_once()
        proxy_a.infer.assert_called_once()
        proxy_b.infer.assert_not_called()

        # Branch should receive the executor payload, not predicate boolean.
        assert proxy_a.infer.call_args[0][0] == payload


# ---------------------------------------------------------------------------
# Executor: error handling
# ---------------------------------------------------------------------------


class TestExecutorErrors:
    async def test_fail_fast(self) -> None:
        g = Graph()
        g.add_node(Node(id="a_1", model_name="a"))
        g.add_node(Node(id="b_1", model_name="b"))
        g.add_edge(Edge(src="a_1", dst="b_1"))

        proxy_a = make_mock_proxy(side_effect=RuntimeError("model crashed"))
        proxy_b = make_mock_proxy(return_value={"out": 1})

        ctx = make_context()
        executor = Executor(g, {"a": proxy_a, "b": proxy_b}, ctx)

        with pytest.raises(RuntimeError, match="DAG execution failed"):
            await executor.execute({"input": 1})

    async def test_missing_proxy_raises(self) -> None:
        g = Graph()
        g.add_node(Node(id="a_1", model_name="a"))

        ctx = make_context()
        executor = Executor(g, {}, ctx)

        with pytest.raises(KeyError, match="No proxy registered for model 'a'"):
            await executor.execute({"input": 1})

    async def test_empty_graph(self) -> None:
        g = Graph()
        ctx = make_context()
        executor = Executor(g, {}, ctx)
        result = await executor.execute({"input": 1})
        assert result == {"input": 1}

    def test_cyclic_graph_raises(self) -> None:
        """Cyclic graphs must be rejected at construction time, not at execute() time.

        Failing fast prevents partial-cycle graphs (with source nodes + cyclic
        nodes) from deadlocking the event loop at request time.
        """
        g = Graph()
        g.add_node(Node(id="a_1", model_name="a"))
        g.add_node(Node(id="b_1", model_name="b"))
        g.add_edge(Edge(src="a_1", dst="b_1"))
        g.add_edge(Edge(src="b_1", dst="a_1"))

        proxy_a = make_mock_proxy(return_value={"x": 1})
        proxy_b = make_mock_proxy(return_value={"y": 2})

        ctx = make_context()
        with pytest.raises(RuntimeError, match="cycle"):
            Executor(g, {"a": proxy_a, "b": proxy_b}, ctx)

        proxy_a.infer.assert_not_called()
        proxy_b.infer.assert_not_called()


# ---------------------------------------------------------------------------
# Helpers for streaming tests
# ---------------------------------------------------------------------------


def make_streaming_proxy(chunks: list[dict[str, Any]]) -> Any:
    """Create an object that satisfies InferableStreamProxy, yielding given chunks."""

    class _StreamProxy:
        async def infer(
            self,
            inputs: dict[str, Any],
            context: InferContext,
            **kwargs: Any,
        ) -> dict[str, Any]:
            return chunks[-1] if chunks else {}

        async def infer_stream(
            self,
            inputs: dict[str, Any],
            context: InferContext,
            **kwargs: Any,
        ) -> AsyncIterator[dict[str, Any]]:
            for chunk in chunks:
                yield chunk

    return _StreamProxy()


# ---------------------------------------------------------------------------
# execute_stream tests
# ---------------------------------------------------------------------------


class TestExecutorStream:
    async def test_execute_stream_single_node(self) -> None:
        """execute_stream() on a single terminal node yields all chunks."""
        g = Graph()
        g.add_node(Node(id="m_1", model_name="m"))

        proxy = make_streaming_proxy([{"tok": 0}, {"tok": 1}, {"tok": 2}])
        ctx = make_context()
        executor = Executor(g, {"m": proxy}, ctx)

        chunks = []
        async for chunk in executor.execute_stream({"input": 1}):
            chunks.append(chunk)

        assert chunks == [{"tok": 0}, {"tok": 1}, {"tok": 2}]

    async def test_execute_stream_linear_chain_terminal_only(self) -> None:
        """Intermediate node uses infer(); terminal node uses infer_stream()."""
        g = Graph()
        g.add_node(Node(id="a_1", model_name="a"))
        g.add_node(Node(id="b_1", model_name="b"))
        g.add_edge(Edge(src="a_1", dst="b_1"))

        proxy_a = make_mock_proxy(return_value={"mid": 42})
        proxy_b = make_streaming_proxy([{"out": 0}, {"out": 1}])

        ctx = make_context()
        executor = Executor(g, {"a": proxy_a, "b": proxy_b}, ctx)

        chunks = []
        async for chunk in executor.execute_stream({"input": 1}):
            chunks.append(chunk)

        assert chunks == [{"out": 0}, {"out": 1}]
        # Intermediate node must use infer(), not infer_stream().
        proxy_a.infer.assert_called_once()

    async def test_execute_stream_non_streaming_proxy_raises_type_error(self) -> None:
        """Terminal proxy without infer_stream() raises TypeError."""
        g = Graph()
        g.add_node(Node(id="m_1", model_name="m"))

        proxy = make_mock_proxy(return_value={"out": 1})
        ctx = make_context()
        executor = Executor(g, {"m": proxy}, ctx)

        with pytest.raises(TypeError, match="InferableStreamProxy"):
            async for _ in executor.execute_stream({}):
                pass

    async def test_execute_stream_non_call_terminal_raises_runtime_error(self) -> None:
        """execute_stream() with a cond/parallel terminal raises RuntimeError, not KeyError."""
        g = Graph()
        true_branch: Graph = Graph()
        true_branch.add_node(Node(id="a_1", model_name="a"))
        false_branch: Graph = Graph()
        false_branch.add_node(Node(id="b_1", model_name="b"))
        cond_node = Node(
            id="cond_1",
            model_name="cond",
            node_type="cond",
            true_branch=true_branch,
            false_branch=false_branch,
        )
        g.add_node(cond_node)

        ctx = make_context()
        executor = Executor(g, {}, ctx)

        with pytest.raises(RuntimeError, match="call"):
            async for _ in executor.execute_stream({}):
                pass

    async def test_infer_stream_protocol_isinstance(self) -> None:
        """InferableStreamProxy is @runtime_checkable."""
        streaming = make_streaming_proxy([])
        non_streaming = make_mock_proxy(return_value={})

        assert isinstance(streaming, InferableStreamProxy)
        assert not isinstance(non_streaming, InferableStreamProxy)
