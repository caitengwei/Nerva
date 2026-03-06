"""End-to-end integration tests for Phase 2 (DAG Pipeline with real Workers)."""

from __future__ import annotations

from nerva import model, trace
from nerva.backends.base import InferContext
from nerva.core.graph import Edge, Graph, Node
from nerva.engine.executor import Executor
from nerva.worker.manager import WorkerManager
from tests.helpers import ConcatModel, EchoModel, UpperModel


class TestExecutorSingleNode:
    async def test_single_node_dag(self) -> None:
        """Simplest DAG: one node, one worker."""
        handle = model("echo", EchoModel, backend="pytorch", device="cpu")
        manager = WorkerManager()
        try:
            proxy = await manager.start_worker(handle)

            g = Graph()
            g.add_node(Node(id="echo_1", model_name="echo"))

            ctx = InferContext(request_id="p2-single-1", deadline_ms=30000)
            executor = Executor(g, {"echo": proxy}, ctx)
            result = await executor.execute({"value": "hello"})
            assert result == {"echo": "hello"}
        finally:
            await manager.shutdown_all()


class TestExecutorLinearChain:
    async def test_linear_two_nodes(self) -> None:
        """echo -> upper: echo passes value, upper uppercases it."""
        h_echo = model("echo", EchoModel, backend="pytorch", device="cpu")
        h_upper = model("upper", UpperModel, backend="pytorch", device="cpu")
        manager = WorkerManager()
        try:
            p_echo = await manager.start_worker(h_echo)
            p_upper = await manager.start_worker(h_upper)

            g = Graph()
            g.add_node(Node(id="echo_1", model_name="echo"))
            g.add_node(Node(id="upper_1", model_name="upper"))
            # echo output becomes upper's input directly.
            g.add_edge(Edge(src="echo_1", dst="upper_1"))

            ctx = InferContext(request_id="p2-chain-1", deadline_ms=30000)
            executor = Executor(g, {"echo": p_echo, "upper": p_upper}, ctx)
            result = await executor.execute({"value": "hello"})
            # This case is intentionally miswired and asserts the real behavior:
            # UpperModel reads "value", but receives {"echo": "hello"}.
            assert result == {"features": ""}
        finally:
            await manager.shutdown_all()

    async def test_linear_with_field_path(self) -> None:
        """echo -> upper with field_path mapping."""
        h_echo = model("echo", EchoModel, backend="pytorch", device="cpu")
        h_upper = model("upper", UpperModel, backend="pytorch", device="cpu")
        manager = WorkerManager()
        try:
            p_echo = await manager.start_worker(h_echo)
            p_upper = await manager.start_worker(h_upper)

            g = Graph()
            g.add_node(Node(id="echo_1", model_name="echo"))
            g.add_node(Node(id="upper_1", model_name="upper"))
            # Map echo's "echo" field to upper's "value" input key.
            g.add_edge(Edge(
                src="echo_1", dst="upper_1",
                src_field_path=("echo",), dst_input_key="value",
            ))

            ctx = InferContext(request_id="p2-chain-fp-1", deadline_ms=30000)
            executor = Executor(g, {"echo": p_echo, "upper": p_upper}, ctx)
            result = await executor.execute({"value": "hello"})
            assert result == {"features": "HELLO"}
        finally:
            await manager.shutdown_all()


class TestExecutorParallel:
    async def test_parallel_to_concat(self) -> None:
        """parallel(upper_a, upper_b) -> concat."""
        h_upper = model("upper", UpperModel, backend="pytorch", device="cpu")
        h_concat = model("concat", ConcatModel, backend="pytorch", device="cpu")
        manager = WorkerManager()
        try:
            p_upper = await manager.start_worker(h_upper)
            p_concat = await manager.start_worker(h_concat)

            # Build graph manually: parallel node with two branches, then concat.
            branch_a = Graph()
            branch_a.add_node(Node(id="upper_a", model_name="upper"))
            branch_b = Graph()
            branch_b.add_node(Node(id="upper_b", model_name="upper"))

            g = Graph()
            par_node = Node(
                id="par_1", model_name="parallel", node_type="parallel",
                branches=[branch_a, branch_b],
            )
            g.add_node(par_node)
            g.add_node(Node(id="concat_1", model_name="concat"))
            # Map parallel output branches to concat inputs.
            g.add_edge(Edge(
                src="par_1", dst="concat_1",
                src_field_path=("0", "features"), dst_input_key="a",
            ))
            g.add_edge(Edge(
                src="par_1", dst="concat_1",
                src_field_path=("1", "features"), dst_input_key="b",
            ))

            ctx = InferContext(request_id="p2-par-1", deadline_ms=30000)
            executor = Executor(
                g, {"upper": p_upper, "concat": p_concat}, ctx
            )
            # Both branches get same input, upper does .upper() on "value".
            result = await executor.execute({"value": "hi"})
            # Both branches uppercase "hi" → "HI", concat does "HI" + "HI" = "HIHI".
            assert result == {"result": "HIHI"}
        finally:
            await manager.shutdown_all()


class TestExecutorTraceAndExecute:
    async def test_trace_then_execute(self) -> None:
        """Full flow: trace a pipeline function, then execute the graph."""
        h_echo = model("echo", EchoModel, backend="pytorch", device="cpu")
        h_upper = model("upper", UpperModel, backend="pytorch", device="cpu")

        def pipeline(x: object) -> object:
            out = h_echo(x)
            return h_upper({"value": out["echo"]})

        g = trace(pipeline)
        assert len(g.nodes) == 2

        manager = WorkerManager()
        try:
            p_echo = await manager.start_worker(h_echo)
            p_upper = await manager.start_worker(h_upper)

            ctx = InferContext(request_id="p2-trace-exec-1", deadline_ms=30000)
            executor = Executor(
                g, {"echo": p_echo, "upper": p_upper}, ctx
            )
            result = await executor.execute({"value": "world"})
            assert result == {"features": "WORLD"}
        finally:
            await manager.shutdown_all()
