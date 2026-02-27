"""Event-driven DAG Executor.

Executes a Graph by scheduling nodes based on an in-degree table.
Each node completion triggers successor scheduling via an asyncio Queue.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from dataclasses import replace
from functools import reduce
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from nerva.backends.base import InferContext
    from nerva.core.graph import Edge, Graph, Node

logger = logging.getLogger(__name__)


class InferableProxy(Protocol):
    """Protocol for objects that can execute inference (WorkerProxy, mock, etc.)."""

    async def infer(
        self,
        inputs: dict[str, Any],
        context: InferContext,
        **kwargs: Any,
    ) -> dict[str, Any]: ...


def resolve_field_path(output: Any, field_path: tuple[str, ...]) -> Any:
    """Traverse nested dicts/objects using a field path tuple.

    >>> resolve_field_path({"a": {"b": 42}}, ("a", "b"))
    42
    """
    return reduce(lambda d, k: d[k], field_path, output)


class Executor:
    """Event-driven DAG executor.

    Uses an in-degree table and a done_queue to maximize concurrency:
    every time a node completes, its successors' in-degrees are decremented
    and any node reaching zero is immediately scheduled.

    Args:
        graph: The computation Graph to execute.
        proxies: Mapping from model_name to an InferableProxy.
        context: InferContext for the entire pipeline execution.
    """

    def __init__(
        self,
        graph: Graph,
        proxies: dict[str, InferableProxy],
        context: InferContext,
    ) -> None:
        self._graph = graph
        self._proxies = proxies
        self._context = context

        # Pre-compute adjacency structures to avoid repeated O(E) scans.
        self._node_map: dict[str, Node] = {n.id: n for n in graph.nodes}
        self._successors: dict[str, list[str]] = {n.id: [] for n in graph.nodes}
        self._incoming: dict[str, list[Edge]] = {n.id: [] for n in graph.nodes}
        for e in graph.edges:
            self._successors.setdefault(e.src, []).append(e.dst)
            self._incoming.setdefault(e.dst, []).append(e)

    async def execute(self, inputs: Any = None) -> Any:
        """Execute the DAG and return the output of the last node.

        Args:
            inputs: Pipeline input data (passed to source nodes).

        Returns:
            Output of the final node in topological order.

        Raises:
            RuntimeError: If any node fails (fail-fast).
            KeyError: If a required proxy is missing.
        """
        graph = self._graph
        if not graph.nodes:
            return inputs

        # Validate proxies before execution.
        for node in graph.nodes:
            if node.node_type == "call" and node.model_name not in self._proxies:
                raise KeyError(
                    f"No proxy registered for model '{node.model_name}'. "
                    f"Available: {list(self._proxies.keys())}"
                )

        # Build in-degree table from pre-computed incoming edges.
        in_degree: dict[str, int] = {}
        for node in graph.nodes:
            in_degree[node.id] = len(self._incoming.get(node.id, []))

        completed: dict[str, Any] = {}
        done_queue: asyncio.Queue[str | BaseException] = asyncio.Queue()
        running: dict[str, asyncio.Task[None]] = {}
        failed: BaseException | None = None

        async def _run_node(node_id: str) -> None:
            try:
                node = self._node_map[node_id]
                node_inputs = self._build_node_inputs(node_id, completed, inputs)

                if node.node_type == "call":
                    proxy = self._proxies[node.model_name]
                    node_ctx = self._make_node_context(node_id)
                    result = await proxy.infer(node_inputs, node_ctx)
                elif node.node_type == "parallel":
                    result = await self._execute_parallel(node, node_inputs)
                elif node.node_type == "cond":
                    result = await self._execute_cond(node, node_inputs)
                else:
                    raise RuntimeError(f"Unknown node type: {node.node_type}")

                completed[node_id] = result
                await done_queue.put(node_id)
            except BaseException as exc:
                await done_queue.put(exc)

        # Start all source nodes (in-degree == 0).
        for node in graph.nodes:
            if in_degree[node.id] == 0:
                task = asyncio.create_task(_run_node(node.id))
                running[node.id] = task

        if not running:
            raise RuntimeError(
                "Graph has no source nodes (in_degree == 0); it may contain a "
                "cycle and cannot be executed by this executor."
            )

        remaining = len(graph.nodes)
        while remaining > 0:
            item = await done_queue.get()

            if isinstance(item, BaseException):
                failed = item
                break

            done_id: str = item
            remaining -= 1
            running.pop(done_id, None)

            # Schedule successors.
            for succ_id in self._successors.get(done_id, []):
                in_degree[succ_id] -= 1
                if in_degree[succ_id] == 0:
                    task = asyncio.create_task(_run_node(succ_id))
                    running[succ_id] = task

        if failed is not None:
            # Cancel all running tasks.
            for task in running.values():
                task.cancel()
            for task in running.values():
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            raise RuntimeError(f"DAG execution failed: {failed}") from failed

        # Return the output of the last node in topological order.
        topo = graph.topological_sort()
        last_node_id = topo[-1].id
        return completed[last_node_id]

    def _make_node_context(self, node_id: str) -> InferContext:
        """Create a per-node InferContext with a unique request_id."""
        short_id = uuid.uuid4().hex[:8]
        return replace(
            self._context,
            request_id=f"{self._context.request_id}:{node_id}:{short_id}",
        )

    def _build_node_inputs(
        self,
        node_id: str,
        completed: dict[str, Any],
        pipeline_inputs: Any,
    ) -> Any:
        """Assemble inputs for a node from completed outputs and pipeline inputs."""
        incoming = self._incoming.get(node_id, [])

        if not incoming:
            # Source node — use pipeline inputs.
            return pipeline_inputs

        # Check if all edges have dst_input_key → build dict.
        all_have_key = all(e.dst_input_key is not None for e in incoming)
        if all_have_key:
            result: dict[str, Any] = {}
            for edge in incoming:
                output = completed[edge.src]
                value = resolve_field_path(output, edge.src_field_path) if edge.src_field_path else output
                # dst_input_key is guaranteed non-None by the all_have_key check above.
                result[edge.dst_input_key] = value  # type: ignore[index]
            return result

        # Single edge without key — pass-through (with optional field_path).
        if len(incoming) == 1:
            edge = incoming[0]
            output = completed[edge.src]
            if edge.src_field_path:
                return resolve_field_path(output, edge.src_field_path)
            return output

        # Multiple edges without keys — shouldn't happen in well-formed graphs,
        # but pass as a list for robustness.
        results = []
        for edge in incoming:
            output = completed[edge.src]
            value = resolve_field_path(output, edge.src_field_path) if edge.src_field_path else output
            results.append(value)
        return results

    async def _execute_parallel(self, node: Node, inputs: Any) -> dict[str, Any]:
        """Execute all branches of a parallel node concurrently."""
        results: dict[str, Any] = {}

        async def _run_branch(idx: int, branch: Graph) -> None:
            sub_executor = Executor(branch, self._proxies, self._context)
            result = await sub_executor.execute(inputs)
            results[str(idx)] = result

        await asyncio.gather(*(
            _run_branch(i, branch) for i, branch in enumerate(node.branches)
        ))
        return results

    async def _execute_cond(self, node: Node, inputs: Any) -> Any:
        """Execute the selected branch of a cond node."""
        # The inputs to a cond node is the predicate value.
        predicate = inputs

        branch = node.true_branch if predicate else node.false_branch

        if branch is None or not branch.nodes:
            return predicate

        sub_executor = Executor(branch, self._proxies, self._context)
        return await sub_executor.execute(inputs)
