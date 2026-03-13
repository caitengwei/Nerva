"""Event-driven DAG Executor.

Executes a Graph by scheduling nodes based on an in-degree table.
Each node completion triggers successor scheduling via an asyncio Queue.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import time
from dataclasses import dataclass, replace
from enum import IntEnum
from functools import reduce
from typing import TYPE_CHECKING, Any, Protocol

import nerva.observability.timing as _timing

if TYPE_CHECKING:
    from nerva.backends.base import InferContext
    from nerva.core.graph import Edge, Graph, Node

logger = logging.getLogger(__name__)

# Process-wide atomic counter for per-node context IDs (replaces uuid4).
_node_ctx_counter = itertools.count()


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


# ---------------------------------------------------------------------------
# Pre-computed graph analysis (shared across requests for same pipeline)
# ---------------------------------------------------------------------------


class _InputStrategy(IntEnum):
    """Pre-computed strategy for assembling a node's inputs."""

    SOURCE = 0  # No incoming edges → use pipeline_inputs
    SINGLE_PASSTHROUGH = 1  # One edge, no dst_input_key
    KEYED_DICT = 2  # All edges have dst_input_key → build dict
    MULTI_LIST = 3  # Multiple edges without keys → build list


@dataclass(frozen=True, slots=True)
class PrecomputedGraph:
    """Immutable cache of graph analysis — computed once, reused per request."""

    node_map: dict[str, Node]
    successors: dict[str, list[str]]
    incoming: dict[str, list[Edge]]
    base_in_degree: dict[str, int]
    source_node_ids: list[str]
    last_node_id: str | None
    input_strategies: dict[str, _InputStrategy]

    @classmethod
    def from_graph(cls, graph: Graph) -> PrecomputedGraph:
        node_map: dict[str, Node] = {n.id: n for n in graph.nodes}
        successors: dict[str, list[str]] = {n.id: [] for n in graph.nodes}
        incoming: dict[str, list[Edge]] = {n.id: [] for n in graph.nodes}
        for e in graph.edges:
            successors.setdefault(e.src, []).append(e.dst)
            incoming.setdefault(e.dst, []).append(e)

        base_in_degree: dict[str, int] = {}
        source_node_ids: list[str] = []
        for node in graph.nodes:
            deg = len(incoming.get(node.id, []))
            base_in_degree[node.id] = deg
            if deg == 0:
                source_node_ids.append(node.id)

        # Pre-compute last node ID (avoids topological_sort per request).
        # If the graph is cyclic, topological_sort() raises ValueError; we set
        # last_node_id = None in that case — execute() will detect "no source nodes"
        # and raise RuntimeError before last_node_id is ever used.
        if graph.nodes:
            try:
                topo = graph.topological_sort()
                last_node_id = topo[-1].id
            except ValueError:
                last_node_id = None
        else:
            last_node_id = None

        # Pre-compute input assembly strategy per node.
        input_strategies: dict[str, _InputStrategy] = {}
        for node in graph.nodes:
            edges = incoming.get(node.id, [])
            if not edges:
                input_strategies[node.id] = _InputStrategy.SOURCE
            elif all(e.dst_input_key is not None for e in edges):
                input_strategies[node.id] = _InputStrategy.KEYED_DICT
            elif len(edges) == 1:
                input_strategies[node.id] = _InputStrategy.SINGLE_PASSTHROUGH
            else:
                input_strategies[node.id] = _InputStrategy.MULTI_LIST

        return cls(
            node_map=node_map,
            successors=successors,
            incoming=incoming,
            base_in_degree=base_in_degree,
            source_node_ids=source_node_ids,
            last_node_id=last_node_id,
            input_strategies=input_strategies,
        )


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class Executor:
    """Event-driven DAG executor.

    Uses an in-degree table and a done_queue to maximize concurrency:
    every time a node completes, its successors' in-degrees are decremented
    and any node reaching zero is immediately scheduled.

    Args:
        graph: The computation Graph to execute.
        proxies: Mapping from model_name to an InferableProxy.
        context: InferContext for the entire pipeline execution.
        _precomputed: Optional pre-computed graph analysis (avoids rebuild per request).
    """

    def __init__(
        self,
        graph: Graph,
        proxies: dict[str, InferableProxy],
        context: InferContext,
        *,
        _precomputed: PrecomputedGraph | None = None,
    ) -> None:
        self._graph = graph
        self._proxies = proxies
        self._context = context

        if _precomputed is not None:
            self._pc = _precomputed
        else:
            self._pc = PrecomputedGraph.from_graph(graph)

        # Aliases for backward compatibility with internal references.
        self._node_map = self._pc.node_map
        self._successors = self._pc.successors
        self._incoming = self._pc.incoming

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
        t_execute_start = time.perf_counter()
        pc = self._pc
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

        # Shallow copy of pre-computed in-degree table (O(N) dict copy vs O(N+E) rebuild).
        in_degree: dict[str, int] = dict(pc.base_in_degree)

        completed: dict[str, Any] = {}
        done_queue: asyncio.Queue[str | BaseException] = asyncio.Queue()
        running: dict[str, asyncio.Task[None]] = {}
        failed: BaseException | None = None
        node_timings: list[dict[str, Any]] = []

        async def _run_node(node_id: str) -> None:
            try:
                node = self._node_map[node_id]
                node_inputs = self._build_node_inputs(node_id, completed, inputs)

                if node.node_type == "call":
                    proxy = self._proxies[node.model_name]
                    node_ctx = self._make_node_context(node_id)
                    t_node = time.perf_counter()
                    result = await proxy.infer(node_inputs, node_ctx)
                    node_timings.append({
                        "node_id": node_id,
                        "model": node.model_name,
                        "infer_ms": round((time.perf_counter() - t_node) * 1000, 3),
                    })
                elif node.node_type == "parallel":
                    result = await self._execute_parallel(node, node_inputs)
                elif node.node_type == "cond":
                    result = await self._execute_cond(
                        node,
                        predicate_input=node_inputs,
                        branch_inputs=inputs,
                    )
                else:
                    raise RuntimeError(f"Unknown node type: {node.node_type}")

                completed[node_id] = result
                await done_queue.put(node_id)
            except BaseException as exc:
                await done_queue.put(exc)

        # Start all source nodes (in-degree == 0) from pre-computed list.
        for node_id in pc.source_node_ids:
            task = asyncio.create_task(_run_node(node_id))
            running[node_id] = task

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

        # Use pre-computed last_node_id (avoids topological_sort per request).
        last_node_id = pc.last_node_id
        assert last_node_id is not None
        result = completed[last_node_id]

        if node_timings:
            total_execute_ms = round((time.perf_counter() - t_execute_start) * 1000, 3)
            total_infer_ms = round(sum(t["infer_ms"] for t in node_timings), 3)
            _timing.write({
                "event": "executor_timing",
                "request_id": self._context.request_id,
                "total_execute_ms": total_execute_ms,
                "total_infer_ms": total_infer_ms,
                "scheduler_overhead_ms": round(total_execute_ms - total_infer_ms, 3),
                "nodes": node_timings,
            })

        return result

    def _make_node_context(self, node_id: str) -> InferContext:
        """Create a per-node InferContext with a unique request_id.

        Uses a process-wide atomic counter instead of uuid4 (~50ns vs ~2μs).
        """
        ctx_id = next(_node_ctx_counter)
        return replace(
            self._context,
            request_id=f"{self._context.request_id}:{node_id}:{ctx_id}",
        )

    def _build_node_inputs(
        self,
        node_id: str,
        completed: dict[str, Any],
        pipeline_inputs: Any,
    ) -> Any:
        """Assemble inputs for a node using pre-computed strategy."""
        strategy = self._pc.input_strategies[node_id]

        if strategy == _InputStrategy.SOURCE:
            return pipeline_inputs

        incoming = self._incoming[node_id]

        if strategy == _InputStrategy.KEYED_DICT:
            result: dict[str, Any] = {}
            for edge in incoming:
                output = completed[edge.src]
                value = (
                    resolve_field_path(output, edge.src_field_path)
                    if edge.src_field_path
                    else output
                )
                # dst_input_key is guaranteed non-None by the KEYED_DICT strategy.
                result[edge.dst_input_key] = value  # type: ignore[index]
            return result

        if strategy == _InputStrategy.SINGLE_PASSTHROUGH:
            edge = incoming[0]
            output = completed[edge.src]
            if edge.src_field_path:
                return resolve_field_path(output, edge.src_field_path)
            return output

        # MULTI_LIST: multiple edges without keys.
        results = []
        for edge in incoming:
            output = completed[edge.src]
            value = (
                resolve_field_path(output, edge.src_field_path)
                if edge.src_field_path
                else output
            )
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

    async def _execute_cond(
        self,
        node: Node,
        *,
        predicate_input: Any,
        branch_inputs: Any,
    ) -> Any:
        """Execute the selected branch of a cond node."""
        # The cond node input is the predicate value. Branch execution should
        # receive the executor input payload (not the predicate scalar).
        predicate = predicate_input

        branch = node.true_branch if predicate else node.false_branch

        if branch is None or not branch.nodes:
            return predicate

        sub_executor = Executor(branch, self._proxies, self._context)
        return await sub_executor.execute(branch_inputs)
