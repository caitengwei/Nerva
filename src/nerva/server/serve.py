"""serve() — top-level API for starting Nerva inference server."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any

import uvicorn

from nerva.backends.base import InferContext
from nerva.core.model import get_model_handle
from nerva.engine.executor import Executor
from nerva.server.app import build_app
from nerva.worker.manager import WorkerManager

if TYPE_CHECKING:
    from nerva.core.graph import Graph
    from nerva.engine.executor import InferableProxy

logger = logging.getLogger(__name__)


def _collect_model_names(pipelines: dict[str, Graph]) -> set[str]:
    """Collect all unique model names from pipeline graphs.

    Skips control-flow nodes (cond, parallel) which are not real models.
    Recurses into cond/parallel sub-graphs to discover models in branches.
    """
    names: set[str] = set()
    visited: set[int] = set()

    def _walk(graph: Graph) -> None:
        gid = id(graph)
        if gid in visited:
            return
        visited.add(gid)
        for node in graph.nodes:
            if node.node_type == "call":
                names.add(node.model_name)
            elif node.node_type == "cond":
                if node.true_branch is not None:
                    _walk(node.true_branch)
                if node.false_branch is not None:
                    _walk(node.false_branch)
            elif node.node_type == "parallel":
                for branch in node.branches:
                    _walk(branch)

    for graph in pipelines.values():
        _walk(graph)
    return names


async def _build_pipelines(
    pipelines: dict[str, Graph],
    manager: WorkerManager,
) -> tuple[dict[str, _PipelineExecutor], list[dict[str, Any]]]:
    """Spawn workers and build executors for each pipeline.

    Returns:
        (executors, model_info) tuple.
    """
    model_names = _collect_model_names(pipelines)

    # Spawn one worker per model.
    # Use Any for proxies dict because WorkerProxy.infer has a named shm_pool
    # param while InferableProxy protocol uses **kwargs — runtime compatible.
    proxies: dict[str, Any] = {}
    model_info: list[dict[str, Any]] = []
    for name in sorted(model_names):
        handle = get_model_handle(name)
        proxy = await manager.start_worker(handle)
        proxies[name] = proxy
        model_info.append({
            "name": handle.name,
            "backend": handle.backend,
            "device": handle.device,
        })

    # Build one executor per pipeline.
    executors: dict[str, _PipelineExecutor] = {}
    for pipe_name, graph in pipelines.items():
        executors[pipe_name] = _PipelineExecutor(graph, proxies)

    return executors, model_info


class _PipelineExecutor:
    """Wraps Graph + proxies to provide execute(inputs) for the RPC handler.

    Creates a fresh Executor with a per-request InferContext on each call.
    """

    def __init__(self, graph: Graph, proxies: dict[str, InferableProxy]) -> None:
        self._graph = graph
        self._proxies = proxies

    async def execute(
        self, inputs: Any, *, deadline_ms: int = 30000, request_id: str = ""
    ) -> Any:
        """Execute the pipeline with a fresh InferContext."""
        if not request_id:
            request_id = str(uuid.uuid4())
        ctx = InferContext(request_id=request_id, deadline_ms=deadline_ms)
        executor = Executor(self._graph, self._proxies, ctx)
        return await executor.execute(inputs)


def serve(
    pipelines: dict[str, Graph],
    *,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
    """Start the Nerva inference server (blocking).

    Scans all Graphs for model declarations, auto-spawns worker processes,
    builds the ASGI application, and starts uvicorn.

    Args:
        pipelines: Mapping from pipeline name to traced Graph.
        host: Bind address.
        port: Bind port.
    """

    async def _run() -> None:
        manager = WorkerManager()
        try:
            executors, model_info = await _build_pipelines(pipelines, manager)
            app = build_app(pipelines=executors, model_info=model_info)
            config = uvicorn.Config(app, host=host, port=port, log_level="info")
            server = uvicorn.Server(config)
            await server.serve()
        finally:
            await manager.shutdown_all()

    asyncio.run(_run())
