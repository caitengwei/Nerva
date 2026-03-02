"""serve() — top-level API for starting Nerva inference server."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import uuid
from typing import TYPE_CHECKING, Any

import uvicorn
from starlette.applications import Starlette

from nerva.backends.base import InferContext
from nerva.core.model import get_model_handle
from nerva.engine.executor import Executor
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


class _NervaASGIApp:
    """ASGI application wrapper that lazily starts workers on first HTTP request.

    This design ensures workers are initialised regardless of whether the ASGI
    host sends a lifespan scope (uvicorn does; httpx ASGITransport does not).
    Workers are shut down when the lifespan scope ends (production servers) or
    when the GC collects the object.
    """

    def __init__(
        self,
        starlette_app: Starlette,
        on_startup: Any,
        on_shutdown: Any,
        *,
        watch_parent: bool = True,
        parent_watch_interval_s: float = 1.0,
    ) -> None:
        self._app = starlette_app
        self._on_startup = on_startup
        self._on_shutdown = on_shutdown
        self._started = False
        self._lock: asyncio.Lock | None = None
        self._lifecycle_loop: asyncio.AbstractEventLoop | None = None
        self._watch_parent = watch_parent
        self._parent_watch_interval_s = parent_watch_interval_s
        self._parent_pid = os.getppid()
        self._parent_watchdog_task: asyncio.Task[None] | None = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            # Lazily create the lock in the current running loop.
            self._lock = asyncio.Lock()
        return self._lock

    async def _ensure_started(self) -> None:
        """Idempotent: run startup exactly once."""
        if self._started:
            return
        async with self._get_lock():
            if not self._started:
                self._lifecycle_loop = asyncio.get_running_loop()
                await self._on_startup()
                self._started = True
                self._start_parent_watchdog()

    def _start_parent_watchdog(self) -> None:
        if not self._watch_parent:
            return
        if self._parent_watchdog_task is not None and not self._parent_watchdog_task.done():
            return
        parent_pid = self._parent_pid
        interval = self._parent_watch_interval_s

        async def _watch_parent_lifecycle() -> None:
            while True:
                await asyncio.sleep(interval)
                if os.getppid() != parent_pid:
                    with contextlib.suppress(Exception):
                        os.kill(os.getpid(), signal.SIGTERM)
                    return

        self._parent_watchdog_task = asyncio.create_task(_watch_parent_lifecycle())

    async def _cancel_parent_watchdog(self) -> None:
        task = self._parent_watchdog_task
        self._parent_watchdog_task = None
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def shutdown(self) -> None:
        """Idempotent shutdown hook for non-lifespan ASGI hosts."""
        if not self._started:
            await self._cancel_parent_watchdog()
            return
        async with self._get_lock():
            if not self._started:
                await self._cancel_parent_watchdog()
                return
            try:
                await self._cancel_parent_watchdog()
                await self._on_shutdown()
            finally:
                self._started = False
                self._lifecycle_loop = None

    def __del__(self) -> None:
        """Best-effort auto cleanup for hosts that never send lifespan events."""
        if not self._started:
            return
        loop = self._lifecycle_loop
        if loop is None or loop.is_closed():
            return
        shutdown_hook = self._on_shutdown
        parent_watchdog_task = self._parent_watchdog_task

        def _schedule_shutdown() -> None:
            if parent_watchdog_task is not None and not parent_watchdog_task.done():
                parent_watchdog_task.cancel()
            task = loop.create_task(shutdown_hook())

            def _swallow_result(done: asyncio.Task[Any]) -> None:
                with contextlib.suppress(BaseException):
                    done.result()

            task.add_done_callback(_swallow_result)

        with contextlib.suppress(Exception):
            loop.call_soon_threadsafe(_schedule_shutdown)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            # Forward lifespan events to the inner Starlette app but also
            # drive our own startup/shutdown hooks.
            await self._handle_lifespan(scope, receive, send)
        else:
            # For http/websocket scopes, ensure workers are up before dispatch.
            await self._ensure_started()
            await self._app(scope, receive, send)

    async def _handle_lifespan(self, scope: Any, receive: Any, send: Any) -> None:
        """Handle lifespan scope: startup → yield → shutdown."""
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                try:
                    await self._ensure_started()
                    await send({"type": "lifespan.startup.complete"})
                except Exception as exc:
                    await send({"type": "lifespan.startup.failed", "message": str(exc)})
                    return
            elif message["type"] == "lifespan.shutdown":
                with contextlib.suppress(Exception):
                    await self.shutdown()
                await send({"type": "lifespan.shutdown.complete"})
                return


def build_nerva_app(pipelines: dict[str, Graph]) -> _NervaASGIApp:
    """Return an ASGI app with worker lifecycle managed automatically.

    Workers are started lazily on the first HTTP request (when running under
    httpx in tests) or on the ASGI lifespan startup event (when running under
    uvicorn in production). Workers are shut down on the lifespan shutdown
    event (production) or when the GC collects the object.

    Use this when you want to control the server yourself:

        app = build_nerva_app({"classify": graph})
        # uvicorn mymodule:app --port 8080

    Args:
        pipelines: Mapping from pipeline name to traced Graph.

    Returns:
        An ASGI application (compatible with Starlette/uvicorn).
    """
    import prometheus_client
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route

    from nerva.server.rpc import RpcHandler

    manager = WorkerManager()
    live_executors: dict[str, Any] = {}
    live_model_info: list[dict[str, Any]] = []

    async def _on_startup() -> None:
        execs, info = await _build_pipelines(pipelines, manager)
        live_executors.update(execs)
        live_model_info.extend(info)

    async def _on_shutdown() -> None:
        await manager.shutdown_all()
        live_executors.clear()
        live_model_info.clear()

    rpc_handler = RpcHandler(live_executors)

    async def _health(request: Any) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def _models(request: Any) -> JSONResponse:
        return JSONResponse({"models": live_model_info})

    async def _metrics(request: Any) -> Response:
        data = prometheus_client.generate_latest()
        return Response(content=data, media_type=prometheus_client.CONTENT_TYPE_LATEST)

    starlette_app = Starlette(
        routes=[
            Route("/rpc/{pipeline_name}", rpc_handler.handle, methods=["POST"]),
            Route("/v1/health", _health, methods=["GET"]),
            Route("/v1/models", _models, methods=["GET"]),
            Route("/metrics", _metrics, methods=["GET"]),
        ],
    )

    return _NervaASGIApp(starlette_app, _on_startup, _on_shutdown)


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
    app = build_nerva_app(pipelines)
    uvicorn.run(app, host=host, port=port, log_level="info")
