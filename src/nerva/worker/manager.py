"""WorkerManager — manages Worker subprocess lifecycles.

Spawns processes, connects proxies, loads models, handles restart and shutdown.
"""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing
import os
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import structlog

from nerva.worker.ipc import class_to_import_path
from nerva.worker.process import worker_entry
from nerva.worker.proxy import WorkerProxy

if TYPE_CHECKING:
    from nerva.core.model import ModelHandle
    from nerva.observability.metrics import NervaMetrics

logger = structlog.get_logger(__name__)

MAX_RESTARTS = 5


class WorkerState(StrEnum):
    """Lifecycle state of a managed worker."""

    STARTING = "STARTING"
    LOADING = "LOADING"
    READY = "READY"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"


@dataclass
class _WorkerEntry:
    """Internal bookkeeping for a single managed worker."""

    handle: ModelHandle
    process: multiprocessing.Process | None  # None when we only connected (not spawned)
    proxy: WorkerProxy
    socket_path: str
    spawned: bool  # True if this instance spawned the process; False if it only connected
    state: WorkerState = WorkerState.STARTING
    restart_count: int = 0


class WorkerManager:
    """Manages Worker subprocess lifecycles.

    Spawns processes, connects proxies, loads models,
    and handles restart and shutdown.
    """

    def __init__(self, metrics: NervaMetrics | None = None) -> None:
        # Socket directory is shared across all uvicorn workers forked from the
        # same parent process (os.getppid() is identical for all forks).
        # Override with NERVA_SOCKET_DIR to isolate multiple server instances.
        socket_dir_env = os.environ.get("NERVA_SOCKET_DIR")
        if socket_dir_env:
            self._socket_dir = socket_dir_env
        else:
            self._socket_dir = os.path.join(
                tempfile.gettempdir(), f"nerva-sockets-{os.getppid()}"
            )
        os.makedirs(self._socket_dir, exist_ok=True)
        self._workers: dict[str, _WorkerEntry] = {}
        self._metrics = metrics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start_worker(self, handle: ModelHandle) -> WorkerProxy:
        """Connect to (or spawn) a worker process, load model if spawned, return proxy.

        Multiple uvicorn workers call this concurrently.  Only the first caller
        (determined by an atomic lock file) spawns the Nerva worker process and
        sends LOAD_MODEL.  Subsequent callers just connect a new DEALER to the
        already-running ROUTER and skip the load step.

        Args:
            handle: The ModelHandle describing the model to load.

        Returns:
            A connected WorkerProxy (model already loaded).
        """
        worker_id = handle.name
        if worker_id in self._workers:
            raise ValueError(
                f"Worker '{worker_id}' already exists. "
                "Use restart_worker() to recreate it."
            )

        socket_path = os.path.join(self._socket_dir, f"nerva-{worker_id}.sock")
        lock_path = socket_path + ".spawning"
        timing_log_dir = os.environ.get("NERVA_TIMING_LOG_DIR") or None

        # Determine whether we are the spawner using an atomic lock file.
        # Limitation (MVP): lock is deleted after startup, so dynamic scaling
        # (uvicorn adding a new worker after initial startup) can cause a new
        # caller to become a second spawner and unlink the existing socket.
        # This design targets static deployments where all workers start together.
        # Also: if the spawner crashes between lock creation and deletion, the lock
        # becomes stale and non-spawners will time out in proxy.start(). In that
        # case, manual cleanup of the lock file is required.
        spawned = False
        proc: multiprocessing.Process | None = None
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(lock_fd)
            spawned = True
        except FileExistsError:
            spawned = False  # another uvicorn worker is spawning (or already done)

        proxy = WorkerProxy(socket_path)
        entry = _WorkerEntry(
            handle=handle,
            process=None,
            proxy=proxy,
            socket_path=socket_path,
            spawned=spawned,
            state=WorkerState.STARTING,
        )

        process_started = False
        try:
            if spawned:
                # Remove stale socket from a previous run if present.
                if os.path.exists(socket_path):
                    os.unlink(socket_path)

                proc = multiprocessing.Process(
                    target=worker_entry,
                    args=(socket_path,),
                    kwargs={"timing_log_dir": timing_log_dir},
                    daemon=False,
                )
                proc.start()
                process_started = True
                entry.process = proc

            # Connect DEALER to the ROUTER (proxy.start() waits for socket file).
            await proxy.start()

            if spawned:
                # Only the spawner sends LOAD_MODEL.
                entry.state = WorkerState.LOADING
                model_class_path = class_to_import_path(handle.model_class)
                await proxy.load_model(
                    model_name=handle.name,
                    model_class_path=model_class_path,
                    backend=handle.backend,
                    device=handle.device,
                    options=handle.options,
                    async_infer=handle.async_infer,
                )
            else:
                # Non-spawner: wait for the spawner's LOAD_MODEL to complete before
                # returning READY.  Without this poll, infer requests arriving while
                # the spawner is still loading would fail with "No model loaded".
                logger.info("worker_connected_existing", worker_id=worker_id)
                loop = asyncio.get_running_loop()
                deadline = loop.time() + 60.0
                while not await proxy.health_check():
                    if loop.time() >= deadline:
                        raise RuntimeError(
                            f"Timed out waiting for model '{worker_id}' to become healthy"
                        )
                    await asyncio.sleep(0.5)

            entry.state = WorkerState.READY
            self._workers[worker_id] = entry
            if self._metrics:
                self._metrics.worker_status.labels(
                    model=handle.name, device=handle.device
                ).set(1)
            logger.info("worker_ready", worker_id=worker_id)
            return proxy
        except Exception:
            logger.exception("worker_start_failed", worker_id=worker_id)
            if process_started and proc is not None:
                with contextlib.suppress(Exception):
                    await self._close_worker(entry)
            else:
                with contextlib.suppress(Exception):
                    await proxy.close()
            raise
        finally:
            # Release spawn lock regardless of outcome.
            if spawned:
                with contextlib.suppress(OSError):
                    os.unlink(lock_path)

    async def restart_worker(self, worker_id: str) -> WorkerProxy:
        """Restart a worker: close old process, start fresh, increment restart count.

        Args:
            worker_id: The name of the worker (same as ModelHandle.name).

        Returns:
            A new connected and model-loaded WorkerProxy.

        Raises:
            KeyError: If worker_id is not found.
            RuntimeError: If max restart count exceeded.
        """
        entry = self._workers.get(worker_id)
        if entry is None:
            raise KeyError(f"Worker '{worker_id}' not found")

        if entry.restart_count >= MAX_RESTARTS:
            raise RuntimeError(
                f"Worker '{worker_id}' exceeded max restarts ({MAX_RESTARTS})"
            )

        # Close old worker.
        await self._close_worker(entry)
        self._workers.pop(worker_id, None)

        # Start fresh.
        old_restart_count = entry.restart_count
        handle = entry.handle
        proxy = await self.start_worker(handle)

        # Preserve and increment restart count.
        new_entry = self._workers[worker_id]
        new_entry.restart_count = old_restart_count + 1

        logger.info("worker_restarted", worker_id=worker_id, restart_count=new_entry.restart_count)
        return proxy

    async def shutdown_all(self) -> None:
        """Gracefully close all proxies, terminate spawned worker processes, and unlink socket files."""
        for worker_id, entry in list(self._workers.items()):
            if entry.state in (WorkerState.STOPPING, WorkerState.STOPPED):
                continue
            try:
                await self._close_worker(entry)
            except Exception:
                logger.exception("worker_shutdown_error", worker_id=worker_id)

        # Clean up socket files left by workers we spawned.
        for entry in self._workers.values():
            if entry.spawned:
                with contextlib.suppress(OSError):
                    os.unlink(entry.socket_path)

        self._workers.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _close_worker(self, entry: _WorkerEntry) -> None:
        """Close this manager's proxy connection.  If we spawned the worker,
        also send SHUTDOWN and terminate the process."""
        entry.state = WorkerState.STOPPING
        if self._metrics:
            self._metrics.worker_status.labels(
                model=entry.handle.name, device=entry.handle.device
            ).set(0)

        if entry.spawned and entry.process is not None:
            # Best-effort SHUTDOWN with timeout — peer may already be dead.
            try:
                await asyncio.wait_for(entry.proxy.shutdown(), timeout=3.0)
            except (TimeoutError, Exception):
                logger.debug("worker_shutdown_send_failed", worker=entry.handle.name)

            entry.process.join(timeout=5)
            if entry.process.is_alive():
                entry.process.kill()
                entry.process.join(timeout=2)

        await entry.proxy.close()
        entry.state = WorkerState.STOPPED
