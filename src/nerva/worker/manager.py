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
    process: multiprocessing.Process
    proxy: WorkerProxy
    socket_path: str
    state: WorkerState = WorkerState.STARTING
    restart_count: int = 0


class WorkerManager:
    """Manages Worker subprocess lifecycles.

    Spawns processes, connects proxies, loads models,
    and handles restart and shutdown.
    """

    def __init__(self, metrics: NervaMetrics | None = None) -> None:
        self._tmpdir = tempfile.mkdtemp(prefix="nerva-mgr-")
        self._pid = os.getpid()
        self._workers: dict[str, _WorkerEntry] = {}
        self._metrics = metrics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start_worker(self, handle: ModelHandle) -> WorkerProxy:
        """Spawn a worker process, connect proxy, load model, return proxy.

        Args:
            handle: The ModelHandle describing the model to load.

        Returns:
            A connected and model-loaded WorkerProxy.
        """
        worker_id = handle.name
        if worker_id in self._workers:
            raise ValueError(
                f"Worker '{worker_id}' already exists. "
                "Use restart_worker() to recreate it."
            )

        os.makedirs(self._tmpdir, exist_ok=True)
        socket_path = os.path.join(self._tmpdir, f"nerva-{self._pid}-{worker_id}.sock")

        # Clean stale socket file if it exists.
        if os.path.exists(socket_path):
            os.unlink(socket_path)

        proxy = WorkerProxy(socket_path)
        proc = multiprocessing.Process(
            target=worker_entry, args=(socket_path,), daemon=False
        )

        entry = _WorkerEntry(
            handle=handle,
            process=proc,
            proxy=proxy,
            socket_path=socket_path,
            state=WorkerState.STARTING,
        )

        process_started = False
        try:
            proc.start()
            process_started = True
            await proxy.start()

            entry.state = WorkerState.LOADING
            model_class_path = class_to_import_path(handle.model_class)
            await proxy.load_model(
                model_name=handle.name,
                model_class_path=model_class_path,
                backend=handle.backend,
                device=handle.device,
                options=handle.options,
            )

            entry.state = WorkerState.READY
            self._workers[worker_id] = entry
            if self._metrics:
                self._metrics.worker_status.labels(
                    model=handle.name, device=handle.device
                ).set(1)
            logger.info("Worker '%s' is READY", worker_id)
            return proxy
        except Exception:
            logger.exception("Failed to start worker '%s'", worker_id)
            if process_started:
                with contextlib.suppress(Exception):
                    await self._close_worker(entry)
            else:
                with contextlib.suppress(Exception):
                    await proxy.close()
            raise

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

        logger.info(
            "Worker '%s' restarted (count=%d)", worker_id, new_entry.restart_count
        )
        return proxy

    async def shutdown_all(self) -> None:
        """Gracefully shutdown all workers and cleanup tmpdir."""
        for worker_id, entry in list(self._workers.items()):
            if entry.state in (WorkerState.STOPPING, WorkerState.STOPPED):
                continue
            try:
                await self._close_worker(entry)
            except Exception:
                logger.exception("Error shutting down worker '%s'", worker_id)

        self._workers.clear()

        # Cleanup tmpdir.
        try:
            for fname in os.listdir(self._tmpdir):
                fpath = os.path.join(self._tmpdir, fname)
                if os.path.isfile(fpath):
                    os.unlink(fpath)
            os.rmdir(self._tmpdir)
        except OSError:
            logger.debug("Failed to cleanup tmpdir %s", self._tmpdir)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _close_worker(self, entry: _WorkerEntry) -> None:
        """Send SHUTDOWN, join process, close proxy."""
        entry.state = WorkerState.STOPPING
        if self._metrics:
            self._metrics.worker_status.labels(
                model=entry.handle.name, device=entry.handle.device
            ).set(0)

        # Best-effort SHUTDOWN with timeout — peer may already be dead.
        try:
            await asyncio.wait_for(entry.proxy.shutdown(), timeout=3.0)
        except (TimeoutError, Exception):
            logger.debug("Failed to send SHUTDOWN to worker '%s'", entry.handle.name)

        entry.process.join(timeout=5)
        if entry.process.is_alive():
            entry.process.kill()
            entry.process.join(timeout=2)

        await entry.proxy.close()
        entry.state = WorkerState.STOPPED
