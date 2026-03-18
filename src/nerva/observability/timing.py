"""Shared timing sink for main-process and worker components.

Provides a module-level TimingSink initialized from NERVA_TIMING_LOG_DIR.
Components (RpcHandler, Executor, Worker) call ``write()`` which is non-blocking.
The background writer runs in a dedicated OS thread — zero asyncio event loop
overhead, so high-concurrency profiling does not perturb measurements.

Usage:
    # Server startup (inside running event loop):
    import nerva.observability.timing as timing
    await timing.setup("/tmp/nerva_timing")

    # Hot path (sync, zero-blocking):
    timing.write({"event": "rpc_timing", ...})

    # Server shutdown:
    await timing.teardown()
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
from typing import IO, Any

_sink: AsyncTimingSink | None = None

_SENTINEL = object()


class AsyncTimingSink:
    """Queue-backed writer using a dedicated OS thread.

    Hot path: ``write()`` does a single non-blocking ``queue.put_nowait()``.
    The OS-thread writer drains the queue and flushes to disk without
    touching the asyncio event loop — critical for accurate profiling under
    burst load where asyncio.to_thread callbacks add unwanted overhead.
    """

    def __init__(self) -> None:
        self._fp: IO[str] | None = None
        self._queue: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None

    async def start(self, log_dir: str, filename: str) -> None:
        """Open log file and start background writer thread."""
        os.makedirs(log_dir, exist_ok=True)
        self._fp = open(os.path.join(log_dir, filename), "a")  # noqa: SIM115
        self._thread = threading.Thread(
            target=self._writer_loop, daemon=True, name=f"timing-writer-{filename}"
        )
        self._thread.start()

    async def stop(self) -> None:
        """Flush pending writes and close.

        Sends a sentinel to the writer thread and waits up to 5 s for it to
        exit.  The writer thread is responsible for closing the file (in its
        finally block), so this method never calls fp.close() directly —
        avoiding the race where join() times out but the thread is still
        mid-write on the same file handle.
        """
        if self._thread is not None:
            self._queue.put(_SENTINEL)
            await asyncio.to_thread(self._thread.join, 5.0)
            self._thread = None
        # fp is closed by _writer_loop's finally clause; null the reference.
        self._fp = None

    def write(self, data: dict[str, Any]) -> None:
        """Non-blocking enqueue. No-op if sink not started or writer thread has died."""
        if self._thread is not None and self._thread.is_alive():
            self._queue.put_nowait(json.dumps(data) + "\n")

    def _writer_loop(self) -> None:
        assert self._fp is not None
        fp = self._fp
        try:
            while True:
                item = self._queue.get()  # blocks until data available
                if item is _SENTINEL:
                    fp.flush()
                    return
                batch = [item]
                # drain any items already queued
                while True:
                    try:
                        nxt = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is _SENTINEL:
                        fp.write("".join(batch))
                        fp.flush()
                        return
                    batch.append(nxt)
                fp.write("".join(batch))
                fp.flush()
        finally:
            # Always close the file in the thread that owns it.  This avoids
            # the race in stop() where join(timeout) may return while the thread
            # is still mid-write, and stop() closes the file under it.
            fp.close()


# ---------------------------------------------------------------------------
# Module-level API
# ---------------------------------------------------------------------------


def get_sink() -> AsyncTimingSink | None:
    return _sink


async def setup(log_dir: str) -> None:
    """Initialize the shared timing sink. Called once on server startup."""
    global _sink
    if _sink is not None:
        return
    sink = AsyncTimingSink()
    await sink.start(log_dir, "nerva_main.log")
    _sink = sink


async def teardown() -> None:
    """Flush and stop the shared timing sink."""
    global _sink
    if _sink is not None:
        await _sink.stop()
        _sink = None


def write(data: dict[str, Any]) -> None:
    """Non-blocking timing write. No-op if sink not initialized."""
    if _sink is not None:
        _sink.write(data)
