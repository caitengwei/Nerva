"""Shared timing sink for main-process and worker components.

Provides a module-level AsyncTimingSink initialized from NERVA_TIMING_LOG_DIR.
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
import contextlib
import json
import logging
import os
import queue
import threading
from typing import IO, Any

logger = logging.getLogger(__name__)

_sink: AsyncTimingSink | None = None

_SENTINEL = object()


class AsyncTimingSink:
    """Queue-backed writer using a dedicated OS thread.

    Hot path: ``write()`` does a single non-blocking ``queue.put_nowait()``.
    The OS-thread writer drains the queue and flushes to disk without
    touching the asyncio event loop — critical for accurate profiling under
    burst load where asyncio.to_thread callbacks add unwanted overhead.
    """

    # Maximum number of unwritten timing entries. Entries beyond this limit are
    # silently dropped to protect the process from OOM during sustained IO stalls.
    _QUEUE_MAXSIZE = 100_000

    def __init__(self) -> None:
        self._fp: IO[str] | None = None
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=self._QUEUE_MAXSIZE)
        self._thread: threading.Thread | None = None
        self._stopping: bool = False

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
            self._stopping = True  # prevent new writes before sentinel is consumed
            self._queue.put(_SENTINEL)
            await asyncio.to_thread(self._thread.join, 5.0)
            if self._thread.is_alive():
                logger.warning(
                    "timing writer thread did not exit within 5 s; log entries may be lost"
                )
            self._thread = None
        # fp is closed by _writer_loop's finally clause; null the reference.
        self._fp = None

    def write(self, data: dict[str, Any]) -> None:
        """Non-blocking enqueue. No-op if sink not started, stopping, or writer thread has died.

        json.dumps is intentionally deferred to the writer thread so that the
        event loop hot path only pays the cost of a single queue.put_nowait().
        Entries are silently dropped when the queue is full (_QUEUE_MAXSIZE).
        """
        if self._thread is not None and not self._stopping and self._thread.is_alive():
            # Silently shed entries when the queue is full (sustained IO stall).
            # contextlib.suppress avoids a try/except on the hot path.
            with contextlib.suppress(queue.Full):
                self._queue.put_nowait(data)

    def _writer_loop(self) -> None:
        assert self._fp is not None
        fp = self._fp
        try:
            while True:
                item = self._queue.get()  # blocks until data available
                if item is _SENTINEL:
                    fp.flush()
                    return
                # json.dumps runs here in the writer thread, off the hot path.
                batch = [json.dumps(item) + "\n"]
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
                    batch.append(json.dumps(nxt) + "\n")
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
