"""Shared async timing sink for main-process components.

Provides a module-level AsyncTimingSink initialized from NERVA_TIMING_LOG_DIR.
Components (RpcHandler, Executor) call ``write()`` which is non-blocking.
The background writer task flushes to disk via asyncio.to_thread().

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
import os
from typing import IO, Any

_sink: AsyncTimingSink | None = None


class AsyncTimingSink:
    """Queue-backed async writer. Hot path is a single put_nowait() call."""

    def __init__(self) -> None:
        self._fp: IO[str] | None = None
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    async def start(self, log_dir: str, filename: str) -> None:
        """Open log file and start background writer task."""
        os.makedirs(log_dir, exist_ok=True)
        self._fp = open(os.path.join(log_dir, filename), "a")  # noqa: SIM115
        self._task = asyncio.create_task(self._writer_loop())

    async def stop(self) -> None:
        """Flush pending writes and close."""
        if self._task is not None:
            self._queue.put_nowait(None)
            with contextlib.suppress(Exception):
                await self._task
            self._task = None
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def write(self, data: dict[str, Any]) -> None:
        """Non-blocking enqueue. No-op if sink not started."""
        if self._task is not None:
            self._queue.put_nowait(json.dumps(data) + "\n")

    async def _writer_loop(self) -> None:
        assert self._fp is not None
        fp = self._fp
        while True:
            line = await self._queue.get()
            if line is None:
                break
            batch = [line]
            while True:
                try:
                    nxt = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if nxt is None:
                    await asyncio.to_thread(fp.write, "".join(batch))
                    await asyncio.to_thread(fp.flush)
                    return
                batch.append(nxt)
            await asyncio.to_thread(fp.write, "".join(batch))
            await asyncio.to_thread(fp.flush)


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
