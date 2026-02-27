"""Dynamic Batcher with deadline-aware admission and backpressure."""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nerva.backends.base import InferContext
    from nerva.engine.executor import InferableProxy


@dataclass
class BatchConfig:
    """Configuration for DynamicBatcher.

    Attributes:
        max_batch_size: Maximum requests per batch before forced dispatch.
        max_delay_ms: Maximum milliseconds to wait for batch to fill.
        queue_capacity: Maximum pending requests in the queue (backpressure).
        queue_timeout_ms: Milliseconds to wait when queue is full before RESOURCE_EXHAUSTED.
        min_remaining_deadline_ms: Minimum remaining deadline to admit a request.
    """

    max_batch_size: int = 32
    max_delay_ms: float = 10.0
    queue_capacity: int = 2048
    queue_timeout_ms: float = 100.0
    min_remaining_deadline_ms: float = 5.0


@dataclass
class _PendingRequest:
    """A request waiting in the batcher queue."""

    inputs: dict[str, Any]
    context: InferContext
    future: asyncio.Future[dict[str, Any]]
    enqueue_time: float = field(default_factory=time.monotonic)


class DynamicBatcher:
    """Transparent InferableProxy wrapper that batches requests.

    Accumulates infer() calls and dispatches them in batches to the
    inner proxy, applying deadline-aware admission and backpressure.
    """

    def __init__(self, inner: InferableProxy, config: BatchConfig) -> None:
        self._inner = inner
        self._config = config
        self._queue: asyncio.Queue[_PendingRequest] = asyncio.Queue(
            maxsize=config.queue_capacity
        )
        self._loop_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the background batch loop."""
        self._loop_task = asyncio.create_task(self._batch_loop())

    async def stop(self) -> None:
        """Stop the batch loop and drain remaining requests."""
        if self._loop_task is not None:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task
            self._loop_task = None
        # Drain remaining requests.
        while not self._queue.empty():
            req = self._queue.get_nowait()
            if not req.future.done():
                req.future.set_exception(RuntimeError("batcher stopped"))

    async def __aenter__(self) -> DynamicBatcher:
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()

    async def infer(
        self,
        inputs: dict[str, Any],
        context: InferContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        raise NotImplementedError  # 在 Task 3 实现

    async def _batch_loop(self) -> None:
        raise NotImplementedError  # 在 Task 3 实现
