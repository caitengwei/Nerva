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
    from nerva.observability.metrics import NervaMetrics


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
    kwargs: dict[str, Any] = field(default_factory=dict)
    enqueue_time: float = field(default_factory=time.monotonic)


class DynamicBatcher:
    """Transparent InferableProxy wrapper that batches requests.

    Accumulates infer() calls and dispatches them in batches to the
    inner proxy, applying deadline-aware admission and backpressure.
    """

    def __init__(
        self,
        inner: InferableProxy,
        config: BatchConfig,
        *,
        model_name: str = "unknown",
        metrics: NervaMetrics | None = None,
    ) -> None:
        self._inner = inner
        self._config = config
        self._model_name = model_name
        self._metrics = metrics
        self._queue: asyncio.Queue[_PendingRequest] = asyncio.Queue(
            maxsize=config.queue_capacity
        )
        self._loop_task: asyncio.Task[None] | None = None
        self._in_flight_futures: list[asyncio.Future[dict[str, Any]]] = []

    async def start(self) -> None:
        """Start the background batch loop."""
        if self._loop_task is not None:
            return  # Already running.
        self._loop_task = asyncio.create_task(self._batch_loop())

    async def stop(self) -> None:
        """Stop the batch loop and drain remaining requests."""
        loop_task = self._loop_task
        self._loop_task = None  # Immediately reject new infer() calls.
        if loop_task is not None:
            loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await loop_task
        # Drain remaining requests.
        while not self._queue.empty():
            req = self._queue.get_nowait()
            if not req.future.done():
                req.future.set_exception(RuntimeError("batcher stopped"))
        # Cancel futures for requests currently in-flight (already dequeued
        # by _batch_loop but not yet resolved by gather).
        for fut in self._in_flight_futures:
            if not fut.done():
                fut.set_exception(RuntimeError("batcher stopped"))
        self._in_flight_futures.clear()

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
        # 0. Fail-fast if batch loop is not running.
        if self._loop_task is None:
            raise RuntimeError("batcher not started; call start() or use 'async with'")

        # 1. Deadline admission check.
        # Admission check: deadline_ms is treated as a total TTL from call time.
        # Elapsed time at this point is negligible (< scheduling jitter),
        # so we compare directly without subtracting elapsed_ms.
        if context.deadline_ms < self._config.min_remaining_deadline_ms:
            raise RuntimeError("DEADLINE_EXCEEDED")

        # 2. Enqueue with backpressure timeout.
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        pending = _PendingRequest(
            inputs=inputs, context=context, future=future, kwargs=dict(kwargs)
        )
        effective_timeout_ms = min(self._config.queue_timeout_ms, context.deadline_ms)
        if self._metrics:
            self._metrics.queue_depth.labels(model=self._model_name).set(self._queue.qsize())
        try:
            await asyncio.wait_for(
                self._queue.put(pending),
                timeout=effective_timeout_ms / 1000.0,
            )
        except TimeoutError as err:
            future.cancel()
            elapsed_ms = (time.monotonic() - pending.enqueue_time) * 1000.0
            if context.deadline_ms - elapsed_ms <= 0:
                raise RuntimeError("DEADLINE_EXCEEDED") from err
            raise RuntimeError("RESOURCE_EXHAUSTED") from err

        # 3. Wait for batch loop to resolve this request.
        return await future

    async def _batch_loop(self) -> None:
        config = self._config
        while True:
            # Wait for the first request (blocking).
            first = await self._queue.get()
            # Register immediately so stop() can drain this future even if
            # CancelledError is raised during the aggregation window below.
            self._in_flight_futures.append(first.future)
            batch: list[_PendingRequest] = [first]
            batch_deadline = time.monotonic() + config.max_delay_ms / 1000.0

            # Accumulate up to max_batch_size within the time window.
            while len(batch) < config.max_batch_size:
                remaining = batch_deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(
                        self._queue.get(), timeout=remaining
                    )
                    # Register immediately after dequeue for the same reason.
                    self._in_flight_futures.append(item.future)
                    batch.append(item)
                except TimeoutError:
                    break

            # Filter expired requests.
            now = time.monotonic()
            valid: list[_PendingRequest] = []
            for req in batch:
                elapsed_ms = (now - req.enqueue_time) * 1000.0
                remaining_ms = req.context.deadline_ms - elapsed_ms
                if remaining_ms <= 0:
                    if not req.future.done():
                        req.future.set_exception(RuntimeError("DEADLINE_EXCEEDED"))
                    # Deregister from in-flight since we are resolving it here.
                    with contextlib.suppress(ValueError):
                        self._in_flight_futures.remove(req.future)
                else:
                    valid.append(req)

            if not valid:
                continue

            # Record metrics for this batch.
            if self._metrics:
                self._metrics.batch_size.labels(model=self._model_name).observe(len(valid))
                dispatch_time = time.monotonic()
                for req in valid:
                    wait_s = dispatch_time - req.enqueue_time
                    self._metrics.batch_wait_seconds.labels(model=self._model_name).observe(wait_s)

            # Dispatch valid requests concurrently (already registered as in-flight above).
            results = await asyncio.gather(
                *(
                    self._inner.infer(req.inputs, req.context, **req.kwargs)
                    for req in valid
                ),
                return_exceptions=True,
            )

            # Distribute results back to futures.
            for req, result in zip(valid, results, strict=True):
                if req.future.done():
                    continue
                if isinstance(result, BaseException):
                    req.future.set_exception(result)
                else:
                    req.future.set_result(result)
            # Deregister futures after successful dispatch.
            for req in valid:
                with contextlib.suppress(ValueError):
                    self._in_flight_futures.remove(req.future)
