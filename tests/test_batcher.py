import asyncio
import contextlib
from typing import Any
from unittest.mock import AsyncMock

import pytest

from nerva.backends.base import InferContext
from nerva.engine.batcher import BatchConfig, DynamicBatcher


def test_batch_config_defaults() -> None:
    cfg = BatchConfig()
    assert cfg.max_batch_size == 32
    assert cfg.max_delay_ms == 10.0
    assert cfg.queue_capacity == 2048
    assert cfg.queue_timeout_ms == 100.0
    assert cfg.min_remaining_deadline_ms == 5.0


def test_batch_config_custom() -> None:
    cfg = BatchConfig(max_batch_size=8, max_delay_ms=5.0)
    assert cfg.max_batch_size == 8
    assert cfg.max_delay_ms == 5.0
    assert cfg.queue_capacity == 2048


def _make_ctx(deadline_ms: int = 30000) -> InferContext:
    return InferContext(request_id="test-req", deadline_ms=deadline_ms)


def _make_inner() -> AsyncMock:
    inner = AsyncMock()
    inner.infer = AsyncMock(return_value={"result": "ok"})
    return inner


async def test_batcher_lifecycle() -> None:
    """start() / stop() は例外を出さない。"""
    inner = _make_inner()
    batcher = DynamicBatcher(inner, BatchConfig())
    await batcher.start()
    await batcher.stop()


async def test_batcher_context_manager() -> None:
    inner = _make_inner()
    async with DynamicBatcher(inner, BatchConfig()) as batcher:
        assert batcher is not None


async def test_infer_deadline_admission_reject() -> None:
    """剩余 deadline 不足时, infer() 立即拒绝。"""
    inner = _make_inner()
    cfg = BatchConfig(min_remaining_deadline_ms=50.0)
    async with DynamicBatcher(inner, cfg) as batcher:
        ctx = _make_ctx(deadline_ms=10)  # 10ms < 50ms threshold
        with pytest.raises(RuntimeError, match="DEADLINE_EXCEEDED"):
            await batcher.infer({"x": 1}, ctx)
    inner.infer.assert_not_called()


async def test_infer_returns_result() -> None:
    """infer() 成功返回 inner 的结果。"""
    inner = _make_inner()
    inner.infer.return_value = {"out": 42}
    async with DynamicBatcher(inner, BatchConfig(max_batch_size=1)) as batcher:
        ctx = _make_ctx()
        result = await batcher.infer({"x": 1}, ctx)
    assert result == {"out": 42}


# ---------------------------------------------------------------------------
# Task 4: size 触发 + timer 触发
# ---------------------------------------------------------------------------


async def test_size_trigger() -> None:
    """max_batch_size full triggers dispatch; inner call count matches batch size."""
    call_count = 0

    async def counting_infer(
        inputs: dict[str, Any], context: InferContext, **kw: Any
    ) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        return {"v": inputs["v"]}

    inner = AsyncMock()
    inner.infer.side_effect = counting_infer

    cfg = BatchConfig(max_batch_size=4, max_delay_ms=1000.0)  # 靠 size 触发
    async with DynamicBatcher(inner, cfg) as batcher:
        ctx = _make_ctx()
        tasks = [
            asyncio.create_task(batcher.infer({"v": i}, ctx))
            for i in range(4)
        ]
        results = await asyncio.gather(*tasks)

    assert len(results) == 4
    assert call_count == 4
    assert {r["v"] for r in results} == {0, 1, 2, 3}


async def test_timer_trigger() -> None:
    """Small batch dispatches automatically after max_delay_ms without blocking."""
    inner = _make_inner()
    inner.infer.return_value = {"done": True}

    cfg = BatchConfig(max_batch_size=100, max_delay_ms=20.0)  # 靠 timer
    async with DynamicBatcher(inner, cfg) as batcher:
        ctx = _make_ctx()
        result = await asyncio.wait_for(
            batcher.infer({"x": 1}, ctx),
            timeout=1.0,
        )

    assert result == {"done": True}


# ---------------------------------------------------------------------------
# Task 5: 批次内 deadline 过滤
# ---------------------------------------------------------------------------


async def test_expired_request_filtered_in_batch() -> None:
    """Requests that expire after enqueue are skipped at dispatch; inner not called for them."""
    inner = _make_inner()
    inner.infer.return_value = {"ok": True}

    # max_batch_size=10 ensures timer (max_delay_ms=30) triggers the batch rather than size.
    # deadline_ms=1 means the expired request's TTL is 1ms; the 30ms delay window far exceeds
    # that, so by the time of dispatch elapsed_ms > 1 and the request is filtered out.
    cfg = BatchConfig(
        max_batch_size=10,
        max_delay_ms=30.0,
        min_remaining_deadline_ms=0.0,  # allow deadline_ms=1 requests to be admitted
    )
    async with DynamicBatcher(inner, cfg) as batcher:
        expired_ctx = _make_ctx(deadline_ms=1)   # 1ms; expires almost immediately after enqueue
        valid_ctx = _make_ctx(deadline_ms=30000)

        expired_task = asyncio.create_task(batcher.infer({"x": "expired"}, expired_ctx))
        valid_task = asyncio.create_task(batcher.infer({"x": "valid"}, valid_ctx))

        expired_result, valid_result = await asyncio.gather(
            expired_task, valid_task, return_exceptions=True
        )

    assert isinstance(expired_result, RuntimeError)
    assert "DEADLINE_EXCEEDED" in str(expired_result)
    assert valid_result == {"ok": True}
    assert inner.infer.call_count == 1  # expired skipped; only one call for valid request


# ---------------------------------------------------------------------------
# Task 6: backpressure — 队列满超时
# ---------------------------------------------------------------------------


async def test_queue_full_raises_resource_exhausted() -> None:
    """infer() raises RESOURCE_EXHAUSTED when queue is full beyond queue_timeout_ms."""
    inner = _make_inner()
    cfg = BatchConfig(queue_capacity=1, queue_timeout_ms=50.0)
    batcher = DynamicBatcher(inner, cfg)
    # start() not called; _batch_loop does not run; queue is not drained

    ctx = _make_ctx()
    # Fill queue to capacity (capacity=1)
    fill_task = asyncio.create_task(batcher.infer({"x": 1}, ctx))
    await asyncio.sleep(0.01)

    # 第二个请求应超时后抛 RESOURCE_EXHAUSTED
    with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
        await asyncio.wait_for(
            batcher.infer({"x": 2}, ctx),
            timeout=1.0,
        )

    fill_task.cancel()
    with contextlib.suppress(BaseException):
        await fill_task


# ---------------------------------------------------------------------------
# Task 7: stop() drain + inner 异常透传
# ---------------------------------------------------------------------------


async def test_stop_drains_pending_requests() -> None:
    """After stop(), pending request futures receive an exception without leaking.

    The request must still be in the queue (not yet dequeued by _batch_loop) when
    stop() is called, so we call stop() immediately after creating the task and
    yielding once to let the enqueue complete.
    """
    inner = _make_inner()
    # Large max_delay_ms so _batch_loop blocks on the first queue.get() indefinitely.
    # queue_capacity=1 so the put() completes after a single yield.
    cfg = BatchConfig(queue_capacity=1, max_delay_ms=10000.0)
    batcher = DynamicBatcher(inner, cfg)
    await batcher.start()

    ctx = _make_ctx()
    pending = asyncio.create_task(batcher.infer({"x": 1}, ctx))
    # Yield to event loop so that:
    #   1. The pending task runs and calls queue.put() (completes immediately, capacity=1).
    #   2. _batch_loop is still waiting on the SECOND queue.get() for more items.
    # At this point the item is IN the batch list (dequeued by _batch_loop's first get()),
    # so we rely on stop() cancel + drain.  If the item was already taken by _batch_loop
    # the future will be cancelled instead; either outcome is acceptable as "stopped".
    await asyncio.sleep(0)

    await batcher.stop()

    result = await asyncio.gather(pending, return_exceptions=True)
    exc = result[0]
    assert isinstance(exc, (RuntimeError, asyncio.CancelledError))
    if isinstance(exc, RuntimeError):
        assert "batcher stopped" in str(exc)


async def test_inner_exception_propagated() -> None:
    """inner.infer() 抛出的异常透传给调用方 future。"""
    inner = _make_inner()
    inner.infer.side_effect = ValueError("model crash")

    cfg = BatchConfig(max_batch_size=1)
    async with DynamicBatcher(inner, cfg) as batcher:
        ctx = _make_ctx()
        with pytest.raises(ValueError, match="model crash"):
            await batcher.infer({"x": 1}, ctx)


# ---------------------------------------------------------------------------
# Task 8: 并发正确性
# ---------------------------------------------------------------------------


async def test_concurrent_requests_no_cross_talk() -> None:
    """Concurrent infer() calls return results matched 1-to-1 with their requests."""
    async def echo_infer(
        inputs: dict[str, Any], context: InferContext, **kw: Any
    ) -> dict[str, Any]:
        await asyncio.sleep(0.001)
        return {"echo": inputs["id"]}

    inner = AsyncMock()
    inner.infer.side_effect = echo_infer

    cfg = BatchConfig(max_batch_size=16, max_delay_ms=10.0)
    async with DynamicBatcher(inner, cfg) as batcher:
        ctx = _make_ctx()
        n = 50
        tasks = [
            asyncio.create_task(batcher.infer({"id": i}, ctx))
            for i in range(n)
        ]
        results = await asyncio.gather(*tasks)

    assert len(results) == n
    returned_ids = {r["echo"] for r in results}
    assert returned_ids == set(range(n))
