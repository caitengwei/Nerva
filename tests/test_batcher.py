from unittest.mock import AsyncMock

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


import pytest


async def test_infer_deadline_admission_reject() -> None:
    """剩余 deadline 不足时，infer() 立即拒绝。"""
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
