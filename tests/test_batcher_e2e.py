"""Phase 3 end-to-end tests: DynamicBatcher + real Worker + Executor."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from nerva import model
from nerva.backends.base import InferContext
from nerva.core.graph import Graph, Node
from nerva.engine.batcher import BatchConfig, DynamicBatcher
from nerva.engine.executor import Executor
from nerva.worker.manager import WorkerManager
from tests.helpers import EchoModel

if TYPE_CHECKING:
    from nerva.worker.proxy import WorkerProxy


@pytest.fixture
async def started_worker() -> Any:
    """Fixture: spawn a real Worker process with EchoModel, yield proxy, shutdown."""
    handle = model("echo", EchoModel, backend="pytorch", device="cpu")
    manager = WorkerManager()
    proxy = await manager.start_worker(handle)
    yield proxy
    await manager.shutdown_all()


async def test_batcher_with_real_worker(started_worker: WorkerProxy) -> None:
    """DynamicBatcher batches 4 concurrent requests to a real EchoModel Worker."""
    proxy = started_worker
    config = BatchConfig(max_batch_size=4, max_delay_ms=20.0)
    async with DynamicBatcher(proxy, config) as batcher:
        # Each request must have a unique request_id — WorkerProxy rejects duplicates.
        ctxs = [
            InferContext(request_id=f"p3-batcher-{i}", deadline_ms=30000)
            for i in range(4)
        ]
        results = await asyncio.gather(
            batcher.infer({"value": "a"}, ctxs[0]),
            batcher.infer({"value": "b"}, ctxs[1]),
            batcher.infer({"value": "c"}, ctxs[2]),
            batcher.infer({"value": "d"}, ctxs[3]),
        )

    assert len(results) == 4
    for result in results:
        assert "echo" in result


async def test_executor_with_batcher(started_worker: WorkerProxy) -> None:
    """Executor routes a single-node Graph through DynamicBatcher to a real Worker."""
    proxy = started_worker
    config = BatchConfig(max_batch_size=1, max_delay_ms=10.0)
    async with DynamicBatcher(proxy, config) as batcher:
        g = Graph()
        g.add_node(Node(id="echo_1", model_name="echo"))

        ctx = InferContext(request_id="p3-executor-1", deadline_ms=30000)
        executor = Executor(g, {"echo": batcher}, ctx)
        result = await executor.execute({"value": "hello"})

    assert "echo" in result
