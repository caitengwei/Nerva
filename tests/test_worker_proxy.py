"""Tests for nerva.worker.proxy — WorkerProxy (Master-side RPC)."""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing
import os
import tempfile
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import pytest

from nerva.backends.base import InferContext
from nerva.worker.process import worker_entry
from nerva.worker.proxy import WorkerProxy

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@asynccontextmanager
async def _started_proxy() -> AsyncIterator[WorkerProxy]:
    """Start a real worker process and return a connected proxy."""
    with tempfile.TemporaryDirectory(prefix="nerva-test-") as tmp_dir:
        socket_path = os.path.join(tmp_dir, "test.sock")
        proxy = WorkerProxy(socket_path)
        proc = multiprocessing.Process(target=worker_entry, args=(socket_path,))
        proc.start()
        try:
            await proxy.start()
            yield proxy
        finally:
            with contextlib.suppress(Exception):
                await proxy.shutdown()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join()
            await proxy.close()


class TestWorkerProxyStartAndLoad:
    """Start proxy + worker, load model, verify no exception."""

    async def test_start_and_load_model(self) -> None:
        async with _started_proxy() as proxy:
            await proxy.load_model(
                model_name="echo",
                model_class_path="tests.helpers:EchoModel",
                backend="pytorch",
                device="cpu",
            )


class TestWorkerProxyInfer:
    """Infer via proxy with inline data."""

    async def test_infer_inline(self) -> None:
        async with _started_proxy() as proxy:
            await proxy.load_model(
                model_name="echo",
                model_class_path="tests.helpers:EchoModel",
                backend="pytorch",
                device="cpu",
            )
            ctx = InferContext(request_id="req-001", deadline_ms=30000)
            result = await proxy.infer({"value": 42}, ctx)
            assert result == {"echo": 42}

    async def test_infer_multiple_sequential(self) -> None:
        async with _started_proxy() as proxy:
            await proxy.load_model(
                model_name="echo",
                model_class_path="tests.helpers:EchoModel",
                backend="pytorch",
                device="cpu",
            )
            for i in range(5):
                ctx = InferContext(request_id=f"req-{i:03d}", deadline_ms=30000)
                result = await proxy.infer({"value": i}, ctx)
                assert result == {"echo": i}

    async def test_infer_rejects_duplicate_inflight_request_id(self) -> None:
        async with _started_proxy() as proxy:
            await proxy.load_model(
                model_name="slow",
                model_class_path="tests.helpers:SlowModel",
                backend="pytorch",
                device="cpu",
            )
            task = asyncio.create_task(
                proxy.infer(
                    {"delay": 0.2},
                    InferContext(request_id="dup-req", deadline_ms=30000),
                )
            )
            await asyncio.sleep(0.05)

            with pytest.raises(RuntimeError, match="Duplicate in-flight request_id"):
                await proxy.infer(
                    {"delay": 0.1},
                    InferContext(request_id="dup-req", deadline_ms=30000),
                )

            assert await task == {"done": True}


class TestWorkerProxyErrorPropagation:
    """Verify error propagation from CrashModel."""

    async def test_infer_error_propagation(self) -> None:
        async with _started_proxy() as proxy:
            await proxy.load_model(
                model_name="crash",
                model_class_path="tests.helpers:CrashModel",
                backend="pytorch",
                device="cpu",
            )
            ctx = InferContext(request_id="req-crash", deadline_ms=30000)
            with pytest.raises(RuntimeError, match="INTERNAL"):
                await proxy.infer({"value": 1}, ctx)


class TestWorkerProxyHealthCheck:
    """Verify health check returns True after model load."""

    async def test_health_check(self) -> None:
        async with _started_proxy() as proxy:
            await proxy.load_model(
                model_name="echo",
                model_class_path="tests.helpers:EchoModel",
                backend="pytorch",
                device="cpu",
            )
            healthy = await proxy.health_check()
            assert healthy is True


class TestWorkerProxyConcurrencyGuards:
    """Guard tests for single-flight control RPCs."""

    async def test_load_model_rejects_concurrent_calls(self) -> None:
        proxy = WorkerProxy("/tmp/unused.sock")
        proxy._load_model_future = asyncio.get_running_loop().create_future()

        with pytest.raises(RuntimeError, match="load_model already in progress"):
            await proxy.load_model(
                model_name="echo",
                model_class_path="tests.helpers:EchoModel",
            )

    async def test_health_check_rejects_concurrent_calls(self) -> None:
        proxy = WorkerProxy("/tmp/unused.sock")
        proxy._health_future = asyncio.get_running_loop().create_future()

        with pytest.raises(RuntimeError, match="health_check already in progress"):
            await proxy.health_check()
