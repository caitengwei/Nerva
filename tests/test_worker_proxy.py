"""Tests for nerva.worker.proxy — WorkerProxy (Master-side RPC)."""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing
import os
import shutil
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import msgpack
import pytest

from nerva.backends.base import InferContext
from nerva.engine.shm_pool import ShmPool
from nerva.worker.ipc import AckStatus, MessageType
from nerva.worker.process import worker_entry
from nerva.worker.proxy import WorkerProxy

StartedWorkerFactory = Callable[[str, str], Awaitable[WorkerProxy]]


@pytest.fixture
async def started_worker() -> AsyncIterator[StartedWorkerFactory]:
    """Factory fixture: start a worker, load model, and auto-cleanup."""
    resources: list[tuple[WorkerProxy, multiprocessing.Process, str]] = []

    async def _start(
        model_name: str,
        model_class_path: str,
        backend: str = "pytorch",
        device: str = "cpu",
    ) -> WorkerProxy:
        tmp_dir = tempfile.mkdtemp(prefix="nerva-test-")
        socket_path = os.path.join(tmp_dir, "test.sock")
        proxy = WorkerProxy(socket_path)
        proc = multiprocessing.Process(target=worker_entry, args=(socket_path,))
        proc.start()
        await proxy.start()
        await proxy.load_model(
            model_name=model_name,
            model_class_path=model_class_path,
            backend=backend,
            device=device,
        )
        resources.append((proxy, proc, tmp_dir))
        return proxy

    yield _start

    for proxy, proc, tmp_dir in resources:
        with contextlib.suppress(Exception):
            await proxy.shutdown()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        with contextlib.suppress(Exception):
            await proxy.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


class TestWorkerProxyStartAndLoad:
    """Start proxy + worker, load model, verify no exception."""

    async def test_start_and_load_model(self, started_worker: StartedWorkerFactory) -> None:
        await started_worker("echo", "tests.helpers:EchoModel")


class TestWorkerProxyInfer:
    """Infer via proxy with inline data."""

    async def test_infer_inline(self, started_worker: StartedWorkerFactory) -> None:
        proxy = await started_worker("echo", "tests.helpers:EchoModel")
        ctx = InferContext(request_id="req-001", deadline_ms=30000)
        result = await proxy.infer({"value": 42}, ctx)
        assert result == {"echo": 42}

    async def test_infer_multiple_sequential(self, started_worker: StartedWorkerFactory) -> None:
        proxy = await started_worker("echo", "tests.helpers:EchoModel")
        for i in range(5):
            ctx = InferContext(request_id=f"req-{i:03d}", deadline_ms=30000)
            result = await proxy.infer({"value": i}, ctx)
            assert result == {"echo": i}

    async def test_infer_single_bytes_uses_raw_fast_path(
        self,
        started_worker: StartedWorkerFactory,
        monkeypatch,  # type: ignore[no-untyped-def]
    ) -> None:
        proxy = await started_worker("echo", "tests.helpers:EchoModel")

        payload = b"\x00\x01\x02\xff"
        expected_input_obj = {"value": payload}
        original_packb = msgpack.packb

        def _guard_packb(obj: Any, *args: Any, **kwargs: Any) -> bytes:
            if obj == expected_input_obj:
                raise AssertionError(
                    "msgpack.packb should not be called for raw bytes fast path"
                )
            return original_packb(obj, *args, **kwargs)

        monkeypatch.setattr("nerva.worker.proxy.msgpack.packb", _guard_packb)

        ctx = InferContext(request_id="req-bytes-fast", deadline_ms=30000)
        result = await proxy.infer(expected_input_obj, ctx)
        assert result == {"echo": payload}


class TestWorkerProxyErrorPropagation:
    """Verify error propagation from CrashModel."""

    async def test_infer_error_propagation(self, started_worker: StartedWorkerFactory) -> None:
        proxy = await started_worker("crash", "tests.helpers:CrashModel")
        ctx = InferContext(request_id="req-crash", deadline_ms=30000)
        with pytest.raises(RuntimeError, match="INTERNAL"):
            await proxy.infer({"value": 1}, ctx)


class TestWorkerProxyHealthCheck:
    """Verify health check returns True after model load."""

    async def test_health_check(self, started_worker: StartedWorkerFactory) -> None:
        proxy = await started_worker("echo", "tests.helpers:EchoModel")
        healthy = await proxy.health_check()
        assert healthy is True


class TestWorkerProxyDeadlineAndCancel:
    async def test_infer_deadline_exceeded(self, started_worker: StartedWorkerFactory) -> None:
        proxy = await started_worker("slow", "tests.helpers:SlowModel")
        ctx = InferContext(request_id="req-deadline", deadline_ms=50)
        with pytest.raises(RuntimeError, match="DEADLINE_EXCEEDED"):
            await proxy.infer({"delay": 0.5}, ctx)

    async def test_infer_deadline_non_positive_rejected(
        self,
        started_worker: StartedWorkerFactory,
    ) -> None:
        proxy = await started_worker("slow", "tests.helpers:SlowModel")
        ctx = InferContext(request_id="req-deadline-zero", deadline_ms=0)
        with pytest.raises(RuntimeError, match="DEADLINE_EXCEEDED"):
            await proxy.infer({"delay": 0.01}, ctx)

    async def test_infer_cancel_aborted(self, started_worker: StartedWorkerFactory) -> None:
        proxy = await started_worker("slow", "tests.helpers:SlowModel")
        request_id = "req-cancel"
        ctx = InferContext(request_id=request_id, deadline_ms=30000)
        infer_task = asyncio.create_task(proxy.infer({"delay": 2.0}, ctx))
        await asyncio.sleep(0.05)
        await proxy.cancel(request_id, reason="user cancelled")

        with pytest.raises(RuntimeError, match="ABORTED"):
            await infer_task


class TestWorkerProxyOutputShmPath:
    async def test_large_output_uses_shm(self, started_worker: StartedWorkerFactory) -> None:
        proxy = await started_worker("big-output", "tests.helpers:BigOutputModel")
        pool = ShmPool(size_classes_kb=[16], slots_per_class=2, name_prefix="out")
        try:
            ctx = InferContext(request_id="req-big-output", deadline_ms=30000)
            result = await proxy.infer({"size": 10000}, ctx, shm_pool=pool)
            assert len(result["blob"]) == 10000
            assert pool.stats[16 * 1024]["in_use"] == 0
        finally:
            pool.close()

    async def test_large_output_shm_exhausted(self, started_worker: StartedWorkerFactory) -> None:
        proxy = await started_worker("big-output", "tests.helpers:BigOutputModel")
        pool = ShmPool(size_classes_kb=[4], slots_per_class=1, name_prefix="outx")
        try:
            ctx = InferContext(request_id="req-big-output-oos", deadline_ms=30000)
            with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
                await proxy.infer({"size": 10000}, ctx, shm_pool=pool)
        finally:
            pool.close()


class TestWorkerProxyInferStream:
    """proxy.infer_stream() yields chunks from a streaming Worker."""

    async def test_infer_stream_collects_all_chunks(
        self, started_worker: StartedWorkerFactory
    ) -> None:
        """infer_stream() yields all 3 chunks from StreamingEchoModel."""
        proxy = await started_worker("streaming-echo", "tests.helpers:StreamingEchoModel")
        ctx = InferContext(request_id="req-stream-proxy-001", deadline_ms=30000)
        chunks = []
        async for chunk in proxy.infer_stream({"value": 5, "count": 3}, ctx):
            chunks.append(chunk)
        assert len(chunks) == 3
        assert chunks[0] == {"chunk": 0, "value": 5}
        assert chunks[1] == {"chunk": 1, "value": 5}
        assert chunks[2] == {"chunk": 2, "value": 5}

    async def test_infer_stream_empty_yields_nothing(
        self, started_worker: StartedWorkerFactory
    ) -> None:
        """count=0 → infer_stream() yields no chunks."""
        proxy = await started_worker("streaming-echo", "tests.helpers:StreamingEchoModel")
        ctx = InferContext(request_id="req-stream-empty-proxy", deadline_ms=30000)
        chunks = []
        async for chunk in proxy.infer_stream({"value": 0, "count": 0}, ctx):
            chunks.append(chunk)
        assert chunks == []

    async def test_infer_stream_error_raises(
        self, started_worker: StartedWorkerFactory
    ) -> None:
        """StreamingCrashModel mid-stream error raises RuntimeError."""
        proxy = await started_worker("streaming-crash", "tests.helpers:StreamingCrashModel")
        ctx = InferContext(request_id="req-stream-crash-proxy", deadline_ms=30000)
        chunks = []
        with pytest.raises(RuntimeError, match="INTERNAL"):
            async for chunk in proxy.infer_stream({}, ctx):
                chunks.append(chunk)
        # One chunk should have arrived before the error.
        assert len(chunks) == 1
        assert chunks[0] == {"chunk": 0}

    async def test_infer_stream_fail_outstanding_terminates(
        self,
        tmp_path,  # type: ignore[no-untyped-def]
    ) -> None:
        """_fail_outstanding() injects poison pill into pending stream queues."""
        proxy = WorkerProxy(os.path.join(tmp_path, "unused.sock"))
        sent: list[dict[str, Any]] = []

        async def _fake_send(msg: dict[str, Any]) -> None:
            sent.append(msg)

        proxy._send = _fake_send  # type: ignore[method-assign]

        # Manually register a pending stream queue.
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        proxy._pending_stream["req-s1"] = q

        proxy._fail_outstanding("worker crashed")

        # Queue should have received a poison pill.
        pill = q.get_nowait()
        assert pill["status"] == AckStatus.UNAVAILABLE.value
        assert pill.get("stream_done") is True
        assert "worker crashed" in pill.get("error", "")

        # _pending_stream should be cleared.
        assert "req-s1" not in proxy._pending_stream


class TestWorkerProxyRequestIdCollision:
    """Cross-dict collision detection between _pending and _pending_stream."""

    async def test_infer_stream_rejects_id_already_in_pending_unary(
        self,
        monkeypatch,  # type: ignore[no-untyped-def]
        tmp_path,  # type: ignore[no-untyped-def]
    ) -> None:
        """infer_stream() raises if request_id is already in _pending (unary in flight)."""
        proxy = WorkerProxy(os.path.join(tmp_path, "unused.sock"))
        fake_future: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()
        proxy._pending["dup-001"] = fake_future
        try:
            ctx = InferContext(request_id="dup-001", deadline_ms=5000)
            with pytest.raises(RuntimeError, match="Duplicate"):
                async for _ in proxy.infer_stream({}, ctx):
                    pass
        finally:
            proxy._pending.pop("dup-001", None)
            fake_future.cancel()

    async def test_infer_rejects_id_already_in_pending_stream(
        self,
        monkeypatch,  # type: ignore[no-untyped-def]
        tmp_path,  # type: ignore[no-untyped-def]
    ) -> None:
        """infer() raises if request_id is already in _pending_stream (stream in flight)."""
        proxy = WorkerProxy(os.path.join(tmp_path, "unused.sock"))
        proxy._pending_stream["dup-002"] = asyncio.Queue()
        try:
            ctx = InferContext(request_id="dup-002", deadline_ms=5000)
            with pytest.raises(RuntimeError, match="Duplicate"):
                await proxy.infer({}, ctx)
        finally:
            proxy._pending_stream.pop("dup-002", None)

    async def test_infer_stream_timeout_maps_to_deadline_exceeded(
        self,
        monkeypatch,  # type: ignore[no-untyped-def]
        tmp_path,  # type: ignore[no-untyped-def]
    ) -> None:
        """asyncio.wait_for timeout in infer_stream() raises RuntimeError with DEADLINE_EXCEEDED."""
        proxy = WorkerProxy(os.path.join(tmp_path, "unused.sock"), submit_timeout=0.01)

        async def _never_send(msg: dict[str, Any]) -> None:
            pass  # Drop message — chunk_queue.get() will time out.

        monkeypatch.setattr(proxy, "_send", _never_send)

        ctx = InferContext(request_id="timeout-001", deadline_ms=5000)
        with pytest.raises(RuntimeError, match="DEADLINE_EXCEEDED"):
            async for _ in proxy.infer_stream({}, ctx):
                pass


class TestWorkerProxyShmAllocContention:
    async def test_competing_alloc_requests(
        self,
        monkeypatch,  # type: ignore[no-untyped-def]
        tmp_path,  # type: ignore[no-untyped-def]
    ) -> None:
        """Two requests competing for one SHM slot: second should exhaust."""
        proxy = WorkerProxy(os.path.join(tmp_path, "unused.sock"))
        sent: list[dict[str, Any]] = []

        async def _fake_send(msg: dict[str, Any]) -> None:
            sent.append(msg)

        monkeypatch.setattr(proxy, "_send", _fake_send)

        pool = ShmPool(size_classes_kb=[16], slots_per_class=1, name_prefix="cmp")
        try:
            proxy._request_pools["r1"] = pool
            proxy._request_pools["r2"] = pool

            await proxy._handle_shm_alloc_request({
                "type": MessageType.SHM_ALLOC_REQUEST.value,
                "request_id": "r1",
                "size": 10000,
            })
            await proxy._handle_shm_alloc_request({
                "type": MessageType.SHM_ALLOC_REQUEST.value,
                "request_id": "r2",
                "size": 10000,
            })

            assert sent[0]["type"] == MessageType.SHM_ALLOC_RESPONSE.value
            assert sent[0]["status"] == AckStatus.OK.value
            assert sent[1]["type"] == MessageType.SHM_ALLOC_RESPONSE.value
            assert sent[1]["status"] == AckStatus.RESOURCE_EXHAUSTED.value
        finally:
            proxy._release_output_slot("r1")
            pool.close()
