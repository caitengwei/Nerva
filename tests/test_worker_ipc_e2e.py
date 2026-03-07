"""Worker-Manager IPC end-to-end integration tests."""

from __future__ import annotations

import time

import pytest

from nerva import model
from nerva.backends.base import InferContext, ModelConfig
from nerva.backends.pytorch import PyTorchBackend
from nerva.engine.shm_pool import ShmPool
from nerva.worker.manager import WorkerManager
from tests.helpers import EchoModel


class TestWorkerManagerE2E:
    async def test_single_model_roundtrip(self) -> None:
        """Full cycle: start manager -> spawn worker -> infer -> shutdown."""
        handle = model("e2e-echo", EchoModel, backend="pytorch", device="cpu")
        manager = WorkerManager()
        try:
            proxy = await manager.start_worker(handle)
            ctx = InferContext(request_id="e2e-1", deadline_ms=99999)
            result = await proxy.infer({"value": "hello"}, ctx)
            assert result == {"echo": "hello"}
        finally:
            await manager.shutdown_all()

    async def test_result_matches_phase0(self) -> None:
        """IPC result should match in-process result."""
        # in-process (direct, no IPC)
        backend = PyTorchBackend()
        config = ModelConfig(
            model_name="echo", model_class=EchoModel, device="cpu"
        )
        await backend.load_model(config)
        ctx0 = InferContext(request_id="p0", deadline_ms=99999)
        result_p0 = await backend.infer({"value": [1, 2, 3]}, ctx0)
        await backend.unload_model()

        # via IPC (WorkerManager)
        handle = model("echo", EchoModel, backend="pytorch", device="cpu")
        manager = WorkerManager()
        try:
            proxy = await manager.start_worker(handle)
            ctx1 = InferContext(request_id="p1", deadline_ms=99999)
            result_p1 = await proxy.infer({"value": [1, 2, 3]}, ctx1)
        finally:
            await manager.shutdown_all()

        assert result_p0 == result_p1

    async def test_multiple_requests(self) -> None:
        handle = model("echo", EchoModel, backend="pytorch", device="cpu")
        manager = WorkerManager()
        try:
            proxy = await manager.start_worker(handle)
            for i in range(10):
                ctx = InferContext(request_id=f"multi-{i}", deadline_ms=99999)
                result = await proxy.infer({"value": i}, ctx)
                assert result == {"echo": i}
        finally:
            await manager.shutdown_all()

    async def test_health_during_idle(self) -> None:
        handle = model("echo", EchoModel, backend="pytorch", device="cpu")
        manager = WorkerManager()
        try:
            proxy = await manager.start_worker(handle)
            assert await proxy.health_check() is True
        finally:
            await manager.shutdown_all()


class TestWorkerManagerShm:
    async def test_infer_with_shm_pool(self) -> None:
        """Verify SHM path works for large payloads."""
        handle = model("echo", EchoModel, backend="pytorch", device="cpu")
        manager = WorkerManager()
        pool = ShmPool(size_classes_kb=[16], slots_per_class=2)
        try:
            proxy = await manager.start_worker(handle)
            # Create input larger than inline threshold (>8KB)
            big_value = "x" * 10000
            ctx = InferContext(request_id="shm-1", deadline_ms=99999)
            result = await proxy.infer({"value": big_value}, ctx, shm_pool=pool)
            assert result == {"echo": big_value}
        finally:
            await manager.shutdown_all()
            pool.close()


@pytest.mark.slow
class TestWorkerManagerPerf:
    async def test_ipc_overhead(self) -> None:
        """Measure IPC overhead vs in-process baseline."""
        n = 100

        # in-process baseline
        backend = PyTorchBackend()
        config = ModelConfig(
            model_name="echo", model_class=EchoModel, device="cpu"
        )
        await backend.load_model(config)

        t0 = time.perf_counter()
        for i in range(n):
            ctx = InferContext(request_id=f"p0-{i}", deadline_ms=99999)
            await backend.infer({"value": i}, ctx)
        p0_total = time.perf_counter() - t0
        await backend.unload_model()

        # via IPC
        handle = model("echo", EchoModel, backend="pytorch", device="cpu")
        manager = WorkerManager()
        try:
            proxy = await manager.start_worker(handle)

            t0 = time.perf_counter()
            for i in range(n):
                ctx = InferContext(request_id=f"p1-{i}", deadline_ms=99999)
                await proxy.infer({"value": i}, ctx)
            p1_total = time.perf_counter() - t0
        finally:
            await manager.shutdown_all()

        p0_avg_us = (p0_total / n) * 1_000_000
        p1_avg_us = (p1_total / n) * 1_000_000
        overhead_us = p1_avg_us - p0_avg_us

        print(f"\nin-process avg: {p0_avg_us:.0f} us")
        print(f"Phase 1 avg: {p1_avg_us:.0f} us")
        print(f"IPC overhead: {overhead_us:.0f} us")

        # Sanity: IPC overhead should be < 10ms per request
        assert overhead_us < 10_000, f"IPC overhead too high: {overhead_us:.0f} us"
