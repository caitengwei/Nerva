"""Tests for nerva.worker.manager — WorkerManager lifecycle management."""

from __future__ import annotations

import signal

from nerva import model
from nerva.backends.base import InferContext
from nerva.worker.manager import WorkerManager, WorkerState
from tests.helpers import EchoModel


class TestWorkerManagerStartAndInfer:
    """Start a worker via manager, run inference, verify result."""

    async def test_start_worker_and_infer(self) -> None:
        mgr = WorkerManager()
        try:
            handle = model("echo", EchoModel)
            proxy = await mgr.start_worker(handle)

            ctx = InferContext(request_id="mgr-req-001", deadline_ms=30000)
            result = await proxy.infer({"value": 42}, ctx)
            assert result == {"echo": 42}

            entry = mgr._workers["echo"]
            assert entry.state == WorkerState.READY
        finally:
            await mgr.shutdown_all()


class TestWorkerManagerShutdownIdempotent:
    """Calling shutdown_all twice should not raise."""

    async def test_shutdown_all_twice(self) -> None:
        mgr = WorkerManager()
        handle = model("echo", EchoModel)
        await mgr.start_worker(handle)

        await mgr.shutdown_all()
        # Second call should be a no-op, no exception.
        await mgr.shutdown_all()


class TestWorkerManagerHealthCheck:
    """Health check via manager returns True after start."""

    async def test_health_check_returns_true(self) -> None:
        mgr = WorkerManager()
        try:
            handle = model("echo", EchoModel)
            proxy = await mgr.start_worker(handle)

            healthy = await proxy.health_check()
            assert healthy is True
        finally:
            await mgr.shutdown_all()


class TestWorkerManagerDetectDead:
    """Kill the worker process, verify manager can detect it's not alive."""

    async def test_kill_process_detect_not_alive(self) -> None:
        mgr = WorkerManager()
        try:
            handle = model("echo", EchoModel)
            await mgr.start_worker(handle)

            entry = mgr._workers["echo"]
            pid = entry.process.pid
            assert pid is not None
            assert entry.process.is_alive()

            # Kill the worker process.
            import os

            os.kill(pid, signal.SIGKILL)
            entry.process.join(timeout=5)

            assert not entry.process.is_alive()
        finally:
            await mgr.shutdown_all()


class TestWorkerManagerRestart:
    """Kill process, restart_worker, verify infer succeeds."""

    async def test_restart_worker_after_kill(self) -> None:
        mgr = WorkerManager()
        try:
            handle = model("echo", EchoModel)
            await mgr.start_worker(handle)

            entry = mgr._workers["echo"]
            pid = entry.process.pid
            assert pid is not None

            # Kill the worker.
            import os

            os.kill(pid, signal.SIGKILL)
            entry.process.join(timeout=5)
            assert not entry.process.is_alive()

            # Restart.
            proxy = await mgr.restart_worker("echo")
            new_entry = mgr._workers["echo"]
            assert new_entry.restart_count == 1
            assert new_entry.state == WorkerState.READY

            # Infer should work on the new worker.
            ctx = InferContext(request_id="mgr-restart-001", deadline_ms=30000)
            result = await proxy.infer({"value": 99}, ctx)
            assert result == {"echo": 99}
        finally:
            await mgr.shutdown_all()


class TestWorkerManagerShutdownProcessDead:
    """After shutdown_all, worker processes should not be alive."""

    async def test_shutdown_all_processes_dead(self) -> None:
        mgr = WorkerManager()
        handle = model("echo", EchoModel)
        await mgr.start_worker(handle)

        entry = mgr._workers["echo"]
        proc = entry.process
        assert proc.is_alive()

        await mgr.shutdown_all()
        assert not proc.is_alive()
