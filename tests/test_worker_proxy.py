"""Tests for nerva.worker.proxy — WorkerProxy (Master-side RPC)."""

from __future__ import annotations

import multiprocessing
import os
import tempfile

import pytest

from nerva.backends.base import InferContext
from nerva.worker.process import worker_entry
from nerva.worker.proxy import WorkerProxy


class TestWorkerProxyStartAndLoad:
    """Start proxy + worker, load model, verify no exception."""

    async def test_start_and_load_model(self) -> None:
        socket_path = os.path.join(tempfile.mkdtemp(prefix="nerva-test-"), "test.sock")
        proxy = WorkerProxy(socket_path)
        proc = multiprocessing.Process(target=worker_entry, args=(socket_path,))
        proc.start()
        try:
            await proxy.start()
            await proxy.load_model(
                model_name="echo",
                model_class_path="tests.helpers:EchoModel",
                backend="pytorch",
                device="cpu",
            )
        finally:
            await proxy.shutdown()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join()
            await proxy.close()


class TestWorkerProxyInfer:
    """Infer via proxy with inline data."""

    async def test_infer_inline(self) -> None:
        socket_path = os.path.join(tempfile.mkdtemp(prefix="nerva-test-"), "test.sock")
        proxy = WorkerProxy(socket_path)
        proc = multiprocessing.Process(target=worker_entry, args=(socket_path,))
        proc.start()
        try:
            await proxy.start()
            await proxy.load_model(
                model_name="echo",
                model_class_path="tests.helpers:EchoModel",
                backend="pytorch",
                device="cpu",
            )
            ctx = InferContext(request_id="req-001", deadline_ms=30000)
            result = await proxy.infer({"value": 42}, ctx)
            assert result == {"echo": 42}
        finally:
            await proxy.shutdown()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join()
            await proxy.close()

    async def test_infer_multiple_sequential(self) -> None:
        socket_path = os.path.join(tempfile.mkdtemp(prefix="nerva-test-"), "test.sock")
        proxy = WorkerProxy(socket_path)
        proc = multiprocessing.Process(target=worker_entry, args=(socket_path,))
        proc.start()
        try:
            await proxy.start()
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
        finally:
            await proxy.shutdown()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join()
            await proxy.close()


class TestWorkerProxyErrorPropagation:
    """Verify error propagation from CrashModel."""

    async def test_infer_error_propagation(self) -> None:
        socket_path = os.path.join(tempfile.mkdtemp(prefix="nerva-test-"), "test.sock")
        proxy = WorkerProxy(socket_path)
        proc = multiprocessing.Process(target=worker_entry, args=(socket_path,))
        proc.start()
        try:
            await proxy.start()
            await proxy.load_model(
                model_name="crash",
                model_class_path="tests.helpers:CrashModel",
                backend="pytorch",
                device="cpu",
            )
            ctx = InferContext(request_id="req-crash", deadline_ms=30000)
            with pytest.raises(RuntimeError, match="INTERNAL"):
                await proxy.infer({"value": 1}, ctx)
        finally:
            await proxy.shutdown()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join()
            await proxy.close()


class TestWorkerProxyHealthCheck:
    """Verify health check returns True after model load."""

    async def test_health_check(self) -> None:
        socket_path = os.path.join(tempfile.mkdtemp(prefix="nerva-test-"), "test.sock")
        proxy = WorkerProxy(socket_path)
        proc = multiprocessing.Process(target=worker_entry, args=(socket_path,))
        proc.start()
        try:
            await proxy.start()
            await proxy.load_model(
                model_name="echo",
                model_class_path="tests.helpers:EchoModel",
                backend="pytorch",
                device="cpu",
            )
            healthy = await proxy.health_check()
            assert healthy is True
        finally:
            await proxy.shutdown()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join()
            await proxy.close()
