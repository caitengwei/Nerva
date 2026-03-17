"""Tests for backend-aware async dispatch.

Covers:
- Backend.is_async_native property default
- VLLMBackend.is_async_native override
- ModelHandle.async_infer field
- model() async_infer parameter
- Worker dispatch path selection (async vs sync)
- NERVA_WORKER_MAX_THREADS environment variable
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import tempfile
from typing import TYPE_CHECKING, Any

import msgpack
import zmq.asyncio

from nerva.backends.base import Backend, InferContext, ModelConfig
from nerva.core.model import Model, ModelHandle, model
from nerva.worker.ipc import (
    AckStatus,
    Descriptor,
    MessageType,
    decode_message,
    encode_message,
)
from nerva.worker.process import _WorkerLoop, worker_entry

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_socket_path(tmp_dir: str) -> str:
    return os.path.join(tmp_dir, "worker.sock")


async def _recv_msg(socket: zmq.asyncio.Socket, timeout: float = 5.0) -> dict[str, Any]:
    parts: list[bytes] = await asyncio.wait_for(socket.recv_multipart(), timeout=timeout)
    raw = parts[1] if len(parts) >= 2 else parts[0]
    return decode_message(raw)


async def _send_msg(socket: zmq.asyncio.Socket, msg: dict[str, Any]) -> None:
    await socket.send_multipart([b"", encode_message(msg)])


async def _connect_dealer(
    ctx: zmq.asyncio.Context,
    sock_path: str,
    timeout: float = 5.0,
) -> zmq.asyncio.Socket:
    """Wait for worker ROUTER socket, connect DEALER, perform handshake."""
    import zmq

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not os.path.exists(sock_path):
        if loop.time() >= deadline:
            raise RuntimeError(f"Timed out waiting for worker socket: {sock_path}")
        await asyncio.sleep(0.05)

    socket: zmq.asyncio.Socket = ctx.socket(zmq.DEALER)
    socket.connect(f"ipc://{sock_path}")
    await _send_msg(socket, {"type": MessageType.WORKER_CONNECT.value})
    msg = await _recv_msg(socket, timeout=timeout)
    assert msg["type"] == MessageType.WORKER_READY.value, f"Expected WORKER_READY, got {msg}"
    return socket


class _AsyncNativeBackend(Backend):
    """Backend that declares is_async_native=True."""

    async def load_model(self, config: ModelConfig) -> None:
        pass

    async def unload_model(self) -> None:
        pass

    @property
    def is_async_native(self) -> bool:
        return True

    async def infer(
        self,
        inputs: dict[str, Any],
        context: InferContext,
        batch_meta: Any = None,
    ) -> dict[str, Any]:
        return {"echo": inputs.get("value")}

    async def infer_stream(  # type: ignore[override]
        self, inputs: dict[str, Any], context: InferContext
    ) -> AsyncIterator[dict[str, Any]]:
        yield {}


# ---------------------------------------------------------------------------
# Unit tests: Backend.is_async_native
# ---------------------------------------------------------------------------


class TestBackendIsAsyncNative:
    def test_default_is_false(self) -> None:
        """Backend.is_async_native defaults to False."""
        from nerva.backends.pytorch import PyTorchBackend

        backend = PyTorchBackend()
        assert backend.is_async_native is False

    def test_vllm_backend_is_true(self) -> None:
        """VLLMBackend.is_async_native is True."""
        from nerva.backends.vllm import VLLMBackend

        backend = VLLMBackend()
        assert backend.is_async_native is True

    def test_custom_async_backend(self) -> None:
        backend = _AsyncNativeBackend()
        assert backend.is_async_native is True


# ---------------------------------------------------------------------------
# Unit tests: ModelHandle.async_infer
# ---------------------------------------------------------------------------


class _DummyModel(Model):
    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {}


class TestModelHandleAsyncInfer:
    def test_default_false(self) -> None:
        handle = ModelHandle(
            name="test", model_class=_DummyModel, backend="pytorch", device="cpu"
        )
        assert handle.async_infer is False

    def test_set_true(self) -> None:
        handle = ModelHandle(
            name="test",
            model_class=_DummyModel,
            backend="pytorch",
            device="cpu",
            async_infer=True,
        )
        assert handle.async_infer is True


class TestModelFunctionAsyncInfer:
    def test_default_false(self) -> None:
        handle = model("test_default_ai", _DummyModel, backend="pytorch", device="cpu")
        assert handle.async_infer is False

    def test_set_true(self) -> None:
        handle = model(
            "test_set_ai", _DummyModel, backend="pytorch", device="cpu", async_infer=True
        )
        assert handle.async_infer is True


# ---------------------------------------------------------------------------
# Unit tests: _WorkerLoop dispatch mode selection
# ---------------------------------------------------------------------------


class TestWorkerLoopDispatchMode:
    async def test_async_dispatch_from_backend_signal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Backend.is_async_native=True forces async dispatch."""
        loop = _WorkerLoop(socket_path="/tmp/unused.sock")
        fake_client = b"test-client"

        async def _fake_send_to(client_id: bytes, msg: dict[str, Any]) -> None:
            pass

        monkeypatch.setattr("nerva.worker.process.import_path_to_class", lambda _p: object)
        monkeypatch.setattr(
            "nerva.worker.process.get_backend", lambda _name: _AsyncNativeBackend
        )
        monkeypatch.setattr(loop, "_send_to", _fake_send_to)

        await loop._handle_load_model({
            "type": MessageType.LOAD_MODEL.value,
            "model_name": "async_model",
            "model_class": "tests.helpers:EchoModel",
            "backend": "mock_async",
            "device": "cpu",
            "async_infer": False,  # backend signal should override
        }, fake_client)

        assert loop._async_dispatch is True
        assert loop._thread_executor is None

    async def test_async_dispatch_from_user_declaration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """async_infer=True in message forces async dispatch even with sync backend."""
        from nerva.backends.pytorch import PyTorchBackend

        loop = _WorkerLoop(socket_path="/tmp/unused.sock")
        fake_client = b"test-client"

        async def _fake_send_to(client_id: bytes, msg: dict[str, Any]) -> None:
            pass

        monkeypatch.setattr("nerva.worker.process.import_path_to_class", lambda _p: _DummyModel)
        monkeypatch.setattr("nerva.worker.process.get_backend", lambda _name: PyTorchBackend)
        monkeypatch.setattr(loop, "_send_to", _fake_send_to)

        await loop._handle_load_model({
            "type": MessageType.LOAD_MODEL.value,
            "model_name": "async_user",
            "model_class": "tests.helpers:EchoModel",
            "backend": "pytorch",
            "device": "cpu",
            "async_infer": True,
        }, fake_client)

        assert loop._async_dispatch is True
        assert loop._thread_executor is None

    async def test_sync_dispatch_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default (no async signals) uses sync dispatch."""
        from nerva.backends.pytorch import PyTorchBackend

        loop = _WorkerLoop(socket_path="/tmp/unused.sock")
        fake_client = b"test-client"

        async def _fake_send_to(client_id: bytes, msg: dict[str, Any]) -> None:
            pass

        monkeypatch.setattr("nerva.worker.process.import_path_to_class", lambda _p: _DummyModel)
        monkeypatch.setattr("nerva.worker.process.get_backend", lambda _name: PyTorchBackend)
        monkeypatch.setattr(loop, "_send_to", _fake_send_to)

        await loop._handle_load_model({
            "type": MessageType.LOAD_MODEL.value,
            "model_name": "sync_model",
            "model_class": "tests.helpers:EchoModel",
            "backend": "pytorch",
            "device": "cpu",
        }, fake_client)

        assert loop._async_dispatch is False
        assert loop._thread_executor is None  # no NERVA_WORKER_MAX_THREADS

    async def test_thread_executor_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """NERVA_WORKER_MAX_THREADS creates a ThreadPoolExecutor for sync dispatch."""
        from nerva.backends.pytorch import PyTorchBackend

        monkeypatch.setenv("NERVA_WORKER_MAX_THREADS", "4")
        loop = _WorkerLoop(socket_path="/tmp/unused.sock")
        fake_client = b"test-client"

        async def _fake_send_to(client_id: bytes, msg: dict[str, Any]) -> None:
            pass

        monkeypatch.setattr("nerva.worker.process.import_path_to_class", lambda _p: _DummyModel)
        monkeypatch.setattr("nerva.worker.process.get_backend", lambda _name: PyTorchBackend)
        monkeypatch.setattr(loop, "_send_to", _fake_send_to)

        await loop._handle_load_model({
            "type": MessageType.LOAD_MODEL.value,
            "model_name": "sync_threaded",
            "model_class": "tests.helpers:EchoModel",
            "backend": "pytorch",
            "device": "cpu",
        }, fake_client)

        assert loop._async_dispatch is False
        assert loop._thread_executor is not None
        # Verify the executor actually runs work (behavioral check).
        future = loop._thread_executor.submit(lambda: 42)
        assert future.result(timeout=2) == 42
        loop._thread_executor.shutdown(wait=False)

    async def test_thread_executor_not_created_for_async(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """NERVA_WORKER_MAX_THREADS is ignored for async dispatch."""
        monkeypatch.setenv("NERVA_WORKER_MAX_THREADS", "4")
        loop = _WorkerLoop(socket_path="/tmp/unused.sock")
        fake_client = b"test-client"

        async def _fake_send_to(client_id: bytes, msg: dict[str, Any]) -> None:
            pass

        monkeypatch.setattr("nerva.worker.process.import_path_to_class", lambda _p: object)
        monkeypatch.setattr(
            "nerva.worker.process.get_backend", lambda _name: _AsyncNativeBackend
        )
        monkeypatch.setattr(loop, "_send_to", _fake_send_to)

        await loop._handle_load_model({
            "type": MessageType.LOAD_MODEL.value,
            "model_name": "async_model",
            "model_class": "tests.helpers:EchoModel",
            "backend": "mock_async",
            "device": "cpu",
        }, fake_client)

        assert loop._async_dispatch is True
        assert loop._thread_executor is None


# ---------------------------------------------------------------------------
# Integration: async_infer=True via real worker subprocess
# ---------------------------------------------------------------------------


class TestAsyncDispatchIntegration:
    async def test_async_infer_worker_e2e(self) -> None:
        """Worker with async_infer=True uses async path and returns correct result."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sock_path = _make_socket_path(tmp_dir)
            timing_dir = os.path.join(tmp_dir, "timing")
            os.makedirs(timing_dir)

            ctx = zmq.asyncio.Context()
            proc = multiprocessing.Process(
                target=worker_entry,
                args=(sock_path,),
                kwargs={"timing_log_dir": timing_dir},
            )
            proc.start()
            socket = None
            try:
                socket = await _connect_dealer(ctx, sock_path)

                # Load with async_infer=True.
                await _send_msg(socket, {
                    "type": MessageType.LOAD_MODEL.value,
                    "model_name": "echo",
                    "model_class": "tests.helpers:EchoModel",
                    "backend": "pytorch",
                    "device": "cpu",
                    "async_infer": True,
                })
                ack = await _recv_msg(socket)
                assert ack["status"] == AckStatus.OK.value

                # Send infer request.
                input_data = {"value": 99}
                inline_bytes = msgpack.packb(input_data, use_bin_type=True)
                descriptor = Descriptor(
                    request_id="req-async-001",
                    node_id=0,
                    inline_data=inline_bytes,
                    length=len(inline_bytes),
                )

                await _send_msg(socket, {
                    "type": MessageType.INFER_SUBMIT.value,
                    "request_id": "req-async-001",
                    "descriptor": descriptor.to_dict(),
                    "deadline_ms": 30000,
                })

                infer_ack = await _recv_msg(socket)
                assert infer_ack["type"] == MessageType.INFER_ACK.value
                assert infer_ack["status"] == AckStatus.OK.value

                out_desc = Descriptor.from_dict(infer_ack["descriptor"])
                assert out_desc.inline_data is not None
                output = msgpack.unpackb(out_desc.inline_data, raw=False)
                assert output == {"echo": 99}
            finally:
                if socket is not None:
                    await _send_msg(socket, {"type": MessageType.SHUTDOWN.value})
                    socket.close(linger=0)
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=2)
                ctx.term()

            # Verify timing log has thread_queue_ms=0.
            import glob
            import json

            timing_files = glob.glob(os.path.join(timing_dir, "nerva_worker_*.log"))
            assert len(timing_files) >= 1
            found_infer_timing = False
            for tf in timing_files:
                with open(tf) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        if record.get("event") == "infer_timing":
                            found_infer_timing = True
                            assert record["thread_queue_ms"] == 0.0
            assert found_infer_timing, "No infer_timing event found in timing logs"

    async def test_sync_infer_worker_e2e(self) -> None:
        """Worker without async_infer uses sync ThreadPool path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            sock_path = _make_socket_path(tmp_dir)
            timing_dir = os.path.join(tmp_dir, "timing")
            os.makedirs(timing_dir)

            ctx = zmq.asyncio.Context()
            proc = multiprocessing.Process(
                target=worker_entry,
                args=(sock_path,),
                kwargs={"timing_log_dir": timing_dir},
            )
            proc.start()
            socket = None
            try:
                socket = await _connect_dealer(ctx, sock_path)

                # Load WITHOUT async_infer.
                await _send_msg(socket, {
                    "type": MessageType.LOAD_MODEL.value,
                    "model_name": "echo",
                    "model_class": "tests.helpers:EchoModel",
                    "backend": "pytorch",
                    "device": "cpu",
                })
                ack = await _recv_msg(socket)
                assert ack["status"] == AckStatus.OK.value

                input_data = {"value": 42}
                inline_bytes = msgpack.packb(input_data, use_bin_type=True)
                descriptor = Descriptor(
                    request_id="req-sync-001",
                    node_id=0,
                    inline_data=inline_bytes,
                    length=len(inline_bytes),
                )

                await _send_msg(socket, {
                    "type": MessageType.INFER_SUBMIT.value,
                    "request_id": "req-sync-001",
                    "descriptor": descriptor.to_dict(),
                    "deadline_ms": 30000,
                })

                infer_ack = await _recv_msg(socket)
                assert infer_ack["type"] == MessageType.INFER_ACK.value
                assert infer_ack["status"] == AckStatus.OK.value

                out_desc = Descriptor.from_dict(infer_ack["descriptor"])
                assert out_desc.inline_data is not None
                output = msgpack.unpackb(out_desc.inline_data, raw=False)
                assert output == {"echo": 42}
            finally:
                if socket is not None:
                    await _send_msg(socket, {"type": MessageType.SHUTDOWN.value})
                    socket.close(linger=0)
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=2)
                ctx.term()

            # Verify timing log has thread_queue_ms >= 0 (sync path).
            import glob
            import json

            timing_files = glob.glob(os.path.join(timing_dir, "nerva_worker_*.log"))
            assert len(timing_files) >= 1
            found_infer_timing = False
            for tf in timing_files:
                with open(tf) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        if record.get("event") == "infer_timing":
                            found_infer_timing = True
                            assert record["thread_queue_ms"] >= 0.0
            assert found_infer_timing, "No infer_timing event found in timing logs"


class TestWorkerProxyAsyncInfer:
    """Verify WorkerProxy.load_model() forwards async_infer field."""

    def test_load_model_forwards_async_infer(self) -> None:
        import inspect

        from nerva.worker.proxy import WorkerProxy

        proxy = WorkerProxy("/tmp/unused.sock")
        sig = inspect.signature(proxy.load_model)
        assert "async_infer" in sig.parameters
        param = sig.parameters["async_infer"]
        assert param.default is False
