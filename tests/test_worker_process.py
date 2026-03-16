"""Tests for nerva.worker.process — Worker subprocess."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import tempfile
from typing import Any

import msgpack
import zmq
import zmq.asyncio

from nerva.backends.base import Backend, InferContext, ModelConfig
from nerva.worker.ipc import (
    AckStatus,
    Descriptor,
    MessageType,
    decode_message,
    encode_message,
)
from nerva.worker.process import worker_entry


def _make_socket_path(tmp_dir: str) -> str:
    """Return a socket path inside the given temp directory."""
    return os.path.join(tmp_dir, "worker.sock")


async def _recv_msg(socket: zmq.asyncio.Socket, timeout: float = 5.0) -> dict[str, Any]:
    """Receive and decode a DEALER message [b"", payload]."""
    parts: list[bytes] = await asyncio.wait_for(socket.recv_multipart(), timeout=timeout)
    raw = parts[1] if len(parts) >= 2 else parts[0]
    return decode_message(raw)


async def _send_msg(socket: zmq.asyncio.Socket, msg: dict[str, Any]) -> None:
    """Send a DEALER message [b"", payload]."""
    await socket.send_multipart([b"", encode_message(msg)])


async def _connect_dealer(
    ctx: zmq.asyncio.Context,
    sock_path: str,
    timeout: float = 5.0,
) -> zmq.asyncio.Socket:
    """Wait for worker to bind its ROUTER socket, then connect a DEALER and handshake."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not os.path.exists(sock_path):
        if loop.time() >= deadline:
            raise RuntimeError(f"Timed out waiting for worker socket: {sock_path}")
        await asyncio.sleep(0.05)

    socket: zmq.asyncio.Socket = ctx.socket(zmq.DEALER)
    socket.connect(f"ipc://{sock_path}")

    # Perform WORKER_CONNECT / WORKER_READY handshake.
    await _send_msg(socket, {"type": MessageType.WORKER_CONNECT.value})
    msg = await _recv_msg(socket, timeout=timeout)
    assert msg["type"] == MessageType.WORKER_READY.value, f"Expected WORKER_READY, got {msg}"
    return socket


class TestWorkerLoadModel:
    """Spawn worker, verify WORKER_CONNECT/WORKER_READY + LOAD_MODEL + ACK."""

    async def test_load_model_ack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            sock_path = _make_socket_path(tmp_dir)
            ctx = zmq.asyncio.Context()

            proc = multiprocessing.Process(target=worker_entry, args=(sock_path,))
            proc.start()
            socket = None
            try:
                socket = await _connect_dealer(ctx, sock_path)

                # Send LOAD_MODEL.
                await _send_msg(socket, {
                    "type": MessageType.LOAD_MODEL.value,
                    "model_name": "echo",
                    "model_class": "tests.helpers:EchoModel",
                    "backend": "pytorch",
                    "device": "cpu",
                })

                # Receive LOAD_MODEL_ACK.
                ack = await _recv_msg(socket)
                assert ack["type"] == MessageType.LOAD_MODEL_ACK.value
                assert ack["status"] == AckStatus.OK.value
                assert ack["model_name"] == "echo"
            finally:
                if socket is not None:
                    await _send_msg(socket, {"type": MessageType.SHUTDOWN.value})
                    socket.close(linger=0)
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=2)
                ctx.term()


class _CaptureOptionsBackend(Backend):
    """Backend used to verify LOAD_MODEL options are passed into ModelConfig."""

    last_config: ModelConfig | None = None

    async def load_model(self, config: ModelConfig) -> None:
        type(self).last_config = config

    async def unload_model(self) -> None:
        pass

    async def infer(
        self,
        inputs: dict[str, Any],
        context: InferContext,
        batch_meta: Any = None,
    ) -> dict[str, Any]:
        return {}

    async def infer_stream(self, inputs: dict[str, Any], context: InferContext):  # type: ignore[override]
        yield {}


class TestWorkerLoadModelOptions:
    async def test_options_are_forwarded_to_model_config(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        from nerva.worker.process import _WorkerLoop

        _CaptureOptionsBackend.last_config = None
        sent: list[dict[str, Any]] = []
        sent_client_ids: list[bytes] = []

        loop = _WorkerLoop(socket_path="/tmp/unused.sock")
        fake_client_id = b"test-client"

        async def _fake_send_to(client_id: bytes, msg: dict[str, Any]) -> None:
            sent_client_ids.append(client_id)
            sent.append(msg)

        monkeypatch.setattr("nerva.worker.process.import_path_to_class", lambda _p: object)
        monkeypatch.setattr(
            "nerva.worker.process.get_backend",
            lambda _name: _CaptureOptionsBackend,
        )
        monkeypatch.setattr(loop, "_send_to", _fake_send_to)

        await loop._handle_load_model(
            {
                "type": MessageType.LOAD_MODEL.value,
                "model_name": "echo",
                "model_class": "tests.helpers:EchoModel",
                "backend": "capture",
                "device": "cpu",
                "options": {"max_batch": 32, "timeout_ms": 2000},
            },
            fake_client_id,
        )

        assert _CaptureOptionsBackend.last_config is not None
        assert _CaptureOptionsBackend.last_config.backend_options == {
            "max_batch": 32,
            "timeout_ms": 2000,
        }
        assert sent[0]["type"] == MessageType.LOAD_MODEL_ACK.value
        assert sent[0]["status"] == AckStatus.OK.value
        assert sent_client_ids[0] == fake_client_id


class TestWorkerInfer:
    """Send INFER_SUBMIT with inline data, verify INFER_ACK."""

    async def test_infer_inline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            sock_path = _make_socket_path(tmp_dir)
            ctx = zmq.asyncio.Context()

            proc = multiprocessing.Process(target=worker_entry, args=(sock_path,))
            proc.start()
            socket = None
            try:
                socket = await _connect_dealer(ctx, sock_path)

                # Load model.
                await _send_msg(socket, {
                    "type": MessageType.LOAD_MODEL.value,
                    "model_name": "echo",
                    "model_class": "tests.helpers:EchoModel",
                    "backend": "pytorch",
                    "device": "cpu",
                })
                ack = await _recv_msg(socket)
                assert ack["status"] == AckStatus.OK.value

                # Send INFER_SUBMIT with inline data.
                input_data = {"value": 42}
                inline_bytes = msgpack.packb(input_data, use_bin_type=True)
                descriptor = Descriptor(
                    request_id="req-001",
                    node_id=0,
                    inline_data=inline_bytes,
                    length=len(inline_bytes),
                )

                await _send_msg(socket, {
                    "type": MessageType.INFER_SUBMIT.value,
                    "request_id": "req-001",
                    "descriptor": descriptor.to_dict(),
                    "deadline_ms": 30000,
                })

                # Receive INFER_ACK.
                infer_ack = await _recv_msg(socket)
                assert infer_ack["type"] == MessageType.INFER_ACK.value
                assert infer_ack["status"] == AckStatus.OK.value
                assert infer_ack["request_id"] == "req-001"

                # Decode output from descriptor inline_data.
                out_desc = Descriptor.from_dict(infer_ack["descriptor"])
                assert out_desc.is_inline
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

    async def test_infer_raw_bytes_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            sock_path = _make_socket_path(tmp_dir)
            ctx = zmq.asyncio.Context()

            proc = multiprocessing.Process(target=worker_entry, args=(sock_path,))
            proc.start()
            socket = None
            try:
                socket = await _connect_dealer(ctx, sock_path)

                await _send_msg(socket, {
                    "type": MessageType.LOAD_MODEL.value,
                    "model_name": "echo",
                    "model_class": "tests.helpers:EchoModel",
                    "backend": "pytorch",
                    "device": "cpu",
                })
                ack = await _recv_msg(socket)
                assert ack["status"] == AckStatus.OK.value

                payload = b"\x00\x01\x02\xff"
                descriptor = Descriptor(
                    request_id="req-bytes-001",
                    node_id=0,
                    inline_data=payload,
                    length=len(payload),
                    payload_codec="raw_bytes_v1",
                    input_key="value",
                )

                await _send_msg(socket, {
                    "type": MessageType.INFER_SUBMIT.value,
                    "request_id": "req-bytes-001",
                    "descriptor": descriptor.to_dict(),
                    "deadline_ms": 30000,
                })

                infer_ack = await _recv_msg(socket)
                assert infer_ack["type"] == MessageType.INFER_ACK.value
                assert infer_ack["status"] == AckStatus.OK.value
                assert infer_ack["request_id"] == "req-bytes-001"

                out_desc = Descriptor.from_dict(infer_ack["descriptor"])
                assert out_desc.is_inline
                assert out_desc.inline_data is not None
                output = msgpack.unpackb(out_desc.inline_data, raw=False)
                assert output == {"echo": payload}
            finally:
                if socket is not None:
                    await _send_msg(socket, {"type": MessageType.SHUTDOWN.value})
                    socket.close(linger=0)
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=2)
                ctx.term()


class TestWorkerHealthCheck:
    """Send HEALTH_CHECK, verify HEALTH_STATUS."""

    async def test_health_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            sock_path = _make_socket_path(tmp_dir)
            ctx = zmq.asyncio.Context()

            proc = multiprocessing.Process(target=worker_entry, args=(sock_path,))
            proc.start()
            socket = None
            try:
                socket = await _connect_dealer(ctx, sock_path)

                # Health check before model load — should be unhealthy.
                await _send_msg(socket, {"type": MessageType.HEALTH_CHECK.value})
                status = await _recv_msg(socket)
                assert status["type"] == MessageType.HEALTH_STATUS.value
                assert status["healthy"] is False

                # Load model.
                await _send_msg(socket, {
                    "type": MessageType.LOAD_MODEL.value,
                    "model_name": "echo",
                    "model_class": "tests.helpers:EchoModel",
                    "backend": "pytorch",
                    "device": "cpu",
                })
                ack = await _recv_msg(socket)
                assert ack["status"] == AckStatus.OK.value

                # Health check after load — should be healthy.
                await _send_msg(socket, {"type": MessageType.HEALTH_CHECK.value})
                status = await _recv_msg(socket)
                assert status["type"] == MessageType.HEALTH_STATUS.value
                assert status["healthy"] is True
            finally:
                if socket is not None:
                    await _send_msg(socket, {"type": MessageType.SHUTDOWN.value})
                    socket.close(linger=0)
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=2)
                ctx.term()


class TestWorkerDecodeRobustness:
    """Malformed control messages should not crash worker loop."""

    async def test_malformed_message_does_not_crash_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            sock_path = _make_socket_path(tmp_dir)
            ctx = zmq.asyncio.Context()

            proc = multiprocessing.Process(target=worker_entry, args=(sock_path,))
            proc.start()
            socket = None
            try:
                socket = await _connect_dealer(ctx, sock_path)

                # Send malformed payload (0xC1 is always invalid msgpack).
                await socket.send_multipart([b"", b"\xc1"])
                await asyncio.sleep(0.1)
                assert proc.is_alive()

                await _send_msg(socket, {"type": MessageType.HEALTH_CHECK.value})
                status = await _recv_msg(socket)
                assert status["type"] == MessageType.HEALTH_STATUS.value
                assert status["healthy"] is False
            finally:
                if socket is not None:
                    await _send_msg(socket, {"type": MessageType.SHUTDOWN.value})
                    socket.close(linger=0)
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=2)
                ctx.term()


class TestWorkerShutdown:
    """Send SHUTDOWN, verify process exits cleanly."""

    async def test_clean_shutdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            sock_path = _make_socket_path(tmp_dir)
            ctx = zmq.asyncio.Context()

            proc = multiprocessing.Process(target=worker_entry, args=(sock_path,))
            proc.start()
            socket = None
            try:
                socket = await _connect_dealer(ctx, sock_path)

                # Send SHUTDOWN.
                await _send_msg(socket, {"type": MessageType.SHUTDOWN.value})

                # Wait for process to exit.
                proc.join(timeout=5)
                assert not proc.is_alive()
                assert proc.exitcode == 0
            finally:
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=2)
                if socket is not None:
                    socket.close(linger=0)
                ctx.term()
