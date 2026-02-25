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
    """Receive and decode a message with timeout."""
    raw: bytes = await asyncio.wait_for(socket.recv(), timeout=timeout)
    return decode_message(raw)


async def _send_msg(socket: zmq.asyncio.Socket, msg: dict[str, Any]) -> None:
    """Encode and send a message."""
    await socket.send(encode_message(msg))


class TestWorkerLoadModel:
    """Spawn worker, verify WORKER_READY + LOAD_MODEL + ACK."""

    async def test_load_model_ack(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            sock_path = _make_socket_path(tmp_dir)
            ctx = zmq.asyncio.Context()
            socket = ctx.socket(zmq.PAIR)
            socket.bind(f"ipc://{sock_path}")

            proc = multiprocessing.Process(target=worker_entry, args=(sock_path,))
            proc.start()
            try:
                # 1. Receive WORKER_READY.
                msg = await _recv_msg(socket)
                assert msg["type"] == MessageType.WORKER_READY.value
                assert "worker_id" in msg

                # 2. Send LOAD_MODEL.
                await _send_msg(socket, {
                    "type": MessageType.LOAD_MODEL.value,
                    "model_name": "echo",
                    "model_class": "tests.helpers:EchoModel",
                    "backend": "pytorch",
                    "device": "cpu",
                })

                # 3. Receive LOAD_MODEL_ACK.
                ack = await _recv_msg(socket)
                assert ack["type"] == MessageType.LOAD_MODEL_ACK.value
                assert ack["status"] == AckStatus.OK.value
                assert ack["model_name"] == "echo"
            finally:
                await _send_msg(socket, {"type": MessageType.SHUTDOWN.value})
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=2)
                socket.close(linger=0)
                ctx.term()


class TestWorkerInfer:
    """Send INFER_SUBMIT with inline data, verify INFER_ACK."""

    async def test_infer_inline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            sock_path = _make_socket_path(tmp_dir)
            ctx = zmq.asyncio.Context()
            socket = ctx.socket(zmq.PAIR)
            socket.bind(f"ipc://{sock_path}")

            proc = multiprocessing.Process(target=worker_entry, args=(sock_path,))
            proc.start()
            try:
                # Wait for WORKER_READY.
                msg = await _recv_msg(socket)
                assert msg["type"] == MessageType.WORKER_READY.value

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
                await _send_msg(socket, {"type": MessageType.SHUTDOWN.value})
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=2)
                socket.close(linger=0)
                ctx.term()


class TestWorkerHealthCheck:
    """Send HEALTH_CHECK, verify HEALTH_STATUS."""

    async def test_health_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            sock_path = _make_socket_path(tmp_dir)
            ctx = zmq.asyncio.Context()
            socket = ctx.socket(zmq.PAIR)
            socket.bind(f"ipc://{sock_path}")

            proc = multiprocessing.Process(target=worker_entry, args=(sock_path,))
            proc.start()
            try:
                # Wait for WORKER_READY.
                msg = await _recv_msg(socket)
                assert msg["type"] == MessageType.WORKER_READY.value

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
                await _send_msg(socket, {"type": MessageType.SHUTDOWN.value})
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=2)
                socket.close(linger=0)
                ctx.term()


class TestWorkerShutdown:
    """Send SHUTDOWN, verify process exits cleanly."""

    async def test_clean_shutdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            sock_path = _make_socket_path(tmp_dir)
            ctx = zmq.asyncio.Context()
            socket = ctx.socket(zmq.PAIR)
            socket.bind(f"ipc://{sock_path}")

            proc = multiprocessing.Process(target=worker_entry, args=(sock_path,))
            proc.start()
            try:
                # Wait for WORKER_READY.
                msg = await _recv_msg(socket)
                assert msg["type"] == MessageType.WORKER_READY.value

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
                socket.close(linger=0)
                ctx.term()
