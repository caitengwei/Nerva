"""WorkerProxy — Master-side async RPC wrapper for a single Worker subprocess.

Communicates with a Worker process over a ZeroMQ PAIR socket.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any

import msgpack
import zmq
import zmq.asyncio

from nerva.engine.shm_pool import IPC_CONTROL_INLINE_MAX_BYTES
from nerva.worker.ipc import (
    AckStatus,
    Descriptor,
    MessageType,
    decode_message,
    encode_message,
)

if TYPE_CHECKING:
    from nerva.backends.base import InferContext
    from nerva.engine.shm_pool import ShmPool

logger = logging.getLogger(__name__)


class WorkerProxy:
    """Master-side async RPC wrapper for a single Worker subprocess.

    Binds a ZeroMQ PAIR socket, waits for WORKER_READY, and provides
    async methods for load_model, infer, health_check, cancel, and shutdown.
    """

    def __init__(self, socket_path: str, submit_timeout: float = 5.0) -> None:
        self._socket_path = socket_path
        self._submit_timeout = submit_timeout

        self._ctx: zmq.asyncio.Context | None = None
        self._socket: zmq.asyncio.Socket | None = None
        self._send_lock = asyncio.Lock()

        # Pending infer futures keyed by request_id.
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}

        # Special one-shot futures for non-request-id messages.
        self._load_model_future: asyncio.Future[dict[str, Any]] | None = None
        self._health_future: asyncio.Future[dict[str, Any]] | None = None

        self._recv_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Bind PAIR socket, wait for WORKER_READY, start background recv loop."""
        self._ctx = zmq.asyncio.Context()
        self._socket = self._ctx.socket(zmq.PAIR)
        self._socket.bind(f"ipc://{self._socket_path}")

        # Wait for WORKER_READY.
        raw: bytes = await asyncio.wait_for(self._socket.recv(), timeout=10.0)
        msg = decode_message(raw)
        if msg.get("type") != MessageType.WORKER_READY.value:
            raise RuntimeError(f"Expected WORKER_READY, got {msg.get('type')}")

        worker_id = msg.get("worker_id", "unknown")
        logger.info("Worker %s connected", worker_id)

        # Start background recv loop.
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def close(self) -> None:
        """Close socket, cancel recv loop, terminate ZMQ context."""
        if self._recv_task is not None and not self._recv_task.done():
            self._recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._recv_task
            self._recv_task = None

        self._fail_outstanding("WorkerProxy closed")

        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        if self._ctx is not None:
            self._ctx.term()
            self._ctx = None

    # ------------------------------------------------------------------
    # RPC methods
    # ------------------------------------------------------------------

    async def load_model(
        self,
        model_name: str,
        model_class_path: str,
        backend: str = "pytorch",
        device: str = "cpu",
        options: dict[str, Any] | None = None,
    ) -> None:
        """Send LOAD_MODEL, wait for ACK. Raise RuntimeError on failure."""
        if self._load_model_future is not None and not self._load_model_future.done():
            raise RuntimeError("load_model already in progress")

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._load_model_future = fut

        try:
            await self._send({
                "type": MessageType.LOAD_MODEL.value,
                "model_name": model_name,
                "model_class": model_class_path,
                "backend": backend,
                "device": device,
                "options": options or {},
            })
            ack = await asyncio.wait_for(fut, timeout=self._submit_timeout * 6)
        except TimeoutError:
            raise RuntimeError(f"Timeout waiting for LOAD_MODEL_ACK for '{model_name}'") from None
        finally:
            if self._load_model_future is fut:
                self._load_model_future = None

        status = ack.get("status", "")
        if status != AckStatus.OK.value:
            error = ack.get("error", "unknown error")
            raise RuntimeError(f"load_model failed: [{status}] {error}")

    async def infer(
        self,
        inputs: dict[str, Any],
        context: InferContext,
        shm_pool: ShmPool | None = None,
    ) -> dict[str, Any]:
        """Serialize inputs, send INFER_SUBMIT, wait for INFER_ACK, return output."""
        request_id = context.request_id
        input_bytes = msgpack.packb(inputs, use_bin_type=True)
        shm_slot = None

        if len(input_bytes) <= IPC_CONTROL_INLINE_MAX_BYTES or shm_pool is None:
            # Inline path.
            descriptor = Descriptor(
                request_id=request_id,
                node_id=0,
                inline_data=input_bytes,
                length=len(input_bytes),
            )
        else:
            # SHM path.
            shm_slot = shm_pool.alloc(len(input_bytes))
            shm_pool.write(shm_slot, input_bytes)
            descriptor = Descriptor(
                request_id=request_id,
                node_id=0,
                shm_id=shm_slot.shm_name,
                offset=shm_slot.offset,
                length=len(input_bytes),
            )

        loop = asyncio.get_running_loop()
        if request_id in self._pending:
            raise RuntimeError(f"Duplicate in-flight request_id '{request_id}'")
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[request_id] = fut

        try:
            await self._send({
                "type": MessageType.INFER_SUBMIT.value,
                "request_id": request_id,
                "descriptor": descriptor.to_dict(),
                "deadline_ms": context.deadline_ms,
            })

            ack = await asyncio.wait_for(fut, timeout=self._submit_timeout)
        except TimeoutError:
            self._pending.pop(request_id, None)
            raise RuntimeError(
                f"Timeout waiting for INFER_ACK for request '{request_id}'"
            ) from None
        finally:
            self._pending.pop(request_id, None)
            # Free SHM slot if allocated.
            if shm_slot is not None and shm_pool is not None:
                shm_pool.free(shm_slot)

        status = ack.get("status", "")
        if status != AckStatus.OK.value:
            error = ack.get("error", "unknown error")
            raise RuntimeError(f"Infer failed: [{status}] {error}")

        # Deserialize output from descriptor.
        out_descriptor = Descriptor.from_dict(ack["descriptor"])
        assert out_descriptor.inline_data is not None
        return msgpack.unpackb(out_descriptor.inline_data, raw=False)  # type: ignore[no-any-return]

    async def cancel(self, request_id: str, reason: str = "") -> None:
        """Send CANCEL (best-effort)."""
        await self._send({
            "type": MessageType.CANCEL.value,
            "request_id": request_id,
            "reason": reason,
        })

    async def health_check(self, timeout: float = 3.0) -> bool:
        """Send HEALTH_CHECK, return True if healthy."""
        if self._health_future is not None and not self._health_future.done():
            raise RuntimeError("health_check already in progress")

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._health_future = fut

        try:
            await self._send({"type": MessageType.HEALTH_CHECK.value})
            status = await asyncio.wait_for(fut, timeout=timeout)
        except TimeoutError:
            return False
        finally:
            if self._health_future is fut:
                self._health_future = None
        return bool(status.get("healthy", False))

    async def shutdown(self) -> None:
        """Send SHUTDOWN to worker."""
        try:
            await self._send({"type": MessageType.SHUTDOWN.value})
        except Exception:
            logger.debug("Failed to send SHUTDOWN (socket may already be closed)")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _send(self, msg: dict[str, Any]) -> None:
        """Thread-safe send via asyncio.Lock."""
        assert self._socket is not None
        async with self._send_lock:
            await self._socket.send(encode_message(msg))

    async def _recv_loop(self) -> None:
        """Background task: receive messages and dispatch to pending futures."""
        assert self._socket is not None
        while True:
            try:
                raw: bytes = await self._socket.recv()
            except zmq.ZMQError:
                logger.debug("ZMQ recv error in recv_loop, exiting")
                self._fail_outstanding("Worker disconnected")
                break
            except asyncio.CancelledError:
                break

            try:
                msg = decode_message(raw)
            except Exception:
                logger.exception("Failed to decode message in recv_loop")
                continue

            msg_type = msg.get("type", "")

            if msg_type == MessageType.LOAD_MODEL_ACK.value:
                if self._load_model_future is not None and not self._load_model_future.done():
                    self._load_model_future.set_result(msg)

            elif msg_type == MessageType.HEALTH_STATUS.value:
                if self._health_future is not None and not self._health_future.done():
                    self._health_future.set_result(msg)

            elif msg_type == MessageType.INFER_ACK.value:
                request_id = msg.get("request_id", "")
                fut = self._pending.get(request_id)
                if fut is not None and not fut.done():
                    fut.set_result(msg)
                else:
                    logger.warning("No pending future for request '%s'", request_id)

            else:
                logger.warning("Unexpected message type in recv_loop: %s", msg_type)

    def _fail_outstanding(self, reason: str) -> None:
        """Fail all pending RPC futures with a uniform runtime error."""
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError(reason))
        self._pending.clear()

        if self._load_model_future is not None and not self._load_model_future.done():
            self._load_model_future.set_exception(RuntimeError(reason))
        self._load_model_future = None

        if self._health_future is not None and not self._health_future.done():
            self._health_future.set_exception(RuntimeError(reason))
        self._health_future = None
