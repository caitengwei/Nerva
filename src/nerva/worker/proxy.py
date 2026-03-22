"""WorkerProxy — Master-side async RPC wrapper for a single Worker subprocess.

Connects a ZeroMQ DEALER socket to the Worker's ROUTER socket, performs a
WORKER_CONNECT/WORKER_READY handshake, and provides async methods for
load_model, infer, health_check, cancel, and shutdown.  Multiple proxies
(from different uvicorn workers) may connect to the same Worker ROUTER.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import os
import time
from collections import deque
from multiprocessing.shared_memory import SharedMemory
from typing import TYPE_CHECKING, Any

import msgpack
import zmq
import zmq.asyncio

from nerva.engine.shm_pool import IPC_CONTROL_INLINE_MAX_BYTES, ShmPoolExhausted
from nerva.observability.timing import AsyncTimingSink
from nerva.worker.ipc import (
    AckStatus,
    Descriptor,
    MessageType,
    decode_message,
    encode_message,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from nerva.backends.base import InferContext
    from nerva.engine.shm_pool import ShmPool, ShmSlot

logger = logging.getLogger(__name__)
MAX_RECENT_COMPLETED_REQUESTS = 2048


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

        # Pending infer futures keyed by request_id (unary).
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        # Pending stream queues keyed by request_id (streaming).
        self._pending_stream: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._request_pools: dict[str, ShmPool | None] = {}
        self._output_slots: dict[str, tuple[ShmPool, ShmSlot]] = {}
        self._recently_completed: deque[str] = deque()
        self._recently_completed_set: set[str] = set()

        # Special one-shot futures for non-request-id messages.
        self._load_model_future: asyncio.Future[dict[str, Any]] | None = None
        self._health_future: asyncio.Future[dict[str, Any]] | None = None

        self._recv_task: asyncio.Task[None] | None = None

        # Per-instance IPC timing log (enabled via NERVA_TIMING_LOG_DIR env var).
        self._model_name: str = ""
        self._timing_sink: AsyncTimingSink | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect DEALER socket to the worker's ROUTER, perform handshake.

        Waits for the worker's socket file to appear (the worker binds first),
        then connects and sends WORKER_CONNECT.  The worker replies with
        WORKER_READY addressed to this specific DEALER.
        """
        self._ctx = zmq.asyncio.Context()
        self._socket = self._ctx.socket(zmq.DEALER)

        # Wait for the worker to bind its ROUTER socket (socket file appears).
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 10.0
        while not os.path.exists(self._socket_path):
            if loop.time() >= deadline:
                raise RuntimeError(
                    f"Timed out waiting for worker socket: {self._socket_path}"
                )
            await asyncio.sleep(0.05)

        self._socket.connect(f"ipc://{self._socket_path}")

        # Start background recv loop before sending the handshake so the reply
        # is not lost.
        self._recv_task = asyncio.create_task(self._recv_loop())

        # Send WORKER_CONNECT and wait for WORKER_READY.
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._load_model_future = fut  # reuse one-shot future slot for handshake
        try:
            await self._send({"type": MessageType.WORKER_CONNECT.value})
            msg = await asyncio.wait_for(fut, timeout=10.0)
        finally:
            if self._load_model_future is fut:
                self._load_model_future = None

        worker_id = msg.get("worker_id", "unknown")
        logger.info("Worker %s connected", worker_id)

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

        if self._timing_sink is not None:
            await self._timing_sink.stop()
            self._timing_sink = None

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
        async_infer: bool = False,
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
                "async_infer": async_infer,
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

        self._model_name = model_name
        timing_log_dir = os.environ.get("NERVA_TIMING_LOG_DIR")
        if timing_log_dir and self._timing_sink is None:
            self._timing_sink = AsyncTimingSink()
            await self._timing_sink.start(timing_log_dir, f"nerva_proxy_{model_name}.log")

    async def infer(
        self,
        inputs: dict[str, Any],
        context: InferContext,
        shm_pool: ShmPool | None = None,
    ) -> dict[str, Any]:
        """Serialize inputs, send INFER_SUBMIT, wait for INFER_ACK, return output."""
        request_id = context.request_id
        t_ser_start = time.perf_counter()
        raw_bytes_input = self._extract_raw_bytes_input(inputs)
        if raw_bytes_input is None:
            input_key = None
            payload_codec = "msgpack_dict_v1"
            input_bytes = msgpack.packb(inputs, use_bin_type=True)
        else:
            input_key, input_bytes = raw_bytes_input
            payload_codec = "raw_bytes_v1"
        proxy_serialize_ms = round((time.perf_counter() - t_ser_start) * 1000, 3)
        shm_slot = None

        if len(input_bytes) <= IPC_CONTROL_INLINE_MAX_BYTES or shm_pool is None:
            # Inline path.
            descriptor = Descriptor(
                request_id=request_id,
                node_id=0,
                inline_data=input_bytes,
                length=len(input_bytes),
                payload_codec=payload_codec,
                input_key=input_key,
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
                payload_codec=payload_codec,
                input_key=input_key,
            )

        loop = asyncio.get_running_loop()
        if request_id in self._pending:
            raise RuntimeError(f"Duplicate in-flight request_id '{request_id}'")
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[request_id] = fut
        self._request_pools[request_id] = shm_pool

        t_send = time.perf_counter()
        try:
            await self._send({
                "type": MessageType.INFER_SUBMIT.value,
                "request_id": request_id,
                "descriptor": descriptor.to_dict(),
                "deadline_ms": context.deadline_ms,
            })

            ack = await asyncio.wait_for(fut, timeout=self._submit_timeout)
            ipc_round_trip_ms = round((time.perf_counter() - t_send) * 1000, 3)
        except TimeoutError:
            self._pending.pop(request_id, None)
            self._release_output_slot(request_id)
            with contextlib.suppress(Exception):
                await self.cancel(request_id, reason="submit timeout")
            raise RuntimeError(
                f"Timeout waiting for INFER_ACK for request '{request_id}'"
            ) from None
        finally:
            self._pending.pop(request_id, None)
            self._request_pools.pop(request_id, None)
            self._mark_request_completed(request_id)
            # Free SHM slot if allocated.
            if shm_slot is not None and shm_pool is not None:
                shm_pool.free(shm_slot)

        status = ack.get("status", "")
        if status == AckStatus.DEADLINE_EXCEEDED.value:
            self._release_output_slot(request_id)
            raise RuntimeError(f"Infer failed: [{status}] deadline exceeded")
        if status == AckStatus.ABORTED.value:
            self._release_output_slot(request_id)
            raise RuntimeError(f"Infer failed: [{status}] request was cancelled")
        if status != AckStatus.OK.value:
            error = ack.get("error", "unknown error")
            self._release_output_slot(request_id)
            raise RuntimeError(f"Infer failed: [{status}] {error}")

        # Deserialize output from descriptor.
        out_descriptor = Descriptor.from_dict(ack["descriptor"])
        t_des_start = time.perf_counter()
        result = self._decode_output(out_descriptor, request_id)
        proxy_deserialize_ms = round((time.perf_counter() - t_des_start) * 1000, 3)
        if self._timing_sink is not None:
            self._timing_sink.write({
                "event": "ipc_timing",
                "request_id": request_id,
                "model": self._model_name,
                "ipc_round_trip_ms": ipc_round_trip_ms,
                "proxy_serialize_ms": proxy_serialize_ms,
                "proxy_deserialize_ms": proxy_deserialize_ms,
            })
        return result

    async def infer_stream(
        self,
        inputs: dict[str, Any],
        context: InferContext,
        shm_pool: ShmPool | None = None,
    ) -> Any:  # AsyncIterator[dict[str, Any]]
        """Send INFER_SUBMIT with stream=True; yield INFER_ACK chunks until stream_done.

        Streaming chunks always use inline Descriptor (SHM disabled for stream path).
        """
        request_id = context.request_id
        raw_bytes_input = self._extract_raw_bytes_input(inputs)
        if raw_bytes_input is None:
            input_key = None
            payload_codec = "msgpack_dict_v1"
            input_bytes = msgpack.packb(inputs, use_bin_type=True)
        else:
            input_key, input_bytes = raw_bytes_input
            payload_codec = "raw_bytes_v1"

        shm_slot = None
        if len(input_bytes) <= IPC_CONTROL_INLINE_MAX_BYTES or shm_pool is None:
            descriptor = Descriptor(
                request_id=request_id,
                node_id=0,
                inline_data=input_bytes,
                length=len(input_bytes),
                payload_codec=payload_codec,
                input_key=input_key,
            )
        else:
            shm_slot = shm_pool.alloc(len(input_bytes))
            shm_pool.write(shm_slot, input_bytes)
            descriptor = Descriptor(
                request_id=request_id,
                node_id=0,
                shm_id=shm_slot.shm_name,
                offset=shm_slot.offset,
                length=len(input_bytes),
                payload_codec=payload_codec,
                input_key=input_key,
            )

        if request_id in self._pending or request_id in self._pending_stream:
            raise RuntimeError(f"Duplicate in-flight request_id '{request_id}'")

        chunk_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._pending_stream[request_id] = chunk_queue
        self._request_pools[request_id] = shm_pool

        try:
            await self._send({
                "type": MessageType.INFER_SUBMIT.value,
                "request_id": request_id,
                "descriptor": descriptor.to_dict(),
                "deadline_ms": context.deadline_ms,
                "stream": True,
            })

            while True:
                ack = await asyncio.wait_for(
                    chunk_queue.get(), timeout=self._submit_timeout
                )
                status = ack.get("status", "")
                stream_done = ack.get("stream_done", True)

                if status != AckStatus.OK.value:
                    error = ack.get("error", "unknown error")
                    raise RuntimeError(f"Infer failed: [{status}] {error}")

                if "descriptor" in ack:
                    out_descriptor = Descriptor.from_dict(ack["descriptor"])
                    yield self._decode_output(out_descriptor, request_id)

                if stream_done:
                    break
        finally:
            self._pending_stream.pop(request_id, None)
            self._request_pools.pop(request_id, None)
            self._mark_request_completed(request_id)
            self._release_output_slot(request_id)
            if shm_slot is not None and shm_pool is not None:
                shm_pool.free(shm_slot)

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
        """Thread-safe send via asyncio.Lock.  DEALER uses [b"", payload] framing."""
        assert self._socket is not None
        async with self._send_lock:
            await self._socket.send_multipart([b"", encode_message(msg)])

    async def _recv_loop(self) -> None:
        """Background task: receive messages and dispatch to pending futures.

        DEALER receives [b"", payload] from the ROUTER (empty delimiter + message).
        """
        assert self._socket is not None
        while True:
            try:
                parts: list[bytes] = await self._socket.recv_multipart()
                raw: bytes = parts[1] if len(parts) >= 2 else parts[0]
            except zmq.ZMQError:
                logger.debug("ZMQ recv error in recv_loop, exiting")
                self._fail_outstanding("UNAVAILABLE: worker disconnected")
                break
            except asyncio.CancelledError:
                break

            try:
                msg = decode_message(raw)
            except Exception:
                logger.exception("Failed to decode message in recv_loop")
                continue

            msg_type = msg.get("type", "")

            if msg_type == MessageType.WORKER_READY.value:
                # Handshake reply to our WORKER_CONNECT (reuses _load_model_future slot).
                if self._load_model_future is not None and not self._load_model_future.done():
                    self._load_model_future.set_result(msg)

            elif msg_type == MessageType.LOAD_MODEL_ACK.value:
                if self._load_model_future is not None and not self._load_model_future.done():
                    self._load_model_future.set_result(msg)

            elif msg_type == MessageType.HEALTH_STATUS.value:
                if self._health_future is not None and not self._health_future.done():
                    self._health_future.set_result(msg)

            elif msg_type == MessageType.INFER_ACK.value:
                request_id = msg.get("request_id", "")
                # Stream path takes priority (same request_id can produce N ACKs).
                q = self._pending_stream.get(request_id)
                if q is not None:
                    await q.put(msg)
                else:
                    fut = self._pending.get(request_id)
                    if fut is not None and not fut.done():
                        fut.set_result(msg)
                    else:
                        if request_id in self._recently_completed_set:
                            logger.debug("Late INFER_ACK for completed request '%s'", request_id)
                        else:
                            logger.warning("No pending future for request '%s'", request_id)

            elif msg_type == MessageType.SHM_ALLOC_REQUEST.value:
                await self._handle_shm_alloc_request(msg)

            else:
                logger.warning("Unexpected message type in recv_loop: %s", msg_type)

    def _fail_outstanding(self, reason: str) -> None:
        """Fail all pending RPC futures with a uniform runtime error."""
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError(reason))
        self._pending.clear()

        # Inject poison pill into all pending stream queues.
        for q in self._pending_stream.values():
            q.put_nowait({
                "status": AckStatus.UNAVAILABLE.value,
                "error": reason,
                "stream_done": True,
            })
        self._pending_stream.clear()

        self._request_pools.clear()

        for request_id in list(self._output_slots.keys()):
            self._release_output_slot(request_id)
        self._recently_completed.clear()
        self._recently_completed_set.clear()

        if self._load_model_future is not None and not self._load_model_future.done():
            self._load_model_future.set_exception(RuntimeError(reason))
        self._load_model_future = None

        if self._health_future is not None and not self._health_future.done():
            self._health_future.set_exception(RuntimeError(reason))
        self._health_future = None

    async def _handle_shm_alloc_request(self, msg: dict[str, Any]) -> None:
        """Allocate an output SHM slot for a worker request and respond."""
        request_id = msg.get("request_id", "")
        size = msg.get("size", 0)

        if not isinstance(size, int) or size <= 0:
            await self._send({
                "type": MessageType.SHM_ALLOC_RESPONSE.value,
                "request_id": request_id,
                "status": AckStatus.INVALID_ARGUMENT.value,
                "error": f"Invalid SHM allocation size: {size!r}",
            })
            return

        pool = self._request_pools.get(request_id)
        if pool is None:
            await self._send({
                "type": MessageType.SHM_ALLOC_RESPONSE.value,
                "request_id": request_id,
                "status": AckStatus.UNAVAILABLE.value,
                "error": "No shm_pool associated with request",
            })
            return

        # Release stale output allocation if worker retries for the same request.
        self._release_output_slot(request_id)

        try:
            slot = pool.alloc(size)
        except ShmPoolExhausted as exc:
            await self._send({
                "type": MessageType.SHM_ALLOC_RESPONSE.value,
                "request_id": request_id,
                "status": AckStatus.RESOURCE_EXHAUSTED.value,
                "error": str(exc),
            })
            return
        except Exception as exc:
            await self._send({
                "type": MessageType.SHM_ALLOC_RESPONSE.value,
                "request_id": request_id,
                "status": AckStatus.INTERNAL.value,
                "error": str(exc),
            })
            return

        self._output_slots[request_id] = (pool, slot)
        await self._send({
            "type": MessageType.SHM_ALLOC_RESPONSE.value,
            "request_id": request_id,
            "status": AckStatus.OK.value,
            "shm_id": slot.shm_name,
            "offset": slot.offset,
            "slot_size": slot.slot_size,
        })

    def _decode_output(
        self,
        descriptor: Descriptor,
        request_id: str,
    ) -> dict[str, Any]:
        """Decode output payload from inline bytes or output SHM slot."""
        if descriptor.is_inline:
            assert descriptor.inline_data is not None
            self._release_output_slot(request_id)
            return msgpack.unpackb(descriptor.inline_data, raw=False)  # type: ignore[no-any-return]

        if descriptor.shm_id is None:
            self._release_output_slot(request_id)
            raise RuntimeError("INFER_ACK descriptor has neither inline_data nor shm_id")

        slot_entry = self._output_slots.pop(request_id, None)
        if slot_entry is not None:
            pool, slot = slot_entry
            view = pool.read_view(slot, descriptor.length)
            try:
                return msgpack.unpackb(view, raw=False)  # type: ignore[no-any-return]
            finally:
                view.release()
                pool.free(slot)

        # Fallback for compatibility: read by shm_id directly if slot metadata is absent.
        logger.warning(
            "Missing output slot metadata for request '%s'; "
            "falling back to direct SharedMemory read",
            request_id,
        )
        shm = SharedMemory(name=descriptor.shm_id, create=False)
        try:
            buf = shm.buf
            assert buf is not None
            view = memoryview(buf)[descriptor.offset : descriptor.offset + descriptor.length]
            try:
                return msgpack.unpackb(view, raw=False)  # type: ignore[no-any-return]
            finally:
                view.release()
        finally:
            shm.close()

    @staticmethod
    def _extract_raw_bytes_input(inputs: dict[str, Any]) -> tuple[str, bytes] | None:
        """Return (input_key, payload) when inputs is a single raw-bytes field."""
        if len(inputs) != 1:
            return None

        input_key, value = next(iter(inputs.items()))
        if not isinstance(value, bytes):
            return None
        return input_key, value

    def _release_output_slot(self, request_id: str) -> None:
        """Free and forget output SHM slot allocated for *request_id*."""
        slot_entry = self._output_slots.pop(request_id, None)
        if slot_entry is None:
            return
        pool, slot = slot_entry
        pool.free(slot)

    def _mark_request_completed(self, request_id: str) -> None:
        """Track recently completed request IDs for late-ACK noise suppression."""
        if request_id in self._recently_completed_set:
            return

        self._recently_completed.append(request_id)
        self._recently_completed_set.add(request_id)
        while len(self._recently_completed) > MAX_RECENT_COMPLETED_REQUESTS:
            old = self._recently_completed.popleft()
            self._recently_completed_set.discard(old)


# ---------------------------------------------------------------------------
# MultiInstanceProxy
# ---------------------------------------------------------------------------


class MultiInstanceProxy:
    """Load-balancing proxy wrapping multiple WorkerProxy instances.

    Implements the InferableProxy protocol. The Executor treats it exactly
    like a single WorkerProxy — multi-instance dispatch is transparent.

    Dispatches infer() calls via round-robin across N WorkerProxy instances,
    each backed by an independent Worker subprocess.
    """

    def __init__(self, proxies: list[WorkerProxy]) -> None:
        if not proxies:
            raise ValueError("MultiInstanceProxy requires at least one proxy")
        self._proxies = proxies
        self._counter = itertools.count()

    async def infer(
        self,
        inputs: dict[str, Any],
        context: InferContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Round-robin dispatch to one of the underlying WorkerProxy instances."""
        idx = next(self._counter) % len(self._proxies)
        return await self._proxies[idx].infer(inputs, context, **kwargs)

    async def infer_stream(
        self,
        inputs: dict[str, Any],
        context: InferContext,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Round-robin dispatch to one proxy, yield its stream chunks."""
        idx = next(self._counter) % len(self._proxies)
        async for chunk in self._proxies[idx].infer_stream(inputs, context, **kwargs):
            yield chunk

    async def health_check(self, timeout: float = 3.0) -> bool:
        """Return True if any instance is healthy."""
        results = await asyncio.gather(
            *(p.health_check(timeout=timeout) for p in self._proxies),
            return_exceptions=True,
        )
        return any(r is True for r in results)

    async def shutdown(self) -> None:
        """Broadcast SHUTDOWN to all instances concurrently."""
        results = await asyncio.gather(
            *(p.shutdown() for p in self._proxies), return_exceptions=True
        )
        for i, r in enumerate(results):
            if isinstance(r, BaseException):
                logger.warning("multi_instance_shutdown_error instance=%d: %s", i, r)

    async def close(self) -> None:
        """Close all underlying proxies concurrently."""
        results = await asyncio.gather(
            *(p.close() for p in self._proxies), return_exceptions=True
        )
        for i, r in enumerate(results):
            if isinstance(r, BaseException):
                logger.warning("multi_instance_close_error instance=%d: %s", i, r)
