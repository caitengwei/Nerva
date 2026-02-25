"""Worker subprocess — runs in a separate process, connects to Master via ZeroMQ PAIR.

Entry point: ``worker_entry(socket_path)`` is passed to ``multiprocessing.Process(target=...)``.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

import msgpack
import zmq
import zmq.asyncio

from nerva.backends.base import Backend, InferContext, ModelConfig
from nerva.backends.registry import get_backend
from nerva.worker.ipc import (
    AckStatus,
    Descriptor,
    MessageType,
    decode_message,
    encode_message,
    import_path_to_class,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal worker loop
# ---------------------------------------------------------------------------


class _WorkerLoop:
    """Async event loop running inside the worker subprocess."""

    def __init__(self, socket_path: str) -> None:
        self._socket_path = socket_path
        self._backend: Backend | None = None
        self._running = False
        self._inflight: dict[str, asyncio.Task[None]] = {}
        self._send_lock: asyncio.Lock | None = None
        self._ctx: zmq.asyncio.Context | None = None
        self._socket: zmq.asyncio.Socket | None = None

    # -- public entry -------------------------------------------------------

    async def run(self) -> None:
        """Connect to master, send WORKER_READY, and enter the recv loop."""
        self._send_lock = asyncio.Lock()
        self._ctx = zmq.asyncio.Context()
        self._socket = self._ctx.socket(zmq.PAIR)
        self._socket.connect(f"ipc://{self._socket_path}")
        self._running = True

        try:
            # Announce readiness.
            await self._send(
                {"type": MessageType.WORKER_READY.value, "worker_id": str(uuid.uuid4())}
            )

            while self._running:
                try:
                    raw: bytes = await asyncio.wait_for(self._socket.recv(), timeout=1.0)
                except TimeoutError:
                    continue

                msg = decode_message(raw)
                msg_type = msg.get("type", "")

                if msg_type == MessageType.LOAD_MODEL.value:
                    await self._handle_load_model(msg)
                elif msg_type == MessageType.INFER_SUBMIT.value:
                    task = asyncio.create_task(self._handle_infer(msg))
                    request_id = msg.get("request_id", "")
                    self._inflight[request_id] = task
                    task.add_done_callback(self._make_cleanup_cb(request_id))
                elif msg_type == MessageType.HEALTH_CHECK.value:
                    await self._handle_health_check(msg)
                elif msg_type == MessageType.CANCEL.value:
                    self._handle_cancel(msg)
                elif msg_type == MessageType.SHUTDOWN.value:
                    logger.info("Received SHUTDOWN, exiting worker loop")
                    self._running = False
                else:
                    logger.warning("Unknown message type: %s", msg_type)
        finally:
            await self._cleanup()

    def _make_cleanup_cb(self, request_id: str) -> Any:
        """Return a done-callback that removes the task from _inflight."""
        def _cb(_t: asyncio.Task[None]) -> None:
            self._inflight.pop(request_id, None)
        return _cb

    # -- message handlers ---------------------------------------------------

    async def _handle_load_model(self, msg: dict[str, Any]) -> None:
        """Import model class, create Backend, load model, send ACK."""
        model_name: str = msg.get("model_name", "unknown")
        try:
            model_class_path: str = msg["model_class"]
            backend_name: str = msg.get("backend", "pytorch")
            device: str = msg.get("device", "cpu")

            model_class = import_path_to_class(model_class_path)
            backend_cls = get_backend(backend_name)
            backend = backend_cls()

            config = ModelConfig(
                model_name=model_name,
                model_class=model_class,
                device=device,
            )
            await backend.load_model(config)
            self._backend = backend

            await self._send({
                "type": MessageType.LOAD_MODEL_ACK.value,
                "model_name": model_name,
                "status": AckStatus.OK.value,
            })
            logger.info("Model '%s' loaded successfully", model_name)
        except Exception as exc:
            logger.exception("Failed to load model '%s'", model_name)
            await self._send({
                "type": MessageType.LOAD_MODEL_ACK.value,
                "model_name": model_name,
                "status": AckStatus.INTERNAL.value,
                "error": str(exc),
            })

    async def _handle_infer(self, msg: dict[str, Any]) -> None:
        """Deserialize inputs, run inference, send INFER_ACK."""
        request_id: str = msg.get("request_id", "")
        try:
            if self._backend is None:
                raise RuntimeError("No model loaded")

            # Deserialize inputs from descriptor.
            descriptor_dict: dict[str, Any] = msg["descriptor"]
            descriptor = Descriptor.from_dict(descriptor_dict)
            inputs = self._read_inputs(descriptor)

            context = InferContext(
                request_id=request_id,
                deadline_ms=msg.get("deadline_ms", 30000),
            )

            # Run inference in a thread to avoid blocking the event loop.
            output = await asyncio.to_thread(self._run_infer_sync, inputs, context)

            # Serialize output as inline data.
            output_bytes = msgpack.packb(output, use_bin_type=True)

            await self._send({
                "type": MessageType.INFER_ACK.value,
                "request_id": request_id,
                "status": AckStatus.OK.value,
                "descriptor": Descriptor(
                    request_id=request_id,
                    node_id=0,
                    inline_data=output_bytes,
                    length=len(output_bytes),
                ).to_dict(),
            })
        except Exception as exc:
            logger.exception("Inference failed for request '%s'", request_id)
            await self._send({
                "type": MessageType.INFER_ACK.value,
                "request_id": request_id,
                "status": AckStatus.INTERNAL.value,
                "error": str(exc),
            })

    def _run_infer_sync(self, inputs: dict[str, Any], context: InferContext) -> dict[str, Any]:
        """Blocking wrapper — runs ``backend.infer()`` via ``asyncio.run()``."""
        assert self._backend is not None
        return asyncio.run(self._backend.infer(inputs, context))

    def _read_inputs(self, descriptor: Descriptor) -> dict[str, Any]:
        """Read inputs from either inline data or SHM."""
        if descriptor.is_inline:
            assert descriptor.inline_data is not None
            return msgpack.unpackb(descriptor.inline_data, raw=False)  # type: ignore[no-any-return]

        # SHM path.
        if descriptor.shm_id is None:
            raise ValueError("Descriptor has neither inline_data nor shm_id")

        from multiprocessing.shared_memory import SharedMemory

        shm = SharedMemory(name=descriptor.shm_id, create=False)
        try:
            buf = shm.buf
            assert buf is not None
            raw = bytes(buf[descriptor.offset : descriptor.offset + descriptor.length])
            return msgpack.unpackb(raw, raw=False)  # type: ignore[no-any-return]
        finally:
            shm.close()

    async def _handle_health_check(self, msg: dict[str, Any]) -> None:
        """Respond with HEALTH_STATUS."""
        healthy = self._backend.health_check() if self._backend is not None else False
        await self._send({
            "type": MessageType.HEALTH_STATUS.value,
            "healthy": healthy,
        })

    def _handle_cancel(self, msg: dict[str, Any]) -> None:
        """Cancel an inflight request by setting context.cancelled (best-effort)."""
        request_id = msg.get("request_id", "")
        task = self._inflight.get(request_id)
        if task is not None and not task.done():
            task.cancel()
            logger.info("Cancelled request '%s'", request_id)
        else:
            logger.warning("No inflight request '%s' to cancel", request_id)

    # -- helpers ------------------------------------------------------------

    async def _send(self, msg: dict[str, Any]) -> None:
        """Thread-safe send via asyncio.Lock."""
        assert self._send_lock is not None
        assert self._socket is not None
        async with self._send_lock:
            await self._socket.send(encode_message(msg))

    async def _cleanup(self) -> None:
        """Close socket, terminate context, unload backend."""
        # Cancel inflight tasks.
        for task in self._inflight.values():
            task.cancel()
        if self._inflight:
            await asyncio.gather(*self._inflight.values(), return_exceptions=True)
        self._inflight.clear()

        # Unload backend.
        if self._backend is not None:
            try:
                await self._backend.unload_model()
            except Exception:
                logger.exception("Error unloading backend")
            self._backend = None

        # Close ZMQ.
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        if self._ctx is not None:
            self._ctx.term()
            self._ctx = None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def worker_entry(socket_path: str) -> None:
    """Entry point for ``multiprocessing.Process(target=worker_entry, args=(path,))``."""
    loop = _WorkerLoop(socket_path)
    asyncio.run(loop.run())
