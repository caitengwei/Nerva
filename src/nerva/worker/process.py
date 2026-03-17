"""Worker subprocess — runs in a separate process, binds a ZeroMQ ROUTER socket.

Multiple WorkerProxy instances (from different uvicorn workers) can connect to the
same Nerva worker via DEALER sockets.  The ROUTER socket routes replies back to the
correct DEALER using the identity frame injected automatically by ZMQ.

Entry point: ``worker_entry(socket_path)`` is passed to ``multiprocessing.Process(target=...)``.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import msgpack
import structlog
import zmq
import zmq.asyncio

from nerva.backends.base import Backend, InferContext, ModelConfig
from nerva.backends.registry import get_backend
from nerva.engine.shm_pool import IPC_CONTROL_INLINE_MAX_BYTES
from nerva.observability.timing import AsyncTimingSink
from nerva.worker.ipc import (
    AckStatus,
    Descriptor,
    MessageType,
    decode_message,
    encode_message,
    import_path_to_class,
)

logger = structlog.get_logger(__name__)
DEFAULT_SHM_ALLOC_TIMEOUT_S = 3.0


class _OutputShmAllocationError(Exception):
    """Raised when output SHM allocation via Master fails."""

    def __init__(self, status: AckStatus, error: str) -> None:
        super().__init__(error)
        self.status = status
        self.error = error

# ---------------------------------------------------------------------------
# Internal worker loop
# ---------------------------------------------------------------------------


class _WorkerLoop:
    """Async event loop running inside the worker subprocess."""

    def __init__(
        self,
        socket_path: str,
        *,
        shm_alloc_timeout_s: float = DEFAULT_SHM_ALLOC_TIMEOUT_S,
        timing_log_dir: str | None = None,
    ) -> None:
        self._socket_path = socket_path
        self._shm_alloc_timeout_s = shm_alloc_timeout_s
        self._worker_id = str(uuid.uuid4())  # stable across all WORKER_CONNECT replies
        self._backend: Backend | None = None
        self._async_dispatch = False
        self._thread_executor: ThreadPoolExecutor | None = None
        self._running = False
        self._inflight: dict[str, asyncio.Task[None]] = {}
        self._contexts: dict[str, InferContext] = {}
        self._shm_alloc_futures: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._send_lock: asyncio.Lock | None = None
        self._ctx: zmq.asyncio.Context | None = None
        self._socket: zmq.asyncio.Socket | None = None
        self._model_name: str = ""
        self._timing_log_dir = timing_log_dir
        self._timing_sink: AsyncTimingSink | None = None

    # -- public entry -------------------------------------------------------

    async def run(self) -> None:
        """Bind ROUTER socket and enter the recv loop.

        Each connecting WorkerProxy sends WORKER_CONNECT; the worker responds
        with WORKER_READY addressed to that specific proxy's DEALER identity.
        Multiple proxies (from different uvicorn workers) may connect concurrently.
        """
        self._send_lock = asyncio.Lock()
        self._ctx = zmq.asyncio.Context()
        self._socket = self._ctx.socket(zmq.ROUTER)
        self._socket.bind(f"ipc://{self._socket_path}")
        self._running = True
        if self._timing_log_dir is not None:
            self._timing_sink = AsyncTimingSink()
            await self._timing_sink.start(self._timing_log_dir, f"nerva_worker_{os.getpid()}.log")

        try:
            while self._running:
                try:
                    # ROUTER receives [client_identity, b"", payload]
                    parts: list[bytes] = await asyncio.wait_for(
                        self._socket.recv_multipart(), timeout=1.0
                    )
                except TimeoutError:
                    continue

                if len(parts) < 3:
                    logger.warning("malformed_router_frame", nparts=len(parts))
                    continue

                client_id = parts[0]
                raw = parts[2]

                try:
                    msg = decode_message(raw)
                except Exception:
                    logger.exception("Failed to decode incoming message")
                    continue
                msg_type = msg.get("type", "")

                if msg_type == MessageType.WORKER_CONNECT.value:
                    await self._send_to(
                        client_id,
                        {"type": MessageType.WORKER_READY.value, "worker_id": self._worker_id},
                    )
                elif msg_type == MessageType.LOAD_MODEL.value:
                    await self._handle_load_model(msg, client_id)
                elif msg_type == MessageType.INFER_SUBMIT.value:
                    request_id = msg.get("request_id", "")
                    task = asyncio.create_task(self._handle_infer(msg, client_id))
                    self._inflight[request_id] = task
                    task.add_done_callback(self._make_cleanup_cb(request_id))
                elif msg_type == MessageType.SHM_ALLOC_RESPONSE.value:
                    self._handle_shm_alloc_response(msg)
                elif msg_type == MessageType.HEALTH_CHECK.value:
                    await self._handle_health_check(msg, client_id)
                elif msg_type == MessageType.CANCEL.value:
                    self._handle_cancel(msg)
                elif msg_type == MessageType.SHUTDOWN.value:
                    logger.info("Received SHUTDOWN, exiting worker loop")
                    self._running = False
                else:
                    logger.warning("unknown_message_type", msg_type=msg_type)
        finally:
            await self._cleanup()

    def _make_cleanup_cb(self, request_id: str) -> Any:
        """Return a done-callback that removes the task from _inflight."""
        def _cb(_t: asyncio.Task[None]) -> None:
            self._inflight.pop(request_id, None)
        return _cb

    # -- message handlers ---------------------------------------------------

    async def _handle_load_model(self, msg: dict[str, Any], client_id: bytes) -> None:
        """Import model class, create Backend, load model, send ACK."""
        model_name: str = msg.get("model_name", "unknown")
        try:
            model_class_path: str = msg["model_class"]
            backend_name: str = msg.get("backend", "pytorch")
            device: str = msg.get("device", "cpu")
            options = msg.get("options", {})
            if not isinstance(options, dict):
                raise TypeError(f"LOAD_MODEL options must be a dict, got {type(options)}")

            model_class = import_path_to_class(model_class_path)
            backend_cls = get_backend(backend_name)
            backend = backend_cls()

            config = ModelConfig(
                model_name=model_name,
                model_class=model_class,
                device=device,
                backend_options=options,
            )
            await backend.load_model(config)
            self._backend = backend
            self._model_name = model_name

            # Determine dispatch mode: backend signal OR user declaration
            self._async_dispatch = (
                getattr(self._backend, "is_async_native", False)
                or msg.get("async_infer", False)
            )

            # Shut down any previous executor (e.g., model reload or dispatch mode change).
            if self._thread_executor is not None:
                self._thread_executor.shutdown(wait=False)
                self._thread_executor = None

            if not self._async_dispatch:
                try:
                    max_threads = int(os.environ.get("NERVA_WORKER_MAX_THREADS", "0")) or None
                except ValueError:
                    logger.warning(
                        "invalid_nerva_worker_max_threads",
                        value=os.environ.get("NERVA_WORKER_MAX_THREADS"),
                    )
                    max_threads = None
                if max_threads:
                    self._thread_executor = ThreadPoolExecutor(max_workers=max_threads)

            await self._send_to(client_id, {
                "type": MessageType.LOAD_MODEL_ACK.value,
                "model_name": model_name,
                "status": AckStatus.OK.value,
            })
            logger.info("model_loaded", model=model_name)
        except Exception as exc:
            logger.exception("load_model_failed", model=model_name)
            await self._send_to(client_id, {
                "type": MessageType.LOAD_MODEL_ACK.value,
                "model_name": model_name,
                "status": AckStatus.INTERNAL.value,
                "error": str(exc),
            })

    async def _handle_infer(self, msg: dict[str, Any], client_id: bytes) -> None:
        """Deserialize inputs, run inference, send INFER_ACK."""
        request_id: str = msg.get("request_id", "")
        context: InferContext | None = None
        t_recv = time.perf_counter()
        structlog.contextvars.bind_contextvars(request_id=request_id)
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
            self._contexts[request_id] = context

            if context.deadline_ms <= 0:
                await self._send_to(client_id, {
                    "type": MessageType.INFER_ACK.value,
                    "request_id": request_id,
                    "status": AckStatus.DEADLINE_EXCEEDED.value,
                    "error": "deadline exceeded before execution",
                })
                return

            try:
                t_dispatch = time.perf_counter()

                if self._async_dispatch:
                    # ── Async path: direct await on Worker event loop ──
                    output = await asyncio.wait_for(
                        self._backend.infer(inputs, context),
                        timeout=context.deadline_ms / 1000.0,
                    )
                    t_infer_end = time.perf_counter()
                else:
                    # ── Sync path: dispatch to ThreadPool ──
                    _thread_ts: list[float] = []

                    def _run_with_timing() -> dict[str, Any]:
                        _thread_ts.append(time.perf_counter())
                        return self._run_infer_sync(inputs, context)

                    if self._thread_executor is not None:
                        loop = asyncio.get_running_loop()
                        output = await asyncio.wait_for(
                            loop.run_in_executor(self._thread_executor, _run_with_timing),
                            timeout=context.deadline_ms / 1000.0,
                        )
                    else:
                        output = await asyncio.wait_for(
                            asyncio.to_thread(_run_with_timing),
                            timeout=context.deadline_ms / 1000.0,
                        )
                    t_infer_end = time.perf_counter()
            except TimeoutError:
                context.cancelled = True
                await self._send_to(client_id, {
                    "type": MessageType.INFER_ACK.value,
                    "request_id": request_id,
                    "status": AckStatus.DEADLINE_EXCEEDED.value,
                    "error": "deadline exceeded",
                })
                return

            # Serialize output and choose inline/SHM output path.
            output_bytes = msgpack.packb(output, use_bin_type=True)
            t_post_infer_start = time.perf_counter()
            out_descriptor = await self._build_output_descriptor(request_id, output_bytes, client_id)

            await self._send_to(client_id, {
                "type": MessageType.INFER_ACK.value,
                "request_id": request_id,
                "status": AckStatus.OK.value,
                "descriptor": out_descriptor.to_dict(),
            })
            t_send_done = time.perf_counter()

            if self._timing_sink is not None:
                if self._async_dispatch:
                    self._timing_sink.write({
                        "event": "infer_timing",
                        "request_id": request_id,
                        "model": self._model_name,
                        "worker_deser_ms": round((t_dispatch - t_recv) * 1000, 3),
                        "thread_queue_ms": 0.0,
                        "backend_infer_ms": round((t_infer_end - t_dispatch) * 1000, 3),
                        "worker_post_infer_ms": round((t_send_done - t_post_infer_start) * 1000, 3),
                    })
                else:
                    t_thread_start = _thread_ts[0] if _thread_ts else t_dispatch
                    self._timing_sink.write({
                        "event": "infer_timing",
                        "request_id": request_id,
                        "model": self._model_name,
                        "worker_deser_ms": round((t_dispatch - t_recv) * 1000, 3),
                        "thread_queue_ms": round((t_thread_start - t_dispatch) * 1000, 3),
                        "backend_infer_ms": round((t_infer_end - t_thread_start) * 1000, 3),
                        "worker_post_infer_ms": round((t_send_done - t_post_infer_start) * 1000, 3),
                    })
        except asyncio.CancelledError:
            if context is not None:
                context.cancelled = True
            await self._send_to(client_id, {
                "type": MessageType.INFER_ACK.value,
                "request_id": request_id,
                "status": AckStatus.ABORTED.value,
                "error": "request cancelled",
            })
            return
        except _OutputShmAllocationError as exc:
            logger.exception("shm_alloc_failed", request_id=request_id)
            await self._send_to(client_id, {
                "type": MessageType.INFER_ACK.value,
                "request_id": request_id,
                "status": exc.status.value,
                "error": exc.error,
            })
            return
        except Exception as exc:
            logger.exception("infer_failed", request_id=request_id)
            await self._send_to(client_id, {
                "type": MessageType.INFER_ACK.value,
                "request_id": request_id,
                "status": AckStatus.INTERNAL.value,
                "error": str(exc),
            })
        finally:
            self._contexts.pop(request_id, None)
            structlog.contextvars.clear_contextvars()

    def _run_infer_sync(self, inputs: dict[str, Any], context: InferContext) -> dict[str, Any]:
        """Blocking wrapper for thread execution of async backend inference.

        This runs inside ``asyncio.to_thread()``, so there is no running loop in
        this thread. ``asyncio.run()`` is safe here and keeps backend.infer()
        isolated from the worker event loop.
        """
        assert self._backend is not None
        return asyncio.run(self._backend.infer(inputs, context))

    def _read_inputs(self, descriptor: Descriptor) -> dict[str, Any]:
        """Read inputs from either inline data or SHM."""
        if descriptor.payload_codec == "raw_bytes_v1":
            if descriptor.input_key is None:
                raise ValueError("raw_bytes_v1 descriptor missing input_key")

            if descriptor.is_inline:
                assert descriptor.inline_data is not None
                return {descriptor.input_key: descriptor.inline_data}

            if descriptor.shm_id is None:
                raise ValueError("Descriptor has neither inline_data nor shm_id")

            shm = SharedMemory(name=descriptor.shm_id, create=False)
            try:
                buf = shm.buf
                assert buf is not None
                view = memoryview(buf)[descriptor.offset : descriptor.offset + descriptor.length]
                try:
                    return {descriptor.input_key: view.tobytes()}
                finally:
                    view.release()
            finally:
                shm.close()

        if descriptor.is_inline:
            assert descriptor.inline_data is not None
            return msgpack.unpackb(descriptor.inline_data, raw=False)  # type: ignore[no-any-return]

        # SHM path.
        if descriptor.shm_id is None:
            raise ValueError("Descriptor has neither inline_data nor shm_id")

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

    async def _handle_health_check(self, msg: dict[str, Any], client_id: bytes) -> None:
        """Respond with HEALTH_STATUS."""
        healthy = self._backend.health_check() if self._backend is not None else False
        await self._send_to(client_id, {
            "type": MessageType.HEALTH_STATUS.value,
            "healthy": healthy,
        })

    def _handle_cancel(self, msg: dict[str, Any]) -> None:
        """Cancel an inflight request by setting context.cancelled (best-effort)."""
        request_id = msg.get("request_id", "")
        context = self._contexts.get(request_id)
        if context is not None:
            context.cancelled = True

        task = self._inflight.get(request_id)
        if task is not None and not task.done():
            task.cancel()
            logger.info("request_cancelled", request_id=request_id)
        else:
            logger.warning("no_inflight_request", request_id=request_id)

    def _handle_shm_alloc_response(self, msg: dict[str, Any]) -> None:
        """Resolve pending SHM allocation request future."""
        request_id = msg.get("request_id", "")
        fut = self._shm_alloc_futures.get(request_id)
        if fut is not None and not fut.done():
            fut.set_result(msg)
        else:
            logger.warning("no_pending_shm_future", request_id=request_id)

    # -- helpers ------------------------------------------------------------

    async def _send_to(self, client_id: bytes, msg: dict[str, Any]) -> None:
        """Route a reply to a specific DEALER client via the ROUTER socket."""
        assert self._send_lock is not None
        assert self._socket is not None
        async with self._send_lock:
            await self._socket.send_multipart([client_id, b"", encode_message(msg)])

    async def _build_output_descriptor(
        self,
        request_id: str,
        output_bytes: bytes,
        client_id: bytes,
    ) -> Descriptor:
        """Build output descriptor. Large payloads request SHM from Master."""
        if len(output_bytes) <= IPC_CONTROL_INLINE_MAX_BYTES:
            return Descriptor(
                request_id=request_id,
                node_id=0,
                inline_data=output_bytes,
                length=len(output_bytes),
                payload_codec="msgpack_dict_v1",
            )

        try:
            alloc = await self._request_output_slot(request_id, len(output_bytes), client_id)
        except _OutputShmAllocationError as exc:
            if exc.status == AckStatus.UNAVAILABLE:
                # No shm_pool associated with this request (e.g. Executor called
                # proxy.infer() without a shm_pool).  Fall back to inline so the
                # request still succeeds; performance is degraded but not broken.
                logger.debug(
                    "no_shm_pool_fallback",
                    request_id=request_id,
                    output_bytes=len(output_bytes),
                )
                return Descriptor(
                    request_id=request_id,
                    node_id=0,
                    inline_data=output_bytes,
                    length=len(output_bytes),
                    payload_codec="msgpack_dict_v1",
                )
            raise
        shm_id = alloc["shm_id"]
        offset = alloc["offset"]
        slot_size = alloc["slot_size"]

        if not isinstance(shm_id, str):
            raise RuntimeError(f"Invalid shm_id in SHM_ALLOC_RESPONSE: {shm_id!r}")
        if not isinstance(offset, int):
            raise RuntimeError(f"Invalid offset in SHM_ALLOC_RESPONSE: {offset!r}")
        if not isinstance(slot_size, int):
            raise RuntimeError(f"Invalid slot_size in SHM_ALLOC_RESPONSE: {slot_size!r}")
        if len(output_bytes) > slot_size:
            raise RuntimeError(
                f"Allocated SHM slot too small: output={len(output_bytes)} slot={slot_size}"
            )

        shm = SharedMemory(name=shm_id, create=False)
        try:
            buf = shm.buf
            assert buf is not None
            buf[offset : offset + len(output_bytes)] = output_bytes
        finally:
            shm.close()

        return Descriptor(
            request_id=request_id,
            node_id=0,
            shm_id=shm_id,
            offset=offset,
            length=len(output_bytes),
            payload_codec="msgpack_dict_v1",
        )

    async def _request_output_slot(
        self, request_id: str, size: int, client_id: bytes
    ) -> dict[str, Any]:
        """Ask Master to allocate an output SHM slot for this request."""
        if request_id in self._shm_alloc_futures:
            raise RuntimeError(f"Duplicate SHM allocation request for '{request_id}'")

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._shm_alloc_futures[request_id] = fut
        try:
            await self._send_to(client_id, {
                "type": MessageType.SHM_ALLOC_REQUEST.value,
                "request_id": request_id,
                "size": size,
            })
            resp = await asyncio.wait_for(fut, timeout=self._shm_alloc_timeout_s)
        finally:
            self._shm_alloc_futures.pop(request_id, None)

        status = resp.get("status", "")
        if status == AckStatus.OK.value:
            return resp

        error = str(resp.get("error", "unknown error"))
        try:
            ack_status = AckStatus(status)
        except Exception:
            ack_status = AckStatus.INTERNAL
        raise _OutputShmAllocationError(ack_status, error)

    async def _cleanup(self) -> None:
        """Close socket, terminate context, unload backend."""
        # Cancel inflight tasks.
        for task in self._inflight.values():
            task.cancel()
        if self._inflight:
            await asyncio.gather(*self._inflight.values(), return_exceptions=True)
        self._inflight.clear()
        self._contexts.clear()

        for fut in self._shm_alloc_futures.values():
            if not fut.done():
                fut.set_exception(RuntimeError("Worker loop shutting down"))
        self._shm_alloc_futures.clear()

        # Shutdown thread executor.
        if self._thread_executor is not None:
            self._thread_executor.shutdown(wait=False)
            self._thread_executor = None

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

        # Stop timing writer and close file.
        if self._timing_sink is not None:
            await self._timing_sink.stop()
            self._timing_sink = None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def worker_entry(
    socket_path: str,
    *,
    shm_alloc_timeout_s: float = DEFAULT_SHM_ALLOC_TIMEOUT_S,
    timing_log_dir: str | None = None,
) -> None:
    """Entry point for ``multiprocessing.Process(target=worker_entry, args=(path,))``."""
    loop = _WorkerLoop(
        socket_path,
        shm_alloc_timeout_s=shm_alloc_timeout_s,
        timing_log_dir=timing_log_dir,
    )
    asyncio.run(loop.run())
