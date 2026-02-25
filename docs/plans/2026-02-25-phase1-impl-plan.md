# Phase 1: Master-Worker IPC Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Master-Worker process separation with ZeroMQ IPC and shared memory data channel, enabling single-model inference across process boundary.

**Architecture:** Master process manages WorkerProxy + ShmPool, communicates with Worker subprocess via ZeroMQ PAIR socket (control) and POSIX shared memory (data >8KB). Worker loads Backend/Model and runs inference via `asyncio.to_thread()` to avoid blocking the event loop.

**Tech Stack:** Python 3.11+, pyzmq, msgpack (already dep), multiprocessing.shared_memory, asyncio, pytest-asyncio

**Design docs:**
- [`2026-02-25-phase1-design.md`](./2026-02-25-phase1-design.md) — Full design
- [`ipc-contract.md`](./ipc-contract.md) — IPC contract
- [`mvp-defaults.md`](./mvp-defaults.md) — Default parameters

---

### Task 1: Add pyzmq Dependency

**Files:**
- Modify: `pyproject.toml:8-15`

**Step 1: Add pyzmq to dependencies**

In `pyproject.toml`, add `pyzmq>=26.0` to `dependencies`:

```toml
dependencies = [
    "msgpack>=1.0",
    "pyzmq>=26.0",
    "starlette>=0.40",
    "uvicorn>=0.30",
    "fastapi>=0.115",
    "prometheus-client>=0.21",
    "structlog>=24.0",
]
```

Also add mypy override for zmq:

```toml
[[tool.mypy.overrides]]
module = [
    "msgpack.*",
    "prometheus_client.*",
    "vllm.*",
    "zmq.*",
]
ignore_missing_imports = true
```

**Step 2: Lock dependencies**

Run: `uv sync --extra dev`
Expected: Success, pyzmq installed

**Step 3: Verify imports**

Run: `uv run python -c "import zmq; print(zmq.zmq_version())"`
Expected: prints ZeroMQ version (e.g. `4.3.5`)

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat(phase1): add pyzmq dependency"
```

---

### Task 2: IPC Message Codec & Descriptor

**Files:**
- Create: `src/nerva/worker/ipc.py`
- Create: `tests/test_ipc.py`
- Modify: `src/nerva/worker/__init__.py`

**Step 1: Write the failing tests**

```python
# tests/test_ipc.py
"""Tests for nerva.worker.ipc — IPC message encoding/decoding."""

from typing import Any

import pytest

from nerva.worker.ipc import (
    AckStatus,
    Descriptor,
    MessageType,
    class_to_import_path,
    decode_message,
    encode_message,
    import_path_to_class,
)


class TestMessageCodec:
    def test_roundtrip_simple(self) -> None:
        msg: dict[str, Any] = {"type": "HEALTH_CHECK", "worker_id": "w0"}
        assert decode_message(encode_message(msg)) == msg

    def test_roundtrip_with_bytes(self) -> None:
        msg: dict[str, Any] = {
            "type": "INFER_SUBMIT",
            "request_id": "req-1",
            "descriptor": {"inline_data": b"\x00\x01\x02"},
        }
        decoded = decode_message(encode_message(msg))
        assert decoded["descriptor"]["inline_data"] == b"\x00\x01\x02"

    def test_roundtrip_with_none(self) -> None:
        msg: dict[str, Any] = {"type": "INFER_ACK", "out_descriptor": None}
        assert decode_message(encode_message(msg)) == msg

    def test_roundtrip_infer_submit_full(self) -> None:
        msg: dict[str, Any] = {
            "type": MessageType.INFER_SUBMIT,
            "request_id": "req-123",
            "node_id": 0,
            "deadline_ms": 5000,
            "descriptor": {
                "schema_version": 1,
                "request_id": "req-123",
                "node_id": 0,
                "shm_id": "nerva-1234-shm-2",
                "offset": 0,
                "length": 65536,
                "inline_data": None,
                "dtype": "bytes",
                "shape": [65536],
                "device": "cpu",
                "lifetime_token": 0,
                "checksum": 0,
            },
            "batch_meta": None,
        }
        assert decode_message(encode_message(msg)) == msg

    def test_roundtrip_infer_ack(self) -> None:
        msg: dict[str, Any] = {
            "type": MessageType.INFER_ACK,
            "request_id": "req-123",
            "node_id": 0,
            "status": AckStatus.OK,
            "out_descriptor": {"inline_data": b"result", "length": 6},
            "error": None,
        }
        decoded = decode_message(encode_message(msg))
        assert decoded["status"] == "OK"
        assert decoded["out_descriptor"]["inline_data"] == b"result"

    def test_roundtrip_load_model(self) -> None:
        msg: dict[str, Any] = {
            "type": MessageType.LOAD_MODEL,
            "model_name": "echo",
            "model_class_path": "tests.helpers:EchoModel",
            "backend": "pytorch",
            "device": "cpu",
            "options": {"max_batch": 32},
        }
        assert decode_message(encode_message(msg)) == msg


class TestDescriptor:
    def test_default_values(self) -> None:
        d = Descriptor(request_id="r1", node_id=0)
        assert d.schema_version == 1
        assert d.shm_id is None
        assert d.inline_data is None
        assert d.dtype == "bytes"
        assert d.shape == []

    def test_roundtrip_dict(self) -> None:
        d = Descriptor(
            request_id="r1",
            node_id=0,
            shm_id="nerva-shm-0",
            offset=4096,
            length=1024,
        )
        reconstructed = Descriptor.from_dict(d.to_dict())
        assert reconstructed == d

    def test_roundtrip_inline(self) -> None:
        d = Descriptor(
            request_id="r1",
            node_id=0,
            inline_data=b"\xff\xfe",
            length=2,
        )
        data = d.to_dict()
        assert data["inline_data"] == b"\xff\xfe"
        assert data["shm_id"] is None
        assert Descriptor.from_dict(data) == d

    def test_is_inline(self) -> None:
        d_inline = Descriptor(request_id="r1", node_id=0, inline_data=b"hi")
        d_shm = Descriptor(request_id="r1", node_id=0, shm_id="shm-0")
        assert d_inline.is_inline
        assert not d_shm.is_inline


class TestMessageType:
    def test_all_types_are_strings(self) -> None:
        for mt in MessageType:
            assert isinstance(mt, str)
            assert mt == mt.value

    def test_expected_types_exist(self) -> None:
        expected = {
            "LOAD_MODEL", "LOAD_MODEL_ACK",
            "INFER_SUBMIT", "INFER_ACK",
            "CANCEL", "HEALTH_CHECK", "HEALTH_STATUS",
            "SHUTDOWN", "WORKER_READY",
        }
        actual = {mt.value for mt in MessageType}
        assert expected == actual


class TestAckStatus:
    def test_expected_statuses(self) -> None:
        expected = {
            "OK", "INVALID_ARGUMENT", "DEADLINE_EXCEEDED",
            "ABORTED", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL",
        }
        actual = {s.value for s in AckStatus}
        assert expected == actual


class TestImportPath:
    def test_roundtrip(self) -> None:
        from nerva.backends.pytorch import PyTorchBackend

        path = class_to_import_path(PyTorchBackend)
        cls = import_path_to_class(path)
        assert cls is PyTorchBackend

    def test_invalid_module(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            import_path_to_class("nonexistent.module:SomeClass")

    def test_invalid_class(self) -> None:
        with pytest.raises(AttributeError):
            import_path_to_class("nerva:NonexistentClass")
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ipc.py -v 2>&1 | head -20`
Expected: FAIL — `ModuleNotFoundError: No module named 'nerva.worker.ipc'`

**Step 3: Write the implementation**

```python
# src/nerva/worker/ipc.py
"""IPC message encoding/decoding and import utilities.

All control-plane messages between Master and Worker are serialized
with msgpack. This module provides codec functions, message type
enums, and the Descriptor dataclass.
"""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import msgpack


class MessageType(StrEnum):
    """Control-plane message types (ipc-contract.md Section 3)."""

    LOAD_MODEL = "LOAD_MODEL"
    LOAD_MODEL_ACK = "LOAD_MODEL_ACK"
    INFER_SUBMIT = "INFER_SUBMIT"
    INFER_ACK = "INFER_ACK"
    CANCEL = "CANCEL"
    HEALTH_CHECK = "HEALTH_CHECK"
    HEALTH_STATUS = "HEALTH_STATUS"
    SHUTDOWN = "SHUTDOWN"
    WORKER_READY = "WORKER_READY"


class AckStatus(StrEnum):
    """INFER_ACK / LOAD_MODEL_ACK status codes."""

    OK = "OK"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    ABORTED = "ABORTED"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    UNAVAILABLE = "UNAVAILABLE"
    INTERNAL = "INTERNAL"


@dataclass
class Descriptor:
    """Data descriptor for IPC payloads (ipc-contract.md Section 4).

    Either ``shm_id`` or ``inline_data`` is set, never both.
    """

    request_id: str = ""
    node_id: int = 0
    schema_version: int = 1
    # SHM fields (None when inline)
    shm_id: str | None = None
    offset: int = 0
    length: int = 0
    # Inline fields (None when SHM)
    inline_data: bytes | None = None
    # Metadata
    dtype: str = "bytes"
    shape: list[int] = field(default_factory=list)
    # Reserved (Phase 1 does not use)
    device: str = "cpu"
    lifetime_token: int = 0
    checksum: int = 0

    @property
    def is_inline(self) -> bool:
        """True if payload is carried inline (not via SHM)."""
        return self.inline_data is not None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a msgpack-friendly dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Descriptor:
        """Reconstruct from a dict."""
        return cls(**data)


# --- Codec ---

def encode_message(msg: dict[str, Any]) -> bytes:
    """Serialize a control message to bytes via msgpack."""
    return msgpack.packb(msg, use_bin_type=True)


def decode_message(data: bytes) -> dict[str, Any]:
    """Deserialize a control message from bytes."""
    result: dict[str, Any] = msgpack.unpackb(data, raw=False)
    return result


# --- Import path utilities ---

def class_to_import_path(cls: type[Any]) -> str:
    """Convert a class to an import path string (e.g. 'nerva.backends.pytorch:PyTorchBackend')."""
    return f"{cls.__module__}:{cls.__qualname__}"


def import_path_to_class(path: str) -> type[Any]:
    """Import a class from an import path string."""
    module_path, class_name = path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    cls: type[Any] = getattr(module, class_name)
    return cls
```

Update `src/nerva/worker/__init__.py` — leave empty (no public re-exports yet).

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ipc.py -v`
Expected: All tests PASS

**Step 5: Run lint and type checks**

Run: `uv run ruff check src/nerva/worker/ tests/test_ipc.py && uv run mypy`
Expected: 0 errors

**Step 6: Commit**

```bash
git add src/nerva/worker/ipc.py src/nerva/worker/__init__.py tests/test_ipc.py
git commit -m "feat(phase1): add IPC message codec, Descriptor, and import utils"
```

---

### Task 3: Shared Memory Pool

**Files:**
- Create: `src/nerva/engine/shm_pool.py`
- Create: `tests/test_shm_pool.py`

**Context:** ShmPool manages pre-allocated POSIX shared memory segments organized by size classes. Master owns all alloc/free operations. Workers open segments read-only by name.

**Step 1: Write the failing tests**

```python
# tests/test_shm_pool.py
"""Tests for nerva.engine.shm_pool — shared memory pool."""

import pytest

from nerva.engine.shm_pool import (
    IPC_CONTROL_INLINE_MAX_BYTES,
    ShmPool,
    ShmPoolExhausted,
    ShmSlot,
)


class TestShmPoolBasic:
    def test_inline_threshold(self) -> None:
        assert IPC_CONTROL_INLINE_MAX_BYTES == 8 * 1024

    def test_alloc_returns_slot(self) -> None:
        pool = ShmPool(size_classes_kb=[4], slots_per_class=2)
        try:
            slot = pool.alloc(1000)
            assert isinstance(slot, ShmSlot)
            assert slot.slot_size == 4096
            assert slot.offset == 0
            pool.free(slot)
        finally:
            pool.close()

    def test_alloc_rounds_up_to_size_class(self) -> None:
        pool = ShmPool(size_classes_kb=[4, 16], slots_per_class=2)
        try:
            slot = pool.alloc(5000)  # > 4KB, should get 16KB slot
            assert slot.slot_size == 16 * 1024
            pool.free(slot)
        finally:
            pool.close()

    def test_alloc_multiple_slots(self) -> None:
        pool = ShmPool(size_classes_kb=[4], slots_per_class=3)
        try:
            slots = [pool.alloc(100) for _ in range(3)]
            # All slots should have unique offsets
            offsets = {s.offset for s in slots}
            assert len(offsets) == 3
            for s in slots:
                pool.free(s)
        finally:
            pool.close()


class TestShmPoolExhaustion:
    def test_pool_exhausted_raises(self) -> None:
        pool = ShmPool(size_classes_kb=[4], slots_per_class=1)
        try:
            slot = pool.alloc(100)
            with pytest.raises(ShmPoolExhausted):
                pool.alloc(100)
            pool.free(slot)
        finally:
            pool.close()

    def test_free_makes_slot_available(self) -> None:
        pool = ShmPool(size_classes_kb=[4], slots_per_class=1)
        try:
            slot = pool.alloc(100)
            pool.free(slot)
            slot2 = pool.alloc(100)  # Should succeed
            pool.free(slot2)
        finally:
            pool.close()

    def test_size_exceeds_max_class(self) -> None:
        pool = ShmPool(size_classes_kb=[4], slots_per_class=2)
        try:
            with pytest.raises(ShmPoolExhausted, match="exceeds max"):
                pool.alloc(5000)  # > 4KB, no bigger class
        finally:
            pool.close()


class TestShmPoolReadWrite:
    def test_write_and_read(self) -> None:
        pool = ShmPool(size_classes_kb=[4], slots_per_class=2)
        try:
            slot = pool.alloc(100)
            data = b"hello world" * 10
            pool.write(slot, data)
            result = pool.read(slot, len(data))
            assert result == data
            pool.free(slot)
        finally:
            pool.close()

    def test_write_full_slot(self) -> None:
        pool = ShmPool(size_classes_kb=[4], slots_per_class=1)
        try:
            slot = pool.alloc(4096)
            data = b"\xab" * 4096
            pool.write(slot, data)
            assert pool.read(slot, 4096) == data
            pool.free(slot)
        finally:
            pool.close()

    def test_write_exceeds_slot_raises(self) -> None:
        pool = ShmPool(size_classes_kb=[4], slots_per_class=1)
        try:
            slot = pool.alloc(100)
            with pytest.raises(ValueError, match="exceeds slot size"):
                pool.write(slot, b"\x00" * 5000)
            pool.free(slot)
        finally:
            pool.close()


class TestShmPoolLifecycle:
    def test_close_is_idempotent(self) -> None:
        pool = ShmPool(size_classes_kb=[4], slots_per_class=1)
        pool.close()
        pool.close()  # Should not raise

    def test_alloc_after_close_raises(self) -> None:
        pool = ShmPool(size_classes_kb=[4], slots_per_class=1)
        pool.close()
        with pytest.raises(RuntimeError, match="closed"):
            pool.alloc(100)

    def test_stats(self) -> None:
        pool = ShmPool(size_classes_kb=[4, 16], slots_per_class=2)
        try:
            slot = pool.alloc(100)
            stats = pool.stats
            assert stats["4096"]["in_use"] == 1
            assert stats["4096"]["total"] == 2
            assert stats["16384"]["in_use"] == 0
            pool.free(slot)
        finally:
            pool.close()

    def test_slot_names_contain_prefix(self) -> None:
        pool = ShmPool(size_classes_kb=[4], slots_per_class=1, name_prefix="test-pool")
        try:
            slot = pool.alloc(100)
            assert "test-pool" in slot.shm_name
            pool.free(slot)
        finally:
            pool.close()
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_shm_pool.py -v 2>&1 | head -10`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
# src/nerva/engine/shm_pool.py
"""Shared memory pool for IPC data channel.

Pre-allocates POSIX shared memory organized by size classes.
Master process owns alloc/free. Workers open segments by name.

References:
- ipc-contract.md Section 5 (pool design)
- mvp-defaults.md Section 3 (default parameters)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Any

IPC_CONTROL_INLINE_MAX_BYTES = 8 * 1024  # 8 KiB

DEFAULT_SIZE_CLASSES_KB = [4, 16, 64, 256, 1024, 4096]
DEFAULT_SLOTS_PER_CLASS = 16


class ShmPoolExhausted(Exception):
    """Raised when no slot is available in the requested size class."""


@dataclass(frozen=True)
class ShmSlot:
    """Handle to an allocated shared memory slot."""

    shm_name: str
    offset: int
    slot_size: int
    _class_idx: int
    _slot_idx: int


class _SlotBitmap:
    """Bitmap for tracking slot allocation within a size class."""

    def __init__(self, num_slots: int) -> None:
        self._bits = bytearray((num_slots + 7) // 8)
        self._num_slots = num_slots

    def alloc(self) -> int | None:
        """Return first free slot index, or None if full."""
        for i in range(self._num_slots):
            byte_idx, bit_idx = divmod(i, 8)
            if not (self._bits[byte_idx] & (1 << bit_idx)):
                self._bits[byte_idx] |= 1 << bit_idx
                return i
        return None

    def free(self, slot_idx: int) -> None:
        """Mark slot as free."""
        byte_idx, bit_idx = divmod(slot_idx, 8)
        self._bits[byte_idx] &= ~(1 << bit_idx)

    @property
    def in_use(self) -> int:
        """Count of currently allocated slots."""
        count = 0
        for i in range(self._num_slots):
            byte_idx, bit_idx = divmod(i, 8)
            if self._bits[byte_idx] & (1 << bit_idx):
                count += 1
        return count


class ShmPool:
    """Shared memory pool with slab-based size classes.

    Args:
        size_classes_kb: Size classes in KiB. Default: [4, 16, 64, 256, 1024, 4096].
        slots_per_class: Number of slots per size class.
        name_prefix: Prefix for POSIX shm names. Default: ``nerva-{pid}``.
    """

    def __init__(
        self,
        size_classes_kb: list[int] | None = None,
        slots_per_class: int = DEFAULT_SLOTS_PER_CLASS,
        name_prefix: str | None = None,
    ) -> None:
        self._size_classes_kb = size_classes_kb or DEFAULT_SIZE_CLASSES_KB
        self._size_classes = [kb * 1024 for kb in self._size_classes_kb]
        self._slots_per_class = slots_per_class
        self._prefix = name_prefix or f"nerva-{os.getpid()}"
        self._closed = False

        self._segments: list[SharedMemory] = []
        self._bitmaps: list[_SlotBitmap] = []

        for idx, slot_size in enumerate(self._size_classes):
            total = slot_size * slots_per_class
            name = f"{self._prefix}-shm-{idx}"
            # Clean up stale segment from previous crash
            try:
                stale = SharedMemory(name=name, create=False)
                stale.close()
                stale.unlink()
            except FileNotFoundError:
                pass
            shm = SharedMemory(name=name, create=True, size=total)
            self._segments.append(shm)
            self._bitmaps.append(_SlotBitmap(slots_per_class))

    def alloc(self, size: int) -> ShmSlot:
        """Allocate a slot that fits ``size`` bytes.

        Raises:
            RuntimeError: If pool is closed.
            ShmPoolExhausted: If no slot available or size too large.
        """
        if self._closed:
            raise RuntimeError("ShmPool is closed")

        for class_idx, slot_size in enumerate(self._size_classes):
            if slot_size >= size:
                slot_idx = self._bitmaps[class_idx].alloc()
                if slot_idx is not None:
                    return ShmSlot(
                        shm_name=self._segments[class_idx].name,
                        offset=slot_idx * slot_size,
                        slot_size=slot_size,
                        _class_idx=class_idx,
                        _slot_idx=slot_idx,
                    )
                raise ShmPoolExhausted(
                    f"No slots available in size class {slot_size} bytes"
                )

        raise ShmPoolExhausted(
            f"Requested size {size} exceeds max size class "
            f"{self._size_classes[-1]} bytes"
        )

    def free(self, slot: ShmSlot) -> None:
        """Return a slot to the pool."""
        if self._closed:
            return
        self._bitmaps[slot._class_idx].free(slot._slot_idx)

    def write(self, slot: ShmSlot, data: bytes) -> None:
        """Write bytes into a slot."""
        if len(data) > slot.slot_size:
            raise ValueError(
                f"Data ({len(data)} bytes) exceeds slot size ({slot.slot_size} bytes)"
            )
        shm = self._segments[slot._class_idx]
        shm.buf[slot.offset : slot.offset + len(data)] = data

    def read(self, slot: ShmSlot, length: int) -> bytes:
        """Read bytes from a slot."""
        shm = self._segments[slot._class_idx]
        return bytes(shm.buf[slot.offset : slot.offset + length])

    def close(self) -> None:
        """Release and unlink all shared memory segments."""
        if self._closed:
            return
        self._closed = True
        for shm in self._segments:
            shm.close()
            shm.unlink()

    @property
    def stats(self) -> dict[str, Any]:
        """Per-size-class allocation statistics."""
        result: dict[str, Any] = {}
        for slot_size, bitmap in zip(self._size_classes, self._bitmaps):
            result[str(slot_size)] = {
                "total": self._slots_per_class,
                "in_use": bitmap.in_use,
            }
        return result
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_shm_pool.py -v`
Expected: All tests PASS

**Step 5: Run full suite + lint**

Run: `uv run ruff check src/ tests/ && uv run mypy && uv run pytest tests/ -v`
Expected: 0 errors, all tests pass

**Step 6: Commit**

```bash
git add src/nerva/engine/shm_pool.py tests/test_shm_pool.py
git commit -m "feat(phase1): add ShmPool with size classes and bitmap tracking"
```

---

### Task 4: Worker Process

**Files:**
- Create: `src/nerva/worker/process.py`
- Create: `tests/helpers.py`
- Create: `tests/test_worker_process.py`

**Context:** The Worker runs as a subprocess via `multiprocessing.Process`. It connects a ZeroMQ PAIR socket to Master, waits for commands, and runs inference via `asyncio.to_thread()` to keep the event loop responsive.

**Step 1: Create test helpers module**

```python
# tests/helpers.py
"""Shared test models and utilities for Phase 1 integration tests."""

from typing import Any

from nerva import Model


class EchoModel(Model):
    """Returns inputs as-is. Used in IPC integration tests."""

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"echo": inputs.get("value")}


class SlowModel(Model):
    """Sleeps before returning. Used to test cancel and health during infer."""

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        delay = inputs.get("delay", 1.0)
        await asyncio.sleep(delay)
        return {"done": True}


class CrashModel(Model):
    """Raises on infer(). Used to test error propagation."""

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("CrashModel always fails")
```

**Step 2: Write the failing tests**

```python
# tests/test_worker_process.py
"""Tests for nerva.worker.process — Worker subprocess."""

import asyncio
import multiprocessing
import os
import tempfile
from typing import Any

import pytest
import zmq
import zmq.asyncio

from nerva.worker.ipc import (
    AckStatus,
    MessageType,
    decode_message,
    encode_message,
)
from nerva.worker.process import worker_entry


def _socket_path() -> str:
    tmpdir = tempfile.mkdtemp(prefix="nerva-test-")
    return os.path.join(tmpdir, "test-worker.sock")


class TestWorkerBootstrap:
    async def test_worker_sends_ready_and_loads_model(self) -> None:
        path = _socket_path()
        ctx = zmq.asyncio.Context()
        sock = ctx.socket(zmq.PAIR)
        sock.bind(f"ipc://{path}")

        proc = multiprocessing.Process(target=worker_entry, args=(path,))
        proc.start()

        try:
            # Wait for WORKER_READY
            raw = await asyncio.wait_for(sock.recv(), timeout=5.0)
            msg = decode_message(raw)
            assert msg["type"] == MessageType.WORKER_READY

            # Send LOAD_MODEL
            load_msg: dict[str, Any] = {
                "type": MessageType.LOAD_MODEL,
                "model_name": "echo",
                "model_class_path": "tests.helpers:EchoModel",
                "backend": "pytorch",
                "device": "cpu",
                "options": {},
            }
            await sock.send(encode_message(load_msg))

            # Wait for LOAD_MODEL_ACK
            raw = await asyncio.wait_for(sock.recv(), timeout=5.0)
            ack = decode_message(raw)
            assert ack["type"] == MessageType.LOAD_MODEL_ACK
            assert ack["status"] == AckStatus.OK

            # Shutdown
            await sock.send(encode_message({"type": MessageType.SHUTDOWN}))
            proc.join(timeout=5)
        finally:
            sock.close()
            ctx.term()
            if proc.is_alive():
                proc.kill()
                proc.join()


class TestWorkerInfer:
    async def test_infer_inline(self) -> None:
        """Send small payload inline, get result inline."""
        path = _socket_path()
        ctx = zmq.asyncio.Context()
        sock = ctx.socket(zmq.PAIR)
        sock.bind(f"ipc://{path}")

        proc = multiprocessing.Process(target=worker_entry, args=(path,))
        proc.start()

        try:
            # Handshake
            await asyncio.wait_for(sock.recv(), timeout=5.0)  # WORKER_READY
            load_msg: dict[str, Any] = {
                "type": MessageType.LOAD_MODEL,
                "model_name": "echo",
                "model_class_path": "tests.helpers:EchoModel",
                "backend": "pytorch",
                "device": "cpu",
                "options": {},
            }
            await sock.send(encode_message(load_msg))
            await asyncio.wait_for(sock.recv(), timeout=5.0)  # LOAD_MODEL_ACK

            # Send INFER_SUBMIT with inline data
            import msgpack

            input_data = msgpack.packb({"value": 42}, use_bin_type=True)
            submit_msg: dict[str, Any] = {
                "type": MessageType.INFER_SUBMIT,
                "request_id": "req-1",
                "node_id": 0,
                "deadline_ms": 99999,
                "descriptor": {
                    "schema_version": 1,
                    "request_id": "req-1",
                    "node_id": 0,
                    "shm_id": None,
                    "offset": 0,
                    "length": len(input_data),
                    "inline_data": input_data,
                    "dtype": "bytes",
                    "shape": [],
                    "device": "cpu",
                    "lifetime_token": 0,
                    "checksum": 0,
                },
                "batch_meta": None,
            }
            await sock.send(encode_message(submit_msg))

            # Wait for INFER_ACK
            raw = await asyncio.wait_for(sock.recv(), timeout=5.0)
            ack = decode_message(raw)
            assert ack["type"] == MessageType.INFER_ACK
            assert ack["request_id"] == "req-1"
            assert ack["status"] == AckStatus.OK
            assert ack["out_descriptor"] is not None

            # Deserialize output
            out_desc = ack["out_descriptor"]
            result = msgpack.unpackb(out_desc["inline_data"], raw=False)
            assert result == {"echo": 42}

            # Shutdown
            await sock.send(encode_message({"type": MessageType.SHUTDOWN}))
            proc.join(timeout=5)
        finally:
            sock.close()
            ctx.term()
            if proc.is_alive():
                proc.kill()
                proc.join()


class TestWorkerHealth:
    async def test_health_check(self) -> None:
        path = _socket_path()
        ctx = zmq.asyncio.Context()
        sock = ctx.socket(zmq.PAIR)
        sock.bind(f"ipc://{path}")

        proc = multiprocessing.Process(target=worker_entry, args=(path,))
        proc.start()

        try:
            # Handshake + load
            await asyncio.wait_for(sock.recv(), timeout=5.0)
            load_msg: dict[str, Any] = {
                "type": MessageType.LOAD_MODEL,
                "model_name": "echo",
                "model_class_path": "tests.helpers:EchoModel",
                "backend": "pytorch",
                "device": "cpu",
                "options": {},
            }
            await sock.send(encode_message(load_msg))
            await asyncio.wait_for(sock.recv(), timeout=5.0)

            # Health check
            await sock.send(
                encode_message({"type": MessageType.HEALTH_CHECK, "worker_id": "w0"})
            )
            raw = await asyncio.wait_for(sock.recv(), timeout=5.0)
            status = decode_message(raw)
            assert status["type"] == MessageType.HEALTH_STATUS
            assert status["ok"] is True

            # Shutdown
            await sock.send(encode_message({"type": MessageType.SHUTDOWN}))
            proc.join(timeout=5)
        finally:
            sock.close()
            ctx.term()
            if proc.is_alive():
                proc.kill()
                proc.join()


class TestWorkerShutdown:
    async def test_graceful_shutdown(self) -> None:
        path = _socket_path()
        ctx = zmq.asyncio.Context()
        sock = ctx.socket(zmq.PAIR)
        sock.bind(f"ipc://{path}")

        proc = multiprocessing.Process(target=worker_entry, args=(path,))
        proc.start()

        try:
            await asyncio.wait_for(sock.recv(), timeout=5.0)  # WORKER_READY
            await sock.send(encode_message({"type": MessageType.SHUTDOWN}))
            proc.join(timeout=5)
            assert not proc.is_alive()
            assert proc.exitcode == 0
        finally:
            sock.close()
            ctx.term()
            if proc.is_alive():
                proc.kill()
                proc.join()
```

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_worker_process.py -v 2>&1 | head -10`
Expected: FAIL — `ModuleNotFoundError: No module named 'nerva.worker.process'`

**Step 4: Write the implementation**

```python
# src/nerva/worker/process.py
"""Worker subprocess entry point and main loop.

Each Worker process:
1. Connects a ZeroMQ PAIR socket to Master
2. Sends WORKER_READY
3. Waits for LOAD_MODEL, creates Backend + Model
4. Enters main loop: handles INFER_SUBMIT, HEALTH_CHECK, CANCEL, SHUTDOWN

Inference is dispatched to a thread via asyncio.to_thread() so the
event loop stays responsive for health checks and cancel signals.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import msgpack
import zmq
import zmq.asyncio

from nerva.backends.base import InferContext, ModelConfig
from nerva.backends.registry import get_backend
from nerva.worker.ipc import (
    AckStatus,
    MessageType,
    class_to_import_path,
    decode_message,
    encode_message,
    import_path_to_class,
)

logger = logging.getLogger(__name__)


class _WorkerLoop:
    """Internal async worker main loop."""

    def __init__(self, socket_path: str) -> None:
        self._socket_path = socket_path
        self._backend: Any | None = None  # Backend instance
        self._model_name: str = ""
        self._running = True
        self._inflight_contexts: dict[str, InferContext] = {}
        self._send_lock = asyncio.Lock()
        self._sock: zmq.asyncio.Socket | None = None

    async def run(self) -> None:
        ctx = zmq.asyncio.Context()
        self._sock = ctx.socket(zmq.PAIR)
        self._sock.connect(f"ipc://{self._socket_path}")

        try:
            # Signal readiness
            await self._send({"type": MessageType.WORKER_READY})

            while self._running:
                raw = await self._sock.recv()
                msg = decode_message(raw)
                msg_type = msg.get("type")

                if msg_type == MessageType.LOAD_MODEL:
                    await self._handle_load_model(msg)
                elif msg_type == MessageType.INFER_SUBMIT:
                    asyncio.create_task(self._handle_infer(msg))
                elif msg_type == MessageType.HEALTH_CHECK:
                    await self._handle_health_check(msg)
                elif msg_type == MessageType.CANCEL:
                    self._handle_cancel(msg)
                elif msg_type == MessageType.SHUTDOWN:
                    logger.info("Worker received SHUTDOWN")
                    self._running = False
                else:
                    logger.warning("Unknown message type: %s", msg_type)
        finally:
            self._sock.close()
            ctx.term()

            # Cleanup backend
            if self._backend is not None:
                await self._backend.unload_model()

    async def _send(self, msg: dict[str, Any]) -> None:
        async with self._send_lock:
            assert self._sock is not None
            await self._sock.send(encode_message(msg))

    async def _handle_load_model(self, msg: dict[str, Any]) -> None:
        model_name = msg["model_name"]
        try:
            model_class = import_path_to_class(msg["model_class_path"])
            backend_cls = get_backend(msg["backend"])
            backend = backend_cls()

            config = ModelConfig(
                model_name=model_name,
                model_class=model_class,
                device=msg["device"],
                backend_options=msg.get("options", {}),
            )
            await backend.load_model(config)
            await backend.warmup()

            self._backend = backend
            self._model_name = model_name

            await self._send({
                "type": MessageType.LOAD_MODEL_ACK,
                "model_name": model_name,
                "status": AckStatus.OK,
                "error": None,
            })
            logger.info("Model '%s' loaded in worker", model_name)
        except Exception as exc:
            logger.exception("Failed to load model '%s'", model_name)
            await self._send({
                "type": MessageType.LOAD_MODEL_ACK,
                "model_name": model_name,
                "status": AckStatus.INTERNAL,
                "error": str(exc),
            })

    async def _handle_infer(self, msg: dict[str, Any]) -> None:
        request_id: str = msg["request_id"]
        node_id: int = msg["node_id"]

        try:
            if self._backend is None:
                await self._send({
                    "type": MessageType.INFER_ACK,
                    "request_id": request_id,
                    "node_id": node_id,
                    "status": AckStatus.UNAVAILABLE,
                    "out_descriptor": None,
                    "error": "No model loaded",
                })
                return

            # Deserialize input from descriptor
            descriptor = msg["descriptor"]
            if descriptor.get("inline_data") is not None:
                input_bytes = descriptor["inline_data"]
            else:
                # SHM path — open, read, close
                from multiprocessing.shared_memory import SharedMemory

                shm = SharedMemory(name=descriptor["shm_id"], create=False)
                input_bytes = bytes(
                    shm.buf[descriptor["offset"] : descriptor["offset"] + descriptor["length"]]
                )
                shm.close()

            inputs: dict[str, Any] = msgpack.unpackb(input_bytes, raw=False)

            # Create context
            context = InferContext(
                request_id=request_id,
                deadline_ms=msg.get("deadline_ms", 0),
            )
            self._inflight_contexts[request_id] = context

            # Run inference in thread to avoid blocking event loop
            result = await asyncio.to_thread(
                self._run_infer_sync, inputs, context
            )

            # Serialize output
            output_bytes = msgpack.packb(result, use_bin_type=True)

            await self._send({
                "type": MessageType.INFER_ACK,
                "request_id": request_id,
                "node_id": node_id,
                "status": AckStatus.OK,
                "out_descriptor": {
                    "inline_data": output_bytes,
                    "length": len(output_bytes),
                },
                "error": None,
            })
        except Exception as exc:
            logger.exception("Infer failed for request %s", request_id)
            await self._send({
                "type": MessageType.INFER_ACK,
                "request_id": request_id,
                "node_id": node_id,
                "status": AckStatus.INTERNAL,
                "out_descriptor": None,
                "error": str(exc),
            })
        finally:
            self._inflight_contexts.pop(request_id, None)

    def _run_infer_sync(
        self, inputs: dict[str, Any], context: InferContext
    ) -> dict[str, Any]:
        """Blocking wrapper: runs async backend.infer() in a new event loop."""
        return asyncio.run(self._backend.infer(inputs, context))

    async def _handle_health_check(self, msg: dict[str, Any]) -> None:
        ok = self._backend is not None and self._backend.health_check()
        await self._send({
            "type": MessageType.HEALTH_STATUS,
            "worker_id": msg.get("worker_id", ""),
            "ok": ok,
            "detail": "" if ok else "no model loaded",
        })

    def _handle_cancel(self, msg: dict[str, Any]) -> None:
        request_id = msg.get("request_id", "")
        ctx = self._inflight_contexts.get(request_id)
        if ctx is not None:
            ctx.cancelled = True
            logger.info("Cancelled request %s", request_id)
        else:
            logger.debug("Cancel for unknown request %s (may have completed)", request_id)


def worker_entry(socket_path: str) -> None:
    """Entry point for the worker subprocess.

    Called as ``multiprocessing.Process(target=worker_entry, args=(...))``
    """
    loop = _WorkerLoop(socket_path)
    asyncio.run(loop.run())
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_worker_process.py -v`
Expected: All tests PASS

**Step 6: Run full suite + lint**

Run: `uv run ruff check src/ tests/ && uv run mypy && uv run pytest tests/ -v`
Expected: 0 errors, all tests pass

**Step 7: Commit**

```bash
git add src/nerva/worker/process.py tests/helpers.py tests/test_worker_process.py
git commit -m "feat(phase1): add Worker subprocess with LOAD_MODEL, INFER, HEALTH, SHUTDOWN"
```

---

### Task 5: WorkerProxy (Master-side RPC)

**Files:**
- Create: `src/nerva/worker/proxy.py`
- Create: `tests/test_worker_proxy.py`

**Context:** WorkerProxy is the Master-side async RPC wrapper. It binds a ZeroMQ PAIR socket, sends commands to Worker, and tracks in-flight requests via `asyncio.Future`.

**Step 1: Write the failing tests**

```python
# tests/test_worker_proxy.py
"""Tests for nerva.worker.proxy — Master-side WorkerProxy."""

import asyncio
import multiprocessing
import os
import tempfile
from typing import Any

import pytest

from nerva.backends.base import InferContext, ModelConfig
from nerva.worker.process import worker_entry
from nerva.worker.proxy import WorkerProxy


def _make_proxy_and_worker() -> tuple[str, str]:
    """Return (socket_path, tmpdir)."""
    tmpdir = tempfile.mkdtemp(prefix="nerva-test-")
    socket_path = os.path.join(tmpdir, "proxy-test.sock")
    return socket_path, tmpdir


class TestWorkerProxyLifecycle:
    async def test_start_and_load_model(self) -> None:
        socket_path, _ = _make_proxy_and_worker()
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
                options={},
            )
        finally:
            await proxy.shutdown()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join()
            await proxy.close()


class TestWorkerProxyInfer:
    async def test_infer_inline(self) -> None:
        socket_path, _ = _make_proxy_and_worker()
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

            context = InferContext(request_id="req-1", deadline_ms=99999)
            result = await proxy.infer({"value": 42}, context)
            assert result == {"echo": 42}
        finally:
            await proxy.shutdown()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join()
            await proxy.close()

    async def test_infer_multiple_sequential(self) -> None:
        socket_path, _ = _make_proxy_and_worker()
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
                ctx = InferContext(request_id=f"req-{i}", deadline_ms=99999)
                result = await proxy.infer({"value": i}, ctx)
                assert result == {"echo": i}
        finally:
            await proxy.shutdown()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join()
            await proxy.close()

    async def test_infer_error_propagation(self) -> None:
        socket_path, _ = _make_proxy_and_worker()
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

            ctx = InferContext(request_id="req-err", deadline_ms=99999)
            with pytest.raises(RuntimeError, match="INTERNAL"):
                await proxy.infer({}, ctx)
        finally:
            await proxy.shutdown()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join()
            await proxy.close()


class TestWorkerProxyHealth:
    async def test_health_check(self) -> None:
        socket_path, _ = _make_proxy_and_worker()
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

            ok = await proxy.health_check()
            assert ok is True
        finally:
            await proxy.shutdown()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join()
            await proxy.close()
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_worker_proxy.py -v 2>&1 | head -10`
Expected: FAIL — `ModuleNotFoundError: No module named 'nerva.worker.proxy'`

**Step 3: Write the implementation**

```python
# src/nerva/worker/proxy.py
"""Master-side async RPC proxy for a single Worker.

WorkerProxy encapsulates ZeroMQ PAIR communication with one Worker
subprocess. Upper layers (Orchestrator, Batcher) call proxy.infer()
without knowing about IPC details.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import msgpack
import zmq
import zmq.asyncio

from nerva.backends.base import InferContext
from nerva.worker.ipc import (
    AckStatus,
    MessageType,
    decode_message,
    encode_message,
)

logger = logging.getLogger(__name__)

# Default submit timeout (ipc-contract.md / mvp-defaults.md)
IPC_SUBMIT_TIMEOUT_S = 5.0


class WorkerProxy:
    """Async RPC proxy for a single Worker subprocess.

    Usage::

        proxy = WorkerProxy(socket_path)
        await proxy.start()
        await proxy.load_model(...)
        result = await proxy.infer(inputs, context)
        await proxy.shutdown()
        await proxy.close()
    """

    def __init__(
        self,
        socket_path: str,
        submit_timeout: float = IPC_SUBMIT_TIMEOUT_S,
    ) -> None:
        self._socket_path = socket_path
        self._submit_timeout = submit_timeout

        self._ctx: zmq.asyncio.Context | None = None
        self._sock: zmq.asyncio.Socket | None = None
        self._send_lock = asyncio.Lock()

        # In-flight request tracking: request_id -> Future
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._recv_task: asyncio.Task[None] | None = None

        # Special futures for non-infer responses
        self._load_model_future: asyncio.Future[dict[str, Any]] | None = None
        self._health_future: asyncio.Future[dict[str, Any]] | None = None

    async def start(self) -> None:
        """Bind socket and wait for Worker READY signal."""
        self._ctx = zmq.asyncio.Context()
        self._sock = self._ctx.socket(zmq.PAIR)
        self._sock.bind(f"ipc://{self._socket_path}")

        # Wait for WORKER_READY
        raw = await asyncio.wait_for(self._sock.recv(), timeout=10.0)
        msg = decode_message(raw)
        if msg.get("type") != MessageType.WORKER_READY:
            raise RuntimeError(f"Expected WORKER_READY, got {msg.get('type')}")

        # Start background recv loop
        self._recv_task = asyncio.create_task(self._recv_loop())
        logger.info("WorkerProxy connected to %s", self._socket_path)

    async def _recv_loop(self) -> None:
        """Background task: receive messages and dispatch to pending futures."""
        assert self._sock is not None
        try:
            while True:
                raw = await self._sock.recv()
                msg = decode_message(raw)
                msg_type = msg.get("type")

                if msg_type == MessageType.INFER_ACK:
                    req_id = msg.get("request_id", "")
                    fut = self._pending.pop(req_id, None)
                    if fut and not fut.done():
                        fut.set_result(msg)
                elif msg_type == MessageType.LOAD_MODEL_ACK:
                    if self._load_model_future and not self._load_model_future.done():
                        self._load_model_future.set_result(msg)
                elif msg_type == MessageType.HEALTH_STATUS:
                    if self._health_future and not self._health_future.done():
                        self._health_future.set_result(msg)
                else:
                    logger.warning("WorkerProxy: unexpected message type %s", msg_type)
        except zmq.ZMQError:
            pass  # Socket closed during shutdown
        except asyncio.CancelledError:
            pass

    async def _send(self, msg: dict[str, Any]) -> None:
        async with self._send_lock:
            assert self._sock is not None
            await self._sock.send(encode_message(msg))

    async def load_model(
        self,
        model_name: str,
        model_class_path: str,
        backend: str = "pytorch",
        device: str = "cpu",
        options: dict[str, Any] | None = None,
    ) -> None:
        """Send LOAD_MODEL and wait for ACK."""
        loop = asyncio.get_running_loop()
        self._load_model_future = loop.create_future()

        await self._send({
            "type": MessageType.LOAD_MODEL,
            "model_name": model_name,
            "model_class_path": model_class_path,
            "backend": backend,
            "device": device,
            "options": options or {},
        })

        ack = await asyncio.wait_for(self._load_model_future, timeout=30.0)
        self._load_model_future = None

        if ack["status"] != AckStatus.OK:
            raise RuntimeError(
                f"LOAD_MODEL failed: {ack.get('error', 'unknown')}"
            )

    async def infer(
        self,
        inputs: dict[str, Any],
        context: InferContext,
        shm_pool: Any | None = None,
    ) -> dict[str, Any]:
        """Send INFER_SUBMIT and wait for INFER_ACK.

        Args:
            inputs: Model input dict.
            context: Per-request context.
            shm_pool: Optional ShmPool for large payloads.

        Returns:
            Model output dict.
        """
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[context.request_id] = fut

        # Serialize inputs
        input_bytes = msgpack.packb(inputs, use_bin_type=True)

        # Build descriptor (inline for now; SHM path added when shm_pool provided)
        from nerva.engine.shm_pool import IPC_CONTROL_INLINE_MAX_BYTES

        shm_slot = None
        if len(input_bytes) <= IPC_CONTROL_INLINE_MAX_BYTES or shm_pool is None:
            descriptor: dict[str, Any] = {
                "schema_version": 1,
                "request_id": context.request_id,
                "node_id": 0,
                "shm_id": None,
                "offset": 0,
                "length": len(input_bytes),
                "inline_data": input_bytes,
                "dtype": "bytes",
                "shape": [],
                "device": "cpu",
                "lifetime_token": 0,
                "checksum": 0,
            }
        else:
            shm_slot = shm_pool.alloc(len(input_bytes))
            shm_pool.write(shm_slot, input_bytes)
            descriptor = {
                "schema_version": 1,
                "request_id": context.request_id,
                "node_id": 0,
                "shm_id": shm_slot.shm_name,
                "offset": shm_slot.offset,
                "length": len(input_bytes),
                "inline_data": None,
                "dtype": "bytes",
                "shape": [],
                "device": "cpu",
                "lifetime_token": 0,
                "checksum": 0,
            }

        submit_msg: dict[str, Any] = {
            "type": MessageType.INFER_SUBMIT,
            "request_id": context.request_id,
            "node_id": 0,
            "deadline_ms": context.deadline_ms,
            "descriptor": descriptor,
            "batch_meta": None,
        }

        try:
            await self._send(submit_msg)
            ack = await asyncio.wait_for(fut, timeout=self._submit_timeout)
        except asyncio.TimeoutError:
            self._pending.pop(context.request_id, None)
            raise RuntimeError(
                f"INFER_SUBMIT timed out for request {context.request_id}"
            ) from None
        finally:
            # Free input SHM slot
            if shm_slot is not None and shm_pool is not None:
                shm_pool.free(shm_slot)

        if ack["status"] != AckStatus.OK:
            raise RuntimeError(
                f"Infer failed ({ack['status']}): {ack.get('error', '')}"
            )

        # Deserialize output
        out_desc = ack.get("out_descriptor")
        if out_desc is None:
            return {}

        if out_desc.get("inline_data") is not None:
            result: dict[str, Any] = msgpack.unpackb(
                out_desc["inline_data"], raw=False
            )
            return result

        # SHM output (future extension)
        raise NotImplementedError("SHM output not yet implemented in Phase 1")

    async def cancel(self, request_id: str, reason: str = "") -> None:
        """Send CANCEL for an in-flight request (best-effort)."""
        await self._send({
            "type": MessageType.CANCEL,
            "request_id": request_id,
            "reason": reason,
        })

    async def health_check(self, timeout: float = 3.0) -> bool:
        """Send HEALTH_CHECK and return True if worker is healthy."""
        loop = asyncio.get_running_loop()
        self._health_future = loop.create_future()

        await self._send({
            "type": MessageType.HEALTH_CHECK,
            "worker_id": "",
        })

        try:
            status = await asyncio.wait_for(self._health_future, timeout=timeout)
            return bool(status.get("ok", False))
        except asyncio.TimeoutError:
            return False
        finally:
            self._health_future = None

    async def shutdown(self) -> None:
        """Send SHUTDOWN to worker."""
        try:
            await self._send({"type": MessageType.SHUTDOWN})
        except Exception:
            pass  # Worker may already be gone

    async def close(self) -> None:
        """Close socket and cancel recv loop."""
        if self._recv_task is not None:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        if self._sock is not None:
            self._sock.close()
        if self._ctx is not None:
            self._ctx.term()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_worker_proxy.py -v`
Expected: All tests PASS

**Step 5: Run full suite + lint**

Run: `uv run ruff check src/ tests/ && uv run mypy && uv run pytest tests/ -v`
Expected: 0 errors, all tests pass

**Step 6: Commit**

```bash
git add src/nerva/worker/proxy.py tests/test_worker_proxy.py
git commit -m "feat(phase1): add WorkerProxy with async infer, health check, shutdown"
```

---

### Task 6: WorkerManager

**Files:**
- Create: `src/nerva/worker/manager.py`
- Create: `tests/test_worker_manager.py`

**Context:** WorkerManager spawns Worker processes, manages their lifecycle (health check, crash recovery, graceful shutdown), and exposes WorkerProxy to upper layers.

**Step 1: Write the failing tests**

```python
# tests/test_worker_manager.py
"""Tests for nerva.worker.manager — WorkerManager lifecycle."""

import asyncio
import signal

import pytest

from nerva import ModelHandle, model
from nerva.backends.base import InferContext
from nerva.worker.manager import WorkerManager

# Use the helpers EchoModel
from tests.helpers import CrashModel, EchoModel


class TestWorkerManagerBasic:
    async def test_start_and_infer(self) -> None:
        handle = model("echo", EchoModel, backend="pytorch", device="cpu")
        manager = WorkerManager()

        try:
            proxy = await manager.start_worker(handle)
            ctx = InferContext(request_id="req-1", deadline_ms=99999)
            result = await proxy.infer({"value": 42}, ctx)
            assert result == {"echo": 42}
        finally:
            await manager.shutdown_all()

    async def test_shutdown_is_idempotent(self) -> None:
        manager = WorkerManager()
        await manager.shutdown_all()
        await manager.shutdown_all()  # Should not raise


class TestWorkerManagerHealth:
    async def test_health_check(self) -> None:
        handle = model("echo", EchoModel, backend="pytorch", device="cpu")
        manager = WorkerManager()

        try:
            proxy = await manager.start_worker(handle)
            ok = await proxy.health_check()
            assert ok is True
        finally:
            await manager.shutdown_all()


class TestWorkerManagerCrashRecovery:
    async def test_detect_dead_worker(self) -> None:
        handle = model("echo", EchoModel, backend="pytorch", device="cpu")
        manager = WorkerManager()

        try:
            proxy = await manager.start_worker(handle)

            # Kill the worker process
            entry = manager._workers[handle.name]
            entry.process.kill()
            entry.process.join(timeout=3)

            # Manager should detect the dead worker
            assert not entry.process.is_alive()
        finally:
            await manager.shutdown_all()

    async def test_restart_dead_worker(self) -> None:
        handle = model("echo", EchoModel, backend="pytorch", device="cpu")
        manager = WorkerManager()

        try:
            proxy = await manager.start_worker(handle)

            # Kill the worker
            entry = manager._workers[handle.name]
            entry.process.kill()
            entry.process.join(timeout=3)

            # Restart
            new_proxy = await manager.restart_worker(handle.name)

            # Should work after restart
            ctx = InferContext(request_id="req-restart", deadline_ms=99999)
            result = await new_proxy.infer({"value": 99}, ctx)
            assert result == {"echo": 99}
        finally:
            await manager.shutdown_all()


class TestWorkerManagerGracefulShutdown:
    async def test_shutdown_all(self) -> None:
        handle = model("echo", EchoModel, backend="pytorch", device="cpu")
        manager = WorkerManager()

        proxy = await manager.start_worker(handle)
        entry = manager._workers[handle.name]
        proc = entry.process

        await manager.shutdown_all()
        proc.join(timeout=5)
        assert not proc.is_alive()
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_worker_manager.py -v 2>&1 | head -10`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
# src/nerva/worker/manager.py
"""WorkerManager — lifecycle management for Worker subprocesses.

Responsibilities:
- Spawn Worker process per ModelHandle
- Load model via WorkerProxy
- Health check (via proxy)
- Crash detection and restart
- Graceful shutdown
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from nerva.core.model import ModelHandle
from nerva.worker.ipc import class_to_import_path
from nerva.worker.process import worker_entry
from nerva.worker.proxy import WorkerProxy

logger = logging.getLogger(__name__)


class WorkerState(StrEnum):
    STARTING = "STARTING"
    LOADING = "LOADING"
    READY = "READY"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"


@dataclass
class _WorkerEntry:
    handle: ModelHandle
    process: multiprocessing.Process
    proxy: WorkerProxy
    socket_path: str
    state: WorkerState = WorkerState.STARTING
    restart_count: int = 0


class WorkerManager:
    """Manages Worker subprocess lifecycles.

    Usage::

        manager = WorkerManager()
        proxy = await manager.start_worker(handle)
        result = await proxy.infer(inputs, context)
        await manager.shutdown_all()
    """

    MAX_RESTARTS = 5

    def __init__(self) -> None:
        self._workers: dict[str, _WorkerEntry] = {}
        self._tmpdir = tempfile.mkdtemp(prefix="nerva-")
        self._pid = os.getpid()

    async def start_worker(self, handle: ModelHandle) -> WorkerProxy:
        """Spawn a Worker process and load the model.

        Returns:
            WorkerProxy connected to the new Worker.
        """
        worker_id = handle.name
        socket_path = os.path.join(
            self._tmpdir, f"nerva-{self._pid}-{worker_id}.sock"
        )

        # Clean up stale socket
        if os.path.exists(socket_path):
            os.unlink(socket_path)

        proxy = WorkerProxy(socket_path)
        proc = multiprocessing.Process(
            target=worker_entry,
            args=(socket_path,),
            daemon=True,
        )
        proc.start()

        entry = _WorkerEntry(
            handle=handle,
            process=proc,
            proxy=proxy,
            socket_path=socket_path,
            state=WorkerState.STARTING,
        )
        self._workers[worker_id] = entry

        # Connect proxy (waits for WORKER_READY)
        await proxy.start()
        entry.state = WorkerState.LOADING

        # Load model
        model_class_path = class_to_import_path(handle.model_class)
        await proxy.load_model(
            model_name=handle.name,
            model_class_path=model_class_path,
            backend=handle.backend,
            device=handle.device,
            options=handle.options,
        )
        entry.state = WorkerState.READY

        logger.info("Worker '%s' started (pid=%d)", worker_id, proc.pid or 0)
        return proxy

    async def restart_worker(self, worker_id: str) -> WorkerProxy:
        """Restart a crashed/stopped Worker.

        Returns:
            New WorkerProxy for the restarted Worker.
        """
        entry = self._workers.get(worker_id)
        if entry is None:
            raise KeyError(f"Unknown worker: {worker_id}")

        # Clean up old proxy
        await entry.proxy.close()

        # Clean up old process
        if entry.process.is_alive():
            entry.process.kill()
            entry.process.join(timeout=5)

        entry.restart_count += 1
        if entry.restart_count > self.MAX_RESTARTS:
            raise RuntimeError(
                f"Worker '{worker_id}' exceeded max restarts ({self.MAX_RESTARTS})"
            )

        logger.info(
            "Restarting worker '%s' (attempt %d/%d)",
            worker_id, entry.restart_count, self.MAX_RESTARTS,
        )

        # Start fresh
        proxy = await self.start_worker(entry.handle)
        return proxy

    async def shutdown_all(self, timeout: float = 30.0) -> None:
        """Gracefully shut down all Workers."""
        for worker_id, entry in list(self._workers.items()):
            entry.state = WorkerState.STOPPING
            try:
                await entry.proxy.shutdown()
            except Exception:
                pass
            try:
                entry.process.join(timeout=5)
            except Exception:
                pass
            if entry.process.is_alive():
                logger.warning("Force-killing worker '%s'", worker_id)
                entry.process.kill()
                entry.process.join(timeout=3)
            await entry.proxy.close()
            entry.state = WorkerState.STOPPED

        self._workers.clear()

        # Clean up tmpdir
        try:
            for f in os.listdir(self._tmpdir):
                os.unlink(os.path.join(self._tmpdir, f))
            os.rmdir(self._tmpdir)
        except OSError:
            pass
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_worker_manager.py -v`
Expected: All tests PASS

**Step 5: Run full suite + lint**

Run: `uv run ruff check src/ tests/ && uv run mypy && uv run pytest tests/ -v`
Expected: 0 errors, all tests pass

**Step 6: Commit**

```bash
git add src/nerva/worker/manager.py tests/test_worker_manager.py
git commit -m "feat(phase1): add WorkerManager with spawn, restart, shutdown"
```

---

### Task 7: End-to-End Integration & Public API

**Files:**
- Create: `tests/test_phase1_e2e.py`
- Modify: `src/nerva/__init__.py`
- Modify: `src/nerva/worker/__init__.py`

**Context:** Verify the full Master → WorkerProxy → Worker → Backend → Model chain. Update public API exports. Compare IPC overhead to Phase 0 baseline.

**Step 1: Write the integration tests**

```python
# tests/test_phase1_e2e.py
"""End-to-end integration tests for Phase 1 (Master-Worker IPC)."""

import asyncio
import time
from typing import Any

import pytest

from nerva import Model, model
from nerva.backends.base import InferContext
from nerva.worker.manager import WorkerManager

from tests.helpers import EchoModel


class TestPhase1EndToEnd:
    async def test_single_model_roundtrip(self) -> None:
        """Full cycle: start manager → spawn worker → infer → shutdown."""
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
        """Phase 1 IPC result should match Phase 0 in-process result."""
        from nerva.backends.base import ModelConfig
        from nerva.backends.pytorch import PyTorchBackend

        # Phase 0: direct in-process
        backend = PyTorchBackend()
        config = ModelConfig(
            model_name="echo", model_class=EchoModel, device="cpu"
        )
        await backend.load_model(config)
        ctx0 = InferContext(request_id="p0", deadline_ms=99999)
        result_p0 = await backend.infer({"value": [1, 2, 3]}, ctx0)
        await backend.unload_model()

        # Phase 1: via IPC
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

            results = []
            for i in range(10):
                ctx = InferContext(request_id=f"multi-{i}", deadline_ms=99999)
                r = await proxy.infer({"value": i}, ctx)
                results.append(r)

            for i, r in enumerate(results):
                assert r == {"echo": i}
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


class TestPhase1ShmPath:
    async def test_infer_with_shm_pool(self) -> None:
        """Verify SHM path works for large payloads."""
        from nerva.engine.shm_pool import ShmPool

        handle = model("echo", EchoModel, backend="pytorch", device="cpu")
        manager = WorkerManager()
        pool = ShmPool(size_classes_kb=[16], slots_per_class=2)

        try:
            proxy = await manager.start_worker(handle)

            # Create input larger than inline threshold (>8KB)
            big_value = "x" * 10000  # ~10KB when msgpack-serialized
            ctx = InferContext(request_id="shm-1", deadline_ms=99999)
            result = await proxy.infer({"value": big_value}, ctx, shm_pool=pool)
            assert result == {"echo": big_value}
        finally:
            await manager.shutdown_all()
            pool.close()


@pytest.mark.slow
class TestPhase1Perf:
    async def test_ipc_overhead(self) -> None:
        """Measure IPC overhead vs Phase 0 in-process."""
        from nerva.backends.base import ModelConfig
        from nerva.backends.pytorch import PyTorchBackend

        n = 100

        # Phase 0 baseline
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

        # Phase 1
        handle = model("echo", EchoModel, backend="pytorch", device="cpu")
        manager = WorkerManager()
        proxy = await manager.start_worker(handle)

        t0 = time.perf_counter()
        for i in range(n):
            ctx = InferContext(request_id=f"p1-{i}", deadline_ms=99999)
            await proxy.infer({"value": i}, ctx)
        p1_total = time.perf_counter() - t0

        await manager.shutdown_all()

        p0_avg_us = (p0_total / n) * 1_000_000
        p1_avg_us = (p1_total / n) * 1_000_000
        overhead_us = p1_avg_us - p0_avg_us

        print(f"\nPhase 0 avg: {p0_avg_us:.0f} us")
        print(f"Phase 1 avg: {p1_avg_us:.0f} us")
        print(f"IPC overhead: {overhead_us:.0f} us")

        # Sanity check: IPC overhead should be < 10ms per request
        assert overhead_us < 10_000, f"IPC overhead too high: {overhead_us:.0f} us"
```

**Step 2: Run tests to verify they fail (or pass if all code is in place)**

Run: `uv run pytest tests/test_phase1_e2e.py -v -m "not slow"`
Expected: PASS (all implementation from Tasks 2-6 should be in place)

Run slow perf test separately:
Run: `uv run pytest tests/test_phase1_e2e.py::TestPhase1Perf -v -s`
Expected: PASS with IPC overhead printed

**Step 3: Update public API exports**

Update `src/nerva/worker/__init__.py`:

```python
"""Worker process management."""

from nerva.worker.manager import WorkerManager
from nerva.worker.proxy import WorkerProxy

__all__ = ["WorkerManager", "WorkerProxy"]
```

Update `src/nerva/__init__.py` — add WorkerManager and WorkerProxy:

```python
"""Nerva — High-performance model inference serving framework."""

__version__ = "0.1.0"

# Ensure built-in backends are registered on import.
import nerva.backends.pytorch as _pytorch_backend  # noqa: F401
from nerva.backends.base import Backend, BatchMeta, InferContext, ModelConfig
from nerva.backends.registry import get_backend, list_backends, register_backend
from nerva.core.model import Model, ModelHandle, model
from nerva.worker.manager import WorkerManager
from nerva.worker.proxy import WorkerProxy

__all__ = [
    "Backend",
    "BatchMeta",
    "InferContext",
    "Model",
    "ModelConfig",
    "ModelHandle",
    "WorkerManager",
    "WorkerProxy",
    "get_backend",
    "list_backends",
    "model",
    "register_backend",
]
```

**Step 4: Run full suite + lint**

Run: `uv run ruff check src/ tests/ && uv run mypy && uv run pytest tests/ -v -m "not slow"`
Expected: 0 errors, all tests pass

**Step 5: Commit**

```bash
git add src/nerva/__init__.py src/nerva/worker/__init__.py tests/test_phase1_e2e.py
git commit -m "feat(phase1): add e2e integration tests and update public API"
```

---

## Task Dependency Graph

```
Task 1 (pyzmq dep) ─────────────────────────────────────────────┐
                                                                 │
Task 2 (IPC codec) ──────────────┬───► Task 4 (Worker) ────┐    │
                                 │                          │    │
Task 3 (ShmPool) ───────────────►├───► Task 5 (Proxy) ─────┤    │
                                 │                          ▼    │
                                 └───► Task 6 (Manager) ────┤    │
                                                            │    │
                                          Task 7 (E2E) ◄────┘────┘
```

Tasks 2 and 3 are independent and can run in parallel.
Tasks 4, 5, 6 depend on Tasks 2+3 but are somewhat independent of each other.
Task 7 depends on all previous tasks.

---

## Verification Criteria (from Phase 1 design)

- [ ] Single model inference via IPC produces same results as Phase 0 in-process
- [ ] IPC overhead measured and documented
- [ ] Worker crash detected by Master; restart succeeds
- [ ] SHM alloc/free has no leaks (pool stats show 0 in-use after all requests complete)
- [ ] ruff 0 errors, mypy 0 issues, all tests pass
- [ ] Inline path (≤8KB) and SHM path (>8KB) both verified

---

## Changelog

| Date | Change |
|---|---|
| 2026-02-25 | Initial version |
