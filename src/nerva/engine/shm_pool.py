"""POSIX shared memory pool with size-class based allocation and bitmap tracking.

Master process owns all alloc/free. Workers open segments by name from descriptors.
Each size class maps to one SharedMemory segment, internally divided into equal-sized
slots tracked by a bitmap.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IPC_CONTROL_INLINE_MAX_BYTES: int = 8 * 1024
"""Data payloads <= this size are sent inline via the IPC control channel."""

DEFAULT_SIZE_CLASSES_KB: list[int] = [4, 16, 64, 256, 1024, 4096]
"""Default size classes in KB for the shared memory pool."""

DEFAULT_SLOTS_PER_CLASS: int = 16
"""Default number of slots per size class."""


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ShmPoolExhausted(Exception):  # noqa: N818
    """Raised when no slot is available in any fitting size class."""


# ---------------------------------------------------------------------------
# ShmSlot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShmSlot:
    """Handle to an allocated slot in the shared memory pool."""

    shm_name: str
    offset: int
    slot_size: int
    _class_idx: int
    _slot_idx: int


# ---------------------------------------------------------------------------
# _SlotBitmap — internal allocation bitmap
# ---------------------------------------------------------------------------


class _SlotBitmap:
    """Bitmap allocator for a fixed number of slots."""

    __slots__ = ("_bits", "_capacity")

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._bits = 0  # each bit: 1 = in-use, 0 = free

    def alloc(self) -> int | None:
        """Return the index of the first free slot, or None if full."""
        for i in range(self._capacity):
            if not (self._bits & (1 << i)):
                self._bits |= 1 << i
                return i
        return None

    def free(self, slot_idx: int) -> None:
        """Mark *slot_idx* as free."""
        self._bits &= ~(1 << slot_idx)

    @property
    def in_use(self) -> int:
        """Number of currently allocated slots."""
        return bin(self._bits).count("1")


# ---------------------------------------------------------------------------
# ShmPool
# ---------------------------------------------------------------------------


class ShmPool:
    """Shared memory pool with size-class based allocation.

    Parameters
    ----------
    size_classes_kb:
        Sorted list of size classes in KB.
    slots_per_class:
        Number of slots per size class.
    name_prefix:
        Short prefix for POSIX SHM segment names.  Segment names follow the
        format ``nv{pid}-{prefix}-{idx}`` and must stay under ~30 chars on
        macOS.
    """

    def __init__(
        self,
        size_classes_kb: list[int] | None = None,
        slots_per_class: int = DEFAULT_SLOTS_PER_CLASS,
        name_prefix: str = "nv",
    ) -> None:
        if size_classes_kb is None:
            size_classes_kb = list(DEFAULT_SIZE_CLASSES_KB)

        self._size_classes_bytes = [k * 1024 for k in size_classes_kb]
        self._slots_per_class = slots_per_class
        self._prefix = name_prefix
        self._closed = False

        pid = os.getpid()
        self._segments: list[SharedMemory] = []
        self._bitmaps: list[_SlotBitmap] = []

        for idx, class_bytes in enumerate(self._size_classes_bytes):
            seg_name = f"nv{pid}-{name_prefix}-{idx}"
            total_bytes = class_bytes * slots_per_class

            # Clean up stale segment with the same name (best-effort).
            self._try_unlink(seg_name)

            shm = SharedMemory(name=seg_name, create=True, size=total_bytes)
            self._segments.append(shm)
            self._bitmaps.append(_SlotBitmap(slots_per_class))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def alloc(self, size: int) -> ShmSlot:
        """Allocate a slot that can hold *size* bytes.

        Rounds up to the smallest fitting size class.
        Raises ``ShmPoolExhausted`` if no slot is available.
        Raises ``RuntimeError`` if the pool has been closed.
        """
        if self._closed:
            raise RuntimeError("ShmPool is closed")

        # Try the smallest fitting class first; if full, promote to the next
        # larger class. This prevents a full class from causing request failures
        # when larger classes still have capacity (e.g. during burst traffic).
        first_fit_idx: int | None = None
        for idx, class_bytes in enumerate(self._size_classes_bytes):
            if class_bytes >= size:
                if first_fit_idx is None:
                    first_fit_idx = idx
                slot_idx = self._bitmaps[idx].alloc()
                if slot_idx is not None:
                    return ShmSlot(
                        shm_name=self._segments[idx].name,
                        offset=slot_idx * class_bytes,
                        slot_size=class_bytes,
                        _class_idx=idx,
                        _slot_idx=slot_idx,
                    )
                # This class is full — try the next bigger class.

        if first_fit_idx is not None:
            raise ShmPoolExhausted(
                f"All size classes >= {size} bytes exhausted "
                f"({self._slots_per_class} slots per class, all in use)"
            )

        raise ShmPoolExhausted(
            f"Requested size {size} exceeds largest size class "
            f"({self._size_classes_bytes[-1]})"
        )

    def free(self, slot: ShmSlot) -> None:
        """Return *slot* to the pool."""
        self._bitmaps[slot._class_idx].free(slot._slot_idx)

    def write(self, slot: ShmSlot, data: bytes) -> None:
        """Write *data* into *slot*.  Raises ``ValueError`` if data exceeds slot size."""
        if len(data) > slot.slot_size:
            raise ValueError(
                f"Data length {len(data)} exceeds slot size {slot.slot_size}"
            )
        seg = self._segments[slot._class_idx]
        buf = seg.buf
        assert buf is not None
        buf[slot.offset : slot.offset + len(data)] = data

    def read_view(self, slot: ShmSlot, length: int) -> memoryview:
        """Return a memoryview of *length* bytes from *slot* without copying."""
        seg = self._segments[slot._class_idx]
        buf = seg.buf
        assert buf is not None
        return memoryview(buf)[slot.offset : slot.offset + length]

    def read(self, slot: ShmSlot, length: int) -> bytes:
        """Read *length* bytes from *slot*."""
        view = self.read_view(slot, length)
        try:
            return view.tobytes()
        finally:
            view.release()

    def close(self) -> None:
        """Release and unlink all shared memory segments.  Idempotent."""
        if self._closed:
            return
        self._closed = True
        for shm in self._segments:
            try:
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
        self._segments.clear()
        self._bitmaps.clear()

    @property
    def stats(self) -> dict[int, dict[str, int]]:
        """Per-size-class statistics: ``{size_bytes: {total, in_use}}``."""
        result: dict[int, dict[str, int]] = {}
        for idx, class_bytes in enumerate(self._size_classes_bytes):
            in_use = self._bitmaps[idx].in_use if idx < len(self._bitmaps) else 0
            result[class_bytes] = {
                "total": self._slots_per_class,
                "in_use": in_use,
            }
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _try_unlink(name: str) -> None:
        """Best-effort cleanup of a stale SHM segment."""
        try:
            stale = SharedMemory(name=name, create=False)
            stale.close()
            stale.unlink()
        except FileNotFoundError:
            pass
