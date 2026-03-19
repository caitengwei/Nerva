"""Tests for nerva.engine.shm_pool — Shared Memory Pool with size classes."""

from __future__ import annotations

import pytest

from nerva.engine.shm_pool import (
    DEFAULT_SIZE_CLASSES_KB,
    DEFAULT_SLOTS_PER_CLASS,
    IPC_CONTROL_INLINE_MAX_BYTES,
    ShmPool,
    ShmPoolExhausted,
    ShmSlot,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_inline_threshold(self) -> None:
        assert IPC_CONTROL_INLINE_MAX_BYTES == 8 * 1024

    def test_default_size_classes(self) -> None:
        assert DEFAULT_SIZE_CLASSES_KB == [4, 16, 64, 256, 1024, 4096]

    def test_default_slots_per_class(self) -> None:
        assert DEFAULT_SLOTS_PER_CLASS == 16


# ---------------------------------------------------------------------------
# Alloc basics
# ---------------------------------------------------------------------------


class TestAlloc:
    def test_alloc_returns_shm_slot(self) -> None:
        pool = ShmPool(size_classes_kb=[4], slots_per_class=2, name_prefix="t1")
        try:
            slot = pool.alloc(100)
            assert isinstance(slot, ShmSlot)
            assert slot.slot_size == 4 * 1024
        finally:
            pool.close()

    def test_alloc_rounds_up_to_next_size_class(self) -> None:
        pool = ShmPool(size_classes_kb=[4, 16, 64], slots_per_class=2, name_prefix="t2")
        try:
            # Request 5 KB — should round up to 16 KB class
            slot = pool.alloc(5 * 1024)
            assert slot.slot_size == 16 * 1024
        finally:
            pool.close()

    def test_alloc_exact_boundary(self) -> None:
        pool = ShmPool(size_classes_kb=[4, 16], slots_per_class=2, name_prefix="t3")
        try:
            slot = pool.alloc(4 * 1024)
            assert slot.slot_size == 4 * 1024
        finally:
            pool.close()

    def test_alloc_multiple_unique_offsets(self) -> None:
        pool = ShmPool(size_classes_kb=[4], slots_per_class=4, name_prefix="t4")
        try:
            slots = [pool.alloc(100) for _ in range(4)]
            offsets = {s.offset for s in slots}
            assert len(offsets) == 4, "All slots should have unique offsets"
        finally:
            pool.close()


# ---------------------------------------------------------------------------
# Pool exhaustion
# ---------------------------------------------------------------------------


class TestExhaustion:
    def test_pool_exhausted(self) -> None:
        pool = ShmPool(size_classes_kb=[4], slots_per_class=2, name_prefix="t5")
        try:
            pool.alloc(100)
            pool.alloc(100)
            with pytest.raises(ShmPoolExhausted):
                pool.alloc(100)
        finally:
            pool.close()

    def test_size_exceeding_max_class_raises(self) -> None:
        pool = ShmPool(size_classes_kb=[4, 16], slots_per_class=2, name_prefix="t6")
        try:
            with pytest.raises(ShmPoolExhausted):
                pool.alloc(17 * 1024)
        finally:
            pool.close()


# ---------------------------------------------------------------------------
# Size class promotion (fallback)
# ---------------------------------------------------------------------------


class TestSizeClassPromotion:
    def test_promotes_to_larger_class_when_full(self) -> None:
        """When the best-fit class is full, alloc promotes to the next larger class."""
        pool = ShmPool(size_classes_kb=[4, 16], slots_per_class=1, name_prefix="tp1")
        try:
            # Fill the 4KB class.
            s1 = pool.alloc(100)
            assert s1.slot_size == 4 * 1024
            # Next 4KB-fitting alloc should promote to 16KB.
            s2 = pool.alloc(100)
            assert s2.slot_size == 16 * 1024
        finally:
            pool.close()

    def test_all_fitting_classes_exhausted(self) -> None:
        """When all fitting classes are full, raises ShmPoolExhausted."""
        pool = ShmPool(size_classes_kb=[4, 16], slots_per_class=1, name_prefix="tp2")
        try:
            pool.alloc(100)   # fills 4KB class
            pool.alloc(100)   # promotes to 16KB class
            with pytest.raises(ShmPoolExhausted, match="All size classes"):
                pool.alloc(100)  # both classes full
        finally:
            pool.close()

    def test_promotion_skips_non_fitting_classes(self) -> None:
        """Alloc for a 5KB request skips the 4KB class entirely."""
        pool = ShmPool(size_classes_kb=[4, 16, 64], slots_per_class=1, name_prefix="tp3")
        try:
            s = pool.alloc(5 * 1024)
            assert s.slot_size == 16 * 1024
            # 16KB full, should promote to 64KB.
            s2 = pool.alloc(5 * 1024)
            assert s2.slot_size == 64 * 1024
        finally:
            pool.close()


# ---------------------------------------------------------------------------
# Free
# ---------------------------------------------------------------------------


class TestFree:
    def test_free_makes_slot_available(self) -> None:
        pool = ShmPool(size_classes_kb=[4], slots_per_class=1, name_prefix="t7")
        try:
            slot = pool.alloc(100)
            pool.free(slot)
            # Should be able to alloc again
            slot2 = pool.alloc(100)
            assert slot2.offset == slot.offset
        finally:
            pool.close()


# ---------------------------------------------------------------------------
# Write / Read roundtrip
# ---------------------------------------------------------------------------


class TestWriteRead:
    def test_roundtrip(self) -> None:
        pool = ShmPool(size_classes_kb=[4], slots_per_class=2, name_prefix="t8")
        try:
            slot = pool.alloc(100)
            data = b"hello shared memory"
            pool.write(slot, data)
            result = pool.read(slot, len(data))
            assert result == data
        finally:
            pool.close()

    def test_read_view_roundtrip(self) -> None:
        pool = ShmPool(size_classes_kb=[4], slots_per_class=2, name_prefix="tv")
        try:
            slot = pool.alloc(100)
            data = b"hello shared memory view"
            pool.write(slot, data)
            view = pool.read_view(slot, len(data))
            try:
                assert isinstance(view, memoryview)
                assert view.tobytes() == data
            finally:
                view.release()
        finally:
            pool.close()

    def test_write_full_slot(self) -> None:
        pool = ShmPool(size_classes_kb=[4], slots_per_class=2, name_prefix="t9")
        try:
            slot = pool.alloc(100)
            data = b"\xab" * (4 * 1024)
            pool.write(slot, data)
            result = pool.read(slot, len(data))
            assert result == data
        finally:
            pool.close()

    def test_write_exceeding_slot_raises(self) -> None:
        pool = ShmPool(size_classes_kb=[4], slots_per_class=2, name_prefix="ta")
        try:
            slot = pool.alloc(100)
            data = b"\x00" * (4 * 1024 + 1)
            with pytest.raises(ValueError, match="exceeds slot size"):
                pool.write(slot, data)
        finally:
            pool.close()


# ---------------------------------------------------------------------------
# Close
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_is_idempotent(self) -> None:
        pool = ShmPool(size_classes_kb=[4], slots_per_class=2, name_prefix="tb")
        pool.close()
        pool.close()  # Should not raise

    def test_alloc_after_close_raises(self) -> None:
        pool = ShmPool(size_classes_kb=[4], slots_per_class=2, name_prefix="tc")
        pool.close()
        with pytest.raises(RuntimeError):
            pool.alloc(100)

    def test_stats_after_close(self) -> None:
        pool = ShmPool(size_classes_kb=[4, 16], slots_per_class=2, name_prefix="te")
        pool.alloc(100)
        pool.close()

        stats = pool.stats
        assert stats[4 * 1024]["total"] == 2
        assert stats[4 * 1024]["in_use"] == 0
        assert stats[16 * 1024]["total"] == 2
        assert stats[16 * 1024]["in_use"] == 0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_in_use(self) -> None:
        pool = ShmPool(size_classes_kb=[4, 16], slots_per_class=4, name_prefix="td")
        try:
            pool.alloc(100)
            pool.alloc(100)
            pool.alloc(5 * 1024)
            stats = pool.stats
            assert stats[4 * 1024]["total"] == 4
            assert stats[4 * 1024]["in_use"] == 2
            assert stats[16 * 1024]["total"] == 4
            assert stats[16 * 1024]["in_use"] == 1
        finally:
            pool.close()


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


class TestNaming:
    def test_slot_names_contain_prefix(self) -> None:
        pool = ShmPool(size_classes_kb=[4], slots_per_class=2, name_prefix="myp")
        try:
            slot = pool.alloc(100)
            assert "myp" in slot.shm_name
        finally:
            pool.close()
