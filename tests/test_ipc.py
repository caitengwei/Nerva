"""Tests for IPC message codec, Descriptor, and import utilities."""

from __future__ import annotations

import uuid

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

# ---------------------------------------------------------------------------
# MessageType enum
# ---------------------------------------------------------------------------

class TestMessageType:
    def test_has_all_expected_types(self) -> None:
        expected = {
            "LOAD_MODEL",
            "LOAD_MODEL_ACK",
            "INFER_SUBMIT",
            "INFER_ACK",
            "SHM_ALLOC_REQUEST",
            "SHM_ALLOC_RESPONSE",
            "CANCEL",
            "HEALTH_CHECK",
            "HEALTH_STATUS",
            "SHUTDOWN",
            "WORKER_READY",
        }
        actual = {m.name for m in MessageType}
        assert actual == expected

    def test_values_are_strings(self) -> None:
        for m in MessageType:
            assert isinstance(m.value, str)


# ---------------------------------------------------------------------------
# AckStatus enum
# ---------------------------------------------------------------------------

class TestAckStatus:
    def test_has_all_expected_statuses(self) -> None:
        expected = {
            "OK",
            "INVALID_ARGUMENT",
            "DEADLINE_EXCEEDED",
            "ABORTED",
            "RESOURCE_EXHAUSTED",
            "UNAVAILABLE",
            "INTERNAL",
        }
        actual = {s.name for s in AckStatus}
        assert actual == expected

    def test_values_are_strings(self) -> None:
        for s in AckStatus:
            assert isinstance(s.value, str)


# ---------------------------------------------------------------------------
# Message roundtrip (encode / decode)
# ---------------------------------------------------------------------------

class TestMessageRoundtrip:
    def test_simple_dict(self) -> None:
        msg = {"type": "HEALTH_CHECK", "ts": 12345}
        assert decode_message(encode_message(msg)) == msg

    def test_with_bytes(self) -> None:
        payload = b"\x00\x01\x02\xff"
        msg = {"type": "INFER_SUBMIT", "data": payload}
        decoded = decode_message(encode_message(msg))
        assert decoded["data"] == payload

    def test_with_none(self) -> None:
        msg = {"type": "CANCEL", "request_id": None}
        decoded = decode_message(encode_message(msg))
        assert decoded["request_id"] is None

    def test_full_infer_submit(self) -> None:
        rid = str(uuid.uuid4())
        msg = {
            "type": MessageType.INFER_SUBMIT.value,
            "request_id": rid,
            "model": "bert-base",
            "descriptor": {
                "request_id": rid,
                "node_id": 0,
                "schema_version": 1,
                "shm_id": None,
                "offset": 0,
                "length": 128,
                "inline_data": b"\x00" * 128,
                "dtype": "float32",
                "shape": [1, 128],
                "device": "cuda:0",
                "lifetime_token": 42,
                "checksum": 0,
            },
        }
        decoded = decode_message(encode_message(msg))
        assert decoded["request_id"] == rid
        assert decoded["descriptor"]["inline_data"] == b"\x00" * 128
        assert decoded["descriptor"]["shape"] == [1, 128]

    def test_infer_ack(self) -> None:
        msg = {
            "type": MessageType.INFER_ACK.value,
            "request_id": "abc-123",
            "status": AckStatus.OK.value,
        }
        decoded = decode_message(encode_message(msg))
        assert decoded["status"] == AckStatus.OK.value

    def test_load_model(self) -> None:
        msg = {
            "type": MessageType.LOAD_MODEL.value,
            "model_name": "gpt2",
            "backend": "nerva.backends.pytorch:PyTorchBackend",
        }
        decoded = decode_message(encode_message(msg))
        assert decoded["model_name"] == "gpt2"


# ---------------------------------------------------------------------------
# Descriptor
# ---------------------------------------------------------------------------

class TestDescriptor:
    def test_default_values(self) -> None:
        d = Descriptor(request_id="r1", node_id=0)
        assert d.schema_version == 1
        assert d.shm_id is None
        assert d.offset == 0
        assert d.length == 0
        assert d.inline_data is None
        assert d.dtype == "bytes"
        assert d.shape == []
        assert d.device == "cpu"
        assert d.lifetime_token == 0
        assert d.checksum == 0

    def test_roundtrip_dict(self) -> None:
        d = Descriptor(
            request_id="r2",
            node_id=1,
            shm_id="shm-abc",
            offset=64,
            length=256,
            dtype="float32",
            shape=[2, 128],
            device="cuda:0",
        )
        d2 = Descriptor.from_dict(d.to_dict())
        assert d2 == d

    def test_inline_check(self) -> None:
        d = Descriptor(
            request_id="r3",
            node_id=0,
            inline_data=b"\x01\x02\x03",
            length=3,
        )
        rebuilt = Descriptor.from_dict(d.to_dict())
        assert rebuilt.inline_data == b"\x01\x02\x03"
        assert rebuilt.length == 3

    def test_is_inline_true(self) -> None:
        d = Descriptor(request_id="r4", node_id=0, inline_data=b"hello")
        assert d.is_inline is True

    def test_is_inline_false_when_none(self) -> None:
        d = Descriptor(request_id="r5", node_id=0)
        assert d.is_inline is False

    def test_is_inline_false_when_empty(self) -> None:
        d = Descriptor(request_id="r6", node_id=0, inline_data=b"")
        assert d.is_inline is False


# ---------------------------------------------------------------------------
# Import path utilities
# ---------------------------------------------------------------------------

class TestImportPath:
    def test_roundtrip(self) -> None:
        path = class_to_import_path(Descriptor)
        assert path == "nerva.worker.ipc:Descriptor"
        cls = import_path_to_class(path)
        assert cls is Descriptor

    def test_invalid_module(self) -> None:
        with pytest.raises((ImportError, ModuleNotFoundError)):
            import_path_to_class("nonexistent.module:Foo")

    def test_invalid_class(self) -> None:
        with pytest.raises(AttributeError):
            import_path_to_class("nerva.worker.ipc:NonExistentClass")
