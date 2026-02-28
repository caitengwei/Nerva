# tests/test_protocol.py
"""Tests for nerva.server.protocol — Binary frame codec."""

import pytest

from nerva.server.protocol import (
    HEADER_SIZE,
    MAGIC,
    MAX_PAYLOAD_BYTES,
    VERSION,
    Frame,
    FrameType,
    ProtocolError,
    decode_frame,
    encode_frame,
)


class TestFrameType:
    def test_all_types_defined(self) -> None:
        assert FrameType.OPEN == 1
        assert FrameType.DATA == 2
        assert FrameType.END == 3
        assert FrameType.ERROR == 4
        assert FrameType.HEARTBEAT == 5


class TestConstants:
    def test_header_size(self) -> None:
        assert HEADER_SIZE == 32

    def test_magic(self) -> None:
        assert MAGIC == 0x4E56

    def test_version(self) -> None:
        assert VERSION == 1

    def test_max_payload(self) -> None:
        assert MAX_PAYLOAD_BYTES == 4 * 1024 * 1024


class TestRoundtrip:
    def test_data_frame_roundtrip(self) -> None:
        frame = Frame(
            frame_type=FrameType.DATA,
            request_id=42,
            flags=0,
            payload=b'{"value": 1}',
        )
        raw = encode_frame(frame)
        decoded, consumed = decode_frame(raw)
        assert consumed == HEADER_SIZE + len(frame.payload)
        assert decoded.frame_type == FrameType.DATA
        assert decoded.request_id == 42
        assert decoded.flags == 0
        assert decoded.payload == b'{"value": 1}'

    def test_open_frame_roundtrip(self) -> None:
        frame = Frame(
            frame_type=FrameType.OPEN,
            request_id=100,
            flags=0,
            payload=b"\x81\xa8pipeline\xa8classify",  # msgpack
        )
        raw = encode_frame(frame)
        decoded, _consumed = decode_frame(raw)
        assert decoded.frame_type == FrameType.OPEN
        assert decoded.request_id == 100
        assert decoded.payload == frame.payload

    def test_empty_payload(self) -> None:
        frame = Frame(frame_type=FrameType.END, request_id=1, flags=0, payload=b"")
        raw = encode_frame(frame)
        decoded, consumed = decode_frame(raw)
        assert decoded.frame_type == FrameType.END
        assert decoded.payload == b""
        assert consumed == HEADER_SIZE

    def test_error_frame_roundtrip(self) -> None:
        frame = Frame(
            frame_type=FrameType.ERROR,
            request_id=99,
            flags=0,
            payload=b"\x83\xa4code\x04\xa7message\xb2DEADLINE_EXCEEDED\xa9retryable\xc3",
        )
        raw = encode_frame(frame)
        decoded, _ = decode_frame(raw)
        assert decoded.frame_type == FrameType.ERROR
        assert decoded.request_id == 99


class TestDecodeValidation:
    def test_short_buffer(self) -> None:
        with pytest.raises(ProtocolError, match="incomplete header"):
            decode_frame(b"\x00" * 10)

    def test_bad_magic(self) -> None:
        raw = b"\xFF\xFF" + b"\x00" * 30
        with pytest.raises(ProtocolError, match="magic"):
            decode_frame(raw)

    def test_bad_version(self) -> None:
        import struct

        header = struct.pack(">HBB", 0x4E56, 99, 2)  # bad version
        header += b"\x00" * (32 - len(header))
        with pytest.raises(ProtocolError, match="version"):
            decode_frame(header)

    def test_payload_exceeds_max(self) -> None:
        frame = Frame(frame_type=FrameType.DATA, request_id=1, flags=0, payload=b"x")
        raw = bytearray(encode_frame(frame))
        # Overwrite payload_length to exceed max
        import struct

        struct.pack_into(">I", raw, 20, MAX_PAYLOAD_BYTES + 1)
        with pytest.raises(ProtocolError, match="payload"):
            decode_frame(bytes(raw))

    def test_incomplete_payload(self) -> None:
        frame = Frame(
            frame_type=FrameType.DATA, request_id=1, flags=0, payload=b"hello"
        )
        raw = encode_frame(frame)
        with pytest.raises(ProtocolError, match="incomplete payload"):
            decode_frame(raw[:-2])  # Truncate payload


class TestEncodeValidation:
    def test_payload_too_large(self) -> None:
        frame = Frame(
            frame_type=FrameType.DATA,
            request_id=1,
            flags=0,
            payload=b"x" * (MAX_PAYLOAD_BYTES + 1),
        )
        with pytest.raises(ProtocolError, match="payload"):
            encode_frame(frame)
