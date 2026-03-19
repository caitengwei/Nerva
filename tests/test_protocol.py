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

    def test_request_id_negative(self) -> None:
        frame = Frame(frame_type=FrameType.DATA, request_id=-1, flags=0, payload=b"")
        with pytest.raises(ProtocolError, match=r"request_id.*out of u64 range"):
            encode_frame(frame)

    def test_request_id_overflow(self) -> None:
        frame = Frame(
            frame_type=FrameType.DATA, request_id=(1 << 64), flags=0, payload=b""
        )
        with pytest.raises(ProtocolError, match=r"request_id.*out of u64 range"):
            encode_frame(frame)


class TestDecodeWithOffset:
    def test_decode_at_nonzero_offset(self) -> None:
        """decode_frame with explicit offset parses the correct frame."""
        f1 = Frame(frame_type=FrameType.OPEN, request_id=1, flags=0, payload=b"first")
        f2 = Frame(frame_type=FrameType.DATA, request_id=2, flags=0, payload=b"second")
        raw = encode_frame(f1) + encode_frame(f2)
        offset1 = HEADER_SIZE + len(f1.payload)
        decoded, consumed = decode_frame(raw, offset1)
        assert decoded.frame_type == FrameType.DATA
        assert decoded.request_id == 2
        assert bytes(decoded.payload) == b"second"
        assert consumed == HEADER_SIZE + len(f2.payload)

    def test_multi_frame_offset_loop(self) -> None:
        """Parse three concatenated frames via offset loop — no slicing."""
        payloads = [b"aaa", b"bbbbbb", b"c"]
        originals = [
            Frame(frame_type=FrameType.DATA, request_id=i, flags=0, payload=p)
            for i, p in enumerate(payloads)
        ]
        raw = b"".join(encode_frame(f) for f in originals)
        parsed: list[Frame] = []
        offset = 0
        while offset < len(raw):
            frame, consumed = decode_frame(raw, offset)
            parsed.append(frame)
            offset += consumed
        assert len(parsed) == 3
        for orig, dec in zip(originals, parsed, strict=True):
            assert dec.request_id == orig.request_id
            assert bytes(dec.payload) == orig.payload

    def test_decode_returns_memoryview_payload(self) -> None:
        """Decoded payload is a memoryview (zero-copy) when input is bytes."""
        frame = Frame(frame_type=FrameType.DATA, request_id=1, flags=0, payload=b"hello")
        raw = encode_frame(frame)
        decoded, _ = decode_frame(raw)
        assert isinstance(decoded.payload, memoryview)
        assert bytes(decoded.payload) == b"hello"

    def test_decode_memoryview_input(self) -> None:
        """decode_frame accepts memoryview input directly."""
        frame = Frame(frame_type=FrameType.DATA, request_id=7, flags=0, payload=b"mv-test")
        raw = encode_frame(frame)
        mv = memoryview(raw)
        decoded, consumed = decode_frame(mv, 0)
        assert isinstance(decoded.payload, memoryview)
        assert bytes(decoded.payload) == b"mv-test"
        assert consumed == HEADER_SIZE + len(b"mv-test")

    def test_incomplete_header_at_offset(self) -> None:
        """Short buffer at offset raises ProtocolError."""
        frame = Frame(frame_type=FrameType.DATA, request_id=1, flags=0, payload=b"x")
        raw = encode_frame(frame)
        with pytest.raises(ProtocolError, match="incomplete header"):
            decode_frame(raw, len(raw) - 5)  # only 5 bytes left


class TestDecodeUnknownFrameType:
    def test_unknown_frame_type_raises_protocol_error(self) -> None:
        """Unknown frame_type byte should raise ProtocolError, not ValueError."""
        import struct

        from nerva.server.protocol import _HEADER_FMT

        header = struct.pack(
            _HEADER_FMT,
            0x4E56,  # magic
            1,  # version
            99,  # unknown frame_type
            0,  # flags
            1,  # request_id
            1,  # stream_id
            0,  # payload_len
            0,  # crc32
            0,  # ext_hdr_len
        )
        with pytest.raises(ProtocolError, match="unknown frame type: 99"):
            decode_frame(header)
