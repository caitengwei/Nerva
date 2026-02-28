"""Binary frame codec for Nerva RPC protocol.

Frame layout (32-byte fixed header + variable payload):
  bytes 0-1:   magic (0x4E56)
  byte  2:     version (1)
  byte  3:     frame_type
  byte  4:     flags
  bytes 5-7:   reserved
  bytes 8-15:  request_id (u64)
  bytes 16-19: stream_id (u32, fixed 1)
  bytes 20-23: payload_length (u32)
  bytes 24-27: crc32 (u32, fixed 0)
  bytes 28-31: ext_hdr_len (u32, fixed 0)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

MAGIC = 0x4E56
VERSION = 1
HEADER_SIZE = 32
MAX_PAYLOAD_BYTES = 4 * 1024 * 1024  # 4 MiB

# Header format: magic(H) version(B) frame_type(B) flags(B) reserved(3x) request_id(Q)
#                stream_id(I) payload_len(I) crc32(I) ext_hdr_len(I)
_HEADER_FMT = ">HBBB3xQIIII"


class FrameType(IntEnum):
    OPEN = 1
    DATA = 2
    END = 3
    ERROR = 4
    HEARTBEAT = 5


class ProtocolError(Exception):
    """Raised on protocol-level encoding/decoding errors."""


@dataclass
class Frame:
    """A single binary RPC frame."""

    frame_type: FrameType
    request_id: int
    flags: int
    payload: bytes


def encode_frame(frame: Frame) -> bytes:
    """Encode a Frame into wire bytes (header + payload)."""
    if len(frame.payload) > MAX_PAYLOAD_BYTES:
        raise ProtocolError(
            f"payload size {len(frame.payload)} exceeds max {MAX_PAYLOAD_BYTES}"
        )
    header = struct.pack(
        _HEADER_FMT,
        MAGIC,
        VERSION,
        frame.frame_type,
        frame.flags,
        frame.request_id,
        1,  # stream_id (fixed)
        len(frame.payload),
        0,  # crc32 (fixed)
        0,  # ext_hdr_len (fixed)
    )
    return header + frame.payload


def decode_frame(data: bytes) -> tuple[Frame, int]:
    """Decode a Frame from wire bytes.

    Returns:
        (frame, consumed_bytes) tuple.

    Raises:
        ProtocolError: On invalid header, bad magic/version, or size violations.
    """
    if len(data) < HEADER_SIZE:
        raise ProtocolError(
            f"incomplete header: got {len(data)} bytes, need {HEADER_SIZE}"
        )

    magic, version, frame_type, flags, request_id, _stream_id, payload_len, _crc32, _ext_hdr_len = (
        struct.unpack_from(_HEADER_FMT, data)
    )

    if magic != MAGIC:
        raise ProtocolError(f"bad magic: expected 0x{MAGIC:04X}, got 0x{magic:04X}")
    if version != VERSION:
        raise ProtocolError(f"unsupported version: {version}")
    if payload_len > MAX_PAYLOAD_BYTES:
        raise ProtocolError(
            f"payload size {payload_len} exceeds max {MAX_PAYLOAD_BYTES}"
        )

    total = HEADER_SIZE + payload_len
    if len(data) < total:
        raise ProtocolError(
            f"incomplete payload: got {len(data)} bytes, need {total}"
        )

    payload = data[HEADER_SIZE:total]
    frame = Frame(
        frame_type=FrameType(frame_type),
        request_id=request_id,
        flags=flags,
        payload=payload,
    )
    return frame, total
