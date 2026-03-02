#!/usr/bin/env python3
"""Nerva demo client: sends Binary RPC requests to a running Nerva server.

Usage:
    # First start the server:
    uvicorn examples.echo_server:app --port 8080

    # Then in another terminal:
    python scripts/demo_client.py

    # Custom host/pipeline/value:
    python scripts/demo_client.py --url http://localhost:8080 --pipeline echo --value "hello"
"""

from __future__ import annotations

import argparse
import struct
import time
import urllib.request
from typing import Any

import msgpack

# Mirrors src/nerva/server/protocol.py header layout.
_MAGIC = 0x4E56
_VERSION = 1
_HEADER_FMT = ">HBBB3xQIIII"
_HEADER_SIZE = 32

_FRAME_OPEN = 1
_FRAME_DATA = 2
_FRAME_END = 3
_FRAME_ERROR = 4


def _encode_frame(frame_type: int, request_id: int, payload: bytes) -> bytes:
    header = struct.pack(
        _HEADER_FMT,
        _MAGIC,
        _VERSION,
        frame_type,
        0,
        request_id,
        1,
        len(payload),
        0,
        0,
    )
    return header + payload


def _decode_frames(data: bytes) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    offset = 0
    while offset < len(data):
        if len(data) - offset < _HEADER_SIZE:
            break
        fields = struct.unpack_from(_HEADER_FMT, data, offset)
        _magic, _ver, frame_type, _flags, req_id, _sid, payload_len, _crc, _ext = fields
        payload = data[offset + _HEADER_SIZE : offset + _HEADER_SIZE + payload_len]
        frames.append({"type": frame_type, "request_id": req_id, "payload": payload})
        offset += _HEADER_SIZE + payload_len
    return frames


def call(
    url: str,
    pipeline: str,
    inputs: dict[str, Any],
    deadline_ms: int = 30000,
) -> dict[str, Any]:
    request_id = 42
    body = (
        _encode_frame(_FRAME_OPEN, request_id, msgpack.packb({"pipeline": pipeline}))
        + _encode_frame(_FRAME_DATA, request_id, msgpack.packb(inputs))
        + _encode_frame(_FRAME_END, request_id, b"")
    )
    deadline_epoch_ms = int(time.time() * 1000) + deadline_ms
    req = urllib.request.Request(
        f"{url}/rpc/{pipeline}",
        data=body,
        headers={
            "Content-Type": "application/x-nerva-rpc",
            "x-nerva-deadline-ms": str(deadline_epoch_ms),
            "x-nerva-stream": "0",
        },
    )
    with urllib.request.urlopen(req) as resp:
        raw = resp.read()

    frames = _decode_frames(raw)
    for frame in frames:
        if frame["type"] == _FRAME_ERROR:
            err = msgpack.unpackb(frame["payload"], raw=False)
            raise RuntimeError(f"RPC error {err.get('code')}: {err.get('message')}")
        if frame["type"] == _FRAME_DATA:
            return msgpack.unpackb(frame["payload"], raw=False)  # type: ignore[no-any-return]
    raise RuntimeError("No DATA frame in response")


def main() -> None:
    parser = argparse.ArgumentParser(description="Nerva demo client")
    parser.add_argument("--url", default="http://localhost:8080", help="Server base URL")
    parser.add_argument("--pipeline", default="echo", help="Pipeline name")
    parser.add_argument("--value", default="hello from demo_client!", help="Input value")
    args = parser.parse_args()

    print(f"Calling {args.url}/rpc/{args.pipeline} ...")
    result = call(args.url, args.pipeline, {"value": args.value})
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
