"""Tests for scripts/demo_client.py frame encoding/decoding behavior."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest


def _load_demo_client_module() -> Any:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "demo_client.py"
    spec = importlib.util.spec_from_file_location("demo_client_script", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decode_frames_round_trip() -> None:
    m = _load_demo_client_module()
    body = (
        m._encode_frame(m._FRAME_OPEN, 7, b"open")
        + m._encode_frame(m._FRAME_DATA, 7, b"data")
        + m._encode_frame(m._FRAME_END, 7, b"")
    )
    frames = m._decode_frames(body)
    assert len(frames) == 3
    assert frames[0]["type"] == m._FRAME_OPEN
    assert frames[1]["type"] == m._FRAME_DATA
    assert frames[1]["payload"] == b"data"
    assert frames[2]["type"] == m._FRAME_END


def test_decode_frames_rejects_incomplete_header() -> None:
    m = _load_demo_client_module()
    with pytest.raises(ValueError, match="Incomplete frame header"):
        m._decode_frames(b"\x00" * (m._HEADER_SIZE - 1))


def test_decode_frames_rejects_invalid_magic() -> None:
    m = _load_demo_client_module()
    frame = m._encode_frame(m._FRAME_DATA, 9, b"x")
    corrupted = b"\x00\x00" + frame[2:]
    with pytest.raises(ValueError, match="Invalid frame magic"):
        m._decode_frames(corrupted)


def test_decode_frames_rejects_incomplete_payload() -> None:
    m = _load_demo_client_module()
    frame = m._encode_frame(m._FRAME_DATA, 5, b"abc")
    truncated = frame[:-1]
    with pytest.raises(ValueError, match="Incomplete frame payload"):
        m._decode_frames(truncated)
