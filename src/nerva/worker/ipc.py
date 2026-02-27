"""IPC message codec, Descriptor dataclass, and import utilities.

Provides the wire format for Master <-> Worker communication over ZeroMQ.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import msgpack

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MessageType(StrEnum):
    """Types of IPC messages exchanged between Master and Worker."""

    LOAD_MODEL = "LOAD_MODEL"
    LOAD_MODEL_ACK = "LOAD_MODEL_ACK"
    INFER_SUBMIT = "INFER_SUBMIT"
    INFER_ACK = "INFER_ACK"
    SHM_ALLOC_REQUEST = "SHM_ALLOC_REQUEST"
    SHM_ALLOC_RESPONSE = "SHM_ALLOC_RESPONSE"
    CANCEL = "CANCEL"
    HEALTH_CHECK = "HEALTH_CHECK"
    HEALTH_STATUS = "HEALTH_STATUS"
    SHUTDOWN = "SHUTDOWN"
    WORKER_READY = "WORKER_READY"


class AckStatus(StrEnum):
    """Status codes returned in acknowledgement messages."""

    OK = "OK"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    ABORTED = "ABORTED"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    UNAVAILABLE = "UNAVAILABLE"
    INTERNAL = "INTERNAL"


# ---------------------------------------------------------------------------
# Descriptor
# ---------------------------------------------------------------------------


@dataclass
class Descriptor:
    """Describes a data payload — either inline bytes or a shared-memory region."""

    request_id: str
    node_id: int
    schema_version: int = 1
    shm_id: str | None = None
    offset: int = 0
    length: int = 0
    inline_data: bytes | None = None
    payload_codec: str = "msgpack_dict_v1"
    input_key: str | None = None
    dtype: str = "bytes"
    shape: list[int] = field(default_factory=list)
    device: str = "cpu"
    lifetime_token: int = 0
    checksum: int = 0

    @property
    def is_inline(self) -> bool:
        """Return True when the payload is carried inline (non-empty bytes)."""
        return self.inline_data is not None and len(self.inline_data) > 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for msgpack encoding."""
        return {
            "request_id": self.request_id,
            "node_id": self.node_id,
            "schema_version": self.schema_version,
            "shm_id": self.shm_id,
            "offset": self.offset,
            "length": self.length,
            "inline_data": self.inline_data,
            "payload_codec": self.payload_codec,
            "input_key": self.input_key,
            "dtype": self.dtype,
            "shape": self.shape,
            "device": self.device,
            "lifetime_token": self.lifetime_token,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Descriptor:
        """Reconstruct a Descriptor from a dict (e.g. decoded msgpack)."""
        return cls(
            request_id=d["request_id"],
            node_id=d["node_id"],
            schema_version=d.get("schema_version", 1),
            shm_id=d.get("shm_id"),
            offset=d.get("offset", 0),
            length=d.get("length", 0),
            inline_data=d.get("inline_data"),
            payload_codec=d.get("payload_codec", "msgpack_dict_v1"),
            input_key=d.get("input_key"),
            dtype=d.get("dtype", "bytes"),
            shape=d.get("shape", []),
            device=d.get("device", "cpu"),
            lifetime_token=d.get("lifetime_token", 0),
            checksum=d.get("checksum", 0),
        )


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------


def encode_message(msg: dict[str, Any]) -> bytes:
    """Encode a message dict to bytes using msgpack."""
    return msgpack.packb(msg, use_bin_type=True)  # type: ignore[no-any-return]


def decode_message(data: bytes) -> dict[str, Any]:
    """Decode msgpack bytes back to a message dict."""
    return msgpack.unpackb(data, raw=False)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Import-path utilities
# ---------------------------------------------------------------------------


def class_to_import_path(cls: type) -> str:
    """Return ``"module:ClassName"`` for the given class."""
    return f"{cls.__module__}:{cls.__qualname__}"


def import_path_to_class(path: str) -> type:
    """Import and return the class identified by ``"module:ClassName"``."""
    module_path, class_name = path.split(":", 1)
    module = importlib.import_module(module_path)
    cls: type = getattr(module, class_name)
    return cls
