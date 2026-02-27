"""Dynamic Batcher with deadline-aware admission and backpressure."""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nerva.backends.base import InferContext
    from nerva.engine.executor import InferableProxy


@dataclass
class BatchConfig:
    """Configuration for DynamicBatcher."""

    max_batch_size: int = 32
    max_delay_ms: float = 10.0
    queue_capacity: int = 2048
    queue_timeout_ms: float = 100.0
    min_remaining_deadline_ms: float = 5.0


@dataclass
class _PendingRequest:
    """A request waiting in the batcher queue."""

    inputs: dict[str, Any]
    context: Any  # InferContext — kept as Any to avoid circular import at runtime
    future: asyncio.Future[dict[str, Any]]
    enqueue_time: float = field(default_factory=time.monotonic)
