"""Backend abstract base class.

Defines the contract between the orchestration layer and inference backends.
Call chain: Orchestrator → Backend → Model
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

T_Input = TypeVar("T_Input")
T_Output = TypeVar("T_Output")


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for loading a model instance."""

    model_name: str
    model_class: type[Any]  # User's Model subclass
    device: str  # "cpu", "cuda:0", "cuda:1", ...
    backend_options: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferContext:
    """Per-request context propagated through the inference chain.

    Not frozen: ``cancelled`` is mutated at runtime by cancel propagation.
    """

    request_id: str
    deadline_ms: int  # Relative TTL from call time (ms). Phase 4: server layer must convert absolute epoch to relative TTL before populating this field.
    trace_id: str | None = None
    cancelled: bool = False


@dataclass(frozen=True)
class BatchMeta:
    """Metadata about the current batch."""

    batch_size: int


class Backend(ABC):
    """Abstract inference backend.

    A Backend bridges the orchestration layer and a user-defined Model.
    It is responsible for:
    - Loading/unloading the Model onto the target device
    - Invoking Model.infer() with proper context
    - Handling batched vs single-request dispatch
    - Device management and error isolation
    """

    @abstractmethod
    async def load_model(self, config: ModelConfig) -> None:
        """Load model onto the specified device."""

    @abstractmethod
    async def unload_model(self) -> None:
        """Unload model and release device resources."""

    @abstractmethod
    async def infer(
        self,
        inputs: dict[str, Any],
        context: InferContext,
        batch_meta: BatchMeta | None = None,
    ) -> dict[str, Any]:
        """Execute inference.

        Args:
            inputs: Model input dict.
            context: Per-request context (deadline, cancel, trace).
            batch_meta: Present when inputs represent a batch.

        Returns:
            Model output dict.
        """

    @abstractmethod
    async def infer_stream(
        self,
        inputs: dict[str, Any],
        context: InferContext,
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming inference — yield intermediate results."""
        # Abstract async generators need a yield to satisfy the type checker.
        yield {}  # pragma: no cover

    async def warmup(self) -> None:  # noqa: B027
        """Optional warmup hook called after load_model."""

    def health_check(self) -> bool:
        """Return True if the backend is healthy."""
        return True

    @property
    def is_async_native(self) -> bool:
        """Whether infer() is truly async (non-blocking on event loop).

        True: Worker dispatches directly on event loop (no ThreadPool).
        False: Worker wraps in asyncio.to_thread() to avoid blocking.
        """
        return False

    @property
    def is_loaded(self) -> bool:
        """Whether a model is currently loaded."""
        return False
