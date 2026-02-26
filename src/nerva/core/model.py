"""Model base class and model() declaration API.

Users subclass Model to define inference logic.
The model() function creates a lazy declaration (ModelHandle)
that is materialized when the pipeline is started.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class Model(ABC):
    """Base class for user-defined models.

    Users implement load() and infer(). The Backend is responsible
    for calling these methods at the appropriate time.

    Lifecycle:
        1. model() creates a ModelHandle (declaration, no loading)
        2. Backend.load_model() instantiates the Model and calls Model.load()
        3. Backend.infer() calls Model.infer() per request
        4. Backend.unload_model() calls Model.unload() for cleanup
    """

    @abstractmethod
    def load(self) -> None:
        """Load model weights and initialize resources.

        Called once by the Backend after instantiation.
        """

    @abstractmethod
    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute inference on a single request (or batched input).

        Args:
            inputs: Input dict with string keys.

        Returns:
            Output dict with string keys.
        """

    async def infer_stream(
        self, inputs: dict[str, Any]
    ) -> Any:  # AsyncIterator[dict[str, Any]]
        """Optional streaming inference. Override to support streaming."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support streaming inference"
        )
        # Make this an async generator so callers can use `async for`.
        yield  # pragma: no cover

    def unload(self) -> None:  # noqa: B027
        """Optional cleanup hook called before unloading."""


@dataclass(frozen=True)
class ModelHandle:
    """Lazy model declaration.

    Created by model(). Does not load the model — just records
    the configuration for later materialization by the Backend.
    """

    name: str
    model_class: type[Model]
    backend: str
    device: str
    options: dict[str, Any] = field(default_factory=dict)

    def __call__(self, inputs: Any) -> Any:
        """Invoke the model.

        During tracing: records a node in the graph (via Proxy).
        During execution: dispatches to the Backend.
        """
        # This will be overridden by the trace/execution runtime.
        # Standalone call for Phase 0: direct in-process execution.
        raise RuntimeError(
            f"ModelHandle '{self.name}' called outside of a pipeline context. "
            "Use a Pipeline or call backend.infer() directly."
        )


def model(
    name: str,
    model_class: type[Model],
    *,
    backend: str = "pytorch",
    device: str = "cpu",
    **options: Any,
) -> ModelHandle:
    """Declare a model for use in a pipeline.

    This is a lazy declaration — the model is not loaded until
    the pipeline is started or serve() is called.

    Args:
        name: Unique model name within the pipeline.
        model_class: User's Model subclass.
        backend: Backend name (e.g., "pytorch", "vllm").
        device: Target device (e.g., "cpu", "cuda:0").
        **options: Backend-specific options.

    Returns:
        A ModelHandle that can be used in pipeline functions.
    """
    if not (isinstance(model_class, type) and issubclass(model_class, Model)):
        raise TypeError(
            f"model_class must be a subclass of nerva.Model, got {model_class}"
        )
    return ModelHandle(
        name=name,
        model_class=model_class,
        backend=backend,
        device=device,
        options=options,
    )
