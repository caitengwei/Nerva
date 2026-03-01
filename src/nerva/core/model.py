"""Model base class and model() declaration API.

Users subclass Model to define inference logic.
The model() function creates a lazy declaration (ModelHandle)
that is materialized when the pipeline is started.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nerva.engine.batcher import BatchConfig

logger = logging.getLogger(__name__)


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

    def __init__(self) -> None:
        self._options: dict[str, Any] = {}

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
    batch_config: BatchConfig | None = None

    def __call__(self, inputs: Any) -> Any:
        """Invoke the model.

        During tracing: records a node in the graph (via Proxy).
        During execution: dispatches to the Backend.
        """
        # Lazy import to avoid circular dependency (model → proxy → graph).
        from nerva.core.proxy import Proxy, _extract_proxy_edges, get_trace_context

        ctx = get_trace_context()
        if ctx is not None:
            # Trace mode: record Node + Edges, return Proxy.
            from nerva.core.graph import Node

            node_id = ctx.next_id(self.name)
            node = Node(id=node_id, model_name=self.name, node_type="call")
            ctx.graph.add_node(node)

            for edge in _extract_proxy_edges(inputs, node_id):
                ctx.graph.add_edge(edge)

            return Proxy(source_node_id=node_id, name=f"{self.name}_out")

        # Not in a trace context — standalone call is not supported.
        raise RuntimeError(
            f"ModelHandle '{self.name}' called outside of a pipeline context. "
            "Use a Pipeline or call backend.infer() directly."
        )


_model_registry: dict[str, ModelHandle] = {}


def get_model_handle(name: str) -> ModelHandle:
    """Look up a registered ModelHandle by name.

    Raises:
        KeyError: If no handle with the given name has been registered.
    """
    if name not in _model_registry:
        raise KeyError(f"No model handle registered for '{name}'")
    return _model_registry[name]


def list_model_handles() -> dict[str, ModelHandle]:
    """Return a copy of all registered model handles."""
    return dict(_model_registry)


def model(
    name: str,
    model_class: type[Model],
    *,
    backend: str = "pytorch",
    device: str = "cpu",
    batch_config: BatchConfig | None = None,
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
        batch_config: Optional batching configuration. If provided, the caller
            should wrap the corresponding WorkerProxy with DynamicBatcher.
        **options: Backend-specific options.

    Returns:
        A ModelHandle that can be used in pipeline functions.
    """
    if not (isinstance(model_class, type) and issubclass(model_class, Model)):
        raise TypeError(
            f"model_class must be a subclass of nerva.Model, got {model_class}"
        )
    handle = ModelHandle(
        name=name,
        model_class=model_class,
        backend=backend,
        device=device,
        options=options,
        batch_config=batch_config,
    )
    if name in _model_registry:
        logger.warning("overwriting existing model handle for '%s'", name)
    _model_registry[name] = handle
    return handle
