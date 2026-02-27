"""Proxy, TraceContext, and trace() for DAG construction via function tracing.

During trace(), ModelHandle calls receive Proxy objects instead of real data.
Each Proxy records which node produced it, enabling automatic Edge construction.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from nerva.core.graph import Edge, Graph

if TYPE_CHECKING:
    from collections.abc import Callable


class Proxy:
    """Proxy object representing a future model output during tracing.

    Attributes:
        source_node_id: ID of the node that produces this value (None for pipeline inputs).
        name: Human-readable name for debugging.
        _field_path: Tuple of keys for __getitem__ traversal.
    """

    __slots__ = ("_field_path", "name", "source_node_id")

    def __init__(
        self,
        source_node_id: str | None = None,
        name: str = "input",
        field_path: tuple[str, ...] = (),
    ) -> None:
        self.source_node_id = source_node_id
        self.name = name
        self._field_path = field_path

    def __getitem__(self, key: str) -> Proxy:
        """Record a field access, returning a new Proxy with extended path."""
        if not isinstance(key, str):
            raise TypeError(
                f"Proxy key must be a string, got {type(key).__name__}"
            )
        return Proxy(
            source_node_id=self.source_node_id,
            name=f"{self.name}[{key!r}]",
            field_path=(*self._field_path, key),
        )

    def __repr__(self) -> str:
        path = "".join(f"[{k!r}]" for k in self._field_path)
        return f"Proxy(src={self.source_node_id}, name={self.name}{path})"


class TraceContext:
    """Holds the mutable state during a trace() call.

    Each trace() invocation creates a fresh TraceContext and sets it
    in a ContextVar, making concurrent traces in different asyncio tasks safe.
    """

    def __init__(self) -> None:
        self.graph = Graph()
        self._counter = 0

    def next_id(self, prefix: str) -> str:
        """Generate a unique node ID."""
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def swap_graph(self, new_graph: Graph) -> Graph:
        """Replace the current graph, returning the old one."""
        old = self.graph
        self.graph = new_graph
        return old


_trace_context_var: ContextVar[TraceContext | None] = ContextVar(
    "_trace_context_var", default=None
)


def get_trace_context() -> TraceContext | None:
    """Return the current TraceContext, or None if not tracing."""
    return _trace_context_var.get()


def _require_trace_context() -> TraceContext:
    """Return the current TraceContext, raising if not in a trace."""
    ctx = _trace_context_var.get()
    if ctx is None:
        raise RuntimeError("Not inside a trace() context")
    return ctx


def _extract_proxy_edges(
    inputs: Any, dst_node_id: str
) -> list[Edge]:
    """Extract Edge objects from ModelHandle call inputs.

    Supports:
    - Single Proxy → one Edge (no dst_input_key)
    - dict[str, Proxy|Any] → one Edge per Proxy value (with dst_input_key)
    """
    edges: list[Edge] = []

    if isinstance(inputs, Proxy):
        if inputs.source_node_id is not None:
            edges.append(Edge(
                src=inputs.source_node_id,
                dst=dst_node_id,
                src_field_path=inputs._field_path,
            ))
    elif isinstance(inputs, dict):
        for key, val in inputs.items():
            if isinstance(val, Proxy) and val.source_node_id is not None:
                edges.append(Edge(
                    src=val.source_node_id,
                    dst=dst_node_id,
                    src_field_path=val._field_path,
                    dst_input_key=key,
                ))

    return edges


def trace(fn: Callable[..., Any], num_inputs: int = 1) -> Graph:
    """Trace a user pipeline function to build a computation Graph.

    Passes Proxy inputs to the function. All ModelHandle.__call__ invocations
    during tracing record Nodes and Edges into the TraceContext's Graph.

    Args:
        fn: Pipeline function to trace.
        num_inputs: Number of input proxies to create.

    Returns:
        The constructed Graph.
    """
    ctx = TraceContext()
    token = _trace_context_var.set(ctx)
    try:
        inputs = [
            Proxy(source_node_id=None, name=f"input_{i}")
            for i in range(num_inputs)
        ]
        if num_inputs == 1:
            fn(inputs[0])
        else:
            fn(*inputs)
        return ctx.graph
    finally:
        _trace_context_var.reset(token)
