"""Control flow primitives for DAG pipelines: cond() and parallel().

These primitives work during tracing to build sub-graphs for conditional
and parallel execution patterns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

from nerva.core.graph import Edge, Graph, Node
from nerva.core.proxy import Proxy, _require_trace_context


def _validate_branch_graph(
    graph: Graph,
    *,
    primitive: str,
    branch_name: str,
) -> None:
    """Validate traced sub-graph boundaries for control-flow branches."""
    node_ids = {node.id for node in graph.nodes}
    if not node_ids:
        raise RuntimeError(
            f"{primitive}({branch_name}): branch traced to an empty sub-graph. "
            "This usually means the branch captured parent Proxies instead of "
            "building operations inside the branch."
        )

    for edge in graph.edges:
        if edge.src not in node_ids or edge.dst not in node_ids:
            raise RuntimeError(
                f"{primitive}({branch_name}): detected cross-graph edge "
                f"{edge.src!r}->{edge.dst!r}. Branch functions must build "
                "their computation only from values produced inside the branch."
            )


def parallel(*fns: Callable[[], Any]) -> tuple[Proxy, ...]:
    """Execute multiple branches in parallel.

    During tracing, each branch function is traced into a separate sub-graph.
    The returned Proxies have initial field_path=(str(i),) so that downstream
    nodes can resolve branch outputs via __getitem__.

    Args:
        *fns: Branch functions, each taking no arguments and returning a Proxy.

    Returns:
        Tuple of Proxy objects, one per branch.
    """
    ctx = _require_trace_context()
    par_id = ctx.next_id("parallel")

    parent_graph = ctx.graph
    branch_graphs: list[Graph] = []

    for idx, fn in enumerate(fns):
        sub_graph = Graph()
        ctx.swap_graph(sub_graph)
        fn()
        branch_graph = ctx.graph
        _validate_branch_graph(
            branch_graph,
            primitive="parallel",
            branch_name=f"branch {idx}",
        )
        branch_graphs.append(branch_graph)

    # Restore parent graph (swap_graph sets ctx.graph = parent_graph).
    ctx.swap_graph(parent_graph)

    par_node = Node(
        id=par_id,
        model_name="parallel",
        node_type="parallel",
        branches=branch_graphs,
    )
    ctx.graph.add_node(par_node)

    # Each branch output gets field_path=(str(i),) so downstream
    # proxy["features"] becomes ("0", "features").
    proxies = tuple(
        Proxy(
            source_node_id=par_id,
            name=f"parallel_out_{i}",
            field_path=(str(i),),
        )
        for i in range(len(fns))
    )
    return proxies


def cond(
    predicate: Any,
    true_fn: Callable[[], Any],
    false_fn: Callable[[], Any],
) -> Proxy:
    """Conditional branch primitive.

    During tracing, both branches are traced into sub-graphs and embedded
    in a cond Node. At execution time, only the selected branch runs.

    Args:
        predicate: A Proxy (during tracing) representing the condition value.
        true_fn: Function to call when predicate is true.
        false_fn: Function to call when predicate is false.

    Returns:
        A Proxy representing the output of the selected branch.
    """
    ctx = _require_trace_context()
    cond_id = ctx.next_id("cond")

    parent_graph = ctx.graph

    # Trace true branch.
    true_sub = Graph()
    ctx.swap_graph(true_sub)
    true_fn()
    true_graph = ctx.graph
    _validate_branch_graph(
        true_graph,
        primitive="cond",
        branch_name="true_fn",
    )

    # Trace false branch.
    false_sub = Graph()
    ctx.swap_graph(false_sub)
    false_fn()
    false_graph = ctx.graph
    _validate_branch_graph(
        false_graph,
        primitive="cond",
        branch_name="false_fn",
    )

    # Restore parent graph (swap_graph sets ctx.graph = parent_graph).
    ctx.swap_graph(parent_graph)

    cond_node = Node(
        id=cond_id,
        model_name="cond",
        node_type="cond",
        true_branch=true_graph,
        false_branch=false_graph,
    )
    ctx.graph.add_node(cond_node)

    # Edge from predicate source to cond node.
    if isinstance(predicate, Proxy) and predicate.source_node_id is not None:
        ctx.graph.add_edge(Edge(
            src=predicate.source_node_id,
            dst=cond_id,
            src_field_path=predicate._field_path,
        ))

    return Proxy(source_node_id=cond_id, name="cond_out")
