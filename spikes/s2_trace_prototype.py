"""
Spike S2: Trace Prototype — 验证通过 Proxy 对象追踪 Python 函数构建 DAG 的可行性。

实现内容:
- Proxy: 代理对象，记录所有 model 调用
- ModelHandle: 可调用对象，调用时在图中记录节点
- Graph IR: Node / Edge / Graph 数据类
- trace(): 追踪用户函数，收集计算图
- nerva_cond(): 条件分支原语
- nerva_parallel(): 并行执行原语
- 拓扑排序验证

运行: uv run python spikes/s2_trace_prototype.py
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Graph IR
# ---------------------------------------------------------------------------


@dataclass
class Node:
    """A model call node in the DAG."""

    id: str
    model_name: str
    node_type: str = "call"  # "call" | "cond" | "parallel"
    # For cond nodes
    true_branch: Graph | None = None
    false_branch: Graph | None = None
    # For parallel nodes
    branches: list[Graph] = field(default_factory=list)

    def __repr__(self) -> str:
        extra = ""
        if self.node_type == "cond":
            t_nodes = [n.model_name for n in (self.true_branch.nodes if self.true_branch else [])]
            f_nodes = [n.model_name for n in (self.false_branch.nodes if self.false_branch else [])]
            extra = f", true={t_nodes}, false={f_nodes}"
        elif self.node_type == "parallel":
            br = [[n.model_name for n in b.nodes] for b in self.branches]
            extra = f", branches={br}"
        return f"Node({self.id}, {self.model_name}, type={self.node_type}{extra})"


@dataclass
class Edge:
    """A data dependency edge: src_node -> dst_node."""

    src: str  # node id
    dst: str  # node id

    def __repr__(self) -> str:
        return f"Edge({self.src} -> {self.dst})"


@dataclass
class Graph:
    """Collection of nodes and edges representing a DAG."""

    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def topological_sort(self) -> list[Node]:
        """Kahn's algorithm for topological sort."""
        node_map = {n.id: n for n in self.nodes}
        in_degree: dict[str, int] = defaultdict(int)
        adj: dict[str, list[str]] = defaultdict(list)

        for n in self.nodes:
            in_degree.setdefault(n.id, 0)

        for e in self.edges:
            adj[e.src].append(e.dst)
            in_degree[e.dst] += 1

        queue: deque[str] = deque()
        for nid, deg in in_degree.items():
            if deg == 0:
                queue.append(nid)

        result: list[Node] = []
        while queue:
            nid = queue.popleft()
            if nid in node_map:
                result.append(node_map[nid])
            for neighbor in adj[nid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.nodes):
            raise ValueError("Graph has a cycle!")
        return result

    def pretty_print(self, title: str = "") -> None:
        if title:
            print(f"\n{'='*60}")
            print(f"  {title}")
            print(f"{'='*60}")
        print(f"  Nodes ({len(self.nodes)}):")
        for n in self.nodes:
            print(f"    {n}")
        print(f"  Edges ({len(self.edges)}):")
        for e in self.edges:
            print(f"    {e}")
        topo = self.topological_sort()
        print(f"  Topological order: {[n.id for n in topo]}")


# ---------------------------------------------------------------------------
# Tracing context (thread-local-like global for simplicity in spike)
# ---------------------------------------------------------------------------


class _TraceContext:
    """Global tracing context that collects nodes and edges during trace()."""

    def __init__(self) -> None:
        self.graph = Graph()
        self._counter = 0

    def next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"


_current_ctx: _TraceContext | None = None


def _get_ctx() -> _TraceContext:
    assert _current_ctx is not None, "Not inside a trace() context"
    return _current_ctx


# ---------------------------------------------------------------------------
# Proxy
# ---------------------------------------------------------------------------


class Proxy:
    """
    Proxy object passed during tracing instead of real data.
    Records which node produced it so we can build edges.
    """

    def __init__(self, source_node_id: str | None = None, name: str = "input") -> None:
        self.source_node_id = source_node_id
        self.name = name

    def __repr__(self) -> str:
        return f"Proxy(src={self.source_node_id}, name={self.name})"


# ---------------------------------------------------------------------------
# ModelHandle
# ---------------------------------------------------------------------------


class ModelHandle:
    """
    A callable representing a model. When called with a Proxy during tracing,
    it records a Node in the graph and returns a new Proxy for its output.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *args: Any) -> Proxy:
        ctx = _get_ctx()
        node_id = ctx.next_id(self.name)
        node = Node(id=node_id, model_name=self.name, node_type="call")
        ctx.graph.add_node(node)

        # Create edges from all Proxy inputs
        for arg in args:
            if isinstance(arg, Proxy) and arg.source_node_id is not None:
                ctx.graph.add_edge(Edge(src=arg.source_node_id, dst=node_id))

        return Proxy(source_node_id=node_id, name=f"{self.name}_out")

    def __repr__(self) -> str:
        return f"ModelHandle({self.name})"


# ---------------------------------------------------------------------------
# Control flow primitives
# ---------------------------------------------------------------------------


def nerva_cond(
    predicate_proxy: Proxy,
    true_fn: Callable[[], Proxy],
    false_fn: Callable[[], Proxy],
) -> Proxy:
    """
    Conditional branch primitive.
    Records a CondNode with both branches traced as sub-graphs.
    """
    ctx = _get_ctx()
    cond_id = ctx.next_id("cond")

    # Save parent graph state, trace true branch into a sub-graph
    parent_graph = ctx.graph
    parent_counter = ctx._counter

    # Trace true branch
    ctx.graph = Graph()
    true_out = true_fn()
    true_graph = ctx.graph

    # Trace false branch
    ctx.graph = Graph()
    false_out = false_fn()
    false_graph = ctx.graph

    # Restore parent graph
    ctx.graph = parent_graph

    cond_node = Node(
        id=cond_id,
        model_name="cond",
        node_type="cond",
        true_branch=true_graph,
        false_branch=false_graph,
    )
    ctx.graph.add_node(cond_node)

    # Edge from predicate source to cond node
    if predicate_proxy.source_node_id is not None:
        ctx.graph.add_edge(Edge(src=predicate_proxy.source_node_id, dst=cond_id))

    return Proxy(source_node_id=cond_id, name="cond_out")


def nerva_parallel(*fns: Callable[[], Proxy]) -> tuple[Proxy, ...]:
    """
    Parallel execution primitive.
    Records a ParallelNode with all branches traced as sub-graphs.
    Returns a tuple of Proxy outputs, one per branch.
    """
    ctx = _get_ctx()
    par_id = ctx.next_id("parallel")

    parent_graph = ctx.graph
    branch_graphs: list[Graph] = []
    branch_outputs: list[Proxy] = []

    for fn in fns:
        ctx.graph = Graph()
        out = fn()
        branch_graphs.append(ctx.graph)
        branch_outputs.append(out)

    # Restore parent graph
    ctx.graph = parent_graph

    par_node = Node(
        id=par_id,
        model_name="parallel",
        node_type="parallel",
        branches=branch_graphs,
    )
    ctx.graph.add_node(par_node)

    # Return proxies sourced from the parallel node
    proxies = tuple(
        Proxy(source_node_id=par_id, name=f"parallel_out_{i}")
        for i in range(len(fns))
    )
    return proxies


# ---------------------------------------------------------------------------
# trace()
# ---------------------------------------------------------------------------


def trace(fn: Callable[..., Any], num_inputs: int = 1) -> Graph:
    """
    Trace a user function by passing Proxy inputs, collecting the DAG.
    """
    global _current_ctx
    _current_ctx = _TraceContext()

    # Create proxy inputs
    inputs = [Proxy(source_node_id=None, name=f"input_{i}") for i in range(num_inputs)]

    # Run the function — all ModelHandle calls will record into the context
    if num_inputs == 1:
        fn(inputs[0])
    else:
        fn(*inputs)

    graph = _current_ctx.graph
    _current_ctx = None
    return graph


# ===========================================================================
# Test Cases
# ===========================================================================


def test_linear_chain() -> None:
    """Test: a -> b -> c (3 nodes, 2 edges)"""
    a = ModelHandle("a")
    b = ModelHandle("b")
    c = ModelHandle("c")

    def pipeline(x: Proxy) -> Proxy:
        x1 = a(x)
        x2 = b(x1)
        x3 = c(x2)
        return x3

    graph = trace(pipeline)
    graph.pretty_print("Test: Linear Chain (a -> b -> c)")

    assert len(graph.nodes) == 3, f"Expected 3 nodes, got {len(graph.nodes)}"
    assert len(graph.edges) == 2, f"Expected 2 edges, got {len(graph.edges)}"

    topo = graph.topological_sort()
    topo_names = [n.model_name for n in topo]
    assert topo_names == ["a", "b", "c"], f"Bad topo order: {topo_names}"
    print("  PASSED")


def test_parallel() -> None:
    """Test: parallel(a, b) -> c (3 nodes at top level, fork/join)"""
    a = ModelHandle("a")
    b = ModelHandle("b")
    c = ModelHandle("c")

    def pipeline(x: Proxy) -> Proxy:
        out_a, out_b = nerva_parallel(
            lambda: a(x),
            lambda: b(x),
        )
        return c(out_a, out_b)

    graph = trace(pipeline)
    graph.pretty_print("Test: Parallel (parallel(a, b) -> c)")

    # Top-level: parallel node + c node
    assert len(graph.nodes) == 2, f"Expected 2 top-level nodes, got {len(graph.nodes)}"
    par_node = graph.nodes[0]
    assert par_node.node_type == "parallel"
    assert len(par_node.branches) == 2
    # Each branch has 1 node
    assert len(par_node.branches[0].nodes) == 1
    assert len(par_node.branches[1].nodes) == 1
    assert par_node.branches[0].nodes[0].model_name == "a"
    assert par_node.branches[1].nodes[0].model_name == "b"

    # Edges from parallel -> c (2 edges: one per output proxy from parallel)
    assert len(graph.edges) == 2, f"Expected 2 edges, got {len(graph.edges)}"
    print("  PASSED")


def test_cond() -> None:
    """Test: cond(pred, a, b) -> c traces both branches"""
    pred_model = ModelHandle("pred")
    a = ModelHandle("a")
    b = ModelHandle("b")
    c = ModelHandle("c")

    def pipeline(x: Proxy) -> Proxy:
        pred_out = pred_model(x)
        features = nerva_cond(
            pred_out,
            lambda: a(x),
            lambda: b(x),
        )
        return c(features)

    graph = trace(pipeline)
    graph.pretty_print("Test: Cond (pred -> cond(a, b) -> c)")

    # Top-level nodes: pred, cond, c
    assert len(graph.nodes) == 3, f"Expected 3 top-level nodes, got {len(graph.nodes)}"
    cond_node = graph.nodes[1]
    assert cond_node.node_type == "cond"
    assert cond_node.true_branch is not None
    assert cond_node.false_branch is not None
    assert len(cond_node.true_branch.nodes) == 1
    assert cond_node.true_branch.nodes[0].model_name == "a"
    assert len(cond_node.false_branch.nodes) == 1
    assert cond_node.false_branch.nodes[0].model_name == "b"

    # Edges: pred -> cond, cond -> c
    assert len(graph.edges) == 2, f"Expected 2 edges, got {len(graph.edges)}"

    topo = graph.topological_sort()
    topo_names = [n.model_name for n in topo]
    assert topo_names == ["pred", "cond", "c"], f"Bad topo order: {topo_names}"
    print("  PASSED")


def test_diamond() -> None:
    """Test: a -> (b, c) -> d where b and c share input from a"""
    a = ModelHandle("a")
    b = ModelHandle("b")
    c = ModelHandle("c")
    d = ModelHandle("d")

    def pipeline(x: Proxy) -> Proxy:
        a_out = a(x)
        b_out = b(a_out)
        c_out = c(a_out)
        d_out = d(b_out, c_out)
        return d_out

    graph = trace(pipeline)
    graph.pretty_print("Test: Diamond (a -> (b, c) -> d)")

    assert len(graph.nodes) == 4, f"Expected 4 nodes, got {len(graph.nodes)}"
    # Edges: a->b, a->c, b->d, c->d
    assert len(graph.edges) == 4, f"Expected 4 edges, got {len(graph.edges)}"

    topo = graph.topological_sort()
    topo_names = [n.model_name for n in topo]
    # a must come first, d must come last, b and c in between
    assert topo_names[0] == "a", f"First should be 'a', got {topo_names[0]}"
    assert topo_names[-1] == "d", f"Last should be 'd', got {topo_names[-1]}"
    assert set(topo_names[1:3]) == {"b", "c"}, f"Middle should be b,c, got {topo_names[1:3]}"
    print("  PASSED")


def test_topological_sort_detects_valid_ordering() -> None:
    """Verify topological sort produces valid ordering for a complex graph."""
    a = ModelHandle("a")
    b = ModelHandle("b")
    c = ModelHandle("c")
    d = ModelHandle("d")
    e = ModelHandle("e")

    def pipeline(x: Proxy) -> Proxy:
        a_out = a(x)
        b_out = b(a_out)
        c_out = c(a_out)
        d_out = d(b_out, c_out)
        e_out = e(c_out, d_out)
        return e_out

    graph = trace(pipeline)
    graph.pretty_print("Test: Topological Sort Validation (a -> b,c -> d -> e)")

    topo = graph.topological_sort()
    topo_ids = [n.id for n in topo]
    id_to_pos = {nid: i for i, nid in enumerate(topo_ids)}

    # Verify: for every edge, src comes before dst
    for edge in graph.edges:
        assert id_to_pos[edge.src] < id_to_pos[edge.dst], (
            f"Topological violation: {edge.src} (pos {id_to_pos[edge.src]}) "
            f"should come before {edge.dst} (pos {id_to_pos[edge.dst]})"
        )
    print("  PASSED")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("Nerva Spike S2: Trace Prototype")
    print("Validating DAG construction via Proxy tracing\n")

    test_linear_chain()
    test_parallel()
    test_cond()
    test_diamond()
    test_topological_sort_detects_valid_ordering()

    print(f"\n{'='*60}")
    print("  All tests passed!")
    print(f"{'='*60}")
