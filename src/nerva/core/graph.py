"""Graph IR for DAG pipeline representation.

Defines Node, Edge, and Graph — the core data structures for representing
multi-model computation graphs built via trace() or explicit API.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Edge:
    """A data dependency edge: src node -> dst node.

    Attributes:
        src: Source node ID.
        dst: Destination node ID.
        src_field_path: Path for Proxy.__getitem__ traversal, e.g. ("features",).
        dst_input_key: Key in destination's input dict, e.g. "img" for fusion({"img": ...}).
    """

    src: str
    dst: str
    src_field_path: tuple[str, ...] = ()
    dst_input_key: str | None = None


@dataclass
class Node:
    """A node in the computation graph.

    Attributes:
        id: Unique node identifier.
        model_name: Name of the model (or "cond"/"parallel" for control flow).
        node_type: One of "call", "cond", "parallel".
        true_branch: Sub-graph for cond true branch.
        false_branch: Sub-graph for cond false branch.
        branches: Sub-graphs for parallel branches.
    """

    id: str
    model_name: str
    node_type: str = "call"
    true_branch: Graph | None = None
    false_branch: Graph | None = None
    branches: list[Graph] = field(default_factory=list)


@dataclass
class Graph:
    """Collection of nodes and edges representing a computation DAG."""

    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self.nodes.append(node)

    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)

    @property
    def node_map(self) -> dict[str, Node]:
        """Map from node ID to Node."""
        return {n.id: n for n in self.nodes}

    def predecessors(self, node_id: str) -> list[str]:
        """Return IDs of all predecessor nodes."""
        return [e.src for e in self.edges if e.dst == node_id]

    def successors(self, node_id: str) -> list[str]:
        """Return IDs of all successor nodes."""
        return [e.dst for e in self.edges if e.src == node_id]

    def incoming_edges(self, node_id: str) -> list[Edge]:
        """Return all edges pointing to the given node."""
        return [e for e in self.edges if e.dst == node_id]

    def topological_sort(self) -> list[Node]:
        """Topological sort using Kahn's algorithm.

        Returns:
            Nodes in topological order.

        Raises:
            ValueError: If the graph contains a cycle.
        """
        nm = self.node_map
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
            if nid in nm:
                result.append(nm[nid])
            for neighbor in adj[nid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.nodes):
            raise ValueError("Graph contains a cycle")
        return result


# Re-export for convenience — used by type annotations elsewhere.
__all__ = ["Edge", "Graph", "Node"]
