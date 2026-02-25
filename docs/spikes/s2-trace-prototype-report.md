# Spike S2: Trace Prototype — 报告

**日期:** 2026-02-25
**文件:** `spikes/s2_trace_prototype.py`
**状态:** 完成，全部测试通过

---

## 1. 目标

验证通过 `Proxy` 对象拦截 `__call__` 调用，能否正确地从普通 Python 函数中构建计算图（DAG）。

具体来说，需要回答以下问题：

- 能否用轻量 Proxy 对象在 tracing 阶段替代真实数据，捕获模型调用依赖关系？
- 能否以足够低的侵入性支持条件分支（`cond`）和并行执行（`parallel`）等控制流原语？
- 构建出的 Graph IR 是否支持正确的拓扑排序？
- 上述方案是否足以作为 Nerva `trace()` 设计的基础？

---

## 2. 测试方法

原型围绕以下几个核心组件构建：

### 2.1 Graph IR

使用 Python `dataclass` 定义三个数据结构：

```python
@dataclass
class Node:
    id: str
    model_name: str
    node_type: str = "call"  # "call" | "cond" | "parallel"
    true_branch: Graph | None = None
    false_branch: Graph | None = None
    branches: list[Graph] = field(default_factory=list)

@dataclass
class Edge:
    src: str  # node id
    dst: str  # node id

@dataclass
class Graph:
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
```

`Node` 统一表示三种类型的节点：普通模型调用节点（`call`）、条件分支节点（`cond`）、并行执行节点（`parallel`）。后两种节点将子图嵌入自身字段中，形成层次化 IR。

### 2.2 Proxy 对象

```python
class Proxy:
    def __init__(self, source_node_id: str | None = None, name: str = "input") -> None:
        self.source_node_id = source_node_id
        self.name = name
```

`Proxy` 是 tracing 阶段在用户函数中流转的占位对象，携带 `source_node_id` 字段，标识产生该值的节点 ID。当输入数据来自用户外部（pipeline 入口）时，`source_node_id` 为 `None`，表示不产生入边。

### 2.3 ModelHandle

```python
class ModelHandle:
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *args: Any) -> Proxy:
        ctx = _get_ctx()
        node_id = ctx.next_id(self.name)
        node = Node(id=node_id, model_name=self.name, node_type="call")
        ctx.graph.add_node(node)
        for arg in args:
            if isinstance(arg, Proxy) and arg.source_node_id is not None:
                ctx.graph.add_edge(Edge(src=arg.source_node_id, dst=node_id))
        return Proxy(source_node_id=node_id, name=f"{self.name}_out")
```

`ModelHandle` 是对模型的可调用封装。在 tracing 阶段，调用 `ModelHandle` 时：

1. 生成唯一节点 ID（格式：`{model_name}_{counter}`）；
2. 将 `Node` 注册到当前上下文的图中；
3. 遍历所有 `Proxy` 类型的参数，为每个有效的 `source_node_id` 创建入边；
4. 返回新的 `Proxy`，携带本节点 ID。

### 2.4 _TraceContext 与全局上下文

```python
class _TraceContext:
    def __init__(self) -> None:
        self.graph = Graph()
        self._counter = 0

    def next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"
```

使用全局变量 `_current_ctx` 存储当前 tracing 上下文。`nerva_cond()` 和 `nerva_parallel()` 通过临时替换 `ctx.graph` 来隔离子图的 tracing，追踪结束后将子图嵌入父节点并恢复父图。

### 2.5 控制流原语

**`nerva_cond(predicate_proxy, true_fn, false_fn)`**

将 true 和 false 两个分支分别 trace 为独立子图，嵌入到 `CondNode` 的 `true_branch` / `false_branch` 字段。向父图中注册该 `CondNode`，并从 predicate 节点连边指向它。

```python
def nerva_cond(predicate_proxy, true_fn, false_fn) -> Proxy:
    # 分别 trace 两个分支为子图
    # 创建 CondNode，嵌入两个子图
    # 在父图添加 predicate -> cond 的边
    ...
```

**`nerva_parallel(*fns)`**

将每个函数 trace 为独立子图，嵌入到 `ParallelNode` 的 `branches` 列表。返回元组，每个元素是一个 `Proxy`，`source_node_id` 均指向该 `ParallelNode`。

```python
def nerva_parallel(*fns) -> tuple[Proxy, ...]:
    # 分别 trace 每个分支为子图
    # 创建 ParallelNode，嵌入所有子图
    # 返回 N 个 Proxy，均 sourced from parallel node
    ...
```

### 2.6 trace() 函数

```python
def trace(fn: Callable[..., Any], num_inputs: int = 1) -> Graph:
    global _current_ctx
    _current_ctx = _TraceContext()
    inputs = [Proxy(source_node_id=None, name=f"input_{i}") for i in range(num_inputs)]
    if num_inputs == 1:
        fn(inputs[0])
    else:
        fn(*inputs)
    graph = _current_ctx.graph
    _current_ctx = None
    return graph
```

创建 `_TraceContext`，构造 `Proxy` 输入，执行用户函数，返回收集到的 `Graph`。

### 2.7 拓扑排序（Kahn's Algorithm）

`Graph.topological_sort()` 使用 Kahn 算法：

1. 统计所有节点的入度；
2. 将入度为 0 的节点加入队列；
3. 依次出队，减少邻居入度，入度降为 0 则入队；
4. 若最终结果节点数与图节点数不等，则抛出 `ValueError("Graph has a cycle!")`。

---

## 3. 测试用例与结果

### Test 1 — Linear Chain (`a -> b -> c`)

**描述:** 最基础的线性串行流水线，三个模型依次调用。

```python
def pipeline(x: Proxy) -> Proxy:
    x1 = a(x)
    x2 = b(x1)
    x3 = c(x2)
    return x3
```

**预期结果:** 3 个节点，2 条边，拓扑顺序为 `[a_1, b_2, c_3]`。

**输出:**
```
============================================================
  Test: Linear Chain (a -> b -> c)
============================================================
  Nodes (3):
    Node(a_1, a, type=call)
    Node(b_2, b, type=call)
    Node(c_3, c, type=call)
  Edges (2):
    Edge(a_1 -> b_2)
    Edge(b_2 -> c_3)
  Topological order: ['a_1', 'b_2', 'c_3']
  PASSED
```

**结果:** PASSED

---

### Test 2 — Parallel (`parallel(a, b) -> c`)

**描述:** 两个模型并行执行，其输出共同作为下游模型 `c` 的输入。

```python
def pipeline(x: Proxy) -> Proxy:
    out_a, out_b = nerva_parallel(
        lambda: a(x),
        lambda: b(x),
    )
    return c(out_a, out_b)
```

**预期结果:** 顶层 2 个节点（1 个 `ParallelNode` 含 2 个子图 + `c`），2 条边（`parallel -> c` 两条，每个输出 Proxy 各一条）。

**输出:**
```
============================================================
  Test: Parallel (parallel(a, b) -> c)
============================================================
  Nodes (2):
    Node(parallel_1, parallel, type=parallel, branches=[['a'], ['b']])
    Node(c_4, c, type=call)
  Edges (2):
    Edge(parallel_1 -> c_4)
    Edge(parallel_1 -> c_4)
  Topological order: ['parallel_1', 'c_4']
  PASSED
```

**结果:** PASSED

---

### Test 3 — Cond (`pred -> cond(a, b) -> c`)

**描述:** 先运行 predicate 模型，再根据结果条件分支到 `a` 或 `b`，最后经 `c` 汇总。

```python
def pipeline(x: Proxy) -> Proxy:
    pred_out = pred_model(x)
    features = nerva_cond(
        pred_out,
        lambda: a(x),
        lambda: b(x),
    )
    return c(features)
```

**预期结果:** 顶层 3 个节点（`pred`、`CondNode` 含 true/false 子图、`c`），2 条边（`pred -> cond`，`cond -> c`）。

**输出:**
```
============================================================
  Test: Cond (pred -> cond(a, b) -> c)
============================================================
  Nodes (3):
    Node(pred_1, pred, type=call)
    Node(cond_2, cond, type=cond, true=['a'], false=['b'])
    Node(c_5, c, type=call)
  Edges (2):
    Edge(pred_1 -> cond_2)
    Edge(cond_2 -> c_5)
  Topological order: ['pred_1', 'cond_2', 'c_5']
  PASSED
```

**结果:** PASSED

---

### Test 4 — Diamond (`a -> (b, c) -> d`)

**描述:** 节点 `a` 的输出同时作为 `b` 和 `c` 的输入（fan-out），`b` 和 `c` 的输出共同作为 `d` 的输入（fan-in），形成菱形 DAG。

```python
def pipeline(x: Proxy) -> Proxy:
    a_out = a(x)
    b_out = b(a_out)
    c_out = c(a_out)
    d_out = d(b_out, c_out)
    return d_out
```

**预期结果:** 4 个节点，4 条边（`a->b`、`a->c`、`b->d`、`c->d`），`a` 的共享输出被正确捕获为两条出边。

**输出:**
```
============================================================
  Test: Diamond (a -> (b, c) -> d)
============================================================
  Nodes (4):
    Node(a_1, a, type=call)
    Node(b_2, b, type=call)
    Node(c_3, c, type=call)
    Node(d_4, d, type=call)
  Edges (4):
    Edge(a_1 -> b_2)
    Edge(a_1 -> c_3)
    Edge(b_2 -> d_4)
    Edge(c_3 -> d_4)
  Topological order: ['a_1', 'b_2', 'c_3', 'd_4']
  PASSED
```

**结果:** PASSED

---

### Test 5 — Topological Sort Validation (`a -> b,c -> d -> e`)

**描述:** 对一个更复杂的图（5 个节点，6 条边）验证 Kahn 算法的正确性：对每一条边，确保 src 节点在拓扑序中位于 dst 节点之前。

```python
def pipeline(x: Proxy) -> Proxy:
    a_out = a(x)
    b_out = b(a_out)
    c_out = c(a_out)
    d_out = d(b_out, c_out)
    e_out = e(c_out, d_out)
    return e_out
```

**预期结果:** 5 个节点，6 条边（`a->b`、`a->c`、`b->d`、`c->d`、`c->e`、`d->e`），所有边均满足拓扑序约束。

**输出:**
```
============================================================
  Test: Topological Sort Validation (a -> b,c -> d -> e)
============================================================
  Nodes (5):
    Node(a_1, a, type=call)
    Node(b_2, b, type=call)
    Node(c_3, c, type=call)
    Node(d_4, d, type=call)
    Node(e_5, e, type=call)
  Edges (6):
    Edge(a_1 -> b_2)
    Edge(a_1 -> c_3)
    Edge(b_2 -> d_4)
    Edge(c_3 -> d_4)
    Edge(c_3 -> e_5)
    Edge(d_4 -> e_5)
  Topological order: ['a_1', 'b_2', 'c_3', 'd_4', 'e_5']
  PASSED
```

**结果:** PASSED

---

## 4. 设计观察

### 4.1 Proxy 方案适合粗粒度模型调用依赖捕获

Proxy 不模拟真实数据，仅携带产生该值的节点 ID。`ModelHandle.__call__` 拦截所有调用，通过检查参数是否为 `Proxy` 类型来自动建边，无需用户显式声明依赖关系。这与 JAX `jit` 的 abstract tracing、`torch.fx` 的 symbolic tracing 思路一致，但实现更轻量——不需要 tensor shape/dtype 信息，只需要调用关系。

### 4.2 cond/parallel 通过嵌入子图实现层次化 IR

`nerva_cond()` 和 `nerva_parallel()` 的核心技巧是临时替换 `ctx.graph`：

```python
parent_graph = ctx.graph
ctx.graph = Graph()       # 开始 trace 子图
out = true_fn()
true_graph = ctx.graph    # 收集子图
ctx.graph = parent_graph  # 恢复父图
```

这使得子图中的节点不污染父图，同时保留了层次结构。对调度器而言，顶层 DAG 决定粗粒度调度顺序，子图决定分支/并行内部的细粒度结构，两层信息均被完整保留。

### 4.3 parallel() 返回多个 Proxy 会产生多条重复边

Test 2 中，`nerva_parallel()` 返回两个 `Proxy`（均 `source_node_id = parallel_1`），当 `c(out_a, out_b)` 被调用时，`ModelHandle.__call__` 对每个 `Proxy` 参数各创建一条 `parallel_1 -> c_4` 的边，导致图中出现两条相同的边：

```
Edge(parallel_1 -> c_4)
Edge(parallel_1 -> c_4)
```

这是当前实现的正确行为：两条边分别代表 `a` 的输出和 `b` 的输出流向 `c`，在语义上是有区别的。但从纯 DAG 结构角度（仅关心节点间是否存在依赖关系），重复边是冗余的。实际实现中可考虑对边进行去重，或在边上增加端口（port）信息以区分不同输入槽位。

### 4.4 Kahn's Algorithm 对所有图形状均正确处理

线性链、菱形、复杂多路 DAG 全部通过拓扑排序验证。Kahn 算法的入度统计逻辑在 `Graph.topological_sort()` 中实现，并通过 `len(result) != len(self.nodes)` 检测环路（尽管当前原型不会主动产生环）。

### 4.5 Proxy 语义天然约束 trace 子集

由于 `Proxy` 对象既没有实现 `__getitem__`、`__getattr__` 等魔术方法，也没有数值，用户在 `trace()` 上下文中无法：

- 对 Proxy 做条件判断（`if proxy`）；
- 对 Proxy 做算术运算（`proxy + 1`）；
- 对 Proxy 做索引访问（`proxy["key"]`）；
- 在 Python 循环中用 Proxy 控制迭代次数。

这些限制正是 trace 子集所需要的：不允许副作用、不允许依赖运行时数据的动态控制流。Proxy 语义不是约束的主动实现，而是其自然结果。

---

## 5. 已知限制

### 5.1 Proxy 不支持属性访问和索引

当用户需要从模型输出中取某个键或字段时，例如：

```python
result = model_a(x)
value = result["logits"]      # AttributeError / TypeError
score = result.score          # AttributeError
```

当前 `Proxy` 未实现 `__getitem__` 和 `__getattr__`，上述调用会直接抛出异常，且错误信息对用户不友好。生产实现中需要在 `Proxy` 上添加这些方法，并返回带有路径信息的新 `Proxy`（例如记录 `source_node_id + key_path`），以支持结构化输出的依赖追踪。

### 5.2 不支持的 Python 构造的错误提示不友好

当用户在 `trace()` 上下文中使用了 Proxy 不支持的操作（如 `if proxy_val > 0.5:`），Python 会抛出通用异常，用户难以判断是哪里不符合 Nerva trace 语义。生产实现需要在 `Proxy` 的相关方法中抛出描述性错误，明确告知用户该操作在 trace 阶段不被支持，以及应使用哪个 Nerva 原语替代（例如 `nerva_cond`）。

### 5.3 当前原型无环路检测的主动防御

拓扑排序算法中有隐式环路检测（`len(result) != len(self.nodes)` 时抛 `ValueError`），但在图构建阶段没有任何防御。如果用户代码（或未来的 `ModelHandle` 实现）错误地将某节点的输出作为其自身的输入，错误只会在拓扑排序时才暴露。生产实现应在 `add_edge()` 时进行实时环路检测，或至少在 `trace()` 完成后立即验证。

---

## 6. 结论与建议

Spike S2 验证了以下核心结论：

**Proxy + `__call__` 拦截的方案可行，足以作为 Nerva `trace()` 的基础设计。** 五个测试用例覆盖了线性、并行、条件、菱形、复杂多路 DAG 等主要图形态，全部正确构建并通过拓扑排序验证。

### 对 Nerva trace 设计的影响

1. **Graph IR 层次化结构确认可用。** `cond`/`parallel` 节点嵌入子图的设计能够完整表达控制流语义，同时不破坏顶层 DAG 的可调度性。后续设计可保留该层次化 IR 结构。

2. **`ModelHandle` 是合适的用户接口抽象。** 用户定义 pipeline 时只需将模型声明为 `ModelHandle`，调用语法与真实推理完全一致，zero-friction。建议在生产实现中将其发展为注册式 API（例如 `@nerva.model` 装饰器）。

3. **需要在 `Proxy` 上扩展 `__getitem__` / `__getattr__`。** 实际模型输出普遍为字典或 dataclass，必须支持结构化输出的路径追踪，否则会严重限制用户表达能力。

4. **`parallel` 边去重或端口化需要决策。** 建议在 `Edge` 上增加可选的 `src_port` / `dst_port` 字段，以区分并行节点的多个输出分别对应下游节点的哪个输入参数，同时消除语义重复边。

5. **trace 子集文档应明确列出不支持的构造。** 由于 Proxy 语义天然限制了动态控制流，需要在用户文档中明确说明 trace 上下文中的约束，并为每类违规提供清晰的错误提示和替代方案（使用 `nerva_cond` / `nerva_parallel`）。

6. **全局上下文应替换为线程本地存储（thread-local）。** 当前使用 `global _current_ctx` 仅适合单线程 spike。生产实现应使用 `threading.local()` 或 Python contextvars，以支持多线程/异步环境下的并发 tracing。

---

## 7. 复现方式

确保已安装 `uv`，然后在项目根目录执行：

```bash
uv run python spikes/s2_trace_prototype.py
```

预期输出末尾为：

```
============================================================
  All tests passed!
============================================================
```

无需额外依赖，原型仅使用 Python 标准库（`collections`、`dataclasses`、`typing`）。
