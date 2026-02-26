# Phase 2 Design — DAG Pipeline 编排

## 1. 目标与范围

### 1.1 目标

将 Phase 1 的单模型 Master-Worker 架构扩展为多模型 DAG 编排：
- 用户通过 `trace()` 或显式 API 构建计算图（Graph IR）
- 支持 `Proxy.__getitem__` 进行结构化输出字段访问
- 提供 `cond()` / `parallel()` 控制流原语
- 事件驱动 Executor 通过 WorkerProxy 调度 DAG 执行

### 1.2 范围

**包含：**
- Graph IR 数据结构（Node、Edge、Graph）
- Proxy 对象 + `trace()` 函数追踪构建 DAG
- `__getitem__` 支持结构化输出路径追踪
- `cond()` / `parallel()` 控制流原语
- 事件驱动 DAG Executor（in-degree table + done_queue）
- ModelHandle 双态行为（trace 模式 / 运行时模式）
- Dict 输入映射（`fusion({"img": out_a, "txt": out_b})`）

**不包含（后续 Phase）：**
- Dynamic Batching（Phase 3）
- HTTP 服务层（Phase 4）
- Edge 端口化去重
- 主动环路检测（add_edge 时）
- 友好的 trace 约束错误提示

### 1.3 验证标准

- trace() 正确构建 linear/diamond/parallel/cond 拓扑的 Graph
- Proxy.__getitem__ 正确记录 field_path，边包含路径信息
- Dict 输入映射正确生成 dst_input_key
- contextvars 隔离并发 trace 调用
- Executor 按拓扑序通过 WorkerProxy（或 mock）执行 DAG
- parallel 节点并行执行子图，输出以 `{"0": ..., "1": ...}` 映射
- cond 节点根据 predicate 选择执行分支
- fail-fast：任一节点失败立即取消剩余并抛出异常
- 全量回归通过（140 tests）

### 1.4 前置依赖

- Phase 0 完成（Model/Backend 接口）
- Phase 1 完成（WorkerProxy/WorkerManager IPC 链路）
- S2 spike 已验证 Proxy + `__call__` 拦截方案可行

---

## 2. 架构概览

```
用户代码                        Nerva 内部
┌─────────────────────┐       ┌───────────────────────────────────────┐
│                     │       │                                       │
│  img = model("img") │       │  trace() 阶段:                        │
│  txt = model("txt") │       │    ModelHandle.__call__ 检测 context   │
│  fuse = model("fuse")│      │    → 创建 Node + Edge + 返回 Proxy     │
│                     │       │    → cond/parallel 创建子图            │
│  def pipeline(x):   │       │    → 产出 Graph IR                    │
│    a, b = parallel(  │      │                                       │
│      λ: img(x),     │       │  execute() 阶段:                      │
│      λ: txt(x))     │       │    Executor 读取 Graph                │
│    return fuse(      │       │    → 按 in-degree 调度节点            │
│      {"img": a[..], │       │    → 通过 WorkerProxy.infer() 执行    │
│       "txt": b[..]})│       │    → 组装输入 / 解析 field_path        │
│                     │       │                                       │
│  g = trace(pipeline) │      │                                       │
│  executor.execute(g) │      │                                       │
└─────────────────────┘       └───────────────────────────────────────┘
```

调用链：`trace()` → `Graph` → `Executor` → `WorkerProxy.infer()`

---

## 3. Graph IR 设计

### 3.1 核心类型

```python
@dataclass(frozen=True)
class Edge:
    src: str                                    # 源节点 ID
    dst: str                                    # 目标节点 ID
    src_field_path: tuple[str, ...] = ()        # proxy["a"]["b"] → ("a", "b")
    dst_input_key: str | None = None            # fusion({"img": ...}) → "img"

@dataclass
class Node:
    id: str                                     # 唯一 ID（格式: {model_name}_{counter}）
    model_name: str                             # 模型名 或 "cond"/"parallel"
    node_type: str = "call"                     # "call" | "cond" | "parallel"
    true_branch: Graph | None = None            # cond 专用
    false_branch: Graph | None = None           # cond 专用
    branches: list[Graph] = field(...)          # parallel 专用

@dataclass
class Graph:
    nodes: list[Node]
    edges: list[Edge]
```

### 3.2 相对 S2 spike 的关键演进

| 变更 | S2 Spike | Phase 2 实现 | 理由 |
|------|----------|-------------|------|
| Edge 字段 | 仅 src/dst | +src_field_path, +dst_input_key | 支持 `proxy["key"]` 和 dict 输入映射 |
| Edge 可变性 | 普通 dataclass | frozen=True | Edge 是不可变值对象 |
| Graph 方法 | add_node/add_edge/topo_sort | +node_map, +predecessors/successors, +incoming_edges | Executor 调度所需 |
| 上下文管理 | 全局变量 _current_ctx | contextvars | 并发安全 |
| Proxy | 无 __getitem__ | __getitem__ 返回新 Proxy，追加 field_path | 结构化输出取值是刚需 |

### 3.3 Graph 查询方法

Executor 需要高效查询图结构，Graph 提供以下方法：

- `node_map` (property) — `{node_id: Node}` 映射
- `predecessors(node_id)` — 返回所有前驱节点 ID
- `successors(node_id)` — 返回所有后继节点 ID
- `incoming_edges(node_id)` — 返回所有指向该节点的 Edge（含 field_path/dst_input_key）
- `topological_sort()` — Kahn 算法，检测环路

---

## 4. Proxy 与 trace() 设计

### 4.1 Proxy

```python
class Proxy:
    __slots__ = ("_field_path", "name", "source_node_id")

    def __getitem__(self, key: str) -> Proxy:
        if not isinstance(key, str):
            raise TypeError(...)
        return Proxy(
            source_node_id=self.source_node_id,
            name=f"{self.name}[{key!r}]",
            field_path=(*self._field_path, key),
        )
```

**设计要点：**
- `__getitem__` 返回新 Proxy，`_field_path` 在父基础上追加 key
- 非 string key 报 TypeError（防止误用整数索引）
- `source_node_id` 不变（仍指向产生输出的同一节点）
- 使用 `__slots__` 减少内存开销

### 4.2 TraceContext

```python
class TraceContext:
    graph: Graph
    _counter: int

    def next_id(prefix: str) -> str      # 生成唯一节点 ID
    def swap_graph(new: Graph) -> Graph   # 用于 cond/parallel 子图隔离
```

使用 `ContextVar[TraceContext | None]`，支持多 asyncio task 并发 trace。

### 4.3 trace()

```python
def trace(fn: Callable[..., Any], num_inputs: int = 1) -> Graph:
    ctx = TraceContext()
    token = _trace_context_var.set(ctx)
    try:
        inputs = [Proxy(source_node_id=None, name=f"input_{i}") for i in range(num_inputs)]
        fn(inputs[0]) if num_inputs == 1 else fn(*inputs)
        return ctx.graph
    finally:
        _trace_context_var.reset(token)
```

### 4.4 _extract_proxy_edges()

从 ModelHandle 的输入参数中提取 Edge 列表：

| 输入形式 | 提取逻辑 |
|---------|----------|
| 单个 Proxy | 一条 Edge，无 dst_input_key |
| dict[str, Proxy\|Any] | 每个 Proxy 值一条 Edge，带 dst_input_key |
| 非 Proxy 值 | 忽略（常量输入，不产生依赖边） |

### 4.5 ModelHandle 双态行为

修改 `ModelHandle.__call__`，通过 `get_trace_context()` 检测当前模式：

- **Trace 模式**：创建 Node + 提取 Edge + 返回 Proxy
- **非 trace 模式**：保持原有 RuntimeError

使用延迟导入（`from nerva.core.proxy import ...`）避免 model.py ↔ proxy.py 循环依赖。

---

## 5. 控制流原语

### 5.1 parallel()

```python
def parallel(*fns: Callable[[], Any]) -> tuple[Proxy, ...]:
```

**执行流程（trace 阶段）：**
1. 为每个 branch 函数临时 swap_graph 到独立子图
2. 执行 branch 函数，收集子图
3. 恢复父图
4. 创建 `parallel` Node，嵌入所有子图
5. 返回 tuple of Proxy，每个 Proxy 的初始 `field_path=(str(i),)`

**关键设计：parallel 输出映射**

每个 branch 的输出 Proxy 初始 `field_path=(str(i),)`，这样：
- `out_a["features"]` 的 field_path 为 `("0", "features")`
- `out_b["features"]` 的 field_path 为 `("1", "features")`

Executor 在运行时将 parallel 输出组装为 `{"0": result_0, "1": result_1, ...}`，与 field_path 解析统一。

### 5.2 cond()

```python
def cond(predicate: Any, true_fn: Callable[[], Any], false_fn: Callable[[], Any]) -> Proxy:
```

**执行流程（trace 阶段）：**
1. 分别 trace true/false branch 为独立子图
2. 创建 `cond` Node，嵌入两个子图
3. predicate 若为 Proxy → 创建 predicate → cond 的边
4. 返回 Proxy(source_node_id=cond_id)

---

## 6. Executor 设计

### 6.1 核心算法

```
初始化:
  topo_order = graph.topological_sort()
  in_degree[node_id] = len(predecessors)
  completed = {}  # node_id → output
  done_queue = asyncio.Queue()

  启动所有 in_degree == 0 的节点

主循环:
  while remaining > 0:
    item = await done_queue.get()
    if isinstance(item, Exception):
      fail-fast: cancel all running, raise
    done_id = item
    remaining -= 1
    for succ in successors(done_id):
      in_degree[succ] -= 1
      if in_degree[succ] == 0:
        启动 succ（build inputs from completed → create_task）

Fail-fast:
  任何节点失败 → cancel all running → raise RuntimeError
```

### 6.2 节点类型执行

| 类型 | 执行逻辑 |
|------|---------|
| `call` | `await proxies[model_name].infer(inputs, node_ctx)` |
| `parallel` | 为每个 branch 创建子 Executor，`asyncio.gather` 并行，输出 `{"0": ..., "1": ...}` |
| `cond` | 取 predicate 真实值，只执行选中分支的子 Executor |

### 6.3 输入组装 (_build_node_inputs)

| 入边情况 | 组装逻辑 |
|---------|---------|
| 无入边 | 使用 pipeline inputs |
| 所有入边都有 dst_input_key | 组装 dict（`{key: resolve(output, field_path)}`） |
| 单条入边无 key | pass-through（可能带 field_path 解析） |
| 多条入边无 key | 组装为 list（兜底） |

### 6.4 resolve_field_path()

```python
def resolve_field_path(output: Any, field_path: tuple[str, ...]) -> Any:
    return reduce(lambda d, k: d[k], field_path, output)
```

### 6.5 Per-node InferContext

Executor 为每个节点生成唯一的 `request_id`（格式 `{base_request_id}:{node_id}:{uuid8}`），避免并发节点共用 WorkerProxy 时的 pending future 冲突。使用 `dataclasses.replace()` 从 pipeline context 派生。

### 6.6 InferableProxy Protocol

```python
class InferableProxy(Protocol):
    async def infer(
        self, inputs: dict[str, Any], context: InferContext, **kwargs: Any,
    ) -> dict[str, Any]: ...
```

WorkerProxy 天然满足此 Protocol，测试中可使用 AsyncMock 替代。

---

## 7. 循环依赖处理

```
model.py ←── proxy.py ←── graph.py
    │              ↑
    └──────────────┘  (ModelHandle.__call__ imports proxy)
```

解决方案：`ModelHandle.__call__` 内部使用延迟导入：

```python
def __call__(self, inputs):
    from nerva.core.proxy import Proxy, _extract_proxy_edges, get_trace_context
    from nerva.core.graph import Node
    ...
```

---

## 8. 文件结构

```
src/nerva/
  core/
    graph.py          # 新建 — Edge, Node, Graph
    proxy.py          # 新建 — Proxy, TraceContext, trace(), _extract_proxy_edges()
    primitives.py     # 新建 — cond(), parallel()
    model.py          # 修改 — ModelHandle.__call__ 双态行为
  engine/
    executor.py       # 新建 — Executor, resolve_field_path(), InferableProxy
  __init__.py         # 修改 — 新增导出

tests/
  test_graph.py       # 新建 — 10 cases
  test_proxy.py       # 新建 — 15 cases
  test_primitives.py  # 新建 — 5 cases
  test_executor.py    # 新建 — 13 cases (mock WorkerProxy)
  test_phase2_e2e.py  # 新建 — 5 cases (真实 Worker 进程)
  helpers.py          # 修改 — 新增 UpperModel, ConcatModel
```

---

## 9. 与前序 Phase 的关系

### 9.1 Phase 0 — 无修改

Model ABC、Backend ABC、Backend registry 保持不变。

### 9.2 Phase 1 — 无修改

WorkerProxy、WorkerManager、IPC 协议、SHM pool 保持不变。Executor 通过 `InferableProxy` Protocol 解耦，不依赖 WorkerProxy 具体实现。

### 9.3 ModelHandle 修改

`ModelHandle.__call__` 是唯一修改的 Phase 0 代码。修改方式：
- 新增 trace 模式分支（延迟导入，检查 ContextVar）
- 原有 RuntimeError 行为保留为 fallback
- `ModelHandle` 仍为 `frozen=True` dataclass

---

## 10. 公共 API

Phase 2 新增以下公共 API（通过 `nerva.__init__` 导出）：

| 符号 | 类型 | 用途 |
|------|------|------|
| `Graph` | class | 计算图容器 |
| `Node` | class | 图节点 |
| `Edge` | class | 图边（含 field_path、input_key） |
| `Proxy` | class | Trace 阶段的占位对象 |
| `Executor` | class | DAG 执行器 |
| `trace()` | function | 函数追踪构建 Graph |
| `cond()` | function | 条件分支原语 |
| `parallel()` | function | 并行执行原语 |

---

## 11. 测试策略

### 11.1 单元测试

- **test_graph.py** (10 cases) — Graph 构建、node_map、predecessors/successors、incoming_edges、拓扑排序（linear/diamond）、环路检测、Edge frozen/field_path/dst_input_key
- **test_proxy.py** (15 cases) — Proxy.__getitem__ 单层/链式/非 string 报错；_extract_proxy_edges 单 Proxy/dict/None 源/混合值；trace linear/getitem/diamond/dict mapping；context 隔离；ModelHandle trace 模式
- **test_primitives.py** (5 cases) — parallel 基本/输出 field_path；cond 基本/predicate 边；嵌套 parallel-in-cond

### 11.2 Executor 测试（mock）

- **test_executor.py** (13 cases) — resolve_field_path（empty/single/nested/missing）；linear 执行/field_path；diamond dict 输入组装；parallel 节点；cond true/false 分支；fail-fast；missing proxy 报错；空图

### 11.3 端到端集成测试

- **test_phase2_e2e.py** (5 cases) — 单节点 DAG；线性链（直传/field_path 映射）；parallel → concat（真实 Worker 进程）；trace → execute 全流程

### 11.4 回归保证

全量测试 140 passed，Phase 0/Phase 1 的 89 个测试无破坏。

---

## 12. 设计决策

| 决策 | 选项 | 理由 |
|------|------|------|
| Proxy.__getitem__ | 记录 field_path tuple，执行时逐层取值 | 模型输出取字段是刚需（`img_feat["features"]`） |
| trace 上下文 | contextvars 替代全局变量 | 并发安全，支持多 asyncio task 同时 trace |
| Executor 并发模型 | 事件驱动（入度表 + done_queue） | 最大化并发度，每个节点完成后立即触发后继 |
| parallel 输出映射 | Proxy 初始 field_path=(str(i),) | 区分不同 branch 的输出，与 __getitem__ 统一 |
| per-node request_id | `{base}:{node_id}:{uuid8}` | 避免 WorkerProxy pending future 冲突 |
| 循环依赖 | ModelHandle.__call__ 内延迟导入 | 最小侵入性，不需要重构模块结构 |
| InferableProxy | Protocol 而非基类 | 结构子类型，WorkerProxy 和 mock 均天然满足 |

---

## 13. 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| 循环导入（model.py ↔ proxy.py ↔ graph.py） | `__call__` 内延迟导入，已验证可行 |
| parallel 输出映射到后续节点 | Proxy 初始 field_path=(str(i),) 统一路径解析，已测试覆盖 |
| 并发节点共用 WorkerProxy request_id 冲突 | per-node unique request_id 生成，e2e 测试已验证 |
| cond predicate 传递 | Executor 对 cond 节点做特殊 input 处理 |
| mypy strict + Proxy 类型 | Proxy.__getitem__ 返回 Proxy，用户 pipeline 可能需要 type: ignore |
| 大 DAG 拓扑排序性能 | Kahn 算法 O(V+E)，MVP 规模不会成为瓶颈 |

---

## 14. 后续优化方向（不在 Phase 2 范围内）

1. **Edge 端口化去重** — parallel 节点的多条出边可通过 src_port/dst_port 区分
2. **主动环路检测** — 在 add_edge() 时实时检测，而非仅在 topological_sort() 时
3. **友好的 trace 约束错误提示** — Proxy 上的不支持操作应给出描述性错误
4. **Graph 序列化** — 支持 JSON/protobuf 序列化，便于可视化和调试
5. **Executor 超时** — per-node deadline 支持
6. **重试策略** — 节点失败后可选重试而非立即 fail-fast

---

## Changelog

| 日期 | 变更 |
|------|------|
| 2026-02-26 | 初始版本（Phase 2 实现完成后补充设计文档） |
