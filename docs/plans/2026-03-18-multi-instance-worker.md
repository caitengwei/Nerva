# Per-Node Multi-Instance Worker 设计

## 1. 背景与目标

### 问题

当前每个 model 固定 1 个 Worker 进程。对于 CPU-bound 的 preprocess/postprocess 节点，
单进程成为瓶颈（在高并发下无法横向扩展处理能力），而 LLM 节点通常只需 1 个
Worker（GPU 由 vLLM 内部调度）。

### 目标

- 任意 DAG 节点可独立配置 Worker 实例数（并发度）
- 配置在 `model()` 声明时传入
- 保持多进程解耦架构：每个实例仍然是独立的 Worker 进程
- Executor 零改动：多实例 proxy 透明实现 `InferableProxy` 协议
- `instances=1`（默认值）行为与现有逻辑完全一致

### 用法示例

```python
mm_preprocess = model(
    "mm_preprocess", MMPreprocessModel,
    backend="pytorch", device="cpu",
    async_infer=True,
    instances=2,          # 2 个 Worker 进程
)
mm_mock_llm = model(
    "mm_mock_llm", MockCPULLMModel,
    backend="pytorch", device="cpu",
    async_infer=True,
    instances=1,          # 默认值，可省略
)
mm_postprocess = model(
    "mm_postprocess", MMPostprocessModel,
    backend="pytorch", device="cpu",
    async_infer=True,
    instances=2,
)
```

## 2. 架构

### 当前

```
_build_pipelines():
  for model_name in models:
    proxy = manager.start_worker(handle)     # 1 proxy per model
    proxies[model_name] = proxy

Executor:
  proxy = self._proxies[node.model_name]     # single proxy
  result = await proxy.infer(inputs, ctx)
```

### 目标

```
_build_pipelines():
  for model_name in models:
    proxy = manager.start_worker(handle)     # instances=1 → WorkerProxy
                                             # instances>1 → MultiInstanceProxy
    proxies[model_name] = proxy              # 对 Executor 透明

Executor:  (零改动)
  proxy = self._proxies[node.model_name]     # InferableProxy
  result = await proxy.infer(inputs, ctx)
```

### 进程拓扑（instances=2 for preprocess）

```
uvicorn worker-0                      Nerva Worker processes
  DEALER ─┐                          ┌── mm_preprocess-0  (ROUTER)
  DEALER ──┤  (2 DEALERs to 2        ├── mm_preprocess-1  (ROUTER)
           │   ROUTER instances)      │
  DEALER ──┼──────────────────────────├── mm_mock_llm-0    (ROUTER)
           │                          │
  DEALER ──┤                          ├── mm_postprocess-0 (ROUTER)
  DEALER ─┘                          └── mm_postprocess-1 (ROUTER)

uvicorn worker-1
  (同样的 DEALER 拓扑，连接到相同的 ROUTER sockets)
```

每个 uvicorn worker 为每个模型创建 `instances` 个 DEALER socket，
每个 DEALER 连接到对应实例的 ROUTER。

## 3. 组件设计

### 3.1 ModelHandle — 新增 `instances` 字段

```python
@dataclass(frozen=True)
class ModelHandle:
    name: str
    model_class: type[Model]
    backend: str
    device: str
    options: dict[str, Any] = field(default_factory=dict)
    batch_config: BatchConfig | None = None
    async_infer: bool = False
    instances: int = 1           # ← new
```

`model()` 函数新增 `instances` 参数，校验 `instances >= 1`。

### 3.2 MultiInstanceProxy — 新增类

实现 `InferableProxy` 协议，内部持有 N 个 `WorkerProxy`，round-robin 分发。

```python
class MultiInstanceProxy:
    """Load-balancing proxy wrapping multiple WorkerProxy instances.

    Implements InferableProxy. The Executor treats it exactly like a
    single WorkerProxy — multi-instance dispatch is transparent.
    """

    def __init__(self, proxies: list[WorkerProxy]) -> None:
        if not proxies:
            raise ValueError("At least one proxy required")
        self._proxies = proxies
        self._counter = itertools.count()

    async def infer(
        self,
        inputs: dict[str, Any],
        context: InferContext,
        shm_pool: ShmPool | None = None,
    ) -> dict[str, Any]:
        idx = next(self._counter) % len(self._proxies)
        return await self._proxies[idx].infer(inputs, context, shm_pool)

    async def health_check(self, timeout: float = 3.0) -> bool:
        """Healthy if any instance is healthy."""
        results = await asyncio.gather(
            *(p.health_check(timeout=timeout) for p in self._proxies),
            return_exceptions=True,
        )
        return any(r is True for r in results)

    async def shutdown(self) -> None:
        for p in self._proxies:
            await p.shutdown()

    async def close(self) -> None:
        for p in self._proxies:
            await p.close()
```

**为什么 round-robin 而不是 least-outstanding？**

- round-robin 无锁、O(1)、可预测
- 所有实例运行相同的 model，延迟分布一致
- 在 closed-loop 压测中，每个 proxy 的 `await infer()` 天然限流——
  caller 只有在前一个请求返回后才会发下一个请求
- 未来如果 DAG 有并行分支同时调用同一 model，可以升级到 least-outstanding

### 3.3 WorkerManager.start_worker() — 多实例逻辑

```python
async def start_worker(self, handle: ModelHandle) -> WorkerProxy | MultiInstanceProxy:
    if handle.instances <= 1:
        return await self._start_single_worker(handle)

    # 多实例：spawn N 个 Worker，创建 N 个 WorkerProxy
    proxies: list[WorkerProxy] = []
    for i in range(handle.instances):
        instance_handle = _instance_handle(handle, i)  # name → name-0, name-1, ...
        proxy = await self._start_single_worker(instance_handle)
        proxies.append(proxy)
    return MultiInstanceProxy(proxies)
```

**Socket path 命名**：

| instances | socket path |
|-----------|-------------|
| 1 | `nerva-mm_preprocess.sock` （不变） |
| 2 | `nerva-mm_preprocess-0.sock`, `nerva-mm_preprocess-1.sock` |

每个实例独立的：
- socket 文件
- spawning lock 文件
- refcount 文件
- Worker 进程

**实例 handle 生成**：

```python
def _instance_handle(handle: ModelHandle, index: int) -> ModelHandle:
    """Create a per-instance ModelHandle with a suffixed name."""
    return ModelHandle(
        name=f"{handle.name}-{index}",
        model_class=handle.model_class,
        backend=handle.backend,
        device=handle.device,
        options=handle.options,
        batch_config=handle.batch_config,
        async_infer=handle.async_infer,
        instances=1,  # individual instance, no recursion
    )
```

注意：`_instance_handle` 的 name 是 `{model_name}-{index}`，但这个 name
只用于 WorkerManager 内部标识（socket path、_workers dict key）。
在 `_build_pipelines()` 里，proxies dict 仍然用原始 `model_name` 作为 key：

```python
# serve.py _build_pipelines():
proxy = await manager.start_worker(handle)
proxies[handle.name] = proxy   # "mm_preprocess" → MultiInstanceProxy
```

Executor 通过 `node.model_name` 查找时用的是原始名字，所以透明。

### 3.4 WorkerManager 内部记账

当前 `_workers: dict[str, _WorkerEntry]` 用 `worker_id = handle.name` 作 key。

多实例后：
- `instances=1`：`_workers["mm_preprocess"]` → 单个 `_WorkerEntry`
- `instances=2`：`_workers["mm_preprocess-0"]` 和 `_workers["mm_preprocess-1"]`

Shutdown/restart 需要感知多实例，新增 `_instance_groups: dict[str, list[str]]`
记录 `model_name → [instance_worker_ids]`。

```python
# shutdown_all() 无需改动（遍历 _workers.items()）
# restart_worker("mm_preprocess") 需要 restart 所有实例
```

### 3.5 serve.py _build_pipelines() — 适配

```python
# 当前代码
proxy = await manager.start_worker(handle)
proxies[name] = proxy

# 改后（不需要改，因为 start_worker 返回的就是正确类型）
# instances=1 → WorkerProxy
# instances>1 → MultiInstanceProxy
# 两者都满足 InferableProxy，proxies dict 无感知
```

唯一需要改的是 `model_info` 列表（记录实例数信息用于 `/v1/health` 等端点）。

### 3.6 Executor — 零改动

`Executor._run_node()` 只调用 `proxy.infer()`，不感知底层是单实例还是多实例。
`PrecomputedGraph` 不受影响（Node 结构不变）。

### 3.7 timing 日志

多实例时每个 Worker 进程写独立的 timing log：
- `nerva_worker_{pid}.log`（已有，按 PID 区分）
- proxy 侧：`nerva_proxy_{model_name}.log`

proxy 侧的 timing log 需要区分实例。`MultiInstanceProxy` 内部的每个
`WorkerProxy` 已经在 `load_model()` 时根据 `model_name`（已经是 `name-{i}`）
创建独立的 timing log 文件。所以 **无需额外改动**。

## 4. 文件改动清单

| 文件 | 操作 | 改动量 | 说明 |
|------|------|--------|------|
| `src/nerva/core/model.py` | 修改 | +5 行 | `ModelHandle.instances` 字段 + `model()` 参数 + 校验 |
| `src/nerva/worker/proxy.py` | 新增类 | ~50 行 | `MultiInstanceProxy` 实现 |
| `src/nerva/worker/manager.py` | 修改 | ~30 行 | 多实例 spawn 逻辑 + `_instance_handle` + `_instance_groups` |
| `src/nerva/server/serve.py` | 修改 | ~5 行 | model_info 增加 instances 信息 |
| `src/nerva/engine/executor.py` | **不改** | 0 | InferableProxy 协议透明 |
| `src/nerva/core/graph.py` | **不改** | 0 | Node 结构不变 |
| `tests/test_multi_instance.py` | 新建 | ~120 行 | 多实例测试 |

## 5. 执行计划

### Step 1: ModelHandle + model() 增加 instances 字段

- `ModelHandle` 新增 `instances: int = 1`
- `model()` 新增 `instances: int = 1` 参数，校验 `>= 1`
- 零破坏性：默认值 1 与现有行为一致

### Step 2: MultiInstanceProxy

- 在 `worker/proxy.py` 底部新增 `MultiInstanceProxy` 类
- 实现 `infer()`, `health_check()`, `shutdown()`, `close()`
- Round-robin 分发

### Step 3: WorkerManager 多实例 spawn

- `start_worker()` 检查 `handle.instances`
- `instances > 1` 时循环创建实例，返回 `MultiInstanceProxy`
- 新增 `_instance_handle()` 辅助函数
- 新增 `_instance_groups` 记录实例组关系
- `shutdown_all()` 无需改动（已遍历 _workers）

### Step 4: serve.py 适配

- `_build_pipelines()` 的 `model_info` 增加 `instances` 字段
- 其余逻辑不变（`start_worker` 返回类型已兼容）

### Step 5: 测试

1. **unit**: `ModelHandle.instances` 默认 1，设置后生效
2. **unit**: `model()` 传递 instances，校验 `instances < 1` 抛异常
3. **unit**: `MultiInstanceProxy` round-robin 分发正确性
4. **unit**: `MultiInstanceProxy.health_check()` any-healthy 语义
5. **integration**: `instances=2` 的 model 启动 2 个 Worker 进程
6. **integration**: 多实例请求均匀分布到各实例
7. **regression**: 现有测试全部通过（`instances=1` 行为不变）

### Step 6: 压测验证

```bash
# preprocess/postprocess 各 2 实例，mock_llm 1 实例
# 修改 examples/mm_vllm_cpu_mock_server.py:
#   mm_preprocess = model(..., instances=2)
#   mm_postprocess = model(..., instances=2)

NERVA_TIMING_LOG_DIR=/private/tmp/nerva_timing_multi \
  uv run uvicorn examples.mm_vllm_cpu_mock_server:app \
    --workers 4 --port 8080

env -u all_proxy -u ALL_PROXY \
  uv run python scripts/bench/run_bench.py \
    --target nerva --concurrency-levels 40 --sample-seconds 60
```

预期结果：
- preprocess/postprocess 的 ipc_transport p99 下降（负载被分摊到 2 个 Worker）
- LLM 延迟不变
- 端到端 QPS 提升或持平

## 6. 设计决策记录

### 为什么不用 least-outstanding 策略？

Round-robin 在 Nerva 的场景下已经足够：
- 每个实例运行相同模型，延迟方差小
- Closed-loop 压测中 caller 天然限流
- 无锁 O(1) 实现，不引入额外竞争

如果未来出现场景（例如不同实例运行在不同速度的设备上），可以升级策略而不影响 API。

### 为什么不在 Node/Graph 层配置 instances？

- `instances` 是运行时部署配置，不是计算图结构
- 同一模型在不同部署环境可能需要不同 instances 数
- 放在 `model()` 与 `backend`, `device`, `async_infer` 一致——都是部署层面的声明

### 为什么不支持 instances=0（跳过部署）？

- 当前 DAG 是静态拓扑，跳过节点需要图重写
- 超出 MVP 范围，不引入复杂度

---

## 7. 实施状态与实测数据（2026-03-18）

**状态：已完成并合并（PR #19）**

### 7.1 实现变更

| 文件 | 说明 |
|------|------|
| `src/nerva/core/model.py` | `ModelHandle.instances` 字段；`model()` 参数 + 校验 |
| `src/nerva/worker/proxy.py` | `MultiInstanceProxy`（round-robin，any-healthy） |
| `src/nerva/worker/manager.py` | 多实例 spawn 逻辑；`_instance_groups`；`restart_worker` 类型修复 |
| `src/nerva/observability/timing.py` | `_writer_loop` finally 关闭文件，消除 race condition |
| `examples/mm_vllm_cpu_mock_server.py` | `NERVA_PRE_POST_INSTANCES` 环境变量控制实例数（默认 3） |
| `scripts/bench/infra/prepare_triton_repo.py` | 新增 pre/post latency 参数和 `--pre-post-instance-count` |
| `scripts/bench/sweep_instances_concurrency.py` | **新建**：instance × C 参数扫描脚本 |
| `tests/test_multi_instance.py` | 27 个测试（unit + integration + regression） |
| `tests/helpers.py` | `PidModel`（用于分布验证） |

### 7.2 Nerva vs Triton 对比压测（C=40，instances=3）

**配置**（两侧完全一致）：
- preprocess：5ms sleep + 10% jitter，3 实例
- LLM mock：256×0.5ms = 128ms sleep，1 实例
- postprocess：2ms sleep + 10% jitter，3 实例
- 理论最低端到端延迟：~135ms

| 指标 | Nerva | Triton | Nerva 优势 |
|------|-------|--------|-----------|
| QPS | **153.8** | 93.7 | +64% |
| p50 | **259.7ms** | 400.6ms | -35% |
| p95 | **303.4ms** | 684.9ms | -56% |
| p99 | **338.2ms** | 795.9ms | -58% |
| error_rate | 0.0 | 0.0 | — |

**差距根因**：Triton Python backend 使用 `time.sleep()`（阻塞），`instance_count=3` 等价于
最多 3 个并发 preprocess 请求；其余 37 个在 OS thread queue 排队。Nerva 使用
`asyncio.sleep()`（`async_infer=True` 路径），Worker 事件循环非阻塞，3 个进程各自多路复用，
无排队等待。

### 7.3 Triton pre/post instance_count 扫描（C=40）

| instance_count | QPS | p50 | p95 | p99 | error_rate |
|---|---|---|---|---|---|
| 1 | 43.9 | 845ms | 1452ms | 1698ms | 0.0 |
| 2 | 43.0 | 881ms | 1235ms | 2602ms | 0.0 |
| 3 | 93.7 | 401ms | 684ms | 796ms | 0.0 |

instance=1→2 无显著提升（LLM 20 实例已足够，pre/post blocking thread 需要更多实例才能摊平）；
instance=3 出现明显跃升，Triton 调度与 pipeline 流量匹配改善。

### 7.4 可重复压测

```bash
# instance × concurrency 全量扫描
uv run python scripts/bench/sweep_instances_concurrency.py \
  --target nerva triton \
  --instances 1 2 3 5 8 10 \
  --concurrency-levels 1,4,8,16,32,64,100 \
  --warmup-seconds 20 \
  --sample-seconds 30
```
