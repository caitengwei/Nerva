# Phase 3 Design — Dynamic Batching + Deadline 调度

## 1. 目标与范围

### 1.1 目标

在 Phase 2 DAG Executor 基础上引入动态批处理能力，通过 `DynamicBatcher` 对 `WorkerProxy` 进行透明包装，实现：

- 请求聚合（size 触发 + timer 触发）
- Deadline 感知准入控制
- 队列满时有界等待 + 超时拒绝（backpressure）
- Cancellation propagation（部分实现：deadline 过期过滤 + stop() drain；主动取消推迟至 Phase 4+）

### 1.2 范围

**包含：**

- `engine/batcher.py` — `BatchConfig`、`DynamicBatcher`（含 deadline 准入，合并 scheduler 职责）
- `core/model.py` — `model()` 新增 `batch_config` 参数，存入 `ModelHandle`
- 单元测试（mock inner proxy）+ 集成测试（真实 Worker）

**不包含（后续 Phase）：**

- `batch()` 用户侧 transform API（组图时透明，无需显式声明）
- `engine/scheduler.py`（deadline 准入合并进 Batcher）
- 多 Worker 副本调度（Phase 4+）
- GPU 级 batched forward pass（见 TODO 节）
- `serve()` 自动 wiring（Phase 4）

### 1.3 验证标准

- size 触发：N 个并发请求在满 `max_batch_size` 时触发批次，inner 调用次数正确
- timer 触发：少量请求提交后等待 `max_delay_ms`，批次按时触发，不永久阻塞
- deadline 准入：剩余 deadline < `min_remaining_deadline_ms` 时进队前立即拒绝
- 批次内过期过滤：进队后过期的请求在批次触发时被跳过，不调用 inner
- backpressure：队列满且超过 `queue_timeout_ms` 后抛 `RESOURCE_EXHAUSTED`
- stop() drain：stop 后剩余请求 future 全部收到异常，无泄漏
- 并发正确性：多 coroutine 并发调用结果与请求一一对应，无串话
- 全量回归通过

### 1.4 前置依赖

- Phase 1 完成（WorkerProxy / WorkerManager IPC 链路）
- Phase 2 完成（DAG Executor，`InferableProxy` Protocol）
- S4 spike 已验证 asyncio Queue + timer 方案可行（14.6x 吞吐提升，调度开销 ~0.2ms）

---

## 2. 架构概览

### 2.1 调用链

```
Executor
  └→ DynamicBatcher.infer()        ← 实现 InferableProxy，对 Executor 透明
       │  持 asyncio.Queue[PendingRequest]
       │  _batch_loop（后台 task）
       ↓ 批次就绪（size 触发 或 timer 触发）
     WorkerProxy.infer() × N        ← asyncio.gather 并发触发
       └→ Worker → Model.infer()
```

Phase 3 对 Executor 完全透明——Executor 只认 `InferableProxy` 协议，不感知 Batcher 存在。

### 2.2 新增 / 修改文件

```
src/nerva/engine/batcher.py     # 新增：DynamicBatcher, BatchConfig, PendingRequest
src/nerva/core/model.py         # 修改：model() 新增 batch_config 参数
tests/test_batcher.py           # 新增：单元测试（mock inner proxy）
tests/test_phase3_e2e.py        # 新增：集成测试（真实 Worker）
```

### 2.3 用户侧 API（组图时透明）

```python
# 组图时与 Phase 2 完全一致，batch_config 只是额外参数
echo = model("echo", EchoModel, max_batch_size=16, max_delay_ms=5)

def pipeline(x):
    return echo(x)

g = trace(pipeline)

# 执行时（Phase 3 MVP 手动 wiring，Phase 4 serve() 自动）
batcher = DynamicBatcher(worker_proxy, config=handle.batch_config)
executor = Executor(g, {"echo": batcher}, ctx)
result = await executor.execute(inputs)
```

---

## 3. 组件详细设计

### 3.1 `BatchConfig`

```python
@dataclass
class BatchConfig:
    max_batch_size: int = 32
    max_delay_ms: float = 10.0
    queue_capacity: int = 2048          # 对应 SCHED_QUEUE_CAPACITY_PER_MODEL
    queue_timeout_ms: float = 100.0     # 队列满时等待上限，超时抛 RESOURCE_EXHAUSTED
    min_remaining_deadline_ms: float = 5.0  # 准入最小剩余 deadline
```

默认值来源：`mvp-defaults.md`（`SCHED_QUEUE_CAPACITY_PER_MODEL`、`SCHED_ADMISSION_MIN_REMAINING_DEADLINE_MS`）及 S4 spike 推荐（`max_batch_size=32`、`max_delay_ms` ≤ 10ms）。

### 3.2 `DynamicBatcher` 接口

```python
class DynamicBatcher:
    def __init__(self, inner: InferableProxy, config: BatchConfig) -> None: ...

    async def start(self) -> None: ...   # 启动 _batch_loop 后台 task
    async def stop(self) -> None: ...    # 取消 _batch_loop，drain 剩余请求

    async def infer(
        self,
        inputs: dict[str, Any],
        context: InferContext,
        **kwargs: Any,
    ) -> dict[str, Any]: ...            # 实现 InferableProxy

    async def __aenter__(self) -> DynamicBatcher: ...
    async def __aexit__(self, *_: Any) -> None: ...
```

### 3.3 `infer()` 提交侧流程

```
infer(inputs, context)
  │
  ├─ 1. deadline 准入检查
  │      remaining_ms = context.deadline_ms - elapsed_ms(context)
  │      if remaining_ms < config.min_remaining_deadline_ms:
  │          raise RuntimeError("DEADLINE_EXCEEDED")
  │
  ├─ 2. 队列满时有界等待
  │      future = asyncio.get_running_loop().create_future()
  │      pending = PendingRequest(inputs, context, future, enqueue_time=now())
  │      try:
  │          await asyncio.wait_for(
  │              queue.put(pending),
  │              timeout=config.queue_timeout_ms / 1000,
  │          )
  │      except asyncio.TimeoutError:
  │          raise RuntimeError("RESOURCE_EXHAUSTED")
  │
  └─ 3. 挂起等待结果
         return await future
```

### 3.4 `_batch_loop` 聚合侧流程

与 S4 spike 结构一致，增加 deadline 过滤：

```
loop forever:
  1. pending = await queue.get()          # 等第一个请求（阻塞）
  2. batch = [pending]
     deadline = now() + config.max_delay_ms / 1000

  3. 在 deadline 内尽量凑满 max_batch_size：
       while len(batch) < max_batch_size:
           remaining = deadline - now()
           if remaining <= 0: break
           try:
               item = await asyncio.wait_for(queue.get(), timeout=remaining)
               batch.append(item)
           except asyncio.TimeoutError:
               break

  4. 过滤批次内已过期请求：
       valid, expired = [], []
       for req in batch:
           remaining = req.context.deadline_ms - elapsed_ms(req.context)
           if remaining < 0:
               expired.append(req)
           else:
               valid.append(req)
       for req in expired:
           req.future.set_exception(RuntimeError("DEADLINE_EXCEEDED"))

  5. 若 valid 为空 → 跳过本轮，continue

  6. asyncio.gather(*[inner.infer(req.inputs, req.context) for req in valid],
                    return_exceptions=True)

  7. 结果分发：
       for req, result in zip(valid, results):
           if isinstance(result, BaseException):
               req.future.set_exception(result)
           else:
               req.future.set_result(result)
```

### 3.5 `stop()` drain

```
1. cancel _batch_loop task
2. await task（suppress CancelledError）
3. drain queue：
     while not queue.empty():
         req = queue.get_nowait()
         req.future.set_exception(RuntimeError("batcher stopped"))
```

### 3.6 `ModelHandle` 集成

`model()` 新增可选参数，存入 `ModelHandle`：

```python
def model(
    name: str,
    model_class: type[Model],
    *,
    backend: str = "pytorch",
    batch_config: BatchConfig | None = None,
    **kwargs: Any,
) -> ModelHandle: ...
```

`ModelHandle.batch_config` 在执行时由调用方用于决定是否包装 `DynamicBatcher`。

---

## 4. TODO：Model-Level Batching

> 当前 Phase 3 的批处理是调度层聚合（`asyncio.gather` 并发触发 N 个单请求 IPC），
> 不是真正的 GPU batched forward pass。

对于 PyTorch 推理和部分自定义 Model，将 batch 直接下放到 `Model.infer()` 是必要的（如 `torch.stack` 合并张量后单次 forward）。届时有两条路径：

**路径 A：修改接口**

```python
# Model.infer 改为 batch-aware
def infer(self, batch: list[dict[str, Any]], context: InferContext) -> list[dict[str, Any]]: ...
```

优点：语义清晰，GPU 利用率最优。缺点：破坏 Phase 0 单请求接口契约，需要全量迁移。

**路径 B：Model 内部异步攒批**

`Model.infer()` 接口不变，Model 自己内部持有 asyncio Queue 做攒批（类似 vLLM `AsyncLLMEngine`）。`DynamicBatcher` 退化为透传，Worker 内部的 Model 自己管理批次窗口。

优点：不破坏现有接口。缺点：Batcher 与 Model 各自管理批次，逻辑分裂，调度可见性差。

**结论：** 两条路径均保留，待 Phase 4/5 有真实模型接入时再决策。`DynamicBatcher` 的 `inner` 设计为 `InferableProxy` 协议，替换 inner 实现时 Executor 不变。

---

## 5. 错误码约定

| 场景 | 异常消息 / 状态码 |
|---|---|
| 准入 deadline 不足 | `DEADLINE_EXCEEDED` |
| 队列满超时 | `RESOURCE_EXHAUSTED` |
| 批次内请求过期 | `DEADLINE_EXCEEDED` |
| batcher 已停止 | `RuntimeError("batcher stopped")` |
| inner proxy 推理失败 | 透传 inner 异常 |

---

## 6. 测试矩阵

### 6.1 单元测试（`tests/test_batcher.py`，mock inner proxy）

| 场景 | 验证点 |
|---|---|
| size 触发 | N 请求同时提交，inner 调用次数 = ceil(N / max_batch_size) |
| timer 触发 | 少量请求提交后等待，批次按时触发，不永久阻塞 |
| 准入 deadline 拒绝 | 剩余 deadline 不足，进队前立即 `DEADLINE_EXCEEDED` |
| 批次内过期过滤 | 进队后 deadline 到期，批次触发时跳过，不调用 inner |
| 队列满等待超时 | 填满 queue 后再提交，超过 `queue_timeout_ms` 抛 `RESOURCE_EXHAUSTED` |
| stop() drain | stop 后剩余请求 future 全部收到异常，不泄漏 |
| 并发正确性 | 多 coroutine 并发调用，结果与请求一一对应，无串话 |
| inner 异常透传 | inner.infer() 抛异常，对应 future 收到同一异常 |

### 6.2 集成测试（`tests/test_phase3_e2e.py`，真实 Worker）

| 场景 | 验证点 |
|---|---|
| e2e batcher + WorkerProxy | N 个并发请求经 DynamicBatcher 正确返回结果 |
| DAG pipeline with batcher | Executor 使用 DynamicBatcher proxy，线性 DAG 端到端正确执行 |

### 6.3 不在 Phase 3 测试范围

- GPU 批推理效果（需真实模型权重）
- 多 Worker 副本场景（Phase 4+）

---

## 7. 关键设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| Batcher 位置 | WorkerProxy 外层 Wrapper（Option A） | InferableProxy 协议解耦，Executor 不变，可独立测试 |
| Scheduler 合并 | deadline 准入内嵌进 Batcher | Phase 3 MVP 复杂度可控，Phase 4+ 再分离 |
| 用户 API | 组图透明，`model()` 参数声明 | 不污染 pipeline 定义，与 Phase 2 trace() 完全兼容 |
| backpressure | 等待 + 超时拒绝（`queue_timeout_ms`） | 比立即拒绝更友好，比无限等待更安全 |
| 批次并发执行 | `asyncio.gather` 并发 N 个单请求 | MVP 不改 Model 接口，model-level batching 留 TODO |

---

## Changelog

| 日期 | 变更 |
|---|---|
| 2026-02-27 | 初始版本 |
