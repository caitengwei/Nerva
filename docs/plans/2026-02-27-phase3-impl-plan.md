# Phase 3 Implementation Plan — Dynamic Batching + Deadline 调度

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 WorkerProxy 外层实现透明 `DynamicBatcher`，支持 size/timer 触发批次聚合、deadline 准入过滤、backpressure 有界等待。

**Architecture:** `DynamicBatcher` 实现 `InferableProxy` 协议，包装任意 inner proxy。Executor 完全不变，batcher 在调用链 `Executor → DynamicBatcher → WorkerProxy` 中透明插入。`BatchConfig` 通过 `model()` 参数声明并存入 `ModelHandle`，不影响组图。

**Tech Stack:** Python 3.11+, asyncio, pytest + pytest-asyncio (asyncio_mode=auto), uv

**Worktree:** `.worktrees/poc_phase_3`（分支 `poc_phase_3`，已从 `poc_phase_2` 检出）

---

## 前置检查

在 worktree 中确认基线干净：

```bash
cd /path/to/Nerva/.worktrees/poc_phase_3
uv run pytest tests/ -q
# 期望：159 passed（含 poc_phase_2 末尾的修复）
```

> **注意：** 若 poc_phase_3 缺少 poc_phase_2 最新提交（设计文档、.gitignore），先 merge：
> ```bash
> git merge poc_phase_2
> ```

---

## Task 1：`BatchConfig` + `_PendingRequest`

**Files:**
- Create: `src/nerva/engine/batcher.py`
- Create: `tests/test_batcher.py`

### Step 1：写失败测试

```python
# tests/test_batcher.py
from nerva.engine.batcher import BatchConfig

def test_batch_config_defaults() -> None:
    cfg = BatchConfig()
    assert cfg.max_batch_size == 32
    assert cfg.max_delay_ms == 10.0
    assert cfg.queue_capacity == 2048
    assert cfg.queue_timeout_ms == 100.0
    assert cfg.min_remaining_deadline_ms == 5.0

def test_batch_config_custom() -> None:
    cfg = BatchConfig(max_batch_size=8, max_delay_ms=5.0)
    assert cfg.max_batch_size == 8
    assert cfg.max_delay_ms == 5.0
    assert cfg.queue_capacity == 2048  # 默认值不变
```

### Step 2：确认测试失败

```bash
uv run pytest tests/test_batcher.py -v
# 期望：ERROR (ImportError: cannot import name 'BatchConfig')
```

### Step 3：实现 `BatchConfig` + `_PendingRequest`

```python
# src/nerva/engine/batcher.py
"""Dynamic Batcher with deadline-aware admission and backpressure."""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nerva.backends.base import InferContext
    from nerva.engine.executor import InferableProxy


@dataclass
class BatchConfig:
    """Configuration for DynamicBatcher."""

    max_batch_size: int = 32
    max_delay_ms: float = 10.0
    queue_capacity: int = 2048
    queue_timeout_ms: float = 100.0
    min_remaining_deadline_ms: float = 5.0


@dataclass
class _PendingRequest:
    """A request waiting in the batcher queue."""

    inputs: dict[str, Any]
    context: InferContext
    future: asyncio.Future[dict[str, Any]]
    enqueue_time: float = field(default_factory=time.monotonic)
```

### Step 4：确认测试通过

```bash
uv run pytest tests/test_batcher.py -v
# 期望：2 passed
```

### Step 5：提交

```bash
git add src/nerva/engine/batcher.py tests/test_batcher.py
git commit -m "feat(batcher): add BatchConfig and _PendingRequest"
```

---

## Task 2：`DynamicBatcher` 骨架 + 生命周期

**Files:**
- Modify: `src/nerva/engine/batcher.py`
- Modify: `tests/test_batcher.py`

### Step 1：写失败测试

```python
# 追加到 tests/test_batcher.py

from unittest.mock import AsyncMock
from nerva.backends.base import InferContext
from nerva.engine.batcher import BatchConfig, DynamicBatcher


def _make_ctx(deadline_ms: int = 30000) -> InferContext:
    return InferContext(request_id="test-req", deadline_ms=deadline_ms)


def _make_inner() -> AsyncMock:
    inner = AsyncMock()
    inner.infer = AsyncMock(return_value={"result": "ok"})
    return inner


async def test_batcher_lifecycle() -> None:
    """start() / stop() は例外を出さない。"""
    inner = _make_inner()
    batcher = DynamicBatcher(inner, BatchConfig())
    await batcher.start()
    await batcher.stop()


async def test_batcher_context_manager() -> None:
    inner = _make_inner()
    async with DynamicBatcher(inner, BatchConfig()) as batcher:
        assert batcher is not None
```

### Step 2：确认测试失败

```bash
uv run pytest tests/test_batcher.py::test_batcher_lifecycle -v
# 期望：ERROR (ImportError: cannot import name 'DynamicBatcher')
```

### Step 3：实现 `DynamicBatcher` 骨架

在 `batcher.py` 中追加（`_PendingRequest` 定义之后）：

```python
class DynamicBatcher:
    """Transparent InferableProxy wrapper that batches requests.

    Accumulates infer() calls and dispatches them in batches to the
    inner proxy, applying deadline-aware admission and backpressure.
    """

    def __init__(self, inner: InferableProxy, config: BatchConfig) -> None:
        self._inner = inner
        self._config = config
        self._queue: asyncio.Queue[_PendingRequest] = asyncio.Queue(
            maxsize=config.queue_capacity
        )
        self._loop_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the background batch loop."""
        self._loop_task = asyncio.create_task(self._batch_loop())

    async def stop(self) -> None:
        """Stop the batch loop and drain remaining requests."""
        if self._loop_task is not None:
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task
            self._loop_task = None
        # Drain remaining requests.
        while not self._queue.empty():
            req = self._queue.get_nowait()
            if not req.future.done():
                req.future.set_exception(RuntimeError("batcher stopped"))

    async def __aenter__(self) -> DynamicBatcher:
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()

    async def infer(
        self,
        inputs: dict[str, Any],
        context: InferContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        raise NotImplementedError  # 后续 Task 实现

    async def _batch_loop(self) -> None:
        raise NotImplementedError  # 后续 Task 实现
```

### Step 4：确认测试通过

```bash
uv run pytest tests/test_batcher.py -v
# 期望：4 passed
```

### Step 5：提交

```bash
git add src/nerva/engine/batcher.py tests/test_batcher.py
git commit -m "feat(batcher): add DynamicBatcher skeleton with lifecycle"
```

---

## Task 3：`infer()` — deadline 准入 + 入队 + future

**Files:**
- Modify: `src/nerva/engine/batcher.py`
- Modify: `tests/test_batcher.py`

### Step 1：写失败测试

```python
# 追加到 tests/test_batcher.py
import pytest

async def test_infer_deadline_admission_reject() -> None:
    """剩余 deadline 不足时，infer() 立即拒绝。"""
    inner = _make_inner()
    cfg = BatchConfig(min_remaining_deadline_ms=50.0)
    async with DynamicBatcher(inner, cfg) as batcher:
        ctx = _make_ctx(deadline_ms=10)  # 10ms < 50ms threshold
        with pytest.raises(RuntimeError, match="DEADLINE_EXCEEDED"):
            await batcher.infer({"x": 1}, ctx)
    inner.infer.assert_not_called()


async def test_infer_returns_result() -> None:
    """infer() 成功返回 inner 的结果。"""
    inner = _make_inner()
    inner.infer.return_value = {"out": 42}
    async with DynamicBatcher(inner, BatchConfig(max_batch_size=1)) as batcher:
        ctx = _make_ctx()
        result = await batcher.infer({"x": 1}, ctx)
    assert result == {"out": 42}
```

### Step 2：确认测试失败

```bash
uv run pytest tests/test_batcher.py::test_infer_deadline_admission_reject -v
# 期望：FAILED (NotImplementedError)
```

### Step 3：实现 `infer()` + `_batch_loop` 基础版

将 `infer()` 和 `_batch_loop` 的 `NotImplementedError` 替换为完整实现：

```python
    async def infer(
        self,
        inputs: dict[str, Any],
        context: InferContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        # 1. Deadline admission check.
        if context.deadline_ms < self._config.min_remaining_deadline_ms:
            raise RuntimeError("DEADLINE_EXCEEDED")

        # 2. Enqueue with backpressure timeout.
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        pending = _PendingRequest(inputs=inputs, context=context, future=future)
        try:
            await asyncio.wait_for(
                self._queue.put(pending),
                timeout=self._config.queue_timeout_ms / 1000.0,
            )
        except asyncio.TimeoutError:
            raise RuntimeError("RESOURCE_EXHAUSTED")

        # 3. Wait for batch loop to resolve this request.
        return await future

    async def _batch_loop(self) -> None:
        config = self._config
        while True:
            # Wait for the first request (blocking).
            first = await self._queue.get()
            batch: list[_PendingRequest] = [first]
            batch_deadline = time.monotonic() + config.max_delay_ms / 1000.0

            # Accumulate up to max_batch_size within the time window.
            while len(batch) < config.max_batch_size:
                remaining = batch_deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(
                        self._queue.get(), timeout=remaining
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            # Filter expired requests.
            now = time.monotonic()
            valid: list[_PendingRequest] = []
            for req in batch:
                elapsed_ms = (now - req.enqueue_time) * 1000.0
                remaining_ms = req.context.deadline_ms - elapsed_ms
                if remaining_ms < 0:
                    if not req.future.done():
                        req.future.set_exception(RuntimeError("DEADLINE_EXCEEDED"))
                else:
                    valid.append(req)

            if not valid:
                continue

            # Dispatch valid requests concurrently.
            results = await asyncio.gather(
                *(
                    self._inner.infer(req.inputs, req.context)
                    for req in valid
                ),
                return_exceptions=True,
            )

            # Distribute results back to futures.
            for req, result in zip(valid, results):
                if req.future.done():
                    continue
                if isinstance(result, BaseException):
                    req.future.set_exception(result)
                else:
                    req.future.set_result(result)
```

### Step 4：确认测试通过

```bash
uv run pytest tests/test_batcher.py -v
# 期望：6 passed
```

### Step 5：提交

```bash
git add src/nerva/engine/batcher.py tests/test_batcher.py
git commit -m "feat(batcher): implement infer() and _batch_loop core logic"
```

---

## Task 4：size 触发 + timer 触发测试

**Files:**
- Modify: `tests/test_batcher.py`

### Step 1：写失败测试

```python
# 追加到 tests/test_batcher.py
import asyncio as _asyncio

async def test_size_trigger() -> None:
    """max_batch_size 满时触发，inner 调用次数正确。"""
    call_count = 0

    async def counting_infer(inputs: dict, context: InferContext, **kw: Any) -> dict:
        nonlocal call_count
        call_count += 1
        return {"v": inputs["v"]}

    inner = AsyncMock()
    inner.infer.side_effect = counting_infer

    cfg = BatchConfig(max_batch_size=4, max_delay_ms=1000.0)  # 大 delay，靠 size 触发
    async with DynamicBatcher(inner, cfg) as batcher:
        ctx = _make_ctx()
        tasks = [
            _asyncio.create_task(batcher.infer({"v": i}, ctx))
            for i in range(4)
        ]
        results = await _asyncio.gather(*tasks)

    assert len(results) == 4
    assert call_count == 4  # 4 个请求，4 次 inner 调用（并发）
    assert {r["v"] for r in results} == {0, 1, 2, 3}


async def test_timer_trigger() -> None:
    """少量请求在 max_delay_ms 到期后自动触发，不永久阻塞。"""
    inner = _make_inner()
    inner.infer.return_value = {"done": True}

    cfg = BatchConfig(max_batch_size=100, max_delay_ms=20.0)  # 大 batch，靠 timer
    async with DynamicBatcher(inner, cfg) as batcher:
        ctx = _make_ctx()
        result = await _asyncio.wait_for(
            batcher.infer({"x": 1}, ctx),
            timeout=1.0,  # 最多等 1s，远大于 20ms
        )

    assert result == {"done": True}
```

### Step 2：确认测试通过（实现已在 Task 3 完成）

```bash
uv run pytest tests/test_batcher.py::test_size_trigger tests/test_batcher.py::test_timer_trigger -v
# 期望：2 passed
```

### Step 3：提交

```bash
git add tests/test_batcher.py
git commit -m "test(batcher): add size trigger and timer trigger tests"
```

---

## Task 5：批次内 deadline 过滤测试

**Files:**
- Modify: `tests/test_batcher.py`

### Step 1：写测试

```python
# 追加到 tests/test_batcher.py

async def test_expired_request_filtered_in_batch() -> None:
    """进队后过期的请求在批次触发时被跳过，不调用 inner。"""
    inner = _make_inner()
    inner.infer.return_value = {"ok": True}

    # deadline_ms=1 意味着进队后 1ms 内就过期
    cfg = BatchConfig(max_batch_size=2, max_delay_ms=50.0, min_remaining_deadline_ms=0.0)
    async with DynamicBatcher(inner, cfg) as batcher:
        # 进队后立刻过期的请求
        expired_ctx = _make_ctx(deadline_ms=1)
        valid_ctx = _make_ctx(deadline_ms=30000)

        expired_task = _asyncio.create_task(batcher.infer({"x": "expired"}, expired_ctx))
        valid_task = _asyncio.create_task(batcher.infer({"x": "valid"}, valid_ctx))

        expired_result, valid_result = await _asyncio.gather(
            expired_task, valid_task, return_exceptions=True
        )

    assert isinstance(expired_result, RuntimeError)
    assert "DEADLINE_EXCEEDED" in str(expired_result)
    assert valid_result == {"ok": True}
    # inner 只被调用了一次（expired 请求被跳过）
    assert inner.infer.call_count == 1
```

### Step 2：确认测试通过

```bash
uv run pytest tests/test_batcher.py::test_expired_request_filtered_in_batch -v
# 期望：1 passed
```

### Step 3：提交

```bash
git add tests/test_batcher.py
git commit -m "test(batcher): add deadline filtering in batch test"
```

---

## Task 6：backpressure — 队列满超时

**Files:**
- Modify: `tests/test_batcher.py`

### Step 1：写测试

```python
# 追加到 tests/test_batcher.py

async def test_queue_full_raises_resource_exhausted() -> None:
    """队列满且超过 queue_timeout_ms 后，infer() 抛 RESOURCE_EXHAUSTED。"""
    # 不启动 _batch_loop（不调用 start()），使队列积压
    inner = _make_inner()
    cfg = BatchConfig(queue_capacity=1, queue_timeout_ms=50.0)
    batcher = DynamicBatcher(inner, cfg)
    # 不 start()，_batch_loop 不消费，queue 会满

    ctx = _make_ctx()
    # 第一个请求填满队列（queue_capacity=1）
    fill_task = _asyncio.create_task(batcher.infer({"x": 1}, ctx))
    await _asyncio.sleep(0.01)  # 给 fill_task 时间入队

    # 第二个请求应在 queue_timeout_ms 后抛 RESOURCE_EXHAUSTED
    with pytest.raises(RuntimeError, match="RESOURCE_EXHAUSTED"):
        await _asyncio.wait_for(
            batcher.infer({"x": 2}, ctx),
            timeout=1.0,
        )

    fill_task.cancel()
    with contextlib.suppress(Exception):
        await fill_task

import contextlib as contextlib  # 已在 batcher.py 导入，测试文件需要单独导入
```

> **注意：** 在测试文件顶部添加 `import contextlib`。

### Step 2：确认测试通过

```bash
uv run pytest tests/test_batcher.py::test_queue_full_raises_resource_exhausted -v
# 期望：1 passed
```

### Step 3：提交

```bash
git add tests/test_batcher.py
git commit -m "test(batcher): add backpressure queue timeout test"
```

---

## Task 7：`stop()` drain + inner 异常透传

**Files:**
- Modify: `tests/test_batcher.py`

### Step 1：写测试

```python
# 追加到 tests/test_batcher.py

async def test_stop_drains_pending_requests() -> None:
    """stop() 后，队列中剩余请求的 future 收到异常，不泄漏。"""
    inner = _make_inner()
    cfg = BatchConfig(queue_capacity=10, max_delay_ms=10000.0)  # 大 delay，不自动触发
    batcher = DynamicBatcher(inner, cfg)
    await batcher.start()

    ctx = _make_ctx()
    pending = _asyncio.create_task(batcher.infer({"x": 1}, ctx))
    await _asyncio.sleep(0.02)  # 让请求入队

    await batcher.stop()

    result = await _asyncio.gather(pending, return_exceptions=True)
    assert isinstance(result[0], RuntimeError)
    assert "batcher stopped" in str(result[0])


async def test_inner_exception_propagated() -> None:
    """inner.infer() 抛出的异常透传给调用方 future。"""
    inner = _make_inner()
    inner.infer.side_effect = ValueError("model crash")

    cfg = BatchConfig(max_batch_size=1)
    async with DynamicBatcher(inner, cfg) as batcher:
        ctx = _make_ctx()
        with pytest.raises(ValueError, match="model crash"):
            await batcher.infer({"x": 1}, ctx)
```

### Step 2：确认测试通过

```bash
uv run pytest tests/test_batcher.py::test_stop_drains_pending_requests \
              tests/test_batcher.py::test_inner_exception_propagated -v
# 期望：2 passed
```

### Step 3：提交

```bash
git add tests/test_batcher.py
git commit -m "test(batcher): add stop drain and exception propagation tests"
```

---

## Task 8：并发正确性测试

**Files:**
- Modify: `tests/test_batcher.py`

### Step 1：写测试

```python
# 追加到 tests/test_batcher.py

async def test_concurrent_requests_no_cross_talk() -> None:
    """多个并发 infer() 调用结果与请求一一对应，无串话。"""
    async def echo_infer(inputs: dict, context: InferContext, **kw: Any) -> dict:
        await _asyncio.sleep(0.001)  # 模拟轻量推理
        return {"echo": inputs["id"]}

    inner = AsyncMock()
    inner.infer.side_effect = echo_infer

    cfg = BatchConfig(max_batch_size=16, max_delay_ms=10.0)
    async with DynamicBatcher(inner, cfg) as batcher:
        ctx = _make_ctx()
        n = 50
        tasks = [
            _asyncio.create_task(batcher.infer({"id": i}, ctx))
            for i in range(n)
        ]
        results = await _asyncio.gather(*tasks)

    assert len(results) == n
    returned_ids = {r["echo"] for r in results}
    assert returned_ids == set(range(n))
```

### Step 2：确认测试通过

```bash
uv run pytest tests/test_batcher.py::test_concurrent_requests_no_cross_talk -v
# 期望：1 passed
```

### Step 3：全量 batcher 单元测试

```bash
uv run pytest tests/test_batcher.py -v
# 期望：全部 passed（≥ 11 个测试）
```

### Step 4：提交

```bash
git add tests/test_batcher.py
git commit -m "test(batcher): add concurrent correctness test"
```

---

## Task 9：`model()` + `ModelHandle` 集成 `batch_config`

**Files:**
- Modify: `src/nerva/core/model.py`
- Modify: `tests/test_model.py`（追加测试）

### Step 1：写失败测试

```python
# 追加到 tests/test_model.py
from nerva.engine.batcher import BatchConfig

def test_model_with_batch_config() -> None:
    """model() 接受 batch_config 参数并存入 ModelHandle。"""
    cfg = BatchConfig(max_batch_size=8, max_delay_ms=5.0)
    handle = model("m", DummyModel, batch_config=cfg)
    assert handle.batch_config is cfg
    assert handle.batch_config.max_batch_size == 8


def test_model_without_batch_config() -> None:
    """batch_config 默认为 None，不影响现有行为。"""
    handle = model("m", DummyModel)
    assert handle.batch_config is None
```

> `DummyModel` 已在 `tests/test_model.py` 中定义（继承 `nerva.Model` 的 stub）。如无，参考同文件中现有 fixture。

### Step 2：确认测试失败

```bash
uv run pytest tests/test_model.py::test_model_with_batch_config -v
# 期望：FAILED (TypeError: model() got an unexpected keyword argument 'batch_config')
```

### Step 3：修改 `ModelHandle` 和 `model()`

在 `src/nerva/core/model.py` 中：

**① `ModelHandle` 新增字段**（`frozen=True` 兼容，加在 `options` 之后）：

```python
# 在 ModelHandle dataclass 中追加
from nerva.engine.batcher import BatchConfig  # 注意循环导入风险，用 TYPE_CHECKING

# 文件顶部改为：
from __future__ import annotations
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from nerva.engine.batcher import BatchConfig

@dataclass(frozen=True)
class ModelHandle:
    name: str
    model_class: type[Model]
    backend: str
    device: str
    options: dict[str, Any] = field(default_factory=dict)
    batch_config: BatchConfig | None = None
```

**② `model()` 新增参数**：

```python
def model(
    name: str,
    model_class: type[Model],
    *,
    backend: str = "pytorch",
    device: str = "cpu",
    batch_config: BatchConfig | None = None,
    **options: Any,
) -> ModelHandle:
    ...
    return ModelHandle(
        name=name,
        model_class=model_class,
        backend=backend,
        device=device,
        options=options,
        batch_config=batch_config,
    )
```

### Step 4：确认测试通过

```bash
uv run pytest tests/test_model.py -v
# 期望：全部 passed
uv run mypy
# 期望：no issues
```

### Step 5：提交

```bash
git add src/nerva/core/model.py tests/test_model.py
git commit -m "feat(model): add batch_config to model() and ModelHandle"
```

---

## Task 10：集成测试 — `DynamicBatcher` + 真实 `Worker`

**Files:**
- Create: `tests/test_phase3_e2e.py`

### Step 1：写集成测试

```python
# tests/test_phase3_e2e.py
"""Phase 3 end-to-end tests: DynamicBatcher + real Worker + Executor."""

from __future__ import annotations

from typing import Any

import pytest

from nerva.backends.base import InferContext
from nerva.engine.batcher import BatchConfig, DynamicBatcher
from nerva.engine.executor import Executor
from nerva.core.graph import Edge, Graph, Node
from nerva.worker.manager import WorkerManager
from nerva.worker.proxy import WorkerProxy

# 复用已有 helpers（EchoModel / UpperModel 定义在 tests/helpers.py 中）
from tests.helpers import EchoModel, MODEL_CLASS_PATH_ECHO


@pytest.fixture
async def manager_and_proxy():
    """Spin up one Worker and yield (manager, WorkerProxy)."""
    manager = WorkerManager()
    proxy = await manager.start_worker(
        model_name="echo",
        model_class_path=MODEL_CLASS_PATH_ECHO,
        device="cpu",
    )
    yield manager, proxy
    await manager.shutdown_all()


async def test_batcher_with_real_worker(manager_and_proxy: Any) -> None:
    """DynamicBatcher 包装真实 WorkerProxy，N 个并发请求正确返回结果。"""
    _, proxy = manager_and_proxy
    cfg = BatchConfig(max_batch_size=4, max_delay_ms=20.0)

    import asyncio

    async with DynamicBatcher(proxy, cfg) as batcher:
        ctx = InferContext(request_id="p3-e2e", deadline_ms=30000)
        tasks = [
            asyncio.create_task(batcher.infer({"value": f"msg-{i}"}, ctx))
            for i in range(4)
        ]
        results = await asyncio.gather(*tasks)

    assert len(results) == 4
    for i, result in enumerate(results):
        assert "echo" in result


async def test_executor_with_batcher(manager_and_proxy: Any) -> None:
    """Executor 使用 DynamicBatcher proxy，线性 DAG 端到端正确执行。"""
    _, proxy = manager_and_proxy
    cfg = BatchConfig(max_batch_size=1, max_delay_ms=10.0)

    g = Graph()
    g.add_node(Node(id="echo_1", model_name="echo"))

    ctx = InferContext(request_id="p3-dag", deadline_ms=30000)

    async with DynamicBatcher(proxy, cfg) as batcher:
        executor = Executor(g, {"echo": batcher}, ctx)
        result = await executor.execute({"value": "hello"})

    assert "echo" in result
```

> **注意：** `MODEL_CLASS_PATH_ECHO` 和 `EchoModel` 参考 `tests/helpers.py` 中现有定义。若不存在，参考 `tests/test_phase2_e2e.py` 中的 import 方式。

### Step 2：确认测试通过

```bash
uv run pytest tests/test_phase3_e2e.py -v
# 期望：2 passed
```

### Step 3：提交

```bash
git add tests/test_phase3_e2e.py
git commit -m "test(phase3): add e2e integration tests for DynamicBatcher + Worker"
```

---

## Task 11：静态检查 + 全量回归 + roadmap 更新

### Step 1：ruff + mypy

```bash
uv run ruff check src/ tests/
# 期望：0 errors

uv run mypy
# 期望：no issues found
```

如有 lint 或类型问题，修复后重新提交：

```bash
git add -p
git commit -m "fix(batcher): resolve ruff/mypy issues"
```

### Step 2：全量测试

```bash
uv run pytest tests/ -v
# 期望：所有测试 passed（≥ 170 passed）
```

### Step 3：更新 roadmap

在 `docs/plans/2026-02-25-mvp-roadmap.md` 中，将 Phase 3 条目改为：

```markdown
**验证结果：** ruff 0 errors, mypy 0 issues, <N> tests passed (<T>s)

**状态：** ✅ 已完成 (2026-02-27)

**产出文件：**
```
src/nerva/engine/batcher.py        # DynamicBatcher, BatchConfig, _PendingRequest
src/nerva/core/model.py            # model() + ModelHandle 新增 batch_config（修改）
tests/test_batcher.py              # 单元测试（mock inner proxy）
tests/test_phase3_e2e.py           # 集成测试（真实 Worker）
```
```

### Step 4：更新 Changelog

在 roadmap Changelog 追加一行：

```
| 2026-02-27 | Phase 3 标记为已完成；补充产出文件清单和关键设计决策 |
```

### Step 5：提交

```bash
git add docs/plans/2026-02-25-mvp-roadmap.md
git commit -m "docs: mark Phase 3 complete in roadmap"
```

---

## 完成检查清单

- [ ] `BatchConfig` defaults 与 `mvp-defaults.md` 一致
- [ ] `DynamicBatcher` 实现 `InferableProxy` 协议（mypy 通过）
- [ ] size 触发、timer 触发均有测试
- [ ] deadline 准入（进队前）有测试
- [ ] 批次内过期过滤有测试
- [ ] backpressure RESOURCE_EXHAUSTED 有测试
- [ ] stop() drain 有测试
- [ ] inner 异常透传有测试
- [ ] 并发无串话有测试
- [ ] `model()` + `ModelHandle.batch_config` 有测试
- [ ] 集成测试（真实 Worker）通过
- [ ] ruff 0 errors，mypy 0 issues
- [ ] 全量回归通过
- [ ] roadmap 更新为已完成

---

## Changelog

| 日期 | 变更 |
|---|---|
| 2026-02-27 | 初始版本 |
