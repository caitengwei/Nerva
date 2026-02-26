# Phase 2 DAG Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现 4 个 benchmark 用例（B1-B4），量化 Phase 2 DAG 编排层的 trace 开销、executor 调度开销、并行效率和端到端延迟。

**Architecture:** 在 `tests/helpers.py` 新增 4 个 benchmark model（BenchImageEncoder / BenchTextEncoder / BenchFusionModel / BenchClassifier），在 `tests/test_phase2_bench.py` 实现 B1-B4 用例。模型通过 `options` 接收 `dim` 和 `delay_ms` 参数，用 `asyncio.sleep` 模拟计算，构造 realistic 大小的 float list 输出。JSON 结果输出到 `bench-results/phase2/`。

**Tech Stack:** Python 3.11+, pytest, pytest-asyncio, asyncio, json, time, statistics

---

### Task 1: 添加 bench-results 到 .gitignore

**Files:**
- Modify: `.gitignore`

**Step 1: 在 .gitignore 末尾添加 bench-results 目录**

在文件末尾添加：

```
# Benchmark results
bench-results/
```

**Step 2: 创建产物目录**

Run: `mkdir -p bench-results/phase2`

**Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore bench-results directory"
```

---

### Task 2: 实现 4 个 Benchmark Model

**Files:**
- Modify: `tests/helpers.py`

**Step 1: 写 BenchImageEncoder / BenchTextEncoder / BenchFusionModel / BenchClassifier 的测试**

在 `tests/test_phase2_bench.py` 中先写模型单元测试（后续 Task 3 会扩展此文件）：

```python
"""Phase 2 DAG Benchmark tests."""

from __future__ import annotations

import pytest

from tests.helpers import (
    BenchClassifier,
    BenchFusionModel,
    BenchImageEncoder,
    BenchTextEncoder,
)


class TestBenchModels:
    """Verify benchmark models produce correct output shapes."""

    async def test_image_encoder_output_shape(self) -> None:
        m = BenchImageEncoder()
        m._options = {"dim": 768, "delay_ms": 0}
        m.load()
        result = await m.infer({"image_bytes": b"\x00" * 4096})
        assert "features" in result
        assert len(result["features"]) == 768
        assert all(isinstance(v, float) for v in result["features"])

    async def test_text_encoder_output_shape(self) -> None:
        m = BenchTextEncoder()
        m._options = {"dim": 768, "delay_ms": 0}
        m.load()
        result = await m.infer({"text": "hello world"})
        assert "features" in result
        assert len(result["features"]) == 768

    async def test_fusion_model_output_shape(self) -> None:
        m = BenchFusionModel()
        m._options = {"dim": 512, "delay_ms": 0}
        m.load()
        result = await m.infer({
            "img_features": [1.0] * 768,
            "txt_features": [2.0] * 768,
        })
        assert "fused_features" in result
        assert len(result["fused_features"]) == 512

    async def test_classifier_output(self) -> None:
        m = BenchClassifier()
        m._options = {"delay_ms": 0}
        m.load()
        result = await m.infer({"fused_features": [1.0] * 512})
        assert "label" in result
        assert "score" in result
        assert isinstance(result["label"], str)
        assert isinstance(result["score"], float)
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_phase2_bench.py::TestBenchModels -v`
Expected: FAIL（ImportError — BenchImageEncoder 等不存在）

**Step 3: 在 helpers.py 实现 4 个 benchmark model**

在 `tests/helpers.py` 末尾追加：

```python
class BenchImageEncoder(Model):
    """Benchmark model: simulates image encoding.

    Options: dim (int), delay_ms (float).
    Input: {"image_bytes": bytes}
    Output: {"features": [float] * dim}
    """

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        dim = int(self._options.get("dim", 768))
        delay_ms = float(self._options.get("delay_ms", 0))
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)
        return {"features": [0.1] * dim}


class BenchTextEncoder(Model):
    """Benchmark model: simulates text encoding.

    Options: dim (int), delay_ms (float).
    Input: {"text": str}
    Output: {"features": [float] * dim}
    """

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        dim = int(self._options.get("dim", 768))
        delay_ms = float(self._options.get("delay_ms", 0))
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)
        return {"features": [0.2] * dim}


class BenchFusionModel(Model):
    """Benchmark model: simulates multimodal fusion.

    Options: dim (int), delay_ms (float).
    Input: {"img_features": list, "txt_features": list}
    Output: {"fused_features": [float] * dim}
    """

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        dim = int(self._options.get("dim", 512))
        delay_ms = float(self._options.get("delay_ms", 0))
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)
        return {"fused_features": [0.3] * dim}


class BenchClassifier(Model):
    """Benchmark model: simulates classification head.

    Options: delay_ms (float).
    Input: {"fused_features": list}
    Output: {"label": str, "score": float}
    """

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        delay_ms = float(self._options.get("delay_ms", 0))
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)
        return {"label": "cat", "score": 0.95}
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_phase2_bench.py::TestBenchModels -v`
Expected: 4 passed

**Step 5: Lint 检查**

Run: `uv run ruff check tests/helpers.py tests/test_phase2_bench.py`

**Step 6: Commit**

```bash
git add tests/helpers.py tests/test_phase2_bench.py
git commit -m "feat(bench): add 4 benchmark models for Phase 2 DAG"
```

---

### Task 3: B1 — trace() 构图开销 benchmark

**Files:**
- Modify: `tests/test_phase2_bench.py`

**Step 1: 实现 B1 测试**

在 `tests/test_phase2_bench.py` 追加：

```python
import json
import platform
import subprocess
import time
from pathlib import Path
from statistics import mean, stdev

from nerva import model, trace
from nerva.core.primitives import parallel

BENCH_DIR = Path("bench-results/phase2")


def _env_info() -> dict[str, str]:
    """Collect environment metadata for benchmark results."""
    sha = "unknown"
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        pass
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "commit_sha": sha,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _save_result(name: str, data: dict) -> None:
    """Save benchmark result as JSON."""
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    path = BENCH_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2))


@pytest.mark.slow
class TestB1TraceOverhead:
    """B1: trace() 构图耗时 — 1000 次取平均，纯 CPU 不启动 Worker。"""

    def test_trace_overhead(self) -> None:
        from tests.helpers import (
            BenchClassifier,
            BenchFusionModel,
            BenchImageEncoder,
            BenchTextEncoder,
        )

        img_enc = model("img_enc", BenchImageEncoder, backend="pytorch", device="cpu")
        txt_enc = model("txt_enc", BenchTextEncoder, backend="pytorch", device="cpu")
        fusion = model("fusion", BenchFusionModel, backend="pytorch", device="cpu")
        classifier = model("classifier", BenchClassifier, backend="pytorch", device="cpu")

        def pipeline(x: object) -> object:
            img_out, txt_out = parallel(
                lambda: img_enc(x),
                lambda: txt_enc(x),
            )
            fused = fusion({"img_features": img_out["features"], "txt_features": txt_out["features"]})
            return classifier({"fused_features": fused["fused_features"]})

        n_iters = 1000
        times_us: list[float] = []

        for _ in range(n_iters):
            t0 = time.perf_counter_ns()
            trace(pipeline)
            t1 = time.perf_counter_ns()
            times_us.append((t1 - t0) / 1000.0)

        avg_us = mean(times_us)
        std_us = stdev(times_us)

        result = {
            "benchmark": "B1_trace_overhead",
            "iterations": n_iters,
            "avg_us": round(avg_us, 2),
            "std_us": round(std_us, 2),
            "min_us": round(min(times_us), 2),
            "max_us": round(max(times_us), 2),
            "env": _env_info(),
        }
        _save_result("B1_trace_overhead", result)

        print(f"\n[B1] trace() overhead: avg={avg_us:.1f}μs, std={std_us:.1f}μs")

        # Sanity gate: < 100μs per trace
        assert avg_us < 100, f"trace() too slow: {avg_us:.1f}μs > 100μs"
```

**Step 2: 运行测试**

Run: `uv run pytest tests/test_phase2_bench.py::TestB1TraceOverhead -v -s`
Expected: PASS，终端打印 trace overhead

**Step 3: Commit**

```bash
git add tests/test_phase2_bench.py
git commit -m "feat(bench): B1 trace overhead benchmark"
```

---

### Task 4: B2 — executor 调度开销 benchmark

**Files:**
- Modify: `tests/test_phase2_bench.py`

**Step 1: 实现 B2 测试**

在 `tests/test_phase2_bench.py` 追加：

```python
from nerva.backends.base import InferContext
from nerva.core.graph import Edge, Graph, Node
from nerva.engine.executor import Executor


@pytest.mark.slow
class TestB2ExecutorOverhead:
    """B2: executor 调度层开销 — delay=0 的模型，executor 总耗时 - 节点 infer 总耗时。"""

    async def test_executor_overhead(self) -> None:
        from unittest.mock import AsyncMock

        # Track per-node infer time via mock.
        total_infer_ns = 0

        async def timed_infer(inputs, context, **kwargs):
            nonlocal total_infer_ns
            t0 = time.perf_counter_ns()
            # Simulate zero-cost inference — just return dummy data.
            result = {"features": [0.1] * 768}
            total_infer_ns += time.perf_counter_ns() - t0
            return result

        # Build the 4-node multimodal DAG.
        g = Graph()
        g.add_node(Node(id="img_enc", model_name="img_enc"))
        g.add_node(Node(id="txt_enc", model_name="txt_enc"))
        g.add_node(Node(id="fusion", model_name="fusion"))
        g.add_node(Node(id="classifier", model_name="classifier"))

        # img_enc + txt_enc → fusion → classifier
        g.add_edge(Edge(src="img_enc", dst="fusion", src_field_path=("features",), dst_input_key="img_features"))
        g.add_edge(Edge(src="txt_enc", dst="fusion", src_field_path=("features",), dst_input_key="txt_features"))
        g.add_edge(Edge(src="fusion", dst="classifier", src_field_path=("fused_features",), dst_input_key="fused_features"))

        n_iters = 100
        overhead_us_list: list[float] = []

        for _ in range(n_iters):
            total_infer_ns = 0

            mock_proxy = AsyncMock()
            mock_proxy.infer = AsyncMock(side_effect=timed_infer)

            proxies = {
                "img_enc": mock_proxy,
                "txt_enc": mock_proxy,
                "fusion": mock_proxy,
                "classifier": mock_proxy,
            }
            ctx = InferContext(request_id="bench-b2", deadline_ms=30000)
            executor = Executor(g, proxies, ctx)

            t0 = time.perf_counter_ns()
            await executor.execute({"image_bytes": b"\x00", "text": "hello"})
            total_ns = time.perf_counter_ns() - t0

            overhead_ns = total_ns - total_infer_ns
            overhead_us_list.append(overhead_ns / 1000.0)

        avg_us = mean(overhead_us_list)
        std_us = stdev(overhead_us_list)

        result = {
            "benchmark": "B2_executor_overhead",
            "iterations": n_iters,
            "avg_us": round(avg_us, 2),
            "std_us": round(std_us, 2),
            "min_us": round(min(overhead_us_list), 2),
            "max_us": round(max(overhead_us_list), 2),
            "env": _env_info(),
        }
        _save_result("B2_executor_overhead", result)

        print(f"\n[B2] executor overhead: avg={avg_us:.1f}μs, std={std_us:.1f}μs")

        # Sanity gate: < 5ms (5000μs)
        assert avg_us < 5000, f"executor overhead too high: {avg_us:.1f}μs > 5000μs"
```

**Step 2: 运行测试**

Run: `uv run pytest tests/test_phase2_bench.py::TestB2ExecutorOverhead -v -s`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_phase2_bench.py
git commit -m "feat(bench): B2 executor overhead benchmark"
```

---

### Task 5: B3 — parallel speedup benchmark

**Files:**
- Modify: `tests/test_phase2_bench.py`

**Step 1: 实现 B3 测试**

在 `tests/test_phase2_bench.py` 追加：

```python
@pytest.mark.slow
class TestB3ParallelSpeedup:
    """B3: parallel 并发增益 — delay=10ms 的模型，parallel vs sequential 耗时比。"""

    async def test_parallel_speedup(self) -> None:
        from unittest.mock import AsyncMock

        delay_s = 0.01  # 10ms

        async def delayed_infer(inputs, context, **kwargs):
            import asyncio
            await asyncio.sleep(delay_s)
            return {"features": [0.1] * 768}

        # -- Sequential: img_enc → txt_enc (linear chain) --
        g_seq = Graph()
        g_seq.add_node(Node(id="img_enc", model_name="img_enc"))
        g_seq.add_node(Node(id="txt_enc", model_name="txt_enc"))
        g_seq.add_edge(Edge(src="img_enc", dst="txt_enc"))

        # -- Parallel: img_enc || txt_enc --
        branch_img = Graph()
        branch_img.add_node(Node(id="img_enc", model_name="img_enc"))
        branch_txt = Graph()
        branch_txt.add_node(Node(id="txt_enc", model_name="txt_enc"))

        g_par = Graph()
        par_node = Node(id="par_1", model_name="parallel", node_type="parallel", branches=[branch_img, branch_txt])
        g_par.add_node(par_node)

        n_iters = 50
        seq_times: list[float] = []
        par_times: list[float] = []

        for _ in range(n_iters):
            mock_proxy = AsyncMock()
            mock_proxy.infer = AsyncMock(side_effect=delayed_infer)
            proxies = {"img_enc": mock_proxy, "txt_enc": mock_proxy}

            ctx = InferContext(request_id="bench-b3-seq", deadline_ms=30000)
            executor = Executor(g_seq, proxies, ctx)
            t0 = time.perf_counter()
            await executor.execute({"image_bytes": b"\x00"})
            seq_times.append(time.perf_counter() - t0)

        for _ in range(n_iters):
            mock_proxy = AsyncMock()
            mock_proxy.infer = AsyncMock(side_effect=delayed_infer)
            proxies = {"img_enc": mock_proxy, "txt_enc": mock_proxy}

            ctx = InferContext(request_id="bench-b3-par", deadline_ms=30000)
            executor = Executor(g_par, proxies, ctx)
            t0 = time.perf_counter()
            await executor.execute({"image_bytes": b"\x00"})
            par_times.append(time.perf_counter() - t0)

        avg_seq = mean(seq_times)
        avg_par = mean(par_times)
        speedup = avg_seq / avg_par

        result = {
            "benchmark": "B3_parallel_speedup",
            "iterations": n_iters,
            "delay_ms": delay_s * 1000,
            "avg_sequential_ms": round(avg_seq * 1000, 2),
            "avg_parallel_ms": round(avg_par * 1000, 2),
            "speedup_ratio": round(speedup, 2),
            "env": _env_info(),
        }
        _save_result("B3_parallel_speedup", result)

        print(f"\n[B3] parallel speedup: seq={avg_seq*1000:.1f}ms, par={avg_par*1000:.1f}ms, speedup={speedup:.2f}x")

        # Sanity gate: > 1.5x speedup
        assert speedup > 1.5, f"parallel speedup too low: {speedup:.2f}x < 1.5x"
```

**Step 2: 运行测试**

Run: `uv run pytest tests/test_phase2_bench.py::TestB3ParallelSpeedup -v -s`
Expected: PASS（speedup 约 1.8-2.0x）

**Step 3: Commit**

```bash
git add tests/test_phase2_bench.py
git commit -m "feat(bench): B3 parallel speedup benchmark"
```

---

### Task 6: B4 — 端到端 pipeline benchmark

**Files:**
- Modify: `tests/test_phase2_bench.py`

**Step 1: 实现 B4 测试**

B4 需要真实 Worker 进程，使用 WorkerManager 启动 4 个 worker。分 Small / Large 两档 payload。

在 `tests/test_phase2_bench.py` 追加：

```python
from statistics import median, quantiles

from nerva.worker.manager import WorkerManager


def _percentiles(data: list[float]) -> dict[str, float]:
    """Compute p50/p95/p99 from a list of values."""
    sorted_data = sorted(data)
    n = len(sorted_data)
    p50 = median(sorted_data)
    p95_idx = min(int(n * 0.95), n - 1)
    p99_idx = min(int(n * 0.99), n - 1)
    return {
        "p50": round(p50, 3),
        "p95": round(sorted_data[p95_idx], 3),
        "p99": round(sorted_data[p99_idx], 3),
    }


@pytest.mark.slow
class TestB4E2EPipeline:
    """B4: 端到端 pipeline — Small + Large 两档，各 100 次。"""

    async def _run_pipeline(
        self,
        *,
        dim: int,
        delay_ms: float,
        image_bytes_size: int,
        text: str,
        n_iters: int,
    ) -> list[float]:
        """Run the full 4-model DAG pipeline and return latencies in ms."""
        from tests.helpers import (
            BenchClassifier,
            BenchFusionModel,
            BenchImageEncoder,
            BenchTextEncoder,
        )

        h_img = model("img_enc", BenchImageEncoder, backend="pytorch", device="cpu", dim=dim, delay_ms=delay_ms)
        h_txt = model("txt_enc", BenchTextEncoder, backend="pytorch", device="cpu", dim=dim, delay_ms=delay_ms)
        h_fusion = model("fusion", BenchFusionModel, backend="pytorch", device="cpu", dim=dim, delay_ms=delay_ms)
        h_cls = model("classifier", BenchClassifier, backend="pytorch", device="cpu", delay_ms=delay_ms)

        manager = WorkerManager()
        try:
            p_img = await manager.start_worker(h_img)
            p_txt = await manager.start_worker(h_txt)
            p_fusion = await manager.start_worker(h_fusion)
            p_cls = await manager.start_worker(h_cls)

            # Build DAG: parallel(img, txt) → fusion → classifier
            branch_img = Graph()
            branch_img.add_node(Node(id="img_enc", model_name="img_enc"))
            branch_txt = Graph()
            branch_txt.add_node(Node(id="txt_enc", model_name="txt_enc"))

            g = Graph()
            par_node = Node(id="par_1", model_name="parallel", node_type="parallel", branches=[branch_img, branch_txt])
            g.add_node(par_node)
            g.add_node(Node(id="fusion", model_name="fusion"))
            g.add_node(Node(id="classifier", model_name="classifier"))

            g.add_edge(Edge(src="par_1", dst="fusion", src_field_path=("0", "features"), dst_input_key="img_features"))
            g.add_edge(Edge(src="par_1", dst="fusion", src_field_path=("1", "features"), dst_input_key="txt_features"))
            g.add_edge(Edge(src="fusion", dst="classifier", src_field_path=("fused_features",), dst_input_key="fused_features"))

            proxies = {
                "img_enc": p_img,
                "txt_enc": p_txt,
                "fusion": p_fusion,
                "classifier": p_cls,
            }

            pipeline_input = {
                "image_bytes": b"\x00" * image_bytes_size,
                "text": text,
            }

            # Warmup
            ctx = InferContext(request_id="warmup", deadline_ms=30000)
            executor = Executor(g, proxies, ctx)
            await executor.execute(pipeline_input)

            latencies_ms: list[float] = []
            for i in range(n_iters):
                ctx = InferContext(request_id=f"b4-{i}", deadline_ms=30000)
                executor = Executor(g, proxies, ctx)
                t0 = time.perf_counter()
                result = await executor.execute(pipeline_input)
                latencies_ms.append((time.perf_counter() - t0) * 1000)

                # Verify correctness.
                assert "label" in result
                assert "score" in result

            return latencies_ms
        finally:
            await manager.shutdown_all()

    async def test_e2e_small_payload(self) -> None:
        """Small payload: 4KB image, 100B text, dim=768 (~3KB features)."""
        latencies = await self._run_pipeline(
            dim=768,
            delay_ms=0,
            image_bytes_size=4096,
            text="a" * 100,
            n_iters=100,
        )

        pcts = _percentiles(latencies)
        result = {
            "benchmark": "B4_e2e_small",
            "iterations": 100,
            "payload": "small",
            "dim": 768,
            "image_bytes_size": 4096,
            "latency_ms": pcts,
            "avg_ms": round(mean(latencies), 3),
            "env": _env_info(),
        }
        _save_result("B4_e2e_small", result)

        print(f"\n[B4-small] p50={pcts['p50']:.1f}ms, p95={pcts['p95']:.1f}ms, p99={pcts['p99']:.1f}ms")

        # Sanity gate: 0% error rate (already verified in loop).

    async def test_e2e_large_payload(self) -> None:
        """Large payload: 256KB image, 1KB text, dim=1280 (~5KB features)."""
        latencies = await self._run_pipeline(
            dim=1280,
            delay_ms=0,
            image_bytes_size=256 * 1024,
            text="a" * 1024,
            n_iters=100,
        )

        pcts = _percentiles(latencies)
        result = {
            "benchmark": "B4_e2e_large",
            "iterations": 100,
            "payload": "large",
            "dim": 1280,
            "image_bytes_size": 256 * 1024,
            "latency_ms": pcts,
            "avg_ms": round(mean(latencies), 3),
            "env": _env_info(),
        }
        _save_result("B4_e2e_large", result)

        print(f"\n[B4-large] p50={pcts['p50']:.1f}ms, p95={pcts['p95']:.1f}ms, p99={pcts['p99']:.1f}ms")
```

**Step 2: 运行测试**

Run: `uv run pytest tests/test_phase2_bench.py::TestB4E2EPipeline -v -s`
Expected: PASS，终端打印 p50/p95/p99

**Step 3: Commit**

```bash
git add tests/test_phase2_bench.py
git commit -m "feat(bench): B4 end-to-end pipeline benchmark (small + large)"
```

---

### Task 7: 全量验证 + lint + mypy

**Files:**
- 可能修改: `tests/test_phase2_bench.py`（lint 修复）

**Step 1: Ruff lint**

Run: `uv run ruff check tests/test_phase2_bench.py tests/helpers.py`
如有错误，修复。

**Step 2: Mypy**

Run: `uv run mypy`
如有错误，修复。

**Step 3: 全量回归测试（不含 slow）**

Run: `uv run pytest tests/ -v --ignore=tests/test_phase2_bench.py -q`
Expected: 140 passed

**Step 4: 运行 benchmark 全套**

Run: `uv run pytest tests/test_phase2_bench.py -v -s -m slow`
Expected: B1-B4 全部 PASS

**Step 5: 验证 JSON 产物**

Run: `ls bench-results/phase2/`
Expected: `B1_trace_overhead.json`, `B2_executor_overhead.json`, `B3_parallel_speedup.json`, `B4_e2e_small.json`, `B4_e2e_large.json`

**Step 6: Commit**

```bash
git add tests/test_phase2_bench.py tests/helpers.py
git commit -m "feat(bench): Phase 2 DAG benchmark suite complete"
```

---

## 注意事项

- `Model._options` 是通过 Backend 在 `load_model` 时注入的。在 benchmark model 的单元测试中，直接设置 `m._options = {...}` 来模拟。在 e2e 测试（B4）中，`model()` 的 `**options` 会通过 Worker 正常传递。
- B1-B3 不启动 Worker 进程，使用 mock 或直接调用，速度快。
- B4 启动真实 Worker 进程，较慢，标记 `@pytest.mark.slow`。
- 所有 benchmark class 都标记 `@pytest.mark.slow`，普通测试运行不会执行这些。
