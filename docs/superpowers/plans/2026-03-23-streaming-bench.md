# Streaming Performance Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为双向流功能添加 5 个性能 benchmark（SB1–SB5），覆盖 Executor 调度开销、WorkerProxy IPC 吞吐、E2E 延迟、HTTP 并发压力，全部 CPU 可运行。

**Architecture:** 新建 `tests/test_stream_bench.py`（与 `test_dag_bench.py` 结构对齐）；在 `tests/helpers.py` 添加 `BenchStreamingModel`。SB1 用纯 mock proxy；SB2/SB3 用真实 Worker（WorkerManager）；SB4/SB5 通过 `httpx.ASGITransport` 打到真实 ASGI app。

**Tech Stack:** pytest-asyncio, httpx (ASGITransport), asyncio.gather, WorkerManager, nerva.server.serve._build_pipelines

---

## 文件变更映射

| 操作 | 路径 | 职责 |
|------|------|------|
| Modify | `tests/helpers.py` | 添加 BenchStreamingModel |
| Create | `tests/test_stream_bench.py` | 5 个 bench class (SB1–SB5) |

---

## Task 1: 添加 BenchStreamingModel 到 helpers.py

**Files:**
- Modify: `tests/helpers.py`

- [ ] **Step 1: 在 helpers.py 末尾追加 BenchStreamingModel**

在 `BenchClassifier` 后面添加：

```python
class BenchStreamingModel(Model):
    """Streaming bench model: yields `count` chunks of `chunk_size` bytes each.

    Options:
        count (int, default 100): number of chunks to yield.
        chunk_size (int, default 1024): bytes per chunk.
        delay_ms (float, default 30): sleep between chunks (simulates inference).
    Output per chunk: {"chunk": i, "payload": bytes}
    """

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        chunk_size = int(self._options.get("chunk_size", 1024))
        return {"chunk": 0, "payload": b"x" * chunk_size}

    async def infer_stream(self, inputs: dict[str, Any]):  # type: ignore[override]
        import asyncio

        count = int(self._options.get("count", 100))
        chunk_size = int(self._options.get("chunk_size", 1024))
        delay_ms = float(self._options.get("delay_ms", 30))
        payload = b"x" * chunk_size
        for i in range(count):
            if delay_ms > 0:
                await asyncio.sleep(delay_ms / 1000.0)
            yield {"chunk": i, "payload": payload}
```

- [ ] **Step 2: 验证 lint 通过**

```bash
cd /Users/twcai/workspace/twcai/Nerva/.worktrees/bidir_stream_perf
uv run ruff check tests/helpers.py
```

Expected: no errors.

- [ ] **Step 3: 验证模型基本行为（Python 交互式快速检查）**

```bash
cd /Users/twcai/workspace/twcai/Nerva/.worktrees/bidir_stream_perf
uv run python -c "
import asyncio
from tests.helpers import BenchStreamingModel
m = BenchStreamingModel()
m._options = {'count': 3, 'chunk_size': 8, 'delay_ms': 0}
m.load()
async def run():
    chunks = []
    async for c in m.infer_stream({}):
        chunks.append(c)
    print('chunks:', len(chunks), chunks[0])
asyncio.run(run())
"
```

Expected: `chunks: 3 {'chunk': 0, 'payload': b'xxxxxxxx'}`

- [ ] **Step 4: Commit**

```bash
git add tests/helpers.py
git commit -m "test(helpers): add BenchStreamingModel for streaming benchmarks"
```

---

## Task 2: 创建 test_stream_bench.py 骨架 + 共享工具函数

**Files:**
- Create: `tests/test_stream_bench.py`

- [ ] **Step 1: 创建文件骨架**

```python
"""Streaming performance benchmarks (SB1–SB5).

Run with: uv run pytest tests/test_stream_bench.py -v -m slow
Results saved to bench-results/stream/
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import platform
import subprocess
import time
from pathlib import Path
from statistics import mean, median, stdev
from typing import TYPE_CHECKING, Any

import httpx
import msgpack
import pytest

from nerva import model, trace
from nerva.backends.base import InferContext
from nerva.core.graph import Graph, Node
from nerva.engine.executor import Executor
from nerva.server.app import build_app
from nerva.server.protocol import Frame, FrameType, encode_frame
from nerva.server.serve import _build_pipelines
from nerva.worker.manager import WorkerManager
from tests.helpers import BenchStreamingModel

if TYPE_CHECKING:
    pass

BENCH_DIR = Path("bench-results/stream")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _env_info() -> dict[str, str]:
    sha = "unknown"
    with contextlib.suppress(Exception):
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "commit_sha": sha,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _save_result(name: str, data: dict[str, Any]) -> None:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    path = BENCH_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2))
    print(f"\n[bench] saved → {path}")


def _percentiles(data: list[float]) -> dict[str, float]:
    sorted_data = sorted(data)
    n = len(sorted_data)
    p50 = median(sorted_data)
    p90_idx = min(int(n * 0.90), n - 1)
    p95_idx = min(int(n * 0.95), n - 1)
    p99_idx = min(int(n * 0.99), n - 1)
    return {
        "p50": round(p50, 3),
        "p90": round(sorted_data[p90_idx], 3),
        "p95": round(sorted_data[p95_idx], 3),
        "p99": round(sorted_data[p99_idx], 3),
    }


def _make_request_body(pipeline: str, inputs: dict[str, Any], request_id: int = 1) -> bytes:
    """Build OPEN + DATA + END frame bytes for RPC requests."""
    frames = b""
    frames += encode_frame(Frame(FrameType.OPEN, request_id, 0, msgpack.packb({"pipeline": pipeline})))
    frames += encode_frame(Frame(FrameType.DATA, request_id, 0, msgpack.packb(inputs)))
    frames += encode_frame(Frame(FrameType.END, request_id, 0, b""))
    return frames


def _stream_headers(mode: str = "1", deadline_offset_ms: int = 60000) -> dict[str, str]:
    deadline = int(time.time() * 1000) + deadline_offset_ms
    return {
        "content-type": "application/x-nerva-rpc",
        "x-nerva-deadline-ms": str(deadline),
        "x-nerva-stream": mode,
    }
```

- [ ] **Step 2: 验证 lint**

```bash
uv run ruff check tests/test_stream_bench.py
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add tests/test_stream_bench.py
git commit -m "test(bench): add test_stream_bench.py skeleton and shared helpers"
```

---

## Task 3: 实现 SB1 — Executor execute_stream 调度开销

**Files:**
- Modify: `tests/test_stream_bench.py`

SB1 目标：纯 mock proxy（无 Worker 进程），100 次串行 `execute_stream`，
测 Executor 调度净开销（μs 级）。

- [ ] **Step 1: 在文件末尾追加 SB1 class**

```python
# ---------------------------------------------------------------------------
# SB1: Executor execute_stream scheduling overhead
# ---------------------------------------------------------------------------


class _MockStreamProxy:
    """Minimal InferableStreamProxy with zero-delay chunks."""

    def __init__(self, n_chunks: int = 5) -> None:
        self._n_chunks = n_chunks

    async def infer(
        self, inputs: dict[str, Any], context: InferContext, **kwargs: Any
    ) -> dict[str, Any]:
        return {}

    async def infer_stream(
        self, inputs: dict[str, Any], context: InferContext, **kwargs: Any
    ):  # type: ignore[return]
        for i in range(self._n_chunks):
            yield {"tok": i}


@pytest.mark.slow
class TestSB1ExecutorStreamOverhead:
    """SB1: execute_stream() scheduling overhead — pure mock proxy, no Worker."""

    async def test_executor_stream_overhead(self) -> None:
        g = Graph()
        g.add_node(Node(id="model", model_name="model"))
        proxy = _MockStreamProxy(n_chunks=5)
        proxies: dict[str, Any] = {"model": proxy}

        n_iters = 100
        overhead_us_list: list[float] = []

        for i in range(n_iters):
            ctx = InferContext(request_id=f"sb1-{i}", deadline_ms=30000)
            executor = Executor(g, proxies, ctx)
            t0 = time.perf_counter_ns()
            async for _ in executor.execute_stream({}):
                pass
            elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
            overhead_us_list.append(elapsed_us)

        avg_us = mean(overhead_us_list)
        std_us = stdev(overhead_us_list)

        result = {
            "benchmark": "SB1_executor_stream_overhead",
            "description": "execute_stream() total time with zero-delay mock proxy (5 chunks)",
            "iterations": n_iters,
            "avg_us": round(avg_us, 2),
            "std_us": round(std_us, 2),
            "min_us": round(min(overhead_us_list), 2),
            "max_us": round(max(overhead_us_list), 2),
            "env": _env_info(),
        }
        _save_result("SB1_executor_stream_overhead", result)
        print(f"\n[SB1] execute_stream overhead: avg={avg_us:.1f}us, std={std_us:.1f}us")
```

- [ ] **Step 2: 运行 SB1 验证可通过**

```bash
uv run pytest tests/test_stream_bench.py::TestSB1ExecutorStreamOverhead -v -m slow -s
```

Expected: PASSED, 打印 `[SB1] execute_stream overhead: avg=...us`，生成 `bench-results/stream/SB1_executor_stream_overhead.json`。

- [ ] **Step 3: Commit**

```bash
git add tests/test_stream_bench.py
git commit -m "test(bench): implement SB1 executor execute_stream overhead"
```

---

## Task 4: 实现 SB2 — WorkerProxy infer_stream IPC 吞吐

**Files:**
- Modify: `tests/test_stream_bench.py`

SB2 目标：真实 Worker 进程，串行 20 次 `infer_stream`（count=5, delay_ms=0），
测 IPC chunk 吞吐（chunks/s）和全流延迟。

- [ ] **Step 1: 追加 SB2 class**

```python
# ---------------------------------------------------------------------------
# SB2: WorkerProxy infer_stream IPC throughput
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSB2WorkerProxyStreamThroughput:
    """SB2: WorkerProxy.infer_stream() IPC throughput — real Worker, delay_ms=0."""

    async def test_worker_proxy_stream_throughput(self) -> None:
        handle = model(
            "sb2-bench-stream",
            BenchStreamingModel,
            backend="pytorch",
            device="cpu",
            count=5,
            chunk_size=1024,
            delay_ms=0,
        )
        manager = WorkerManager()
        try:
            proxy = await manager.start_worker(handle)

            n_iters = 20
            latencies_ms: list[float] = []

            # Warmup
            ctx = InferContext(request_id="sb2-warmup", deadline_ms=30000)
            async for _ in proxy.infer_stream({}, ctx):
                pass

            for i in range(n_iters):
                ctx = InferContext(request_id=f"sb2-{i}", deadline_ms=30000)
                t0 = time.perf_counter()
                chunk_count = 0
                async for _ in proxy.infer_stream({}, ctx):
                    chunk_count += 1
                elapsed_ms = (time.perf_counter() - t0) * 1000
                latencies_ms.append(elapsed_ms)

            pcts = _percentiles(latencies_ms)
            n_chunks_per_stream = 5  # matches count=5 model option above
            chunks_per_sec = round(n_chunks_per_stream * 1000 / mean(latencies_ms), 1)

            result = {
                "benchmark": "SB2_worker_proxy_stream_throughput",
                "description": "WorkerProxy.infer_stream() serial, count=5, chunk_size=1024, delay_ms=0",
                "iterations": n_iters,
                "avg_ms": round(mean(latencies_ms), 3),
                "latency_ms": pcts,
                "chunks_per_sec": chunks_per_sec,
                "env": _env_info(),
            }
            _save_result("SB2_worker_proxy_stream_throughput", result)
            print(
                f"\n[SB2] IPC throughput: avg={mean(latencies_ms):.1f}ms, "
                f"p95={pcts['p95']:.1f}ms, {chunks_per_sec} chunks/s"
            )
        finally:
            await manager.shutdown_all()
```

- [ ] **Step 2: 运行 SB2**

```bash
uv run pytest tests/test_stream_bench.py::TestSB2WorkerProxyStreamThroughput -v -m slow -s
```

Expected: PASSED，打印 `[SB2] IPC throughput: avg=...ms`，生成 JSON。

- [ ] **Step 3: Commit**

```bash
git add tests/test_stream_bench.py
git commit -m "test(bench): implement SB2 WorkerProxy infer_stream IPC throughput"
```

---

## Task 5: 实现 SB3 — E2E 单流延迟（TTFT + 全流）

**Files:**
- Modify: `tests/test_stream_bench.py`

SB3 目标：WorkerProxy + Executor（_PipelineExecutor），串行 20 次 `execute_stream`，
分别记录 TTFT 和全流延迟（delay_ms=0，测纯 IPC + executor 开销）。

- [ ] **Step 1: 追加 SB3 class**

```python
# ---------------------------------------------------------------------------
# SB3: E2E single-stream latency (TTFT + full stream)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSB3E2EStreamLatency:
    """SB3: E2E execute_stream() TTFT and full-stream latency — real Worker."""

    async def test_e2e_stream_latency(self) -> None:
        from nerva.server.serve import _PipelineExecutor

        handle = model(
            "sb3-bench-stream",
            BenchStreamingModel,
            backend="pytorch",
            device="cpu",
            count=5,
            chunk_size=1024,
            delay_ms=0,
        )
        graph = trace(lambda inp: handle(inp))
        manager = WorkerManager()
        try:
            executors, _ = await _build_pipelines({"sb3": graph}, manager)
            pipeline_executor = executors["sb3"]

            n_iters = 20
            ttft_ms_list: list[float] = []
            full_ms_list: list[float] = []

            # Warmup
            async for _ in pipeline_executor.execute_stream({}, deadline_ms=30000, request_id="sb3-warmup"):
                pass

            for i in range(n_iters):
                t_start = time.perf_counter()
                t_first: float | None = None
                async for _ in pipeline_executor.execute_stream(
                    {}, deadline_ms=30000, request_id=f"sb3-{i}"
                ):
                    if t_first is None:
                        t_first = time.perf_counter()
                t_end = time.perf_counter()

                ttft_ms_list.append(((t_first or t_end) - t_start) * 1000)
                full_ms_list.append((t_end - t_start) * 1000)

            ttft_pcts = _percentiles(ttft_ms_list)
            full_pcts = _percentiles(full_ms_list)

            result = {
                "benchmark": "SB3_e2e_stream_latency",
                "description": "E2E execute_stream() serial, count=5, chunk_size=1024, delay_ms=0",
                "iterations": n_iters,
                "ttft_ms": {
                    "avg": round(mean(ttft_ms_list), 3),
                    **ttft_pcts,
                },
                "full_stream_ms": {
                    "avg": round(mean(full_ms_list), 3),
                    **full_pcts,
                },
                "env": _env_info(),
            }
            _save_result("SB3_e2e_stream_latency", result)
            print(
                f"\n[SB3] TTFT: avg={mean(ttft_ms_list):.1f}ms p95={ttft_pcts['p95']:.1f}ms | "
                f"Full: avg={mean(full_ms_list):.1f}ms p95={full_pcts['p95']:.1f}ms"
            )
        finally:
            await manager.shutdown_all()
```

- [ ] **Step 2: 运行 SB3**

```bash
uv run pytest tests/test_stream_bench.py::TestSB3E2EStreamLatency -v -m slow -s
```

Expected: PASSED，打印 `[SB3] TTFT: ...`，生成 JSON。

- [ ] **Step 3: Commit**

```bash
git add tests/test_stream_bench.py
git commit -m "test(bench): implement SB3 E2E stream latency (TTFT + full stream)"
```

---

## Task 6: 实现 SB4 — HTTP 并发流压力（自动瓶颈检测）

**Files:**
- Modify: `tests/test_stream_bench.py`

SB4 目标：`x-nerva-stream=1`，C 从 1 指数递增，p90 > 4500ms 时停止，记录瓶颈 C。

- [ ] **Step 1: 追加 SB4 class**

```python
# ---------------------------------------------------------------------------
# SB4: HTTP concurrent streams — output streaming, auto bottleneck detection
# ---------------------------------------------------------------------------

_C_SEQUENCE = [1, 2, 4, 8, 16, 32, 64, 128]
_THEORETICAL_LATENCY_MS = 3000.0   # count=100, delay_ms=30
_BOTTLENECK_THRESHOLD_MS = 4500.0  # 1.5x theoretical
_ROUNDS_PER_C = 3


async def _one_stream_request(
    client: httpx.AsyncClient,
    body: bytes,
    headers: dict[str, str],
    url: str,
) -> float:
    """Issue one streaming POST and return total latency in ms."""
    t0 = time.perf_counter()
    await client.post(url, content=body, headers=headers)
    return (time.perf_counter() - t0) * 1000


@pytest.mark.slow
class TestSB4HttpConcurrentStreams:
    """SB4: x-nerva-stream=1, escalating concurrency until p90 > 4500ms."""

    async def test_http_concurrent_streams(self) -> None:
        handle = model(
            "sb4-bench-stream",
            BenchStreamingModel,
            backend="pytorch",
            device="cpu",
            count=100,
            chunk_size=1024,
            delay_ms=30,
        )
        graph = trace(lambda inp: handle(inp))
        manager = WorkerManager()
        try:
            executors, model_info = await _build_pipelines({"sb4": graph}, manager)
            app = build_app(pipelines=executors, model_info=model_info)
            transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]

            body = _make_request_body("sb4", {"value": "x"})
            results: list[dict[str, Any]] = []
            bottleneck_c: int | None = None

            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver", timeout=120.0
            ) as client:
                # Warmup with C=1
                headers = _stream_headers("1", deadline_offset_ms=120000)
                await _one_stream_request(client, body, headers, "/rpc/sb4")

                for c in _C_SEQUENCE:
                    round_latencies: list[float] = []
                    round_qps_list: list[float] = []
                    headers = _stream_headers("1", deadline_offset_ms=120000)

                    for _ in range(_ROUNDS_PER_C):
                        tasks = [
                            _one_stream_request(client, body, headers, "/rpc/sb4")
                            for _ in range(c)
                        ]
                        t_round = time.perf_counter()
                        latencies = await asyncio.gather(*tasks)
                        wall_clock_s = time.perf_counter() - t_round
                        round_latencies.extend(latencies)
                        round_qps_list.append(c / (wall_clock_s + 1e-9))

                    pcts = _percentiles(round_latencies)
                    qps = round(mean(round_qps_list), 2)

                    entry = {
                        "c": c,
                        "p50_ms": pcts["p50"],
                        "p90_ms": pcts["p90"],
                        "p99_ms": pcts["p99"],
                        "qps": qps,
                    }
                    results.append(entry)
                    print(
                        f"\n[SB4] C={c:3d}: p50={pcts['p50']:.0f}ms "
                        f"p90={pcts['p90']:.0f}ms p99={pcts['p99']:.0f}ms qps={qps}"
                    )

                    if pcts["p90"] > _BOTTLENECK_THRESHOLD_MS:
                        bottleneck_c = c
                        print(f"[SB4] Bottleneck detected at C={c} (p90={pcts['p90']:.0f}ms > {_BOTTLENECK_THRESHOLD_MS:.0f}ms)")
                        break

            result = {
                "benchmark": "SB4_http_concurrent_streams",
                "stream_mode": "1",
                "theoretical_latency_ms": _THEORETICAL_LATENCY_MS,
                "bottleneck_threshold_ms": _BOTTLENECK_THRESHOLD_MS,
                "bottleneck_c": bottleneck_c,
                "rounds_per_c": _ROUNDS_PER_C,
                "results": results,
                "env": _env_info(),
            }
            _save_result("SB4_http_concurrent_streams", result)
        finally:
            await manager.shutdown_all()
```

- [ ] **Step 2: 运行 SB4（注意：会跑较长时间）**

```bash
uv run pytest tests/test_stream_bench.py::TestSB4HttpConcurrentStreams -v -m slow -s
```

Expected: PASSED，逐行打印每个 C 的结果，最终生成 JSON，并打印瓶颈 C。

- [ ] **Step 3: Commit**

```bash
git add tests/test_stream_bench.py
git commit -m "test(bench): implement SB4 HTTP concurrent streaming with auto bottleneck detection"
```

---

## Task 7: 实现 SB5 — HTTP Full-Duplex 大 Body 并发压力

**Files:**
- Modify: `tests/test_stream_bench.py`

SB5 与 SB4 逻辑相同，区别：`x-nerva-stream=2`，input payload 为 64KB。

- [ ] **Step 1: 追加 SB5 class**

```python
# ---------------------------------------------------------------------------
# SB5: HTTP full-duplex large body — x-nerva-stream=2
# ---------------------------------------------------------------------------

_SB5_INPUT_PAYLOAD_BYTES = 65536  # 64 KB


@pytest.mark.slow
class TestSB5HttpFullDuplexLargeBody:
    """SB5: x-nerva-stream=2, 64KB body, escalating concurrency until p90 > 4500ms."""

    async def test_http_fullduplex_large_body(self) -> None:
        handle = model(
            "sb5-bench-stream",
            BenchStreamingModel,
            backend="pytorch",
            device="cpu",
            count=100,
            chunk_size=1024,
            delay_ms=30,
        )
        graph = trace(lambda inp: handle(inp))
        manager = WorkerManager()
        try:
            executors, model_info = await _build_pipelines({"sb5": graph}, manager)
            app = build_app(pipelines=executors, model_info=model_info)
            transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]

            # 64KB input payload embedded in DATA frame
            large_input = {"payload": b"x" * _SB5_INPUT_PAYLOAD_BYTES}
            body = _make_request_body("sb5", large_input)
            results: list[dict[str, Any]] = []
            bottleneck_c: int | None = None

            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver", timeout=120.0
            ) as client:
                # Warmup
                headers = _stream_headers("2", deadline_offset_ms=120000)
                await _one_stream_request(client, body, headers, "/rpc/sb5")

                for c in _C_SEQUENCE:
                    round_latencies: list[float] = []
                    round_qps_list: list[float] = []
                    headers = _stream_headers("2", deadline_offset_ms=120000)

                    for _ in range(_ROUNDS_PER_C):
                        tasks = [
                            _one_stream_request(client, body, headers, "/rpc/sb5")
                            for _ in range(c)
                        ]
                        t_round = time.perf_counter()
                        latencies = await asyncio.gather(*tasks)
                        wall_clock_s = time.perf_counter() - t_round
                        round_latencies.extend(latencies)
                        round_qps_list.append(c / (wall_clock_s + 1e-9))

                    pcts = _percentiles(round_latencies)
                    qps = round(mean(round_qps_list), 2)

                    entry = {
                        "c": c,
                        "p50_ms": pcts["p50"],
                        "p90_ms": pcts["p90"],
                        "p99_ms": pcts["p99"],
                        "qps": qps,
                    }
                    results.append(entry)
                    print(
                        f"\n[SB5] C={c:3d}: p50={pcts['p50']:.0f}ms "
                        f"p90={pcts['p90']:.0f}ms p99={pcts['p99']:.0f}ms qps={qps}"
                    )

                    if pcts["p90"] > _BOTTLENECK_THRESHOLD_MS:
                        bottleneck_c = c
                        print(f"[SB5] Bottleneck at C={c} (p90={pcts['p90']:.0f}ms > {_BOTTLENECK_THRESHOLD_MS:.0f}ms)")
                        break

            result = {
                "benchmark": "SB5_http_fullduplex_large_body",
                "stream_mode": "2",
                "input_payload_bytes": _SB5_INPUT_PAYLOAD_BYTES,
                "theoretical_latency_ms": _THEORETICAL_LATENCY_MS,
                "bottleneck_threshold_ms": _BOTTLENECK_THRESHOLD_MS,
                "bottleneck_c": bottleneck_c,
                "rounds_per_c": _ROUNDS_PER_C,
                "results": results,
                "env": _env_info(),
            }
            _save_result("SB5_http_fullduplex_large_body", result)
        finally:
            await manager.shutdown_all()
```

- [ ] **Step 2: 运行 SB5**

```bash
uv run pytest tests/test_stream_bench.py::TestSB5HttpFullDuplexLargeBody -v -m slow -s
```

Expected: PASSED，逐行打印每个 C 结果，生成 JSON。

- [ ] **Step 3: Commit**

```bash
git add tests/test_stream_bench.py
git commit -m "test(bench): implement SB5 HTTP full-duplex large body benchmark"
```

---

## Task 8: 全套验证 + lint + type check

**Files:** (no changes)

- [ ] **Step 1: 运行全套 slow bench 确认可通过**

```bash
uv run pytest tests/test_stream_bench.py -v -m slow -s
```

Expected: 5 tests PASSED (SB1–SB5)，`bench-results/stream/` 下生成 5 个 JSON 文件。

- [ ] **Step 2: 验证 lint**

```bash
uv run ruff check tests/helpers.py tests/test_stream_bench.py
```

Expected: no errors.

- [ ] **Step 3: 验证 type check（允许有 ignore 注释，不新增 error）**

```bash
uv run mypy tests/test_stream_bench.py tests/helpers.py
```

Expected: no new errors beyond existing ignores.

- [ ] **Step 4: 确认非 slow 测试不受影响**

```bash
uv run pytest tests/ -v --ignore=tests/test_stream_bench.py -x -q 2>&1 | tail -5
```

Expected: all existing tests still pass.

- [ ] **Step 5: Final commit**

```bash
git add .
git commit -m "test(bench): complete streaming benchmark suite SB1-SB5"
```
