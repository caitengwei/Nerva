# Streaming Performance Benchmark Design

**Date:** 2026-03-23
**Branch:** dev_bidir_stream_perf
**Scope:** 双向流功能（PR #22）的性能测试用例

---

## 目标

为已实现的双向流功能（`infer_stream` / `execute_stream` / HTTP streaming）添加性能测试，
覆盖以下维度：

- 各层调度开销（Executor、WorkerProxy IPC）
- 端到端流延迟（TTFT、全流 p50/p95/p99）
- HTTP 层并发压力（逐步升 C 直到出现性能瓶颈）

所有测试在 CPU 上运行（无 GPU 依赖），模型用 `asyncio.sleep` 模拟推理耗时。
测试结果以 JSON 文件保存，不设硬断言。

---

## Mock 模型

在 `tests/helpers.py` 新增：

```python
class BenchStreamingModel(Model):
    """Streaming bench model: yields `count` chunks of `chunk_size` bytes each.

    Options:
        count (int, default 100): number of chunks to yield.
        chunk_size (int, default 1024): bytes per chunk.
        delay_ms (float, default 30): sleep between chunks (simulates inference).
    Output per chunk: {"chunk": i, "payload": bytes of chunk_size}
    """
```

参数默认值：`count=100, chunk_size=1024 (1KB), delay_ms=30`。
chunk 内容为 `b"x" * chunk_size`（固定字节，不引入额外 CPU 开销）。
理论全流耗时：`count × delay_ms = 100 × 30ms = 3000ms`。

---

## 测试文件

`tests/test_stream_bench.py`（新文件，与 `test_dag_bench.py` 并列）

所有 bench class 均标记 `@pytest.mark.slow`。
结果保存到 `bench-results/stream/` 目录。

---

## Bench Cases

### SB1 — Executor `execute_stream` 调度开销

**层：** Executor（纯 mock proxy，无 Worker 进程）
**方法：** 构造一个含 1 个 call node 的 Graph，proxy 是 `AsyncMock`，
`infer_stream` yield 5 个小 chunk（`{"tok": i}`）。
跑 100 次串行 `executor.execute_stream()`，每次计时全流总耗时（含 mock yield），
反映 Executor 调度净开销（μs 级）。

**输出字段：** `iterations, avg_us, std_us, min_us, max_us`

---

### SB2 — WorkerProxy `infer_stream` IPC 延迟一致性

**层：** WorkerProxy + 真实 Worker 子进程
**方法：** 启动一个 `BenchStreamingModel` Worker（通过 `WorkerManager`），
串行发 5 次 `infer_stream` 请求（count=100, chunk_size=1024, delay_ms=30ms）。
每次记录全流耗时（ms），与理论耗时（count × delay_ms = 3000ms）对比，
`overhead_ms = avg_ms - theoretical_ms` 反映 IPC + 序列化净开销。

**输出字段：** `iterations, theoretical_ms, avg_ms, overhead_ms, latency_ms {p50, p90, p95, p99}`

---

### SB3 — E2E 单流延迟（TTFT + 全流）

**层：** WorkerProxy + Executor + 真实 Worker
**方法：** 启动 Worker，套 Executor（`_PipelineExecutor`），串行发 5 次 `execute_stream`。
每次分别记录：
- **TTFT**：`execute_stream()` 开始 → 第一个 chunk 到达的时间（理论 ≈ 30ms）
- **全流延迟**：`execute_stream()` 开始 → 最后一个 chunk 到达的时间（理论 ≈ 3000ms）

模型参数：`count=100, chunk_size=1024, delay_ms=30ms`（贴近真实场景，理论延迟可计算）。

**输出字段：** `iterations, theoretical_ttft_ms, theoretical_full_ms, ttft_ms {avg, p50, p95}, full_stream_ms {avg, p50, p95}`

---

### SB4 — HTTP 并发流压力（自动瓶颈检测）

**层：** HTTP RpcHandler（ASGI）
**方法：** 用 `httpx.AsyncClient(app=asgi_app, base_url="http://test")` 发流请求，
`x-nerva-stream=1`，pipeline 中使用 `BenchStreamingModel`（count=100, delay_ms=30ms）。

C 从 1 开始按指数递增（`[1, 2, 4, 8, 16, 32, 64, 128]`），每轮：
- 发 C 个并发流请求（`asyncio.gather`）
- 重复 3 次取均值
- 计算 p50/p90/p99 全流延迟和 QPS

**停止条件：** `p90 > 3000ms × 1.5 = 4500ms`，记录 `bottleneck_c`。
最大 C = 128（安全上限）。

**输出字段：**
```json
{
  "theoretical_latency_ms": 3000,
  "bottleneck_threshold_ms": 4500,
  "bottleneck_c": <int | null>,
  "results": [
    {"c": 1, "p50_ms": ..., "p90_ms": ..., "p99_ms": ..., "qps": ...},
    ...
  ]
}
```

---

### SB5 — HTTP Full-Duplex 大 Body 并发压力（x-nerva-stream=2）

**层：** HTTP RpcHandler（ASGI），`x-nerva-stream=2`
**目的：** 模拟 full-duplex 使用场景，压测大 body 读取 + 输出流的组合路径；
为未来真正实现 input streaming 时留下基准对比数据。

**与 SB4 的区别：**
- Header：`x-nerva-stream=2`（而非 `1`）
- 输入 payload：64KB（`b"x" * 65536`），模拟大型 input（如图像/音频帧）
- 模型参数：`count=100, chunk_size=1024, delay_ms=30`（与 SB4 相同，理论延迟 3000ms）

**C 递增逻辑：** 与 SB4 完全相同（`[1, 2, 4, 8, 16, 32, 64, 128]`，3 轮取均值，
停止条件 `p90 > 4500ms`，记录 `bottleneck_c`）。

**输出字段：** 与 SB4 相同结构，额外增加 `input_payload_bytes: 65536`。

---

## 结果目录结构

```
bench-results/stream/
  SB1_executor_stream_overhead.json
  SB2_worker_proxy_stream_latency.json
  SB3_e2e_stream_latency.json
  SB4_http_concurrent_streams.json
  SB5_http_fullduplex_large_body.json
```

---

## 依赖约束

- 无 GPU 依赖，全部 CPU 可运行
- 不新增 pyproject 依赖（`httpx` 已在 dev 依赖中）
- Worker 进程使用 `WorkerManager`，测试结束后调用 `shutdown_all()`
- 与现有 `test_dag_bench.py` 风格对齐（`@pytest.mark.slow`、`_percentiles()`、`_env_info()`、`_save_result()`）
