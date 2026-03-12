# Nerva 框架开销分析报告

**日期:** 2026-03-12
**环境:** CPU mock，3-stage pipeline（mm_preprocess → mm_mock_llm → mm_postprocess）
**Mock 配置:** `MOCK_TOKEN_LATENCY_MS=0.5`，256 tokens → 128ms 模拟推理延迟
**并发测试:** concurrency = 1, 4, 16

---

## 一、性能基准

无框架开销参考（排除 128ms 模拟推理后）：

| 并发 | e2e p50 | e2e p99 | QPS |
|---|---|---|---|
| 1 | 137.6ms | 157.6ms | 7.1 |
| 4 | 139.6ms | 156.7ms | 28.4 |
| 16 | 138.9ms | 156.8ms | 113.8 |

框架线性扩展正常，无并发退化。

---

## 二、框架各层开销明细（p50）

总 e2e p50 ≈ 138ms，其中：

| 层级 | p50 | p99 | 说明 |
|---|---|---|---|
| `backend_infer` mm_mock_llm | 129ms | 133ms | 已知设计值，非瓶颈 |
| `backend_infer` pre/postprocess | ~0.5ms | ~3ms | 可忽略 |
| `worker_deser_ms`（Worker msgpack 解包 + InferContext 构造） | 0.012ms | 0.051ms | 可忽略 |
| `proxy_serialize_ms`（Proxy msgpack encode inputs，per stage） | 0.005ms | 0.026ms | 可忽略 |
| `proxy_deserialize_ms`（Proxy msgpack decode outputs，per stage） | 0.002ms | 0.009ms | 可忽略 |
| **IPC 纯传输**（ZMQ ipc:// round-trip，per stage） | **~0.75ms** | **~4ms** | 3 stage 合计 p50 ≈ 2.25ms |
| **Executor DAG 调度**（create_task + done_queue，per request） | **0.51ms** | **3.1ms** | 含 _build_node_inputs、_make_node_context 等 |
| **RPC body_read**（`await request.body()`） | **1.56ms** | **13.2ms** | ← **最大单项，高并发尾延迟瓶颈** |
| RPC frame parse + msgpack decode inputs | 0.023ms | 0.035ms | 可忽略 |
| RPC response 序列化（frame encode + msgpack encode） | 0.008ms | 0.026ms | 可忽略 |

**真实框架开销合计（排除 backend.infer）p50 ≈ 8ms**：
- RPC body_read: 1.56ms
- ZMQ IPC 传输（3 stage）: 2.25ms
- Executor 调度: 0.51ms
- 序列化/反序列化（全链路）: ~0.05ms
- 其余（uvicorn 路由、structlog、Prometheus metrics 等）: ~3.6ms

---

## 三、关键发现

### 1. RPC body_read 是 p99 最大单项
`await request.body()` p99 = **13.2ms**，是框架开销中尾延迟最大来源。原因是 uvicorn 的 HTTP body 读取受 asyncio 调度影响，高并发下等待时间放大。

### 2. ZMQ IPC 传输每次 ~0.75ms（p50）
3 个 stage × 0.75ms = ~2.25ms。ZMQ ipc:// 已是本机最快 IPC 方案，可通过减少 stage 数量降低总传输开销，或引入 SHM 绕过序列化（大 payload 已有 SHM 路径）。

### 3. Executor DAG 调度 ~0.5ms（p50）
主要来自 Python 层的 asyncio.create_task、done_queue 处理、_build_node_inputs 字典构建。p99 为 3.1ms 说明有偶发调度抖动。

### 4. 数据序列化/反序列化开销可忽略
msgpack encode/decode 全链路合计 < 0.1ms。Worker 侧反序列化 < 0.02ms。

### 5. timing 异步化修复
原始实现使用 `buffering=1`（行缓冲）在 async 热路径同步写文件，导致高并发下 p99 退化 16×~265×，concurrency=4 时 error_rate 达 33%。已修复为 Queue + background asyncio.to_thread writer，修复后性能恢复正常。

---

## 四、优化方向优先级

| 优先级 | 方向 | 预期收益 |
|---|---|---|
| 高 | 调查 RPC body_read 高 p99（uvicorn HTTP/1.1 vs HTTP/2，或预读 body） | p99 从 13ms → <5ms |
| 中 | ZMQ IPC：大 payload 已有 SHM 路径，小 payload 考虑 inline 优化 | p50 -0.2ms/stage |
| 低 | Executor 调度：预计算 node_inputs 模板，减少 dict 构建 | p50 -0.1ms |
| 低 | 减少 structlog contextvars bind/clear 调用频次 | p99 -0.5ms |

---

## 五、埋点覆盖范围

```
HTTP Request
  │  rpc_body_read_ms          ← await request.body()
  │  rpc_parse_ms              ← frame parse + msgpack decode
  ▼
[Executor.execute()]
  │  executor_scheduler_overhead_ms  ← DAG dispatch (excl. infer)
  │  executor_total_execute_ms
  │
  ├─ [WorkerProxy.infer() × N stages]
  │    proxy_serialize_ms      ← msgpack encode inputs
  │    ipc_round_trip_ms       ← ZMQ send → ACK
  │    ipc_transport_ms        ← ipc_round_trip - worker_deser - backend_infer
  │    proxy_deserialize_ms    ← msgpack decode outputs
  │
  │    [Worker._handle_infer()]
  │      worker_deser_ms       ← msgpack unpack + InferContext
  │      backend_infer_ms      ← backend.infer() 实际执行
  │
  rpc_serialize_ms             ← frame encode + msgpack encode response
```

---

## 六、分析工具

```bash
# 启动带完整埋点的服务器
NERVA_TIMING_LOG_DIR=/tmp/nerva_timing MOCK_TOKEN_LATENCY_MS=0.5 \
  uv run uvicorn examples.mm_vllm_cpu_mock_server:app --host 127.0.0.1 --port 8080

# 运行压测
env -u http_proxy -u https_proxy -u all_proxy \
  uv run python scripts/bench/run_bench.py \
    --target nerva --concurrency-levels 1,4,16 \
    --warmup-seconds 5 --sample-seconds 15 \
    --output-root /tmp/bench-results

# 分析完整埋点
uv run python scripts/bench/analyze_timing_log.py --log-dir /tmp/nerva_timing
```
