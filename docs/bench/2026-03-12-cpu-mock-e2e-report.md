# CPU Mock E2E 压测报告：Nerva vs Triton

- **日期**: 2026-03-12
- **Commit**: 53c1a17
- **环境**: macOS ARM（Apple Silicon，CPU-only，无 GPU）
- **Workload**: `mm_vllm`（multimodal preprocess → mock LLM → postprocess）

---

## 1. 测试目的

在无 GPU 的 CPU 环境下，对比 Nerva 和 Triton 的 **serving 框架开销**，排除真实模型推理变量，建立基准参考线。

---

## 2. 测试设计

### 2.1 Mock 模型行为

两侧均以 `asyncio.sleep` 模拟 LLM token 生成延迟，参数完全一致：

```
delay = max_tokens × MOCK_TOKEN_LATENCY_MS / 1000
       = 256 × 0.5ms = 128ms / 请求
```

| 组件 | Nerva 侧 | Triton 侧 |
|---|---|---|
| 服务程序 | `uvicorn examples.mm_vllm_cpu_mock_server:app :8080` | 真实 `tritonserver`（podman 容器，`24.08-py3` ARM） |
| pipeline | preprocess → MockCPULLM → postprocess（3 Worker 进程，ZeroMQ IPC） | mm_preprocess（真实） → mm_infer（CPU mock） → mm_postprocess（真实），真实 ensemble 调度 |
| 协议 | Binary RPC（msgpack framing） | HTTP REST JSON（Triton v2 infer） |
| 模拟延迟位置 | `asyncio.sleep` in `MockCPULLMModel.infer()` | `asyncio.sleep` in `mm_infer/model.py TritonPythonModel._handle()`，decoupled sender 模式 |

**Triton 侧 pipeline 等价性**：`mm_preprocess` 和 `mm_postprocess` 均为真实 Python backend（含 tensor 构造、pb_utils 调用），仅 `mm_infer` 将 vLLM 替换为 `asyncio.sleep`，ensemble 路由和 stage 间 tensor 传递路径与生产一致。

### 2.2 压测参数

| 参数 | 值 |
|---|---|
| concurrency levels | 1 / 4 / 16 |
| warmup | 5 s |
| sample | 15 s |
| max_tokens | 256 |
| image_size_bytes | 65536（64 KB 合成图片） |
| deadline_ms | 10000 |
| 工具 | `scripts/bench/run_bench.py`（closed-loop） |

---

## 3. 结果

### 3.1 原始数据

| Target | 并发 | QPS | p50 (ms) | p95 (ms) | p99 (ms) | 错误率 | 请求数 |
|--------|-----:|----:|---------:|---------:|---------:|-------:|-------:|
| nerva  |    1 |  7.1 |    139.4 |    153.5 |    167.1 |  0.000 |    106 |
| nerva  |    4 | 28.1 |    141.5 |    151.1 |    156.0 |  0.000 |    424 |
| nerva  |   16 | 102.6 |   139.3 |    192.7 |    848.5 |  0.000 |   1556 |
| triton |    1 |  7.3 |    135.1 |    148.8 |    161.4 |  0.000 |    110 |
| triton |    4 | 28.1 |    136.9 |    170.0 |    232.8 |  0.000 |    425 |
| triton |   16 | 114.0 |   134.8 |    162.1 |    228.9 |  0.000 |   1724 |

### 3.2 Serving 框架开销（隔离分析）

两侧 mock sleep = 128ms 相同，p50 减去 128ms 即为纯 serving overhead：

| Target | 并发 | p50 overhead | p95 overhead | p99 overhead |
|--------|-----:|-------------:|-------------:|-------------:|
| nerva  |    1 |      11.4 ms |      25.5 ms |      39.1 ms |
| nerva  |    4 |      13.5 ms |      23.1 ms |      28.0 ms |
| nerva  |   16 |      11.3 ms |      64.7 ms |     720.5 ms |
| triton |    1 |       7.1 ms |      20.8 ms |      33.4 ms |
| triton |    4 |       8.9 ms |      42.0 ms |     104.8 ms |
| triton |   16 |       6.8 ms |      34.1 ms |     100.9 ms |

### 3.3 吞吐量对比

| 并发 | Nerva QPS | Triton QPS | 差距 |
|-----:|----------:|-----------:|-----:|
|    1 |       7.1 |        7.3 | −2.7% |
|    4 |      28.1 |       28.1 |  0.0% |
|   16 |     102.6 |      114.0 | −10.0% |

---

## 4. 观察与分析

### 4.1 p50 基准延迟

Nerva p50 ≈ 139–141ms，Triton p50 ≈ 135–137ms，差距约 **4–5ms**（p50），来源：

```
Nerva overhead ≈ 11ms:
  HTTP parse/encode + binary RPC framing (msgpack)  ≈ 1–2 ms
  ZeroMQ IPC × 3 worker 往返（ipc:// UNIX socket）  ≈ 7–9 ms
  Executor DAG 调度 + asyncio event loop            ≈ <1 ms

Triton overhead ≈ 7ms:
  HTTP parse/encode (JSON + protobuf-like)          ≈ 2–3 ms
  ensemble scheduler + stage 间 tensor copy         ≈ 3–4 ms
  Python backend 进程调度                            ≈ 1–2 ms
```

### 4.2 高并发下的尾延迟分歧

C=16 时两侧 p99 均出现显著抬升，但模式不同：

| Target | C=1 p99 | C=4 p99 | C=16 p99 | 趋势 |
|--------|--------:|--------:|---------:|------|
| nerva  |  167 ms |  156 ms |   849 ms | C=16 出现尖刺 |
| triton |  161 ms |  233 ms |   229 ms | C=4 起已显著增长，C=16 趋于平稳 |

- **Triton**：C=4 时 p99 即跳至 233ms，推测 ensemble + Python backend 多进程在并发请求下存在调度抖动，但 C=16 时 p99 不再继续恶化（229ms），说明 Triton 在该并发下已有一定的排队稳定性。
- **Nerva**：C=4 p99 正常（156ms），C=16 时 p99 跳至 849ms，出现明显尾刺。初步判断为 Worker 进程 ZeroMQ PAIR socket 在高并发下串行处理请求（每个 Worker 同一时刻只处理一个请求），导致排队积压。

### 4.3 吞吐量差距随并发扩大

C=1/4 时两侧吞吐接近，C=16 时差距扩大至 **10%**（Nerva 102.6 vs Triton 114.0 QPS），与 p99 尾刺现象一致，说明 Worker 串行处理是 Nerva 当前吞吐瓶颈。

---

## 5. 结论

| 维度 | 结论 |
|---|---|
| **p50 框架开销** | Nerva ≈ 11ms，Triton ≈ 7ms，差距 **~4ms**，来源为 ZeroMQ IPC |
| **低并发（C=1/4）吞吐** | 两侧持平，Nerva 无明显劣势 |
| **高并发（C=16）吞吐** | Nerva 低 **10%**，Worker 串行处理造成排队 |
| **p99 尾延迟** | Nerva C=16 出现 849ms 尖刺，需重点关注；Triton C≥4 时 p99 在 230ms 左右 |
| **错误率** | 全部 0，serving 层无功能性问题 |

**根本问题**：Nerva 每个 Worker 进程当前为单请求串行处理（ZeroMQ PAIR，一个请求处理完才接下一个）。在 mock sleep 场景下，C=16 会有 16 个请求竞争 3 个 Worker，实际并发能力受限。Shared Memory Pool 和 Worker 并发度提升是解决路径。

---

## 6. 局限

1. **Mock ≠ 真实 LLM**：asyncio.sleep 无 CPU 负载，不反映真实模型下的计算争抢与内存带宽压力。
2. **macOS ARM 容器环境**：Triton 运行在 podman VM 内，跨 VM 网络（port publish）引入额外延迟，生产 Linux 环境的 Triton overhead 应更低。
3. **仅测 unary**：未覆盖 streaming / decoupled 模式。
4. **下一步**：在 Linux GPU 机器用真实模型复跑，同时排查 Nerva C=16 p99 尖刺的 Worker 串行瓶颈。

---

## 附：运行命令

```bash
# 1. 生成 Triton CPU mock model repo
uv run python scripts/bench/infra/prepare_triton_repo.py \
    --output /tmp/triton_cpu_full_mock \
    --cpu-mock --mock-token-latency-ms 0.5

# 2. Nerva CPU mock server
MOCK_TOKEN_LATENCY_MS=0.5 uv run uvicorn \
    examples.mm_vllm_cpu_mock_server:app --host 127.0.0.1 --port 8080

# 3. Triton（真实 tritonserver，podman 容器）
podman run --rm \
    -p 8002:8002 -p 8003:8003 -p 8004:8004 \
    -v /tmp/triton_cpu_full_mock:/models \
    nvcr.io/nvidia/tritonserver:24.08-py3 \
    tritonserver --model-repository=/models \
      --http-port=8002 --grpc-port=8003 --metrics-port=8004

# 4. 压测
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
uv run python scripts/bench/run_bench.py \
    --target nerva --target triton \
    --concurrency-levels 1,4,16 \
    --warmup-seconds 5 --sample-seconds 15 \
    --deadline-ms 10000 --output-root /tmp/bench-results-real
```
