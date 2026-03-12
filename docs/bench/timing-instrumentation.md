# Nerva 请求链路计时埋点

## 概述

通过设置环境变量 `NERVA_TIMING_LOG_DIR`，Nerva 会在请求处理过程中记录三个关键计时点，写入 JSON 格式的 per-component 日志文件，供离线分析各阶段耗时分布。

---

## 埋点位置

```
Client HTTP Request
  │
  ▼
[RPC Handler]           ← 现有 Prometheus e2e 计时（不变）
  │
  ▼
[WorkerProxy.infer()]
  │  t_send = perf_counter()         ← IPC send 前
  │  await send(INFER_SUBMIT)
  │  await wait_for(fut)             ← 等待 Worker 回包
  │  ipc_round_trip_ms = now - t_send  ← 包含 ZMQ send + Worker 处理 + ZMQ recv
  │
  ▼  [ZeroMQ PAIR ipc://]
  │
  ▼
[Worker._handle_infer()]
  │  t_recv = perf_counter()         ← 任务开始（IPC 消息已收到）
  │  read_inputs() / Descriptor 解析
  │  t_infer_start = perf_counter()  ← backend.infer() 前
  │  asyncio.to_thread(backend.infer(...))
  │  t_infer_end = perf_counter()    ← backend.infer() 后
  │
  │  worker_deser_ms  = t_infer_start - t_recv    （反序列化 + 调度开销）
  │  backend_infer_ms = t_infer_end  - t_infer_start（模型推理耗时）
```

### 两类日志事件

| 事件 | 写入方 | 关键字段 |
|---|---|---|
| `ipc_timing` | WorkerProxy（主进程） | `ipc_round_trip_ms` |
| `infer_timing` | Worker 子进程 | `worker_deser_ms`, `backend_infer_ms` |

两类事件均包含 `request_id` 和 `model` 字段，可跨文件关联。

---

## 启用方法

```bash
export NERVA_TIMING_LOG_DIR=/tmp/nerva_timing
mkdir -p /tmp/nerva_timing
```

设置后启动 Nerva server，计时日志自动写入该目录，**不设置时零开销**。

### 日志文件命名规则

| 文件 | 来源 | 每进程独立 |
|---|---|---|
| `nerva_proxy_{model_name}.log` | WorkerProxy（主进程，每 model 一个） | 是（不同 model 写不同文件） |
| `nerva_worker_{pid}.log` | Worker 子进程（每个 Worker 一个） | 是（PID 唯一，无写冲突） |

> **多 Worker 无写冲突**：每个 Worker 是独立子进程，写自己的 timing 文件，不共享文件句柄。

---

## 完整压测流程示例

```bash
# 1. 清端口、准备 log 目录
lsof -ti :8080 2>/dev/null | xargs kill -9 2>/dev/null
mkdir -p /tmp/nerva_timing

# 2. 启动 Nerva（带 timing）
NERVA_TIMING_LOG_DIR=/tmp/nerva_timing \
  MOCK_TOKEN_LATENCY_MS=0.5 \
  uv run uvicorn examples.mm_vllm_cpu_mock_server:app \
    --host 127.0.0.1 --port 8080 \
    > /tmp/nerva_main.log 2>&1 &

# 等待就绪
for i in $(seq 1 30); do
  curl -sf http://127.0.0.1:8080/v1/health | grep -q status && break
  sleep 2
done

# 3. 压测（注意 unset 代理，见 bench-preflight-checklist.md）
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  uv run python scripts/bench/run_bench.py \
    --target nerva \
    --concurrency-levels 1,4,16 \
    --warmup-seconds 5 --sample-seconds 15 \
    --deadline-ms 10000 \
    --output-root /tmp/bench-results

# 4. 分析 timing 日志
uv run python scripts/bench/analyze_timing_log.py \
    --log-dir /tmp/nerva_timing
```

---

## 分析脚本输出说明

```
Parsing 4 file(s):
  /tmp/nerva_timing/nerva_proxy_mm_preprocess.log
  /tmp/nerva_timing/nerva_proxy_mm_infer.log
  /tmp/nerva_timing/nerva_worker_12345.log
  /tmp/nerva_timing/nerva_worker_12346.log

Parsed 1556 ipc_timing events, 4668 infer_timing events (1556 unique request_ids)

================================================================================
Overall latency breakdown
================================================================================
  ipc_round_trip_ms (WorkerProxy)     n= 1556  p50=   8.213 ms  p95=  14.312 ms  p99=  21.501 ms  mean=   8.891 ms
  worker_deser_ms   (all models)      n= 4668  p50=   0.412 ms  p95=   0.831 ms  p99=   1.203 ms  mean=   0.435 ms
  backend_infer_ms  (all models)      n= 4668  p50= 128.102 ms  p95= 133.201 ms  p99= 148.712 ms  mean= 129.034 ms

================================================================================
Per-model breakdown
================================================================================

  [mm_infer]
    worker_deser_ms                   n= 1556  p50=   0.389 ms  ...
    backend_infer_ms                  n= 1556  p50= 128.102 ms  ...

  [mm_postprocess]
    ...

================================================================================
Derived estimates (p50)
================================================================================
  ipc_round_trip_ms p50           =   8.213 ms
  backend_infer_ms  p50           = 128.102 ms
  estimated IPC transport overhead =  -119.889 ms  (round_trip - backend_infer, single-stage approximation)
```

### 指标含义

| 指标 | 含义 | 预期（CPU mock，256 token × 0.5ms） |
|---|---|---|
| `ipc_round_trip_ms` | WorkerProxy 视角的单次 Worker 调用总耗时（含 ZMQ 传输 + Worker 处理） | ≈ 130–140ms（含 128ms mock sleep） |
| `worker_deser_ms` | Worker 收到消息→开始推理的准备时间（Descriptor 解析 + InferContext 构造） | < 1ms |
| `backend_infer_ms` | `backend.infer()` 实际执行时间（含 `asyncio.to_thread` 调度） | ≈ 128ms |

> **注意**：`ipc_round_trip_ms` 覆盖的是整个 Worker 处理周期（含推理），不是纯 ZMQ 传输时延。
> 纯 ZMQ 传输时延 ≈ `ipc_round_trip_ms` − `backend_infer_ms` − `worker_deser_ms`，在多阶段 pipeline 中需按阶段分别计算。

---

## 与 Prometheus 指标的关系

| 维度 | Prometheus（`/metrics`） | Timing 日志 |
|---|---|---|
| 覆盖范围 | 全程 e2e（RPC 入口到出口） | per-component（IPC + 反序列化 + 推理） |
| 粒度 | 请求级聚合（histogram bucket） | 逐请求 JSON 行 |
| 适用场景 | 监控、告警、实时 dashboard | 离线瓶颈定位、p99 尖刺溯源 |
| 启用条件 | 始终开启 | 需设 `NERVA_TIMING_LOG_DIR` |

两者结合使用：Prometheus 发现问题（如 p99 升高），timing 日志定位根因（是 IPC 抖动还是 backend 慢）。
