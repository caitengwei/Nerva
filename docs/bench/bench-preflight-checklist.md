# 压测前置检查清单

从 2026-03-12 CPU mock e2e 压测中总结，下次跑压测前逐项确认。

---

## 1. 环境变量：代理

**问题**：`all_proxy=socks5://127.0.0.1:6153` 导致 httpx 对 localhost 也走 SOCKS 代理，所有请求立即失败（error_rate=1.0），但 latency 显示 ~17ms（快速失败），容易误判为服务正常。

**症状**：error_rate=1.0，latency 远低于预期（mock sleep 128ms，实测 17ms）。

**修复**：运行压测前清空代理变量：

```bash
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    uv run python scripts/bench/run_bench.py ...
```

> ⚠️ 注意：设置 `no_proxy=127.0.0.1` **不够**，httpx 在 `all_proxy` 存在时不遵守 `no_proxy`，必须 unset。

---

## 2. uv worktree 依赖隔离

**问题**：新建 worktree 后 `uv run` 会在该目录创建独立 `.venv`，`uv sync` 只安装基础包，`httpx`、`msgpack`、`structlog` 等运行时依赖缺失。

**修复**：在 worktree 中首次跑 bench 前执行：

```bash
uv sync --all-extras
# 如有缺失再补：
uv pip install httpx msgpack structlog prometheus-client uvicorn starlette
```

---

## 3. 端口冲突

**问题**：前次压测的服务器进程可能仍在后台运行，新启动会失败或静默绑定到旧进程。

**修复**：启动前先清理：

```bash
lsof -ti :8080 :8002 :8003 :8004 2>/dev/null | xargs kill -9 2>/dev/null
```

---

## 4. Triton 侧 pipeline 等价性

**问题**：`start_triton_server.py --allow-mock` 是单个 Starlette handler，完全绕过 ensemble 和 Python backend，与 Nerva 的 3-worker pipeline 深度不对等，对比无意义。

**正确做法**：

- 使用 `prepare_triton_repo.py --cpu-mock` 生成真实 ensemble + CPU mock mm_infer
- 用真实 `tritonserver`（podman 容器）运行，而非 `--allow-mock`

```bash
uv run python scripts/bench/infra/prepare_triton_repo.py \
    --output /tmp/triton_cpu_full_mock \
    --cpu-mock --mock-token-latency-ms 0.5

podman run --rm \
    -p 8002:8002 -p 8003:8003 -p 8004:8004 \
    -v /tmp/triton_cpu_full_mock:/models \
    nvcr.io/nvidia/tritonserver:24.08-py3 \
    tritonserver --model-repository=/models \
      --http-port=8002 --grpc-port=8003 --metrics-port=8004
```

> `--allow-mock` 仅适合快速冒烟测试（验证 bench 脚本流程），不适合结论性对比。

---

## 5. 服务就绪判断

**问题**：`wait_service_ready.py` 只支持 `--kind vllm|triton`，不支持 Nerva；Triton 容器启动加载模型需要时间（首次可能 30–60s）。

**修复**：Nerva 用 curl 轮询 `/v1/health`，Triton 轮询 `/v2/health/ready`，给 Triton 足够等待时间：

```bash
# Nerva
for i in $(seq 1 30); do
  curl -sf http://127.0.0.1:8080/v1/health | grep -q status && break
  sleep 2
done

# Triton（模型加载最多 2min）
for i in $(seq 1 60); do
  curl -sf http://127.0.0.1:8002/v2/health/ready | grep -q '"ready"' && break
  sleep 2
done
```

---

## 6. 结果合理性校验

跑完第一组数据后，先做以下快速校验再继续：

| 检查项 | 预期 | 异常提示 |
|---|---|---|
| `error_rate` | 0.000 | 非 0 → 先查代理、端口、服务健康 |
| p50 ≈ mock_sleep | 128ms ± 20ms | 远低于 128ms → 请求未到达服务；远高于 → deadline 太短或 Worker 积压 |
| QPS ≈ concurrency / p50 | C=1 → ~7.8，C=16 → ~125 | 远低于理论值 → 服务侧有瓶颈或客户端有阻塞 |

---

## 7. macOS podman 网络

**问题**：macOS 上 podman 运行在 Linux VM 内，`--network host` 指的是 VM 的网络，不是 Mac host。容器内服务无法直接通过 `127.0.0.1` 被 Mac 上的 bench 访问。

**修复**：用端口发布代替 `--network host`：

```bash
podman run --rm -p 8002:8002 -p 8003:8003 -p 8004:8004 ...
```

---

## 8. 并发度选取与稳定性边界（2026-03-17 实测）

**关键发现**：Nerva 和 Triton 都存在饱和阈值，超过后结果剧烈抖动，两轮对比方向甚至可能反转。

| 系统 | 稳定 C 上限 | 原因 |
|------|-------------|------|
| Nerva N-worker | `N × 4`（每 worker ≤4 并发）| asyncio event loop thundering herd，超过后超线性退化 |
| Triton cpu_mock | `instance_count`（默认=20）| 每个 Python instance 同步阻塞，超过后排队 |
| Triton HTTP 层 | 需 `--http-thread-count ≥ C` | 默认 8 线程会成为瓶颈，但增大会加剧 VM CPU 压力 |

**建议对比并发度**：`C = min(N×4, instance_count)` = `min(3×4, 20) = 12`（3-worker Nerva）

实测稳定结果（C=20，warmup=30s，sample=60s）：
- Nerva (3-worker): QPS≈133, p50≈138ms
- Triton (20 inst, http-thread-count=20): QPS≈127, p50≈149ms
- Nerva 领先 7% p50，两轮抖动 <1%

C=60 结果（仅供参考，不稳定）：两轮方向相反，不可作为结论性数据。

**Triton `--http-thread-count` 设置**：设为 `instance_count`（如 20）而不是 C。HTTP 线程数过多会占 VM CPU 反而让所有服务变慢（macOS podman VM 只有 6 个核）。

---

## 快速启动模板

```bash
# 0. 清端口、装依赖
lsof -ti :8080 :8002 :8003 :8004 2>/dev/null | xargs kill -9 2>/dev/null
uv sync --extra dev

# 1. 生成 model repo（仅首次或参数变更时）
uv run python scripts/bench/infra/prepare_triton_repo.py \
    --output /private/tmp/triton_cpu_mock --cpu-mock --mock-token-latency-ms 0.5
# 注意：macOS 上 podman volume mount 需用 /private/tmp（不能用 /tmp 符号链接）

# 2. 启动 Nerva（workers 数量按需调整）
MOCK_TOKEN_LATENCY_MS=0.5 uv run uvicorn \
    examples.mm_vllm_cpu_mock_server:app \
    --host 127.0.0.1 --port 8080 --workers 3 > /tmp/nerva.log 2>&1 &

# 3. 启动 Triton（http-thread-count 设为 instance_count=20，不要过大）
podman run --rm \
    -p 8002:8002 -p 8003:8003 -p 8004:8004 \
    -v /private/tmp/triton_cpu_mock:/models \
    nvcr.io/nvidia/tritonserver:24.08-py3 \
    tritonserver --model-repository=/models \
      --http-port=8002 --grpc-port=8003 --metrics-port=8004 \
      --http-thread-count=20 \
    > /tmp/triton.log 2>&1 &

# 4. 等待就绪（见第 5 节）

# 5. 压测（stable C = workers×4; 注意 unset 代理）
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  uv run python scripts/bench/run_bench.py \
    --target nerva --target triton \
    --concurrency-levels 12,20 \
    --warmup-seconds 30 --sample-seconds 60 \
    --deadline-ms 30000 \
    --output-root /tmp/bench-results

# 6. 校验 error_rate（应全为 0）
grep -rh error_rate /tmp/bench-results/mm_vllm/*/*/*/*/summary.json
```

---

## Nerva 单侧压测（含 timing 埋点）

不含 Triton 对比的最小命令集，适合框架开销分析和优化验证。

```bash
# 1. 装依赖（含 httptools 等 perf extras）
uv sync --extra dev --extra perf

# 2. 清端口 + 启动带 timing 埋点的服务器
lsof -ti :8080 2>/dev/null | xargs kill -9 2>/dev/null
mkdir -p /tmp/nerva_timing
NERVA_TIMING_LOG_DIR=/tmp/nerva_timing MOCK_TOKEN_LATENCY_MS=0.5 \
  uv run uvicorn examples.mm_vllm_cpu_mock_server:app \
    --host 127.0.0.1 --port 8080 &

# 3. 等待就绪
for i in $(seq 1 30); do
  curl -sf http://127.0.0.1:8080/v1/health && break; sleep 2
done

# 4. 压测（unset 代理）
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  uv run python scripts/bench/run_bench.py \
    --target nerva --concurrency-levels 1,4,16 \
    --warmup-seconds 5 --sample-seconds 15

# 5. 分析 timing 明细（各层开销 p50/p95/p99）
uv run python scripts/bench/analyze_timing_log.py --log-dir /tmp/nerva_timing

# 6. 校验 error_rate（应全为 0）
grep error_rate bench-results/mm_vllm/*/*/nerva/*/summary.json
```

### Linux 上验证 uvloop（可选）

macOS 上 uvloop 与 ZMQ asyncio ipc:// 不兼容（见分析报告第七章），Linux 上需单独验证：

```bash
uv pip install uvloop
NERVA_TIMING_LOG_DIR=/tmp/nerva_timing MOCK_TOKEN_LATENCY_MS=0.5 \
  uv run uvicorn examples.mm_vllm_cpu_mock_server:app \
    --host 127.0.0.1 --port 8080 --loop uvloop &
```

### 结果合理性快速校验

| 检查项 | 预期值 | 异常排查 |
|---|---|---|
| `error_rate` | 0 | 非 0 → 查代理变量、端口、服务健康 |
| e2e p50 | 128ms ± 15ms | 远低 → 请求未到达；远高 → Worker 积压 |
| rpc_body_read p50 | < 1ms | > 3ms → asyncio 调度瓶颈 |
| ipc_transport p50/stage | < 1ms | > 2ms → ZMQ 或 Worker 进程调度问题 |
| executor_scheduler p50 | < 1ms | > 2ms → 事件循环过载 |
