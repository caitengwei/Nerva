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

## 快速启动模板

```bash
# 0. 清端口、装依赖
lsof -ti :8080 :8002 :8003 :8004 2>/dev/null | xargs kill -9 2>/dev/null
uv sync --all-extras

# 1. 生成 model repo（仅首次或参数变更时）
uv run python scripts/bench/infra/prepare_triton_repo.py \
    --output /tmp/triton_cpu_full_mock --cpu-mock --mock-token-latency-ms 0.5

# 2. 启动 Nerva
MOCK_TOKEN_LATENCY_MS=0.5 uv run uvicorn \
    examples.mm_vllm_cpu_mock_server:app \
    --host 127.0.0.1 --port 8080 > /tmp/nerva.log 2>&1 &

# 3. 启动 Triton
podman run --rm \
    -p 8002:8002 -p 8003:8003 -p 8004:8004 \
    -v /tmp/triton_cpu_full_mock:/models \
    nvcr.io/nvidia/tritonserver:24.08-py3 \
    tritonserver --model-repository=/models \
      --http-port=8002 --grpc-port=8003 --metrics-port=8004 \
    > /tmp/triton.log 2>&1 &

# 4. 等待就绪（见第 5 节）

# 5. 压测（注意 unset 代理）
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  uv run python scripts/bench/run_bench.py \
    --target nerva --target triton \
    --concurrency-levels 1,4,16 \
    --warmup-seconds 5 --sample-seconds 15 \
    --deadline-ms 10000 \
    --output-root /tmp/bench-results

# 6. 校验 error_rate（应全为 0）
grep -h error_rate /tmp/bench-results/mm_vllm/*/*/*/*/summary.json
```
