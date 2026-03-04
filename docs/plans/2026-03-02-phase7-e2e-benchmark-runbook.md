# Phase 7 E2E Benchmark Runbook

## 1. 启动 Nerva Phase 7 服务

```bash
uv run uvicorn examples.phase7_multimodal_vllm_server:app --host 127.0.0.1 --port 8080
```

## 2. 启动 Native vLLM

```bash
uv run python scripts/bench/infra/start_vllm_server.py \
  --model <MODEL_PATH> \
  --host 127.0.0.1 \
  --port 8001
```

Linux + GPU（容器，`nerdctl`）：

```bash
nerdctl run --rm --gpus all --network host --ipc host \
  -v <MODEL_PATH>:/models:ro \
  vllm/vllm-openai:v0.6.0 \
  --model /models \
  --host 0.0.0.0 \
  --port 8001
```

说明：
- 默认要求本机可执行 `vllm` 二进制；若缺失会直接报错退出（fail-fast）。
- 仅本地联调可使用 mock 兜底：追加 `--allow-mock`（不要用于正式对照数据采集）。

就绪检查：

```bash
uv run python scripts/bench/infra/wait_service_ready.py \
  --kind vllm \
  --url http://127.0.0.1:8001/health \
  --timeout-seconds 120
```

## 3. 启动 Triton

```bash
uv run python scripts/bench/infra/prepare_triton_repo.py --output /tmp/phase7-triton-repo

uv run python scripts/bench/infra/start_triton_server.py \
  --model-repo /tmp/phase7-triton-repo \
  --http-port 8002 \
  --grpc-port 8003 \
  --metrics-port 8004
```

Linux + GPU（容器，`nerdctl`）：

```bash
uv run python scripts/bench/infra/prepare_triton_repo.py --output /tmp/phase7-triton-repo

nerdctl run --rm --gpus all --network host --ipc host \
  -v /tmp/phase7-triton-repo:/models \
  nvcr.io/nvidia/tritonserver:24.08-py3 \
  tritonserver \
  --model-repository=/models \
  --http-port=8002 \
  --grpc-port=8003 \
  --metrics-port=8004
```

说明：
- 默认要求本机可执行 `tritonserver` 二进制；若缺失会直接报错退出（fail-fast）。
- 仅本地联调可使用 mock 兜底：追加 `--allow-mock`（不要用于正式对照数据采集）。

就绪检查：

```bash
uv run python scripts/bench/infra/wait_service_ready.py \
  --kind triton \
  --url http://127.0.0.1:8002/v2/health/ready \
  --timeout-seconds 120
```

## 4. 统一压测命令

小流量冒烟（C=1,32）：

```bash
uv run python scripts/bench/run_phase7.py \
  --target nerva --target vllm --target triton \
  --workload phase7_mm_vllm \
  --concurrency-levels 1,32 \
  --warmup-seconds 10 \
  --sample-seconds 30
```

全矩阵（含 C=1000）：

```bash
uv run python scripts/bench/run_phase7.py \
  --target nerva --target vllm --target triton \
  --workload phase7_mm_vllm \
  --concurrency-levels 1,32,128,512,1000 \
  --warmup-seconds 60 \
  --sample-seconds 300
```

容器模式（若 vLLM 用 `--model /models` 启动）请追加模型名参数：

```bash
uv run python scripts/bench/run_phase7.py \
  --target nerva --target vllm --target triton \
  --vllm-model /models \
  --workload phase7_mm_vllm \
  --concurrency-levels 1,32,128,512,1000 \
  --warmup-seconds 60 \
  --sample-seconds 300
```

## 5. 产物与报告

压测产物目录：

```text
bench-results/phase7/<date>/<commit>/<target>/<concurrency>/
```

生成聚合报告：

```bash
uv run python scripts/bench/report_phase7.py \
  --input-root bench-results/phase7 \
  --output docs/plans/phase7-benchmark-report.md
```

## 6. 常见故障排查

- client 打不满：先降低单请求 payload 大小，检查 loadgen 端 CPU 与 event loop 饱和。必要时采用多进程 loadgen。
- server 饱和：检查 QPS 上升时 p95/p99 异常抖动，观察 `GET /metrics` 与 worker 日志 `request_id` 链路。
- GPU OOM：降低 `--gpu-memory-utilization`，减少 `max_tokens`，或降低并发档位后逐级爬升。
