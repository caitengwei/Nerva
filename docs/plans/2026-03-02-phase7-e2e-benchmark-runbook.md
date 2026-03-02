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
