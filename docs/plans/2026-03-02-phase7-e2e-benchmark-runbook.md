# Phase 7 E2E Benchmark Runbook

口径说明：本 runbook 只保留一套口径 `full-e2e`（不再区分 infer-only）。

## 1. 启动 Nerva Phase 7 服务

```bash
MM_VLLM_MODEL_PATH=<MODEL_PATH> \
uv run uvicorn examples.mm_vllm_server:app --host 127.0.0.1 --port 8080
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
uv run python scripts/bench/infra/prepare_triton_repo.py \
  --output /tmp/phase7-triton-repo \
  --vllm-url http://127.0.0.1:8001 \
  --vllm-model /models

uv run python scripts/bench/infra/start_triton_server.py \
  --model-repo /tmp/phase7-triton-repo \
  --http-port 8002 \
  --grpc-port 8003 \
  --metrics-port 8004
```

Linux + GPU（容器，`nerdctl`）：

```bash
uv run python scripts/bench/infra/prepare_triton_repo.py \
  --output /tmp/phase7-triton-repo \
  --vllm-url http://127.0.0.1:8001 \
  --vllm-model /models

nerdctl run --rm --gpus all --network host --ipc host \
  -v /tmp/phase7-triton-repo:/models \
  nvcr.io/nvidia/tritonserver:26.03-py3 \
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

`full-e2e` 口径定义：
- `nerva`：前后处理在服务端 DAG（`mm_preprocess -> mm_vllm -> mm_postprocess`）内执行。
- `vllm`：target adapter 使用与 Nerva 同语义的 pre/post 逻辑，连同推理一起计入端到端延迟。
- `triton`：在 Triton model repository 内通过 ensemble 串联 `phase7_preprocess -> phase7_infer -> phase7_postprocess`，其中 `phase7_infer` 通过 OpenAI-compatible `/v1/completions` 调用 vLLM，统一计入端到端延迟。
- 三目标统一采样参数：`--max-tokens` / `--temperature` / `--top-p`。
- 建议压测命令启用 `--require-real-backend`，避免 mock 结果混入对照数据。
- 采样原则：按目标分三次执行，每次仅启动当前目标所需服务，避免同机多服务并存导致资源争用污染结果。
- TODO（下一个迭代）：`triton` 目标在 `--require-real-backend` 下，增加对其上游 vLLM 实例是否为 real 的联合校验。

小流量冒烟（C=1,32），按目标分别执行：

```bash
# nerva（仅启动 nerva）
uv run python scripts/bench/run_phase7.py \
  --target nerva \
  --max-tokens 256 \
  --temperature 1.0 \
  --top-p 1.0 \
  --require-real-backend \
  --workload phase7_mm_vllm \
  --concurrency-levels 1,32 \
  --warmup-seconds 10 \
  --sample-seconds 30
```

```bash
# vllm（仅启动 vllm）
uv run python scripts/bench/run_phase7.py \
  --target vllm \
  --max-tokens 256 \
  --temperature 1.0 \
  --top-p 1.0 \
  --require-real-backend \
  --vllm-model /models \
  --workload phase7_mm_vllm \
  --concurrency-levels 1,32 \
  --warmup-seconds 10 \
  --sample-seconds 30
```

```bash
# triton（启动 vllm + triton）
uv run python scripts/bench/run_phase7.py \
  --target triton \
  --max-tokens 256 \
  --temperature 1.0 \
  --top-p 1.0 \
  --require-real-backend \
  --vllm-model /models \
  --workload phase7_mm_vllm \
  --concurrency-levels 1,32 \
  --warmup-seconds 10 \
  --sample-seconds 30
```

全矩阵（含 C=1000），按目标分别执行：

```bash
# nerva（仅启动 nerva）
uv run python scripts/bench/run_phase7.py \
  --target nerva \
  --max-tokens 256 \
  --temperature 1.0 \
  --top-p 1.0 \
  --require-real-backend \
  --workload phase7_mm_vllm \
  --concurrency-levels 1,32,128,512,1000 \
  --warmup-seconds 60 \
  --sample-seconds 300
```

```bash
# vllm（仅启动 vllm）
uv run python scripts/bench/run_phase7.py \
  --target vllm \
  --max-tokens 256 \
  --temperature 1.0 \
  --top-p 1.0 \
  --require-real-backend \
  --vllm-model /models \
  --workload phase7_mm_vllm \
  --concurrency-levels 1,32,128,512,1000 \
  --warmup-seconds 60 \
  --sample-seconds 300
```

```bash
# triton（启动 vllm + triton）
uv run python scripts/bench/run_phase7.py \
  --target triton \
  --max-tokens 256 \
  --temperature 1.0 \
  --top-p 1.0 \
  --require-real-backend \
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
