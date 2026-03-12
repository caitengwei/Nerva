# Nerva

High-performance model inference serving framework for Python.

Nerva lets ML engineers define multi-model pipelines as plain Python functions, then serves them over a binary RPC protocol with process-isolated workers — no YAML configs, no C++ plugins, no leaving the Python ecosystem.

## Why Nerva

Existing serving frameworks (Triton, TorchServe) require learning framework-specific configuration, have limited DAG orchestration support, and make Python-native development difficult. Nerva takes a different approach:

- **Pipeline = Python function.** Define arbitrarily complex DAGs with `trace()`, `cond()`, and `parallel()` — the framework infers the computation graph automatically.
- **Process isolation by default.** Each model runs in its own worker process with a dedicated CUDA context. One model crashing or OOM-ing doesn't take down the service.
- **IPC designed for ML payloads.** Small tensors inline via ZeroMQ; large payloads (images, embeddings) go through shared memory with near-zero copy overhead.

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management.

## Installation

```bash
uv sync --dev
```

Optional backend extras:

```bash
uv sync --dev --extra pytorch
uv sync --dev --extra vllm
```

## Quick Start

**Terminal 1** — start an echo server:

```bash
uv run uvicorn examples.echo_server:app --host 127.0.0.1 --port 8080
```

**Terminal 2** — send a request:

```bash
uv run python scripts/demo_client.py --url http://127.0.0.1:8080 --pipeline echo --value "hello"
```

Expected output:

```text
Calling http://127.0.0.1:8080/rpc/echo ...
Result: {'echo': 'hello'}
```

Health and model listing:

```bash
curl http://127.0.0.1:8080/v1/health
curl http://127.0.0.1:8080/v1/models
```

## Core API

### Define a model

Subclass `Model` and implement `load()` + `infer()`:

```python
from typing import Any
from nerva import Model, build_nerva_app, model, trace


class EchoModel(Model):
    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"echo": inputs["value"]}
```

### Build and serve a pipeline

```python
echo = model("echo", EchoModel, backend="pytorch", device="cpu")
graph = trace(lambda inp: echo(inp))

# Option 1: ASGI factory (recommended — works with uvicorn, tests, etc.)
app = build_nerva_app({"echo": graph})

# Option 2: Blocking startup
# from nerva import serve
# serve({"echo": graph}, host="0.0.0.0", port=8080)
```

Run the ASGI app:

```bash
uv run uvicorn your_module:app --port 8080
```

### Multi-model DAG with control flow

```python
from nerva import cond, model, parallel, trace

text_enc = model("text_enc", TextEncoder, backend="pytorch", device="cpu")
img_enc  = model("img_enc", ImageEncoder, backend="pytorch", device="cuda:0")
fusion   = model("fusion", Fusion, backend="pytorch", device="cuda:1")

def pipeline(inp):
    t, i = parallel(
        lambda: text_enc({"text": inp["text"]}),
        lambda: img_enc({"image": inp["image"]}),
    )
    return fusion({"text_feat": t["features"], "img_feat": i["features"]})

graph = trace(pipeline)
app = build_nerva_app({"mm": graph})
```

### Built-in endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/rpc/{pipeline_name}` | POST | Binary RPC inference |
| `/v1/health` | GET | Health check |
| `/v1/models` | GET | List loaded models |
| `/metrics` | GET | Prometheus metrics |

## Worker Lifecycle

- **With ASGI lifespan** (e.g. `uvicorn`): workers start on app startup and shut down on app shutdown.
- **Without lifespan** (e.g. `httpx.ASGITransport` in tests): workers start lazily on first request. Call `await app.shutdown()` after use for deterministic cleanup.
- **Parent-process watchdog**: if the parent process exits unexpectedly, workers self-terminate to prevent orphan processes.

## Examples

| File | Description |
|---|---|
| `examples/echo_server.py` | Minimal runnable E2E server |
| `examples/01_single_model.py` | Single-model serving flow |
| `examples/02_multi_model_pipeline.py` | Multi-stage pipeline |
| `examples/03_parallel_dag.py` | `parallel` / `cond` flow composition |
| `examples/mm_vllm_server.py` | Multimodal + vLLM benchmark DAG service |
| `scripts/demo_client.py` | Standalone Binary RPC client |

## Benchmarking

Full runbook: [`docs/plans/2026-03-02-phase7-e2e-benchmark-runbook.md`](docs/plans/2026-03-02-phase7-e2e-benchmark-runbook.md)

Benchmarks compare Nerva against native vLLM and Triton on the same workload.
Startup scripts fail-fast if real backend dependencies are missing.

```bash
# Start Nerva
MM_VLLM_MODEL_PATH=<MODEL_PATH> \
uv run uvicorn examples.mm_vllm_server:app --host 127.0.0.1 --port 8080

# Start vLLM
uv run python scripts/bench/infra/start_vllm_server.py --model <MODEL_PATH> --host 127.0.0.1 --port 8001
uv run python scripts/bench/infra/wait_service_ready.py --kind vllm --url http://127.0.0.1:8001/health --timeout-seconds 120

# Start Triton (embeds vLLM in-process, no separate vLLM needed)
uv run python scripts/bench/infra/prepare_triton_repo.py --output /tmp/mm_vllm-triton-repo --vllm-model <MODEL_PATH>
uv run python scripts/bench/infra/start_triton_server.py --model-repo /tmp/mm_vllm-triton-repo --http-port 8002 --grpc-port 8003 --metrics-port 8004
uv run python scripts/bench/infra/wait_service_ready.py --kind triton --url http://127.0.0.1:8002/v2/health/ready --timeout-seconds 300

# Run benchmark matrix
uv run python scripts/bench/run_bench.py \
  --target nerva --target vllm --target triton \
  --concurrency-levels 1,32,128,512,1000 \
  --warmup-seconds 60 --sample-seconds 300 \
  --require-real-backend
```

See [`docs/design/性能测试指南.md`](docs/design/性能测试指南.md) for detailed execution steps, result interpretation, and metrics-based bottleneck diagnosis.

## Development

```bash
export PATH="$HOME/.local/bin:$PATH"
uv run ruff check src/ tests/ examples/ scripts/   # lint
uv run mypy                                         # type check
uv run pytest tests/ -v                             # test
```

## Documentation

| Directory | Content |
|---|---|
| [`docs/design/`](docs/design/) | Architecture, module design, testing and benchmarking guides (Chinese) |
| [`docs/plans/`](docs/plans/) | Implementation plans, ADRs, protocol specs, roadmap |
| [`docs/spikes/`](docs/spikes/) | Technical spike reports (IPC benchmarks, trace prototype, batcher benchmarks) |

## License

See [`LICENSE`](LICENSE).
