# Nerva

Nerva is a Python-native model inference serving framework focused on low-latency, high-throughput workloads.  
It provides graph-based pipeline composition, process-isolated model workers, and a binary RPC serving path over HTTP.

## Highlights

- Graph tracing API (`trace`, `cond`, `parallel`) for pipeline orchestration.
- Worker-process isolation with explicit lifecycle management.
- Binary RPC protocol with msgpack payloads.
- Two serving entry points:
  - `build_nerva_app(...)` for ASGI hosts (`uvicorn`, tests).
  - `serve(...)` for direct blocking startup.
- Built-in endpoints:
  - `POST /rpc/{pipeline_name}`
  - `GET /v1/health`
  - `GET /v1/models`
  - `GET /metrics`

## Requirements

- Python 3.11+
- `uv` recommended for environment and dependency management.

## Installation

```bash
uv sync --dev
```

Optional extras:

```bash
uv sync --dev --extra pytorch
uv sync --dev --extra vllm
```

## Quick Start: End-to-End Echo Server

Terminal 1 (start server):

```bash
uv run uvicorn examples.echo_server:app --host 127.0.0.1 --port 8080
```

Terminal 2 (send request):

```bash
uv run python scripts/demo_client.py --url http://127.0.0.1:8080 --pipeline echo --value "hello"
```

Expected output:

```text
Calling http://127.0.0.1:8080/rpc/echo ...
Result: {'echo': 'hello'}
```

Optional checks:

```bash
curl http://127.0.0.1:8080/v1/health
curl http://127.0.0.1:8080/v1/models
```

## Core API

### ASGI factory (recommended for hosting)

```python
from typing import Any

from nerva import Model, build_nerva_app, model, trace


class EchoModel(Model):
    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"echo": inputs["value"]}


echo = model("echo", EchoModel, backend="pytorch", device="cpu")
graph = trace(lambda inp: echo(inp))
app = build_nerva_app({"echo": graph})
```

Run it:

```bash
uv run uvicorn your_module:app --port 8080
```

### Blocking startup

```python
from nerva import serve

serve({"echo": graph}, host="0.0.0.0", port=8080)
```

## Lifecycle Notes

- With ASGI lifespan support (for example `uvicorn`), workers are started on startup and closed on shutdown.
- For hosts that do not send lifespan (for example `httpx.ASGITransport`), Nerva lazily starts workers on first request.
- In no-lifespan test scenarios, call `await app.shutdown()` after requests for deterministic cleanup.
- A best-effort GC cleanup path is included for no-lifespan usage.
- Parent-process watchdog is enabled by default: if launcher parent exits unexpectedly (for example `uv run ...` process is killed), the serving process self-terminates to avoid orphan processes.

## Examples

- `examples/echo_server.py`: minimal runnable E2E server.
- `examples/01_single_model.py`: single-model serving flow.
- `examples/02_multi_model_pipeline.py`: multi-stage pipeline.
- `examples/03_parallel_dag.py`: `parallel` / `cond` flow composition.
- `examples/phase7_multimodal_vllm_server.py`: phase7 multimodal + vLLM benchmark DAG service.
- `scripts/demo_client.py`: standalone Binary RPC client script.

## Benchmark Quick Start (Phase 7)

- Runbook: `docs/plans/2026-03-02-phase7-e2e-benchmark-runbook.md`
- 默认口径：真实 vLLM / Triton 二进制。启动脚本在缺失依赖时会 fail-fast。
- `--allow-mock` 仅用于本地联调，不可用于正式对照数据采集。

```bash
# Start native vLLM (real backend by default)
uv run python scripts/bench/infra/start_vllm_server.py --model <MODEL_PATH> --host 127.0.0.1 --port 8001
uv run python scripts/bench/infra/wait_service_ready.py --kind vllm --url http://127.0.0.1:8001/health --timeout-seconds 120

# Start Triton (real backend by default)
uv run python scripts/bench/infra/prepare_triton_repo.py --output /tmp/phase7-triton-repo
uv run python scripts/bench/infra/start_triton_server.py --model-repo /tmp/phase7-triton-repo --http-port 8002 --grpc-port 8003 --metrics-port 8004
uv run python scripts/bench/infra/wait_service_ready.py --kind triton --url http://127.0.0.1:8002/v2/health/ready --timeout-seconds 120

# Benchmark matrix (includes concurrency 1000)
uv run python scripts/bench/run_phase7.py --target nerva --target vllm --target triton --concurrency-levels 1,32,128,512,1000 --warmup-seconds 60 --sample-seconds 300
```

## Development

Common checks:

```bash
export PATH="$HOME/.local/bin:$PATH"
uv run ruff check src/ tests/ examples/ scripts/
uv run mypy
uv run pytest tests/ -v
```

## Documentation

- Implementation plans and design notes: `docs/plans/`
- Spikes and benchmark reports: `docs/spikes/`

## License

See `LICENSE`.
