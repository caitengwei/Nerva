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
- `scripts/demo_client.py`: standalone Binary RPC client script.

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
