# Nerva

High-performance model inference serving framework. Python-first philosophy with Rust/C++ for performance-critical components. Targets low-latency, high-throughput serving as a flexible alternative to Triton Inference Server.

## Status

Phase 0 complete. Phase 1 design complete, pending implementation.

## Project Layout

```
src/nerva/                  # 主包（src layout）
  core/model.py             # Model ABC, ModelHandle, model()
  backends/base.py          # Backend ABC, ModelConfig, InferContext, BatchMeta
  backends/registry.py      # register_backend / get_backend / list_backends
  backends/pytorch.py       # PyTorchBackend
  engine/                   # (Phase 1+) batcher, scheduler, shm_pool
  worker/                   # (Phase 1+) process, manager, ipc
  server/                   # (Phase 4+) ASGI app, RPC, protocol
  observability/            # (Phase 5+) metrics, logging
tests/                      # pytest + pytest-asyncio (asyncio_mode=auto)
examples/                   # 目标 API 示例（01_single_model, 02_multi_model, 03_parallel_dag）
spikes/                     # 技术验证脚本（s1_ipc, s2_trace, s4_batcher）
docs/plans/                 # 设计文档、review、roadmap
docs/spikes/                # spike 报告
```

## Tech Stack

- **Language:** Python 3.11+ (dev environment: 3.14)
- **Package manager:** uv
- **Build:** hatchling (src layout)
- **Lint:** ruff (line-length=100)
- **Type check:** mypy (strict)
- **Test:** pytest + pytest-asyncio (asyncio_mode=auto)
- **Performance-critical (future):** Rust / C++

## Key Commands

```bash
uv run pytest tests/ -v       # run tests
uv run ruff check src/ tests/ # lint
uv run mypy                   # type check
```

**Note:** Ensure `uv` is in your PATH (`export PATH="$HOME/.local/bin:$PATH"` if installed via official script).

## Architecture

Call chain: Orchestrator → Backend → Model

- **Model** — user subclass, implements `load()` + `infer()`
- **Backend** — execution backend (PyTorchBackend wraps Model, manages lifecycle)
- **ModelHandle** — lazy declaration from `model()`, not loaded until pipeline starts
- **Backend registry** — `@register_backend("name")` decorator pattern

## Design Decisions (from Design Review)

- IPC: ZeroMQ PAIR over ipc:// (Phase 1), SHM for data >8KB
- Trace: restricted trace as default + explicit DAG fallback
- Batching: no padding, group by size-matching; default delay 10ms
- serve(): separated from transform chain; Pipeline independently executable
- vLLM: thin wrapper in Worker, Nerva doesn't manage internal scheduling
- Pipeline failure: fail-fast semantics (MVP)
- Config: pure code config for MVP

## Development Guidelines

- Use Chinese for all communication (except code and technical terms)
- Conservative approach to code changes — avoid introducing bugs
- Focus on ML/AI Infra/HPC domain concerns

## Roadmap

See [`docs/plans/2026-02-25-mvp-roadmap.md`](docs/plans/2026-02-25-mvp-roadmap.md) for full details.
