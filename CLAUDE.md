# Nerva

High-performance model inference serving framework. Python-first philosophy with Rust/C++ for performance-critical components. Targets low-latency, high-throughput serving as a flexible alternative to Triton Inference Server.

## Status

Phase 0, 1, 2 complete. 140 tests passing.

## Project Layout

```
src/nerva/                  # 主包（src layout）
  core/model.py             # Model ABC, ModelHandle, model()
  core/graph.py             # Edge, Node, Graph IR
  core/proxy.py             # Proxy, TraceContext, trace()
  core/primitives.py        # cond(), parallel()
  backends/base.py          # Backend ABC, ModelConfig, InferContext, BatchMeta
  backends/registry.py      # register_backend / get_backend / list_backends
  backends/pytorch.py       # PyTorchBackend
  engine/executor.py        # DAG Executor (event-driven)
  engine/shm_pool.py        # Shared Memory Pool
  worker/process.py         # Worker process main loop
  worker/proxy.py           # WorkerProxy (async RPC)
  worker/manager.py         # WorkerManager (lifecycle)
  worker/ipc.py             # IPC message codec
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

Call chain: trace() → Graph → Executor → WorkerProxy → Worker → Backend → Model

- **Model** — user subclass, implements `load()` + `infer()`
- **Backend** — execution backend (PyTorchBackend wraps Model, manages lifecycle)
- **ModelHandle** — lazy declaration from `model()`, dual-mode `__call__` (trace/runtime)
- **Backend registry** — `@register_backend("name")` decorator pattern
- **Graph IR** — Edge (with field_path/input_key), Node, Graph (topo sort)
- **Proxy** — trace-time placeholder, records dependencies via `__getitem__`
- **trace()** — builds Graph by tracing user pipeline function
- **cond()/parallel()** — control flow primitives (sub-graph embedding)
- **Executor** — event-driven DAG executor (in-degree table + done_queue)
- **WorkerProxy** — async RPC wrapper for Worker subprocess (ZeroMQ PAIR)
- **WorkerManager** — spawns/restarts/shuts down Worker processes

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

## 压测注意事项

详细检查清单见 [`docs/bench/bench-preflight-checklist.md`](docs/bench/bench-preflight-checklist.md)。关键坑：

1. **代理变量**：本机 `all_proxy=socks5://...` 会导致 httpx 对 localhost 也走代理，error_rate=1.0 且 latency 异常低（~17ms）。压测命令必须加 `env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY`；`no_proxy` 对 httpx 无效。

2. **Triton 侧必须用真实 tritonserver**：`--allow-mock` 是单函数 HTTP handler，绕过 ensemble 和 Python backend，pipeline 深度与 Nerva 不对等，对比结论无意义。正确做法：`prepare_triton_repo.py --cpu-mock` 生成 model repo，podman 起真实容器。

3. **macOS podman 用 `-p` 不用 `--network host`**：podman 跑在 Linux VM 内，`--network host` 是 VM 的网络，容器端口不可达。

4. **worktree 首次跑前执行 `uv sync --all-extras`**：新 worktree 的 `.venv` 缺少 httpx/msgpack 等运行时依赖。

5. **第一组结果出来后先校验**：error_rate 应为 0，p50 应接近 mock_sleep（256×0.5ms=128ms），QPS ≈ concurrency/p50。任何偏差先排查再继续。
