# Nerva v0.1 Design Document

> High-performance model inference serving framework with Python-first philosophy.

## 1. Overview

### 1.1 What is Nerva

Nerva is a high-performance model inference serving framework designed as a flexible alternative to Triton Inference Server. It provides a unified orchestration layer across multiple inference backends, exposing a clean, Python-native API for algorithm engineers.

### 1.2 Design Goals

- **Multi-model orchestration**: Support complex inference pipelines with DAG topology
- **High-performance single-model serving**: Dynamic batching, async scheduling, multi-GPU
- **Unified backend abstraction**: Plug-in backends (vLLM, PyTorch first; ONNX, TensorRT later)
- **Python-first**: Algorithm engineers can define and deploy models with minimal boilerplate
- **JAX-style function transforms**: Pipeline = pure function; capabilities injected via composable transforms

### 1.3 Target Users

Algorithm engineers who want to deploy models to production without dealing with infrastructure complexity.

### 1.4 MVP Scope

A working low-latency multi-model orchestration pipeline on single-node multi-GPU:
user defines models and pipeline function, applies transforms (`trace`, `batch`, `stream`), and serves via ASGI binary streaming RPC.

**MVP includes:**
- `model()`, `trace()`, `batch()`, `stream()`, `serve()` core transforms
- `nerva.cond()`, `nerva.parallel()` control flow primitives
- PyTorch Backend + vLLM Backend
- Master-Worker architecture (multi-process)
- Dynamic batching with deadline awareness
- Starlette + ASGI binary streaming RPC data plane
- FastAPI-based management API (model load/unload, health, metrics)
- Request-level deadline / cancel / backpressure
- IPC low-latency path: UDS control channel + POSIX shm descriptor passing
- Prometheus metrics + structlog logging

**MVP excludes (future iterations):**
- YAML declarative pipeline config
- `fori_loop`, `switch` control flow primitives
- `cache()`, `timeout()`, `retry()` transforms
- `model_group` load balancing
- Distributed cluster deployment
- Model repository / version management
- Client SDK

### 1.5 PoC Success Criteria

- Validate multi-model DAG orchestration under single-node multi-GPU
- Achieve low tail latency (`p95` / `p99`) under target concurrency
- Maintain stable streaming TTFT under high load
- Verify backpressure and cancellation behavior under overload
- Quantify IPC overhead split: queue wait / scheduling / transport / compute

### 1.6 Spec Set And Freeze Status

This design document is the architecture overview. Detailed contracts are frozen in companion specs:

| Spec | Purpose | Status |
|---|---|---|
| [`protocol-v1.md`](./protocol-v1.md) | Binary wire protocol and RPC state machine | Freeze Candidate |
| [`ipc-contract.md`](./ipc-contract.md) | Master-Worker IPC contract and lifecycle | Freeze Candidate |
| [`mvp-defaults.md`](./mvp-defaults.md) | Single source of default parameters | Active |
| [`config-mapping.md`](./config-mapping.md) | Defaults to runtime config/env mapping | Active |
| [`mvp-benchmark-plan.md`](./mvp-benchmark-plan.md) | Benchmark matrix, SLO gates, fault injection | Freeze Candidate |
| [`mvp-tuning-playbook.md`](./mvp-tuning-playbook.md) | Post-baseline tuning sequence | Active |
| [`benchmark-report-template.md`](./benchmark-report-template.md) | Benchmark output template | Active |
| [`adr-index.md`](./adr-index.md) | Architecture decision registry | Active |

### 1.7 Locked MVP Decisions (ADR)

The following decisions are accepted and treated as frozen for MVP:
- [`ADR-001`](./adr/ADR-001-payload-encoding.md): Canonical RPC payload encoding
- [`ADR-002`](./adr/ADR-002-worker-granularity.md): Worker granularity (`model() -> worker`)
- [`ADR-003`](./adr/ADR-003-cancel-propagation.md): Cancel propagation and completion status semantics

---

## 2. Core Concepts

### 2.1 Model

The smallest inference unit. Users define models by subclassing `nerva.Model`, implementing `load()` and `infer()` methods.

```python
from nerva import Model

class MyModel(Model):
    def load(self):
        self.model = torch.load("model.pt")

    async def infer(self, request: InferRequest) -> InferResponse:
        return self.model(request.tensor)
```

### 2.2 Backend

Inference backend abstraction. Each backend (vLLM, PyTorch, ONNX, etc.) implements a unified interface, bridging Model and the underlying engine.

### 2.3 Pipeline

Orchestration unit describing data flow between multiple Models. Defined as a pure Python function; the framework builds the DAG via tracing.

### 2.4 Worker

Execution unit. Each Worker is an independent process loading one Model instance, bound to a specific GPU. Master dispatches requests to Workers via IPC.

### 2.5 Server

Service entry point. ASGI server receives binary RPC requests, routes them to the corresponding Pipeline or single Model.

**Concept relationships:**

```
Server ─receives→ Request ─routes→ Pipeline ─dispatches→ Model(s) ─runs on→ Worker(s) ─uses→ Backend
```

---

## 3. Architecture: Master-Worker

### 3.1 Overview

```
                    ┌─────────────────────────────┐
                    │        Master Process        │
                    │  ┌───────────────────────┐   │
 Binary RPC Req ──▶│  │ Starlette + ASGI Server │   │
                    │  └──────────┬────────────┘   │
                    │             ▼                 │
                    │  ┌──────────────────────┐    │
                    │  │   Pipeline Router     │    │
                    │  └──────────┬────────────┘   │
                    │             ▼                 │
                    │  ┌──────────────────────┐    │
                    │  │ Pipeline Orchestrator │    │
                    │  └───┬──────┬──────┬────┘    │
                    │      ▼      ▼      ▼         │
                    │  ┌──────┐┌──────┐┌──────┐    │
                    │  │Batch0││Batch1││Batch2│    │
                    │  └──┬───┘└──┬───┘└──┬───┘    │
                    └─────┼───────┼───────┼────────┘
                          ▼       ▼       ▼
                    ┌─────────┐┌─────────┐┌─────────┐
                    │Worker 0 ││Worker 1 ││Worker 2 │
                    │(GPU 0)  ││(GPU 1)  ││(CPU)    │
                    │ vLLM    ││PyTorch  ││PyTorch  │
                    └─────────┘└─────────┘└─────────┘
```

### 3.2 Master Process Responsibilities

1. **Binary RPC Server** — Starlette + ASGI server (Uvicorn), handles unary / server-streaming binary RPC
2. **Pipeline Router** — Routes requests by pipeline/model name
3. **Pipeline Orchestrator** — Executes Pipeline DAG in topological order, manages inter-node data passing
4. **Dynamic Batcher** — One Batcher per Model, aggregates requests within time window / batch size threshold
5. **Deadline & Backpressure Controller** — Enforces queue limits, deadline admission checks, and cancellation propagation
6. **Worker Manager** — Manages Worker process lifecycle (start, health check, restart, graceful shutdown)

### 3.3 Worker Process Responsibilities

Each Worker is a single-purpose independent process:

1. Load one Model instance (via corresponding Backend)
2. Listen on IPC channel for batch requests
3. Invoke Backend to execute inference
4. Return results to Master

### 3.4 IPC Design

Two-layer communication between Master and Worker:

- **Control channel**: Unix Domain Socket + msgpack serialization, for management commands and descriptor exchange
- **Data channel**: POSIX shared memory pool. Large payload/tensor bytes stay in shared memory; only descriptor is sent via control channel

Descriptor schema (MVP):
- `request_id`, `node_id`
- `shm_id`, `offset`, `length`
- `inline_data` (small payload inline)
- `payload_codec` (`msgpack_dict_v1` / `raw_bytes_v1`)
- `input_key` (target input field for `raw_bytes_v1`)
- `dtype`, `shape`
- `lifetime_token` (for ref-count style reclaim)

Payload codec convention (Phase 1.1):
- Default `msgpack_dict_v1`: payload is msgpack-encoded dict.
- Single-field bytes can use `raw_bytes_v1`: skip dict-level `msgpack.packb`; Worker reconstructs `{input_key: bytes}`.

Reduced-copy convention (Phase 1.1):
- For SHM + `msgpack_dict_v1`, prefer `memoryview -> msgpack.unpackb` to avoid `bytes(buf[slice])` temporary copies.

Lifecycle (MVP):
1. Master allocates slot from shm pool and writes payload
2. Master sends descriptor to Worker via UDS
3. Worker reads from shm and returns ack/error
4. Master reclaims slot after ack or timeout-driven GC

---

## 4. JAX-Style Function Transform Pipeline API

### 4.1 Design Philosophy

1. **Users write functions only** — A Pipeline is just a function; model calls are function calls
2. **Transforms = capabilities** — `trace()` builds the graph, `batch()` adds batching, `stream()` adds streaming; all injected via function transforms
3. **Control flow via primitives** — `nerva.cond()` and `nerva.parallel()` are traceable in MVP

### 4.2 User API

```python
import nerva
from nerva import model, trace, batch, stream, serve

# Define models (declarative, not loaded immediately)
tokenizer = model("tokenizer", TokenizerImpl, backend="pytorch", device="cpu")
llm = model("llm", LLMImpl, backend="vllm", device="cuda:0")
detokenizer = model("detokenizer", DetokenizerImpl, backend="pytorch", device="cpu")

# Define Pipeline (just a regular function)
def text_generation(text):
    tokens = tokenizer(text)
    output = llm(tokens)
    return detokenizer(output)

# Function transforms — layer capabilities
graph = trace(text_generation)
graph = batch(graph, targets=["llm"], max_size=32, max_delay_ms=10)
graph = stream(graph)
app = serve(graph, route="/rpc/text_generation")

# Or composed (read inside-out)
app = serve(stream(batch(trace(text_generation), targets=["llm"])))
```

### 4.3 Control Flow Primitives

```python
# cond — conditional routing
def multimodal_router(request):
    features = nerva.cond(
        request.media_type == "image",
        lambda: image_encoder(request.data),
        lambda: text_encoder(request.data),
    )
    return classifier(features)

# parallel — concurrent execution with result collection
def multimodal_fusion(request):
    img_feat, txt_feat = nerva.parallel(
        lambda: image_encoder(request.image),
        lambda: text_encoder(request.text),
    )
    return fusion_model(img_feat, txt_feat)
```

### 4.4 Transform Reference

| Transform | Input | Output | Purpose |
|---|---|---|---|
| `trace(fn)` | Function | `Graph` (DAG IR) | Trace function calls, build computation graph |
| `batch(g, ...)` | Graph | Graph | Inject dynamic batching on specified nodes |
| `stream(g)` | Graph | Graph | Terminal node output becomes async iterator |
| `serve(g, ...)` | Graph | ASGI App | Generate deployable binary RPC service |

Transform composition in MVP is **ordered**, not commutative:
`serve(stream(batch(trace(fn))))`.

### 4.5 Trace Internals

```
User function
    │  trace() passes Proxy objects instead of real data
    ▼
Proxy records all model() calls and control flow primitives
    │
    ▼
Generate DAG IR (Graph)
    ├── Node: model call node
    ├── Edge: data dependency
    ├── CondNode: nerva.cond branch node
    └── ParallelNode: nerva.parallel fork/join node
    │
    ▼
Graph optimization passes
    ├── Auto-parallelize independent nodes
    ├── Device affinity scheduling (merge communication for same-GPU nodes)
    └── Dead node elimination
```

### 4.6 Trace Subset Constraints (MVP)

To keep predictable latency and implementation complexity under control, `trace()` supports a restricted subset:
- model function calls
- `nerva.cond()` and `nerva.parallel()`
- simple Python expressions over traceable values

Not supported in MVP:
- arbitrary Python side effects during tracing
- dynamic loops with data-dependent trip count
- runtime graph mutation

---

## 5. Backend Abstraction Layer

### 5.1 Abstract Interface

```python
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    model_path: str
    device: str  # "cpu", "cuda:0", "cuda:1", ...
    backend_options: dict[str, Any] = field(default_factory=dict)

@dataclass
class InferContext:
    request_id: str
    deadline_ms: int
    trace_id: str | None = None
    cancelled: bool = False

@dataclass
class BatchMeta:
    batch_size: int
    padded: bool = False
    pad_to_multiple: int | None = None

class Backend(ABC):
    @abstractmethod
    async def load_model(self, config: ModelConfig) -> None:
        """Load model to specified device."""

    @abstractmethod
    async def unload_model(self) -> None:
        """Unload model, release resources."""

    @abstractmethod
    async def infer(
        self,
        inputs: dict[str, Any],
        context: InferContext,
        batch_meta: BatchMeta | None = None,
    ) -> dict[str, Any]:
        """Execute single inference (batched input)."""

    @abstractmethod
    async def infer_stream(
        self,
        inputs: dict[str, Any],
        context: InferContext,
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming inference, yield intermediate results."""

    async def warmup(self) -> None:
        """Optional warmup hook."""

    def health_check(self) -> bool:
        return True
```

### 5.2 MVP Backends

**PyTorchBackend:**
- Loads user-provided `torch.nn.Module`
- `torch.inference_mode()` optimization
- User-defined pre/post processing

**VLLMBackend:**
- Wraps vLLM's `AsyncLLMEngine`
- Native continuous batching (vLLM manages its own batching; Nerva's Batcher is bypassed)
- Streaming output via vLLM's streaming API
- Coordinates Nerva Worker process with vLLM's internal processes

### 5.3 Registration Mechanism

```python
from nerva import register_backend

@register_backend("pytorch")
class PyTorchBackend(Backend):
    ...

@register_backend("vllm")
class VLLMBackend(Backend):
    ...
```

---

## 6. Dynamic Batching & Multi-GPU Scheduling

### 6.1 Dynamic Batching

One Batcher instance per Model node in the Master process.

```
Request A ──▶ ┌──────────────────┐
Request B ──▶ │  Dynamic Batcher │──batch──▶ Worker (GPU)
Request C ──▶ │                  │
              │ Trigger:          │
              │ 1. batch_size=N  │
              │ 2. max_delay=Tms │
              │ (whichever first) │
              └──────────────────┘
```

**Parameters** (configured via `batch()` transform):
```python
graph = batch(graph, targets=["llm"],
    max_size=32,         # max batch size
    max_delay_ms=10,     # max wait time
    pad_to_multiple=8,   # pad to multiples of 8 (GPU tensor core alignment)
)
```

**Implementation:**
- Bounded `asyncio.Queue` + timer in Master process
- Requests enqueue on arrival; batch fires when full or timer expires
- Variable-length inputs padded, original lengths recorded for unpadding
- Results split per-request and dispatched back to respective `asyncio.Future`s
- Admission control checks remaining deadline before enqueue
- Queue overflow returns immediate overload error (no unbounded waiting)

**vLLM special handling:** vLLM manages its own continuous batching internally. Nerva's Batcher is bypassed for vLLM nodes.

### 6.2 Multi-GPU Scheduling

**Model placement** — user explicitly specifies device:
```python
llm = model("llm", LLMImpl, backend="vllm", device="cuda:0")
encoder = model("encoder", EncoderImpl, backend="pytorch", device="cuda:1")
```

**Worker allocation:**
- Each `model()` declaration corresponds to one Worker process
- Worker binds GPU via `CUDA_VISIBLE_DEVICES` at startup
- Multiple Workers can share a GPU (for CPU models or when VRAM allows)

**Load balancing (multi-replica, future iteration):**
```python
llm_group = model_group("llm", LLMImpl, backend="vllm",
    devices=["cuda:0", "cuda:1"],
    strategy="round_robin",  # or "least_pending"
)
```

### 6.3 Latency-First Scheduling (MVP)

- Request scheduling priority: earliest deadline first (EDF)
- Cancellation propagates from ingress to queued and running nodes
- Per-model queue isolation to avoid head-of-line blocking across models
- Streaming requests reserve bounded output buffer to protect global latency

---

## 7. Binary Streaming RPC Service Layer (ASGI)

### 7.1 Tech Stack

- **Data plane framework**: Starlette (raw ASGI request/response streaming path)
- **ASGI server**: Uvicorn (single Nerva master process)
- **Management API**: FastAPI (model control, health, metrics)
- `serve()` transform compiles Graph into ASGI binary RPC app
- Process-level parallelism is provided by Nerva Worker processes, not Uvicorn worker mode

### 7.2 API Endpoints

```
POST /rpc/{pipeline_name}               # Unary or server-streaming binary RPC
POST /rpc/{request_id}/cancel           # Explicit cancel (optional)

GET  /v1/models                         # List loaded models
GET  /v1/models/{model_name}            # Model details + status
POST /v1/models/{model_name}/load       # Load model
POST /v1/models/{model_name}/unload     # Unload model

GET  /v1/health                         # Health check
GET  /v1/metrics                        # Prometheus metrics
```

### 7.3 Binary Wire Protocol (MVP)

Request headers:
- `Content-Type: application/x-nerva-rpc`
- `Accept: application/x-nerva-rpc`
- `x-nerva-request-id` (optional; server generates if absent)
- `x-nerva-deadline-ms` (required for latency-aware scheduling)
- `x-nerva-stream` (`0` unary, `1` server-streaming)

Frame format:

| Field | Size | Description |
|---|---:|---|
| `magic` | 2B | `0x4E56` (`NV`) |
| `version` | 1B | Protocol version (`1`) |
| `type` | 1B | `OPEN=1`, `DATA=2`, `END=3`, `ERROR=4`, `HEARTBEAT=5` |
| `flags` | 2B | bit flags (`compressed`, `eos`) |
| `reserved` | 2B | reserved |
| `request_id` | 8B | request correlation id |
| `stream_id` | 4B | stream id (MVP fixed to `1`) |
| `payload_len` | 4B | payload length |
| `crc32` | 4B | payload checksum (MVP optional) |
| `header_ext_len` | 4B | extension header length (MVP `0`) |

Payload:
- `OPEN`: method/pipeline metadata
- `DATA`: binary input/output chunk
- `END`: final metadata (latency, token count, etc.)
- `ERROR`: error code + message + retryable flag

### 7.4 RPC Lifecycle

Unary:
1. Client sends `OPEN + DATA + END`
2. Server executes DAG and returns `DATA + END`

Server-streaming:
1. Client sends `OPEN + DATA + END`
2. Server returns multiple `DATA` frames
3. Server finalizes with `END`

Cancel:
- Client connection close or explicit `/rpc/{request_id}/cancel`
- Master propagates cancel signal to queue and Worker execution path

---

## 8. Observability & Error Handling

### 8.1 Metrics (Prometheus)

```
# Request level
nerva_request_total{pipeline, status}
nerva_request_duration_seconds{pipeline}          # histogram
nerva_request_in_flight{pipeline}
nerva_request_deadline_exceeded_total{pipeline}
nerva_request_cancel_total{pipeline, source}

# Batching
nerva_batch_size{model}
nerva_batch_wait_seconds{model}

# Scheduler / Queue
nerva_queue_depth{model}
nerva_queue_drop_total{model, reason}
nerva_scheduler_wait_seconds{pipeline, node}

# IPC
nerva_ipc_control_latency_seconds{worker}
nerva_ipc_data_bytes_total{worker, direction}
nerva_ipc_shm_alloc_fail_total{worker}

# Worker
nerva_worker_status{model, device}
nerva_gpu_memory_used_bytes{device}
nerva_worker_infer_seconds{model}

# Streaming
nerva_stream_tokens_total{pipeline}
nerva_stream_first_token_seconds{pipeline}        # TTFT
```

### 8.2 Logging

- `structlog` with structured JSON output
- Per-request `request_id` propagated across full pipeline chain
- Levels: ERROR (exceptions), WARNING (deadline/cancel/overload), INFO (request summary), DEBUG (node-level detail)

### 8.3 Error Handling

| Failure Type | Strategy |
|---|---|
| Invalid frame / protocol mismatch | Return `ERROR(INVALID_ARGUMENT)` and close stream |
| Queue overload | Reject early with `ERROR(RESOURCE_EXHAUSTED)` |
| Request deadline exceeded | Cancel downstream work and return `ERROR(DEADLINE_EXCEEDED)` |
| Client-initiated cancel | Propagate cancel, mark request as `ABORTED` |
| Worker process crash | Worker Manager auto-restarts; in-flight requests return `UNAVAILABLE` |
| GPU OOM | Isolate failing request, record metrics, return `RESOURCE_EXHAUSTED` |
| Model load failure | Model marked `error` state; health endpoint reports anomaly |

---

## 9. Project Structure

```
nerva/
├── nerva/
│   ├── __init__.py              # Public API: model, trace, batch, stream, serve...
│   ├── core/
│   │   ├── graph.py             # DAG IR: Node, Edge, Graph
│   │   ├── proxy.py             # Proxy objects (for trace)
│   │   ├── transforms.py        # Function transforms: trace, batch, stream, serve
│   │   └── primitives.py        # Control flow primitives: cond, parallel
│   ├── engine/
│   │   ├── executor.py          # DAG execution engine (topological scheduling, asyncio)
│   │   ├── batcher.py           # Dynamic Batcher
│   │   └── scheduler.py         # Deadline-aware scheduling, backpressure
│   ├── worker/
│   │   ├── manager.py           # Worker lifecycle management
│   │   ├── process.py           # Worker process main loop
│   │   └── ipc.py               # IPC communication (Unix socket + shared memory)
│   ├── backends/
│   │   ├── base.py              # Backend abstract base class
│   │   ├── pytorch.py           # PyTorch Backend
│   │   ├── vllm.py              # vLLM Backend
│   │   └── registry.py          # Backend registration
│   ├── server/
│   │   ├── app.py               # ASGI application entrypoint
│   │   ├── rpc.py               # Binary streaming RPC data plane handlers
│   │   ├── protocol.py          # Binary frame encode/decode definitions
│   │   ├── routes.py            # Management API endpoints
│   │   └── middleware.py        # Middleware (metrics, logging, error handling)
│   └── observability/
│       ├── metrics.py           # Prometheus metrics
│       └── logging.py           # Structured logging
├── tests/
├── examples/
├── pyproject.toml
└── README.md
```

## 10. Tech Stack Summary

| Component | Choice | Rationale |
|---|---|---|
| Data Plane Server | Starlette + Uvicorn (ASGI) | Low-overhead binary streaming path, compatible with Nerva Master-Worker model |
| Management API | FastAPI | Operational ergonomics: OpenAPI, validation, admin endpoints |
| Async Runtime | asyncio | Python standard, widest ecosystem |
| IPC | Unix Domain Socket + POSIX shm | Low latency, zero-copy for large tensors |
| RPC Protocol | Nerva Binary Streaming RPC | Unary + server-streaming, deadline/cancel support |
| Serialization | msgpack (control) + protobuf/msgpack (payload) | Reduce hot-path JSON overhead |
| Metrics | prometheus_client | Industry standard |
| Logging | structlog | Structured JSON, good performance |
| Type Checking | mypy (strict) | Catch bugs early |
| Linting | ruff | Fast, comprehensive rules |
| Testing | pytest + pytest-asyncio | Native async test support |
| Package Management | uv | Fast, modern Python toolchain |
