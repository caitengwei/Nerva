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

A working multi-model pipeline: user defines models and pipeline function, applies transforms (`trace`, `batch`, `stream`), and serves via HTTP/2 with streaming support.

**MVP includes:**
- `model()`, `trace()`, `batch()`, `stream()`, `serve()` core transforms
- `nerva.cond()`, `nerva.parallel()` control flow primitives
- PyTorch Backend + vLLM Backend
- Master-Worker architecture (multi-process)
- Dynamic Batching
- HTTP/2 serving + SSE streaming
- Basic model load/unload API
- Prometheus metrics + structlog logging

**MVP excludes (future iterations):**
- YAML declarative pipeline config
- `fori_loop`, `switch` control flow primitives
- `cache()`, `timeout()` transforms
- `model_group` load balancing
- Distributed cluster deployment
- Model repository / version management
- Client SDK

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

Service entry point. HTTP/2 server receiving external requests, routing them to the corresponding Pipeline or single Model.

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
  HTTP/2 Request ──▶│  │  FastAPI + Hypercorn   │   │
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

1. **HTTP/2 Server** — FastAPI + Hypercorn, handles request/response, supports SSE streaming output and streaming request body input
2. **Pipeline Router** — Routes requests by pipeline/model name
3. **Pipeline Orchestrator** — Executes Pipeline DAG in topological order, manages inter-node data passing
4. **Dynamic Batcher** — One Batcher per Model, aggregates requests within time window / batch size threshold
5. **Worker Manager** — Manages Worker process lifecycle (start, health check, restart, graceful shutdown)

### 3.3 Worker Process Responsibilities

Each Worker is a single-purpose independent process:

1. Load one Model instance (via corresponding Backend)
2. Listen on IPC channel for batch requests
3. Invoke Backend to execute inference
4. Return results to Master

### 3.4 IPC Design

Two-layer communication between Master and Worker:

- **Control channel**: Unix Domain Socket + msgpack serialization, for management commands (load model, health check, shutdown) and small inference requests
- **Data channel**: For large tensor data, use POSIX shared memory + reference passing to avoid serialization overhead. Master writes to shared memory, passes only the handle to Worker

---

## 4. JAX-Style Function Transform Pipeline API

### 4.1 Design Philosophy

1. **Users write functions only** — A Pipeline is just a function; model calls are function calls
2. **Transforms = capabilities** — `trace()` builds the graph, `batch()` adds batching, `stream()` adds streaming; all injected via function transforms
3. **Control flow via primitives** — `nerva.cond()`, `nerva.switch()` replace `if/else` and are traceable

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
app = serve(graph, route="/v1/generate")

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
| `retry(g, ...)` | Graph | Graph | Inject retry logic on specified nodes |
| `timeout(g, ...)` | Graph | Graph | Inject timeout control |
| `cache(g, ...)` | Graph | Graph | Inject result caching on specified nodes |
| `serve(g, ...)` | Graph | ASGI App | Generate deployable HTTP/2 service |

Transforms satisfy **composition**: `f(g(x))` and `g(f(x))` are both valid (some ordering constraints apply, e.g. `serve` must be outermost).

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

class Backend(ABC):
    @abstractmethod
    async def load_model(self, config: ModelConfig) -> None:
        """Load model to specified device."""

    @abstractmethod
    async def unload_model(self) -> None:
        """Unload model, release resources."""

    @abstractmethod
    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute single inference (batched input)."""

    @abstractmethod
    async def infer_stream(self, inputs: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Streaming inference, yield intermediate results."""

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
- `asyncio.Queue` + timer in Master process
- Requests enqueue on arrival; batch fires when full or timer expires
- Variable-length inputs padded, original lengths recorded for unpadding
- Results split per-request and dispatched back to respective `asyncio.Future`s

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

---

## 7. HTTP/2 Service Layer

### 7.1 Tech Stack

- **Framework**: FastAPI (Pydantic validation, auto OpenAPI docs, dependency injection)
- **HTTP/2 Server**: Hypercorn (native HTTP/2 + SSE support)
- `serve()` transform compiles Graph into FastAPI ASGI app

### 7.2 API Endpoints

```
POST /v1/infer/{pipeline_name}          # Synchronous inference
POST /v1/infer/{pipeline_name}/stream   # Streaming inference (SSE)

GET  /v1/models                         # List loaded models
GET  /v1/models/{model_name}            # Model details + status
POST /v1/models/{model_name}/load       # Load model
POST /v1/models/{model_name}/unload     # Unload model

GET  /v1/health                         # Health check
GET  /v1/metrics                        # Prometheus metrics
```

### 7.3 Request/Response Format

```python
# Synchronous inference
POST /v1/infer/text_generation
Content-Type: application/json

{
    "inputs": {"text": "Hello, world!"},
    "parameters": {"max_tokens": 100, "temperature": 0.7}
}

# Response
{
    "id": "req_abc123",
    "outputs": {"text": "Hello! How can I help you?"},
    "metadata": {"latency_ms": 42, "tokens_generated": 8}
}

# Streaming — Server-Sent Events
POST /v1/infer/text_generation/stream

data: {"token": "Hello", "index": 0}
data: {"token": "!", "index": 1}
data: {"token": " How", "index": 2}
data: [DONE]
```

### 7.4 Streaming Input (HTTP/2)

Leverages HTTP/2 streaming request body for large file / multimedia input:

```
POST /v1/infer/speech_to_text/stream
Content-Type: application/octet-stream
Transfer-Encoding: chunked
```

---

## 8. Observability & Error Handling

### 8.1 Metrics (Prometheus)

```
# Request level
nerva_request_total{pipeline, status}
nerva_request_duration_seconds{pipeline, node}    # histogram
nerva_request_in_flight{pipeline}

# Batching
nerva_batch_size{model}
nerva_batch_wait_seconds{model}

# Worker
nerva_worker_status{model, device}
nerva_gpu_memory_used_bytes{device}

# Streaming
nerva_stream_tokens_total{pipeline}
nerva_stream_first_token_seconds{pipeline}        # TTFT
```

### 8.2 Logging

- `structlog` with structured JSON output
- Per-request `request_id` propagated across full pipeline chain
- Levels: ERROR (exceptions), WARNING (timeout/retry), INFO (request summary), DEBUG (node-level detail)

### 8.3 Error Handling

| Failure Type | Strategy |
|---|---|
| Single inference failure | Retry per node `retry()` transform config |
| Worker process crash | Worker Manager auto-restarts; in-flight requests return 503 |
| GPU OOM | Worker isolated, no impact on other models; metrics recorded, return 503 |
| Pipeline timeout | `timeout()` transform controls; downstream nodes cancelled on timeout |
| Model load failure | Model marked `error` state; health check reports anomaly |

---

## 9. Project Structure

```
nerva/
├── nerva/
│   ├── __init__.py              # Public API: model, trace, batch, stream, serve...
│   ├── core/
│   │   ├── graph.py             # DAG IR: Node, Edge, Graph
│   │   ├── proxy.py             # Proxy objects (for trace)
│   │   ├── transforms.py        # Function transforms: trace, batch, stream, retry, timeout, cache
│   │   └── primitives.py        # Control flow primitives: cond, switch, parallel, fori_loop
│   ├── engine/
│   │   ├── executor.py          # DAG execution engine (topological scheduling, asyncio)
│   │   ├── batcher.py           # Dynamic Batcher
│   │   └── scheduler.py         # Multi-GPU scheduling, load balancing
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
│   │   ├── app.py               # FastAPI application (serve transform output)
│   │   ├── routes.py            # API endpoints
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
| HTTP Server | FastAPI + Hypercorn | Native HTTP/2, familiar to algorithm engineers |
| Async Runtime | asyncio | Python standard, widest ecosystem |
| IPC | Unix Domain Socket + POSIX shm | Low latency, zero-copy for large tensors |
| Serialization | msgpack (control messages) | Faster than JSON, lighter than protobuf |
| Metrics | prometheus_client | Industry standard |
| Logging | structlog | Structured JSON, good performance |
| Type Checking | mypy (strict) | Catch bugs early |
| Linting | ruff | Fast, comprehensive rules |
| Testing | pytest + pytest-asyncio | Native async test support |
| Package Management | uv | Fast, modern Python toolchain |
