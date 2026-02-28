# Phase 4 Design — Binary RPC Serving Layer

## Goal

Expose Nerva pipelines via HTTP with a binary RPC protocol. Clients send framed requests (OPEN+DATA+END), the server routes to the correct pipeline, executes via Executor, and returns framed responses. Phase 4 covers unary mode only; streaming deferred to Phase 5.

## Architecture

```
Client
  │
  │  POST /rpc/{pipeline_name}  (Binary RPC, unary)
  │  GET  /v1/health
  │  GET  /v1/models
  ▼
┌─────────────────────────────────────────────┐
│  Starlette ASGI App (uvicorn)               │
│  ├── /rpc/{pipeline_name}  → RpcHandler     │
│  ├── /v1/health            → health_route   │
│  └── /v1/models            → models_route   │
├─────────────────────────────────────────────┤
│  PipelineRegistry                           │
│  { "classify": (Graph, {model→proxy}) }     │
├─────────────────────────────────────────────┤
│  WorkerManager (auto lifecycle)             │
│  ├── WorkerProxy("sentiment") → Worker[0]  │
│  └── WorkerProxy("encoder")   → Worker[1]  │
└─────────────────────────────────────────────┘
```

### Data Flow (unary request)

```
Client → OPEN frame + DATA frame(s) + END frame
       → RpcHandler decodes → deadline check
       → Executor.execute(inputs)
       → Response DATA frame + END frame → Client
```

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Payload encoding | msgpack | Already a project dependency; consistent with IPC layer; avoids protobuf compile chain |
| serve() API | Blocking startup (`serve(pipelines, host, port)`) | Simplest API; multi-pipeline via dict + URL routing |
| Multi-pipeline routing | Single entry `POST /rpc/{pipeline_name}` | Client selects pipeline via URL path; OPEN frame contains pipeline_name for validation |
| stream() transform | Deferred to Phase 5 | Real streaming need comes with vLLM integration |
| Management API | Basic: GET /v1/health + GET /v1/models | No dynamic load/unload in MVP |
| Worker management | Automatic inside serve() | serve() scans graphs, spawns workers, shuts down on exit |
| Deadline handling | Server converts absolute epoch → relative TTL | InferContext.deadline_ms is relative TTL; server layer does the conversion |

## Binary Protocol

### Frame Format (32-byte fixed header + variable payload)

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|    Magic (0x4E56)   | Version |  FrameType  |     Flags       |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                        Request ID (8 bytes)                   |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                  Stream ID (4B)             | Payload Len (4B)|
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                   CRC32 (4B)                | Ext Hdr Len (4B)|
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                     Payload (variable)                        |
```

### Frame Types

| Type | Value | Payload |
|---|---|---|
| OPEN | 1 | msgpack: `{"pipeline": "classify", "trace_id": "..."}` |
| DATA | 2 | msgpack: user inference data (dict or raw bytes) |
| END | 3 | msgpack: `{"status": 0}` or empty |
| ERROR | 4 | msgpack: `{"code": 4, "message": "...", "retryable": true}` |
| HEARTBEAT | 5 | none (not implemented in MVP) |

### MVP Simplifications

- Stream ID: fixed to 1 (no multiplexing)
- CRC32: fixed to 0 (no checksum)
- Ext Hdr Len: fixed to 0
- HEARTBEAT: not implemented

### Request Headers

- `x-nerva-request-id` (optional): decimal u64; server generates if absent
- `x-nerva-deadline-ms` (required): Unix epoch milliseconds
- `x-nerva-stream` (required): `0` (unary only in Phase 4)
- `x-nerva-trace-id` (optional): trace ID passthrough

### Error Codes

| Code | Name | Retryable | Trigger |
|---|---|---|---|
| 0 | OK | - | Success |
| 3 | INVALID_ARGUMENT | No | Bad frame, unknown pipeline |
| 4 | DEADLINE_EXCEEDED | Yes | Expired before/during execution |
| 8 | RESOURCE_EXHAUSTED | Yes | Queue full / backpressure |
| 13 | INTERNAL | Maybe | Unhandled execution error |

### protocol.py Interface

```python
@dataclass
class Frame:
    frame_type: FrameType
    request_id: int
    flags: int
    payload: bytes  # msgpack-encoded

def encode_frame(frame: Frame) -> bytes: ...
def decode_frame(data: bytes) -> tuple[Frame, int]: ...  # (frame, consumed_bytes)
```

## RPC Handler

### Unary State Machine

```
INIT → read frames → OPEN_RECV → read DATA → DATA_RECV → read END → RUNNING → RESPOND → HALT
```

### Handler Flow

1. Read complete frame stream from ASGI request body (OPEN + DATA + END)
2. Validate OPEN frame: pipeline_name exists in registry
3. Validate deadline: `x-nerva-deadline-ms` header, absolute epoch → relative TTL
4. Construct `InferContext` (request_id, deadline_ms, trace_id)
5. Call `Executor.execute(inputs)`
6. Success: respond with DATA frame + END frame
7. Failure: respond with ERROR frame (mapped error code)

### Error Mapping

- Pipeline not found → `INVALID_ARGUMENT(3)`
- Deadline already expired → `DEADLINE_EXCEEDED(4)`
- Exception containing `RESOURCE_EXHAUSTED` → `RESOURCE_EXHAUSTED(8)`
- Exception containing `DEADLINE_EXCEEDED` → `DEADLINE_EXCEEDED(4)`
- Other exceptions → `INTERNAL(13)`

## serve() API

```python
def serve(
    pipelines: dict[str, Graph],
    *,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
```

Internal flow:
1. Scan all Graphs, collect model_name → ModelHandle mapping
2. Via WorkerManager, auto-spawn a worker for each model
3. Construct Executor for each pipeline (graph + proxies)
4. Assemble Starlette ASGI app (RPC route + management routes)
5. `uvicorn.run(app, host, port)`
6. On shutdown, automatically cleanup all workers

## Management Routes

- `GET /v1/health` → `{"status": "ok"}`
- `GET /v1/models` → `{"models": [{"name": "sentiment", "backend": "pytorch", "device": "cpu"}]}`

## Testing Strategy

### Unit Tests

| File | Coverage |
|---|---|
| `test_protocol.py` | Frame encode/decode roundtrip, magic validation, payload size limit, all frame types |
| `test_rpc.py` | Unary full flow (mock Executor), unknown pipeline, deadline expired, exception→error code mapping |
| `test_serve.py` | Pipeline registry construction, model collection, ASGI app assembly |

### Integration Tests (`test_phase4_e2e.py`)

Using real Workers + httpx AsyncClient (ASGI transport, no real port):

1. Unary happy path: send OPEN+DATA+END → receive DATA+END, result correct
2. Unknown pipeline → ERROR frame INVALID_ARGUMENT
3. Deadline expired: `x-nerva-deadline-ms` set to past → ERROR frame DEADLINE_EXCEEDED
4. Health & models: GET /v1/health → 200, GET /v1/models → lists registered models

### Not in Phase 4 Test Scope

- Streaming mode (Phase 5)
- Cancel propagation (Phase 5, with vLLM)
- CRC32 checksum, HEARTBEAT
