# Full-Duplex Streaming Design

> 目标场景：Qwen3-Omni speech-to-speech 推理。客户端发送音频流，服务端边推理边返回音频/文本流。

## 1. 方案综述

### 1.1 问题定义

Qwen3-Omni speech-to-speech 推理场景要求：
- 客户端发送音频（1-10s），服务端流式返回文本+音频
- First-token-latency 要低（用户体感的"反应速度"）
- 中间 chunk 延迟要稳定（流畅播放不卡顿）

核心挑战：在 Nerva 现有的 Unary RPC 架构（请求-响应一对一）上，以最小变更支持流式输出。

### 1.2 传输层方案对比

| 方案 | 描述 | 优势 | 劣势 | 适配成本 |
|------|------|------|------|----------|
| **HTTP/2 StreamingResponse** | Starlette StreamingResponse, chunked transfer | 复用现有 ASGI 栈；帧协议天然支持多 DATA 帧 | 不支持真正的双向流（HTTP 请求体必须先发完） | 低 |
| WebSocket | 独立 WS 端点，双向消息 | 真正全双工；低 overhead | 需要新的连接管理、认证、状态机；与现有帧协议不兼容 | 高 |
| gRPC bidirectional streaming | protobuf + HTTP/2 | 成熟的流式语义；代码生成 | 引入 protobuf 编译链；与 Nerva 自有二进制协议冲突 | 高 |
| Server-Sent Events (SSE) | text/event-stream | 简单；浏览器友好 | 仅文本；无二进制支持；单向 | 不适用 |

**选择：HTTP/2 StreamingResponse**
- 与现有帧协议（OPEN/DATA/END）和 ASGI 栈完全一致
- protocol-v1 的状态机已预定义了 `RESP_STREAMING` 状态（`RUNNING → RESP_STREAMING → HALT`）
- MVP 不需要真正的双向流（音频输入 1-10s，缓冲后处理即可）

### 1.3 IPC 层方案对比

Worker 需要向 Proxy 发送多个 response chunk，三种方案：

| 方案 | 描述 | 优势 | 劣势 |
|------|------|------|------|
| **A: 复用 INFER_SUBMIT/ACK + 扩展字段** | `stream: bool` + `stream_done: bool` | 最小变更；向后兼容（`msg.get` 默认值） | 消息类型语义略有 overload |
| B: 新增消息类型 | INFER_STREAM_SUBMIT, STREAM_CHUNK, STREAM_END | 语义清晰；类型安全 | 增加 MessageType 枚举、dispatch 分支、测试面 |
| C: 独立数据通道 | 流式 chunk 走 ZMQ PUSH/PULL，控制消息走 ROUTER/DEALER | 数据面/控制面分离 | 连接管理复杂度翻倍；Worker 需要 bind 两个 socket |

**选择：方案 A**
- msgpack dict 天然支持可选字段，不存在版本兼容问题
- 语义本质不变：SUBMIT = 发起推理，ACK = 返回结果
- 最小变更原则：Proxy 的 recv_loop 只需增加一个 `isinstance` 分支

### 1.4 Executor 层方案对比

DAG 中的流式输出可以在不同粒度实现：

| 方案 | 描述 | 优势 | 劣势 |
|------|------|------|------|
| **A: 仅终端节点流式** | DAG 中间节点用 `infer()`，最后一个节点用 `infer_stream()` | 简单；DAG 调度逻辑不变 | 不能 pipeline 中间结果 |
| B: 节点间流式传递 | 上游节点 yield → 下游节点消费 AsyncIterator | 端到端流式延迟最优 | 极高复杂度；DAG 调度模型需重写（从 batch 到 streaming） |
| C: 全节点缓冲，仅 RPC 层分 chunk | Executor 仍返回完整结果，RPC 层切分后发送 | 零 Executor 变更 | 假流式——first-token-latency 等于整体推理延迟 |

**选择：方案 A**
- LLM/multimodal 场景中，流式输出的价值主要在最后一个生成节点
- 中间节点（如 embedding、预处理）通常是低延迟、非生成式的
- 未来如需节点间流式，可在方案 A 基础上渐进演进

## 2. 核心设计决策

### 2.1 传输层：HTTP/2 Streaming

使用 HTTP/2 的双向流式能力，基于 ASGI（Starlette + `StreamingResponse`）实现：

| 模式 | `x-nerva-stream` | 输入 | 输出 | 状态 |
|------|-------------------|------|------|------|
| Unary | `0` | 缓冲 | 缓冲 | **已实现** |
| Output-streaming | `1` | 缓冲 | 流式 | **本次实现** |
| Full-duplex | `2` | 缓冲（降级） | 流式 | **本次实现（输入降级）** |

> **MVP 中 `x-nerva-stream=2` 的行为**：RPC 层接受 `2` 并按 `1` 处理（`await request.body()` 缓冲输入，流式输出）。客户端可以 chunked 发送，服务端缓冲后处理。这是有意的优雅降级——客户端声明 full-duplex 意图，服务端尽力而为。真正的 streaming input（`request.stream()` 逐 chunk 解帧 + Backend API 变更）留待后续版本，届时客户端零改动。

帧格式不变。现有的 OPEN/DATA/END/ERROR/HEARTBEAT 帧类型天然支持流式：
- 客户端：`OPEN → DATA* → END`（chunked 发送）
- 服务端：`DATA* → END`（chunked 发送）

`flags` 字节（当前固定 0）保留用于流控扩展（如 `LAST_CHUNK` 标志），本次不使用。

### 2.2 Worker API：不修改 `infer()` 签名

**关键洞察**：流式场景分两个维度——流式输入和流式输出。两者可以独立处理。

#### 流式输入：在 RPC 层缓冲，不修改 Backend API

```
Client ──chunk──chunk──chunk──> RPC Handler (缓冲) ──complete dict──> Executor ──> Worker
```

- RPC Handler 通过 `request.stream()` 增量读取客户端帧
- 积累所有 DATA 帧后组装为完整 `inputs: dict`
- 然后正常调用 `executor.execute()` 或 `executor.execute_stream()`

**优势**：
- **零 API 变更**：Backend.infer() / infer_stream() 的签名不变
- **完全兼容**：Executor、Batcher、WorkerProxy、Worker 的非流式路径不受影响
- **简化实现**：一次只需要关心输出流式

**劣势**：
- 无法实现「边收边推理」——必须等完整输入到达后才开始推理
- 对于长音频，first-token-latency 线性增加

**为什么 MVP 可以接受**：
1. Qwen3-Omni 的 audio encoder 需要完整音频段才能编码（不是 token 级流式输入）
2. 典型 speech2speech 的输入音频为 1-10s，传输延迟 <100ms，远小于推理延迟
3. vLLM 的 multimodal input 也是缓冲后一次性传入
4. 真正的流式输入（VAD 分段 → 增量推理）是 model 层的能力，不是 serving 框架的核心职责

#### 流式输出：使用已有的 `infer_stream()` ABC

```python
# 已存在于 backends/base.py
@abstractmethod
async def infer_stream(
    self, inputs: dict[str, Any], context: InferContext,
) -> AsyncIterator[dict[str, Any]]:
    """Streaming inference — yield intermediate results."""
```

PyTorchBackend 和 VLLMBackend 都已实现此接口。不需要任何修改。

### 2.3 DynamicBatcher：不参与流式路径

用户明确：流式场景不需要 batch 合并。

- Batcher 仍然服务于非流式高吞吐场景（如 embedding 计算）
- 流式请求绕过 Batcher，直接走 Executor → WorkerProxy → Worker

### 2.4 IPC：复用 INFER_SUBMIT / INFER_ACK，不新增消息类型

**关键洞察**：流式输出只是「同一个 request_id 产生多个 INFER_ACK」。

当前（非流式）：
```
Proxy → Worker:  INFER_SUBMIT { request_id, descriptor, ... }
Worker → Proxy:  INFER_ACK    { request_id, status=OK, descriptor }     ← 单次
```

流式（lookahead 模式，最后一个数据 ACK 携带 `stream_done: true`）：
```
Proxy → Worker:  INFER_SUBMIT { request_id, descriptor, stream: true, ... }
Worker → Proxy:  INFER_ACK    { request_id, status=OK, descriptor, stream_done: false }  ← chunk 1
Worker → Proxy:  INFER_ACK    { request_id, status=OK, descriptor, stream_done: false }  ← chunk 2
...
Worker → Proxy:  INFER_ACK    { request_id, status=OK, descriptor, stream_done: true }   ← 最后一个（带数据）
```

不变量：`status=OK` 的 INFER_ACK **总是**携带 `descriptor`（无论 unary 还是 streaming）。
例外：空流（Backend 未 yield 任何 chunk）时发送 `stream_done: true` 无 descriptor。

**变更点**：
- `INFER_SUBMIT` 增加 `stream: bool` 字段（默认 false，向后兼容）
- `INFER_ACK` 增加 `stream_done: bool` 字段（默认 true，向后兼容）
- 不新增 MessageType 枚举值

**不需要新消息类型的原因**：
1. 语义不变——INFER_SUBMIT 仍是「发起推理」，INFER_ACK 仍是「推理结果」
2. 序列化格式是 msgpack dict，天然支持可选字段
3. Worker 和 Proxy 都通过 `msg.get("stream", False)` 读取，不存在无法处理的版本

## 3. 流式架构设计

### 3.1 分层架构

```
┌─────────────────────────────────────────────────────────────────────┐
│  Client                                                             │
│  x-nerva-stream: 0 (Unary) │ 1 (Output-Stream) │ 2 (Full-Duplex)  │
└──────────────────┬──────────────────┬──────────────────────────────┘
                   │                  │
┌──────────────────▼──────────────────▼──────────────────────────────┐
│  RPC Layer (server/rpc.py)                                         │
│                                                                    │
│  stream=0 → handle()         → Response(DATA+END)                  │
│  stream=1 → _handle_stream() → StreamingResponse(DATA*+END)       │
│  stream=2 → _handle_stream() → (buffer input) + StreamingResponse │
│                                                                    │
│  ★ 分流点：根据 x-nerva-stream header 路由                         │
└──────────────────┬──────────────────┬──────────────────────────────┘
                   │                  │
              execute()        execute_stream()
                   │                  │
┌──────────────────▼──────────────────▼──────────────────────────────┐
│  Executor Layer (engine/executor.py)                               │
│                                                                    │
│  execute()         → 全节点 infer()，返回 dict                     │
│  execute_stream()  → 中间节点 infer()，终端节点 infer_stream()     │
│                      → yield AsyncIterator[dict]                   │
│                                                                    │
│  ★ 分流点：终端节点判断 proxy 是否实现 InferableStreamProxy        │
└──────────────────┬──────────────────┬──────────────────────────────┘
                   │                  │
              infer()          infer_stream()
                   │                  │
┌──────────────────▼──────────────────▼──────────────────────────────┐
│  WorkerProxy Layer (worker/proxy.py)                               │
│                                                                    │
│  infer()         → Future[dict]  (1:1 request/response)            │
│  infer_stream()  → Queue → yield (1:N request/response)            │
│                                                                    │
│  _pending 类型: Future (unary) | Queue (stream)                    │
│  recv_loop: isinstance dispatch                                    │
└──────────────────┬──────────────────┬──────────────────────────────┘
                   │                  │
            INFER_SUBMIT        INFER_SUBMIT
            {stream:false}      {stream:true}
                   │                  │
              ─────┼── ZMQ ipc:// ────┼─────
                   │                  │
            INFER_ACK ×1       INFER_ACK ×N
                   │           {stream_done:false}*
                   │           {stream_done:true}
                   │                  │
┌──────────────────▼──────────────────▼──────────────────────────────┐
│  Worker Process (worker/process.py)                                │
│                                                                    │
│  _handle_infer()        → backend.infer()        → 1× INFER_ACK   │
│  _handle_infer_stream() → backend.infer_stream() → N× INFER_ACK   │
│                                                                    │
│  ★ 分流点：msg.get("stream", False)                                │
└──────────────────┬──────────────────┬──────────────────────────────┘
                   │                  │
               infer()          infer_stream()
                   │                  │
┌──────────────────▼──────────────────▼──────────────────────────────┐
│  Backend (backends/base.py)                                        │
│                                                                    │
│  infer()         → dict[str, Any]                                  │
│  infer_stream()  → AsyncIterator[dict[str, Any]]                   │
│                                                                    │
│  PyTorchBackend / VLLMBackend 已实现两个方法                        │
└────────────────────────────────────────────────────────────────────┘
```

### 3.2 Streaming 与 Unary 的对称性

每一层都保持对称的 unary/stream 双路径，分流由上层传入的 `stream` 标志决定：

| 维度 | Unary | Streaming |
|------|-------|-----------|
| RPC 入口 | `handle()` | `_handle_stream()` |
| Executor 方法 | `execute()` → `dict` | `execute_stream()` → `AsyncIterator[dict]` |
| Proxy 方法 | `infer()` → `Future` | `infer_stream()` → `Queue` → yield |
| IPC 消息 | INFER_SUBMIT → 1× INFER_ACK | INFER_SUBMIT{stream} → N× INFER_ACK |
| Worker 处理 | `_handle_infer()` | `_handle_infer_stream()` |
| Backend 方法 | `infer()` | `infer_stream()` |
| 响应格式 | `Response(DATA+END)` | `StreamingResponse(DATA*+END)` |

### 3.3 边界情况与约束

#### SHM 策略

流式 chunk 通常较小（LLM token < 1KB，audio chunk ~几 KB），大概率走 inline 路径（< `IPC_CONTROL_INLINE_MAX_BYTES` = 8KB 阈值）。

**MVP 决策**：流式 chunk **禁止使用 SHM**，Worker 侧 `_handle_infer_stream()` 直接构造 inline Descriptor，不调用 `_build_output_descriptor()`。

**禁止而非 fallback 的原因**——`_output_slots` 存在竞态：

```
_output_slots[request_id] 只存一个 (pool, slot) tuple。
如果 Worker 连续发送两个 SHM chunk：

1. Worker 发 chunk 1 INFER_ACK (shm_id=A)
2. Worker 立即开始 chunk 2, 发 SHM_ALLOC_REQUEST
3. Proxy recv_loop 收到 chunk 1 ACK → put 进 Queue
4. Proxy recv_loop 收到 chunk 2 SHM_ALLOC_REQUEST
   → _handle_shm_alloc_request() 调用 _release_output_slot(request_id)
   → 释放了 chunk 1 的 slot A（此时 infer_stream 还没消费 chunk 1！）
5. infer_stream() 从 Queue 取 chunk 1 → _decode_output → 读已释放的 SHM → use-after-free
```

依赖 `no_shm_pool_fallback` 不够安全——它只在 Proxy 端没有 shm_pool 时生效，而 `_request_pools` 仍然会为流式请求关联 pool（如果调用方传入了 shm_pool 参数）。必须在 Worker 侧主动阻断 SHM 路径。

**后续优化**：如果 audio chunk 稳定超过 8KB，需要重新设计 SHM 分配模型。两个方向：
- (a) `_output_slots` 从 `dict[str, (pool, slot)]` 改为 `dict[str, deque[(pool, slot)]]`，支持同一 request_id 多个并发 slot
- (b) stream 开始时预分配 ring buffer SHM slot，Worker 循环写入、Proxy 循环读取，避免 per-chunk alloc/free

#### Sync dispatch 路径

Worker 当前有两个 dispatch 路径：
- `_async_dispatch = True` → 直接 `await backend.infer()`
- `_async_dispatch = False` → `loop.run_in_executor(self._thread_executor, ...)`

`infer_stream()` 返回 `AsyncIterator`，无法在 `ThreadPoolExecutor` 中使用。

**设计约束**：流式路径**强制使用 async dispatch**。Worker 收到 `stream=true` 的 INFER_SUBMIT 时，无论 `_async_dispatch` 设置如何，直接执行 `async for chunk in backend.infer_stream(inputs, context)`。如果 Backend 内部需要调用同步代码，由 Backend 自行 `await asyncio.to_thread()`。

#### 流式中途错误处理

```
正常流:  ACK{ok,done=false} → ACK{ok,done=false} → ... → ACK{ok,done=true}
中途错误: ACK{ok,done=false} → ACK{ok,done=false} → ACK{INTERNAL,done=true}
```

- **Worker 层**：`async for` 循环中 Backend 抛异常 → 捕获后发送 `INFER_ACK{status=INTERNAL/DEADLINE_EXCEEDED, stream_done=true, error=...}`
- **Proxy 层**：收到 `status != OK` 的 ACK → 从 Queue 取出后抛 `RuntimeError` → 终止 yield
- **Executor 层**：`execute_stream()` 的 `async for` 被异常中断 → propagate
- **RPC 层**：`generate()` **内部 catch 异常** → yield ERROR frame → **正常 return**（不 re-raise）

  **为什么必须 catch 而非 propagate**：`StreamingResponse` 通过 `async for chunk in body_iterator` 消费 generator。如果异常从 generator 传播出去，Starlette 直接将异常交给 ASGI server（uvicorn），uvicorn 关闭 TCP 连接。此时 HTTP 响应头（status=200）和之前的 DATA frame 已经在 wire 上，**没有机会再 yield ERROR frame 或 END frame**。客户端看到的是：若干 DATA frame → 连接断开（无结构化错误信息）。

  正确做法：generator 内部 try/except → yield ERROR frame → return。Generator 正常结束，Starlette 发送 `more_body=False` 关闭响应。客户端能收到完整的 ERROR frame 并解析错误码。

  **边界情况**：如果客户端在 ERROR frame yield 之前已断连（ASGI send 失败），ERROR frame 丢失。这是 best-effort——客户端通过连接断开 + 缺少 END frame 也能判断流异常终止。

**关键原则**：
1. 无论正常还是异常，最后一个 INFER_ACK 的 `stream_done` 必须为 `true`，确保 Proxy 不会永久阻塞在 `queue.get()` 上。
2. `generate()` **永远不让异常传播出 generator**——要么 yield END（正常），要么 yield ERROR（异常），然后 return。

#### 取消传播

同 Unary 路径：`context.cancelled = True` + `task.cancel()`。

流式特有考虑：Backend 的 `infer_stream()` 在每次 `yield` 前应检查 `context.cancelled`。Worker 的 `async for` 被 `CancelledError` 中断后，发送 `INFER_ACK{status=ABORTED, stream_done=true}`。

#### 背压

**MVP 不实现显式背压。** 理由：
- LLM token 生成速度（~50 tokens/s）远低于 ZMQ IPC 传输能力
- ZMQ ROUTER socket 的 HWM (High Water Mark) 提供隐式缓冲（默认 1000 条消息）
- HTTP/2 的 flow control 在传输层提供了 client → server 的背压

**风险场景**：如果 audio chunk 生成速度极快且 chunk 较大，ZMQ 缓冲可能增长。后续版本可引入基于 credit 的流控（Proxy 发送 credit → Worker 按 credit 发送 chunk）。

#### MultiInstanceProxy

需要增加 `infer_stream()` 方法。实现：round-robin 选择一个 `WorkerProxy`，然后调用其 `infer_stream()` 并直接 yield。

**约束**：一个流式请求全程绑定同一个 Worker 实例，不能中途切换。

#### 指标采集

与 unary 路径一致，分两套采集机制：

**RPC 层（Prometheus `/metrics`）**：在 uvicorn 进程内，通过 `NervaMetrics` 采集 HTTP 可见延迟：

| 指标名 | 类型 | 描述 |
|--------|------|------|
| `nerva_stream_time_to_first_chunk_seconds` | Histogram | 从请求进入到 `generate()` yield 第一个 DATA frame |
| `nerva_stream_inter_chunk_latency_seconds` | Histogram | `generate()` 中相邻 yield 间延迟 |
| `nerva_stream_chunks_total` | Counter | 每个请求 yield 的 chunk 数 |
| `nerva_stream_duration_seconds` | Histogram | 流式响应总时长（首个 yield 到最后一个 yield） |

**Worker 层（`AsyncTimingSink` 文件日志）**：在 Worker 子进程内，扩展现有 `_timing_sink` 写入 `nerva_worker_{pid}.log`，记录推理侧延迟（不含 IPC overhead）：

```python
# _handle_infer_stream 中写入
self._timing_sink.write({
    "event": "infer_stream_timing",
    "request_id": request_id,
    "model": self._model_name,
    "chunk_count": chunk_count,
    "first_chunk_ms": ...,       # backend.infer_stream() 首次 yield 耗时
    "total_stream_ms": ...,      # 整个 async for 循环耗时
})
```

Worker 是独立子进程，没有 HTTP 端点，prometheus_client 的 counter/histogram 是 process-local 的。现有 unary 路径也是同样模式（Worker timing → 文件日志，RPC timing → Prometheus）。

## 4. 各层 API 定义

### 4.1 RPC Layer（server/rpc.py）

```python
# 新增：流式响应处理
async def _handle_stream(
    self, request: Request, pipeline_name: str, request_id: int, deadline_ms: int,
) -> StreamingResponse:
    """Handle x-nerva-stream=1 (output streaming) and x-nerva-stream=2 (full-duplex, input degraded)."""

    # 1. 读取输入（缓冲模式，stream=2 也走此路径——优雅降级）
    #    await request.body() 在 chunked transfer 下仍然正确：Starlette 持续
    #    读取直到客户端关闭请求体（HTTP/2 END_STREAM），返回拼接后的完整 bytes。
    body = await request.body()
    frames = _parse_frames(body)
    ...  # 同 unary path 的帧解析

    # 2. 构造 Executor（同 unary path，context 包含 deadline_ms / request_id）
    #    pipeline wrapper 内部创建 Executor(graph, proxies, context)
    #    调用签名与 execute() 一致：仅传 inputs
    async def generate() -> AsyncIterator[bytes]:
        # ★ 异常必须在 generator 内部 catch，不能传播出去。
        #   原因：StreamingResponse 对 generator 异常的处理是直接关闭连接，
        #   此时 HTTP 200 + 之前的 DATA frame 已在 wire 上，无法再发 ERROR。
        try:
            async for chunk in executor.execute_stream(inputs):
                yield encode_frame(Frame(FrameType.DATA, request_id, 0,
                                         msgpack.packb(chunk, use_bin_type=True)))
            yield encode_frame(Frame(FrameType.END, request_id, 0,
                                     msgpack.packb({"status": 0}, use_bin_type=True)))
        except Exception as exc:
            code, message = _map_exception(exc)
            yield encode_frame(Frame(FrameType.ERROR, request_id, 0,
                                     msgpack.packb({"code": int(code), "message": message},
                                                   use_bin_type=True)))
            # 不 re-raise：generator 正常 return → Starlette 正常关闭响应

    return StreamingResponse(generate(), media_type=CONTENT_TYPE)
```

### 4.2 Executor Layer（engine/executor.py）

**签名对齐**：`execute_stream()` 与 `execute()` 保持一致——`deadline_ms` 和 `request_id` 已在构造时通过 `self._context: InferContext` 传入，不在方法签名中重复。输入类型也保持 `Any`（与 `execute()` 一致），因为 source 节点的输入可以是任意结构。

```python
# 现有（不变）
class InferableProxy(Protocol):
    async def infer(
        self, inputs: dict[str, Any], context: InferContext, **kwargs: Any,
    ) -> dict[str, Any]: ...

# 新增：继承 InferableProxy，添加 infer_stream()
@runtime_checkable
class InferableStreamProxy(InferableProxy, Protocol):
    """Protocol for proxies that support both unary and streaming inference.

    继承关系：InferableStreamProxy is-a InferableProxy。
    - 中间节点通过 InferableProxy.infer() 调用（非流式）
    - 终端节点通过 InferableStreamProxy.infer_stream() 调用（流式）

    @runtime_checkable 允许 execute_stream() 做 isinstance 检查，
    在终端节点 proxy 不支持流式时给出明确 TypeError。
    """
    async def infer_stream(
        self, inputs: dict[str, Any], context: InferContext, **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]: ...
```

**实现方**：`WorkerProxy` 和 `MultiInstanceProxy` 都实现 `InferableStreamProxy`（同时有 `infer()` 和 `infer_stream()`）。

**Executor 签名不变**：`__init__` 仍接收 `dict[str, InferableProxy]`。`execute_stream()` 在终端节点做运行时窄化：

```python
# Executor 新增方法（签名与 execute() 对称）：
async def execute_stream(self, inputs: Any = None) -> AsyncIterator[dict[str, Any]]:
    """Execute pipeline, streaming the terminal node's output.

    与 execute() 的区别仅在终端节点：
    - execute()        → proxy.infer()        → 返回 dict
    - execute_stream() → proxy.infer_stream() → yield dict chunks

    中间节点仍使用 infer()（非流式）。

    deadline_ms / request_id 来自 self._context（构造时传入），
    与 execute() 一致。
    """
    ...
    # 终端节点运行时检查
    terminal_proxy = self._proxies[terminal_node.model_name]
    if not isinstance(terminal_proxy, InferableStreamProxy):
        raise TypeError(
            f"Proxy for '{terminal_node.model_name}' does not implement "
            f"InferableStreamProxy — cannot use execute_stream(). "
            f"Either use execute() or register a streaming-capable proxy."
        )
    # 中间节点正常 infer()，终端节点 infer_stream()
    async for chunk in terminal_proxy.infer_stream(node_inputs, node_ctx):
        yield chunk
```

### 4.3 WorkerProxy Layer（worker/proxy.py）

#### 数据结构变更：拆分 `_pending` 为两个 dict

**不使用 `Future | Queue` 类型联合**。原因：
1. `_fail_outstanding()` 需要对 Future 调用 `set_exception()`、对 Queue 注入 poison pill——类型联合使得每个操作都要 isinstance 分支
2. `_output_slots` 清理逻辑在 unary 和 stream 路径中语义不同（见 §3.3 SHM 竞态分析）
3. mypy strict 模式下 `dict[str, Future | Queue]` 会导致每个 `.get()` 返回值都需要 narrow

```python
# 现有（不变）
self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}

# 新增
self._pending_stream: dict[str, asyncio.Queue[dict[str, Any]]] = {}
```

#### `infer_stream()` 完整实现（含 try/finally 资源清理）

```python
async def infer_stream(
    self,
    inputs: dict[str, Any],
    context: InferContext,
    shm_pool: ShmPool | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Send INFER_SUBMIT with stream=True, yield INFER_ACK chunks."""
    request_id = context.request_id

    # ── 序列化输入（同 infer()）──
    raw_bytes_input = self._extract_raw_bytes_input(inputs)
    if raw_bytes_input is None:
        input_key = None
        payload_codec = "msgpack_dict_v1"
        input_bytes = msgpack.packb(inputs, use_bin_type=True)
    else:
        input_key, input_bytes = raw_bytes_input
        payload_codec = "raw_bytes_v1"

    shm_slot = None
    if len(input_bytes) <= IPC_CONTROL_INLINE_MAX_BYTES or shm_pool is None:
        descriptor = Descriptor(
            request_id=request_id, node_id=0,
            inline_data=input_bytes, length=len(input_bytes),
            payload_codec=payload_codec, input_key=input_key,
        )
    else:
        shm_slot = shm_pool.alloc(len(input_bytes))
        shm_pool.write(shm_slot, input_bytes)
        descriptor = Descriptor(
            request_id=request_id, node_id=0,
            shm_id=shm_slot.shm_name, offset=shm_slot.offset,
            length=len(input_bytes),
            payload_codec=payload_codec, input_key=input_key,
        )

    # ── 注册 Queue + 发送 ──
    if request_id in self._pending or request_id in self._pending_stream:
        raise RuntimeError(f"Duplicate in-flight request_id '{request_id}'")

    chunk_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    self._pending_stream[request_id] = chunk_queue
    self._request_pools[request_id] = shm_pool

    try:
        await self._send({
            "type": MessageType.INFER_SUBMIT.value,
            "request_id": request_id,
            "descriptor": descriptor.to_dict(),
            "deadline_ms": context.deadline_ms,
            "stream": True,
        })

        # ── yield chunks 直到 stream_done=True ──
        while True:
            ack = await asyncio.wait_for(
                chunk_queue.get(), timeout=self._submit_timeout,
            )
            status = ack.get("status", "")
            stream_done = ack.get("stream_done", True)

            if status != AckStatus.OK.value:
                error = ack.get("error", "unknown error")
                raise RuntimeError(f"Stream failed: [{status}] {error}")

            # 不变量：status=OK 的 ACK 总是带 descriptor（lookahead 保证）。
            # 仅空流（Backend 未 yield 任何 chunk）的终止 ACK 无 descriptor。
            if "descriptor" in ack:
                out_descriptor = Descriptor.from_dict(ack["descriptor"])
                output = self._decode_output(out_descriptor, request_id)
                yield output

            if stream_done:
                break
    finally:
        # ── 资源清理（对标 infer() 的 finally 块 L259-265）──
        self._pending_stream.pop(request_id, None)
        self._request_pools.pop(request_id, None)
        self._mark_request_completed(request_id)
        self._release_output_slot(request_id)  # 防御性清理
        if shm_slot is not None and shm_pool is not None:
            shm_pool.free(shm_slot)
```

#### `_fail_outstanding()` 变更

```python
def _fail_outstanding(self, reason: str) -> None:
    # ── 现有：fail unary futures ──
    for fut in self._pending.values():
        if not fut.done():
            fut.set_exception(RuntimeError(reason))
    self._pending.clear()

    # ── 新增：poison-pill streaming queues ──
    # 注入一条 status=UNAVAILABLE 的终止消息，
    # infer_stream() 的 while 循环会在 queue.get() 后检查 status 并抛异常。
    for q in self._pending_stream.values():
        q.put_nowait({
            "status": AckStatus.UNAVAILABLE.value,
            "error": reason,
            "stream_done": True,
        })
    self._pending_stream.clear()

    # ... 其余清理逻辑不变（_output_slots, _request_pools, 等）
```

#### `recv_loop` 变更

```python
elif msg_type == MessageType.INFER_ACK.value:
    request_id = msg.get("request_id", "")

    # 先查 stream dict（新路径）
    q = self._pending_stream.get(request_id)
    if q is not None:
        await q.put(msg)
    else:
        # 再查 unary dict（现有行为）
        fut = self._pending.get(request_id)
        if fut is not None and not fut.done():
            fut.set_result(msg)
        else:
            if request_id in self._recently_completed_set:
                logger.debug("Late INFER_ACK for completed request '%s'", request_id)
            else:
                logger.warning("No pending future for request '%s'", request_id)
```

### 4.4 Worker Process Layer（worker/process.py）

```python
async def _handle_infer(self, msg: dict[str, Any], client_id: bytes) -> None:
    stream = msg.get("stream", False)
    ...
    if stream:
        await self._handle_infer_stream(msg, client_id, request_id, inputs, context)
    else:
        # 现有非流式逻辑（不变）
        ...

async def _handle_infer_stream(
    self, msg, client_id, request_id, inputs, context
) -> None:
    """Call backend.infer_stream(), send multiple INFER_ACK chunks.

    流式 chunk 强制走 inline Descriptor，不调用 _build_output_descriptor()。
    原因：_output_slots[request_id] 只存一个 (pool, slot)，连续 chunk 的
    SHM_ALLOC_REQUEST 会释放前一个尚未被 Proxy 消费的 slot（use-after-free）。
    详见 §3.3 SHM 策略。
    """
    # ★ _handle_infer_stream 自己 catch 所有异常，不让异常传播到
    #   _handle_infer 的外层 CancelledError/Exception handler——
    #   外层 handler 没有 stream_done 字段，会导致 Proxy 依赖隐式默认值。
    try:
        # ── Lookahead: 缓冲一个 chunk，使最后一个 ACK 能携带 descriptor ──
        # 保持不变量：status=OK 的 INFER_ACK 总是带 descriptor。
        # 代价：最后一个 chunk 多等一个迭代周期（~20ms，LLM token 间隔级别）。
        prev_bytes: bytes | None = None
        async for chunk in self._backend.infer_stream(inputs, context):
            output_bytes = msgpack.packb(chunk, use_bin_type=True)
            if prev_bytes is not None:
                # 发送前一个 chunk（确认不是最后一个）
                desc = Descriptor(
                    request_id=request_id, node_id=0,
                    inline_data=prev_bytes, length=len(prev_bytes),
                    payload_codec="msgpack_dict_v1",
                )
                await self._send_to(client_id, {
                    "type": MessageType.INFER_ACK.value,
                    "request_id": request_id,
                    "status": AckStatus.OK.value,
                    "descriptor": desc.to_dict(),
                    "stream_done": False,
                })
            prev_bytes = output_bytes

        # 迭代器耗尽——prev_bytes 就是最后一个 chunk
        if prev_bytes is not None:
            desc = Descriptor(
                request_id=request_id, node_id=0,
                inline_data=prev_bytes, length=len(prev_bytes),
                payload_codec="msgpack_dict_v1",
            )
            await self._send_to(client_id, {
                "type": MessageType.INFER_ACK.value,
                "request_id": request_id,
                "status": AckStatus.OK.value,
                "descriptor": desc.to_dict(),
                "stream_done": True,
            })
        else:
            # 空流（Backend 未 yield 任何 chunk）——发送纯终止信号
            await self._send_to(client_id, {
                "type": MessageType.INFER_ACK.value,
                "request_id": request_id,
                "status": AckStatus.OK.value,
                "stream_done": True,
            })

    except asyncio.CancelledError:
        if context is not None:
            context.cancelled = True
        await self._send_to(client_id, {
            "type": MessageType.INFER_ACK.value,
            "request_id": request_id,
            "status": AckStatus.ABORTED.value,
            "error": "request cancelled",
            "stream_done": True,
        })
    except Exception as exc:
        await self._send_to(client_id, {
            "type": MessageType.INFER_ACK.value,
            "request_id": request_id,
            "status": AckStatus.INTERNAL.value,
            "error": str(exc),
            "stream_done": True,
        })
```

## 5. 数据流全景

### 5.1 Output-Streaming（x-nerva-stream=1）

```
Client                    RPC Handler              Executor        WorkerProxy           Worker
  |── POST /rpc/pipe ───>|                         |                |                      |
  |   OPEN+DATA+END      |── parse frames ────────>|                |                      |
  |   (buffered body)     |                         |── execute_stream()                    |
  |                       |                         |── infer_stream()                      |
  |                       |                         |                |── INFER_SUBMIT ──────>|
  |                       |                         |                |   {stream: true}      |
  |                       |                         |                |                      |
  |                       |                         |                |<── INFER_ACK ────────|
  |                       |                         |                |   {stream_done:false} |
  |<── DATA frame ───────|<── chunk ──────────────|<── yield ──────|                      |
  |                       |                         |                |<── INFER_ACK ────────|
  |<── DATA frame ───────|<── chunk ──────────────|<── yield ──────|   {stream_done:false} |
  |                       |                         |                |<── INFER_ACK ────────|
  |<── END frame ────────|<── done ───────────────|<── done ───────|   {stream_done:true}  |
```

### 5.2 Full-Duplex（x-nerva-stream=2）

**MVP 行为**：输入侧降级为缓冲，与 5.1 数据流一致。客户端以 chunked transfer 发送，RPC 层 `await request.body()` 缓冲完整后再进入 execute_stream()。

```
Client                    RPC Handler              Executor        ...
  |── OPEN ──────────────>|                         |
  |── DATA chunk 1 ──────>| (accumulate)            |
  |── DATA chunk 2 ──────>| (accumulate)            |
  |── END ───────────────>| ── assemble inputs ────>| ... (same as 5.1)
  |<── DATA frame ───────|<── ...
```

**未来真正的 Full-Duplex** 需要：
1. RPC 层使用 `async for chunk in request.stream()` 逐 chunk 解帧（替代 `await request.body()`）
2. Backend API 变更：`infer_stream()` 接受 `AsyncIterator[dict]` 输入（替代 `dict`）
3. Executor 支持输入/输出同时流式的节点调度模型

## 6. 实现计划

### Phase 1: IPC + Worker（无外部依赖变更）
1. `worker/process.py`: 新增 `_handle_infer_stream()`，根据 `msg["stream"]` 分发
2. `worker/proxy.py`: 新增 `infer_stream()`，recv_loop 支持 Queue 模式
3. 单元测试：mock backend 验证多 ACK 流

### Phase 2: Executor 流式路径
4. `engine/executor.py`: 新增 `execute_stream()` 方法
5. 新增 `InferableStreamProxy` 协议
6. 单元测试：DAG 终端节点流式输出

### Phase 3: RPC + 端到端
7. `server/rpc.py`: `x-nerva-stream` 为 `1` 或 `2` 时路由到 `_handle_stream()`（`2` 按 `1` 处理，输入缓冲降级）
8. 使用 Starlette `StreamingResponse` 逐帧发送
9. 端到端测试：mock streaming backend → HTTP 客户端接收流式帧

### Phase 4: 集成验证
10. Mock streaming model（模拟 Qwen3-Omni 的 chunk-by-chunk 输出）
11. 压测：first-token-latency、chunk 间延迟、吞吐量

## 7. 关键约束

- **向后兼容**：所有 `x-nerva-stream=0` 请求行为完全不变
- **IPC 兼容**：新增字段使用 `msg.get("stream", False)` 读取，旧 Worker 忽略新字段
- **不修改 Frame 格式**：帧头 32 字节布局不变
- **不修改 Backend API**：`infer()` 和 `infer_stream()` 签名不变
- **不修改 DynamicBatcher**：流式请求绕过 batcher

## 8. 待讨论事项

1. **protocol-v1.md 更新**：当前协议文档 `x-nerva-stream` 只定义了 `0` 和 `1`，但 RPC 实现只接受 `0`。Phase 3 需要更新协议文档：`1`（Output-streaming）正式化，`2`（Full-duplex）标注为"accepted, input buffered in v1"。

2. **`execute_stream()` 的 fallback**：如果 RPC 收到 `x-nerva-stream=1` 但终端节点的 proxy 不实现 `InferableStreamProxy`，应该报错还是 fallback 到 unary？建议报错（`ERROR(INVALID_ARGUMENT)`），因为 silent fallback 会让客户端误以为收到了完整流。

3. **Heartbeat during long streams**：protocol-v1 定义了 HEARTBEAT 帧类型（当前未使用）。对于长时间的流式响应（如生成长篇文本），chunk 间隔可能较长（几秒），是否需要插入 HEARTBEAT 帧维持连接活性？MVP 可以不实现，但需要确认 HTTP keep-alive 超时配置。

4. **~~`_pending` 类型联合~~**（已决策）：拆为 `_pending`（Future，不变）+ `_pending_stream`（Queue，新增）。理由见 §4.3。`_fail_outstanding` 对 Queue 注入 poison pill 而非调用 `set_exception`；`recv_loop` 先查 `_pending_stream` 再查 `_pending`。
