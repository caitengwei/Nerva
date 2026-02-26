# Nerva v0.1 设计文档

> 以 Python 为先理念的高性能模型推理服务框架。

## 1. 概述

### 1.1 什么是 Nerva

Nerva 是一个高性能模型推理服务框架，定位为 Triton Inference Server 的灵活替代方案。它在多个推理后端之上提供统一编排层，并向算法工程师暴露简洁、Python 原生的 API。

### 1.2 设计目标

- **多模型编排**：支持 DAG 拓扑的复杂推理流水线
- **高性能单模型服务**：动态批处理、异步调度、多 GPU
- **统一后端抽象**：插件式后端（先支持 vLLM、PyTorch；后续 ONNX、TensorRT）
- **Python 优先**：算法工程师可用最少模板代码定义与部署模型
- **JAX 风格函数变换**：Pipeline = 纯函数；能力通过可组合 transforms 注入

### 1.3 目标用户

希望将模型部署到生产环境、但不想处理复杂基础设施细节的算法工程师。

### 1.4 MVP 范围

在单机多 GPU 上构建可工作的低延迟多模型编排流水线：
用户定义模型和 pipeline 函数，应用 transforms（`trace`、`batch`、`stream`），并通过 ASGI 二进制流式 RPC 提供服务。

**MVP 包含：**
- `model()`、`trace()`、`batch()`、`stream()`、`serve()` 核心 transforms
- `nerva.cond()`、`nerva.parallel()` 控制流原语
- PyTorch Backend + vLLM Backend
- Master-Worker 多进程架构
- 具备 deadline 感知的动态批处理
- Starlette + ASGI 二进制流式 RPC 数据面
- 基于 FastAPI 的管理 API（模型加载/卸载、健康检查、指标）
- 请求级 deadline / cancel / backpressure
- 低延迟 IPC 路径：UDS 控制通道 + POSIX shm 描述符传递
- Prometheus 指标 + structlog 日志

**MVP 不包含（后续迭代）：**
- YAML 声明式 pipeline 配置
- `fori_loop`、`switch` 控制流原语
- `cache()`、`timeout()`、`retry()` transforms
- `model_group` 负载均衡
- 分布式集群部署
- 模型仓库 / 版本管理
- 客户端 SDK

### 1.5 PoC 成功标准

- 验证单机多 GPU 下的多模型 DAG 编排能力
- 在目标并发下达到低尾延迟（`p95` / `p99`）
- 在高负载下保持稳定的流式 TTFT
- 验证过载场景下 backpressure 与 cancellation 行为
- 量化 IPC 开销拆分：排队等待 / 调度 / 传输 / 计算

---

## 2. 核心概念

### 2.1 Model

最小推理单元。用户通过继承 `nerva.Model` 定义模型，实现 `load()` 和 `infer()` 方法。

```python
from nerva import Model

class MyModel(Model):
    def load(self):
        self.model = torch.load("model.pt")

    async def infer(self, request: InferRequest) -> InferResponse:
        return self.model(request.tensor)
```

### 2.2 Backend

推理后端抽象。每个后端（vLLM、PyTorch、ONNX 等）实现统一接口，衔接 Model 与底层引擎。

### 2.3 Pipeline

描述多个 Model 之间数据流的编排单元。以纯 Python 函数定义；框架通过 tracing 构建 DAG。

### 2.4 Worker

执行单元。每个 Worker 是独立进程，加载一个 Model 实例并绑定特定 GPU。Master 通过 IPC 将请求分发给 Worker。

### 2.5 Server

服务入口。ASGI 服务器接收二进制 RPC 请求，并路由到对应的 Pipeline 或单个 Model。

**概念关系：**

```
Server ─receives→ Request ─routes→ Pipeline ─dispatches→ Model(s) ─runs on→ Worker(s) ─uses→ Backend
```

---

## 3. 架构：Master-Worker

### 3.1 总览

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

### 3.2 Master 进程职责

1. **Binary RPC Server** — Starlette + ASGI 服务器（Uvicorn），处理 unary / server-streaming 二进制 RPC
2. **Pipeline Router** — 按 pipeline/model 名称路由请求
3. **Pipeline Orchestrator** — 按拓扑序执行 Pipeline DAG，管理节点间数据传递
4. **Dynamic Batcher** — 每个 Model 一个 Batcher，在时间窗口 / batch 大小阈值内聚合请求
5. **Deadline & Backpressure Controller** — 执行队列上限、deadline 准入检查和取消传播
6. **Worker Manager** — 管理 Worker 生命周期（启动、健康检查、重启、优雅停机）

### 3.3 Worker 进程职责

每个 Worker 是单职责独立进程：

1. 加载一个 Model 实例（通过对应 Backend）
2. 监听 IPC 通道中的 batch 请求
3. 调用 Backend 执行推理
4. 将结果返回 Master

### 3.4 IPC 设计

Master 与 Worker 之间采用双层通信：

- **控制通道**：Unix Domain Socket + msgpack 序列化，用于管理命令和描述符交换
- **数据通道**：POSIX 共享内存池。大 payload/tensor 字节保留在共享内存中，仅通过控制通道传递描述符

描述符结构（MVP）：
- `request_id`、`node_id`
- `shm_id`、`offset`、`length`
- `inline_data`（小 payload inline）
- `payload_codec`（`msgpack_dict_v1` / `raw_bytes_v1`）
- `input_key`（`raw_bytes_v1` 场景下 bytes 对应的输入字段）
- `dtype`、`shape`
- `lifetime_token`（用于类似引用计数的回收）

payload codec 约定（Phase 1.1）：
- 默认 `msgpack_dict_v1`：输入按 dict 做 msgpack 编解码。
- 单字段 bytes 可走 `raw_bytes_v1`：跳过 dict 级 `msgpack.packb`，由 `input_key` 在 Worker 侧还原为 `{input_key: bytes}`。

减少中间拷贝约定（Phase 1.1）：
- SHM + `msgpack_dict_v1` 解码优先使用 `memoryview -> msgpack.unpackb`，避免 `bytes(buf[slice])` 临时副本。

生命周期（MVP）：
1. Master 从 shm 池分配槽位并写入 payload
2. Master 通过 UDS 发送描述符到 Worker
3. Worker 从 shm 读取并返回 ack/error
4. Master 在 ack 或超时触发 GC 后回收槽位

---

## 4. JAX 风格函数变换 Pipeline API

### 4.1 设计理念

1. **用户只写函数** — Pipeline 就是函数；模型调用就是函数调用
2. **Transforms = 能力注入** — `trace()` 构图，`batch()` 注入批处理，`stream()` 注入流式输出；全部通过函数变换注入
3. **控制流通过原语表达** — MVP 中 `nerva.cond()` 与 `nerva.parallel()` 可被 tracing

### 4.2 用户 API

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

### 4.3 控制流原语

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

### 4.4 Transform 参考

| Transform | 输入 | 输出 | 作用 |
|---|---|---|---|
| `trace(fn)` | Function | `Graph` (DAG IR) | 追踪函数调用并构建计算图 |
| `batch(g, ...)` | Graph | Graph | 在指定节点注入动态批处理 |
| `stream(g)` | Graph | Graph | 将终端节点输出改为异步迭代 |
| `serve(g, ...)` | Graph | ASGI App | 生成可部署的二进制 RPC 服务 |

MVP 中 transform 组合是**有序**的、非交换的：
`serve(stream(batch(trace(fn))))`。

### 4.5 Trace 内部机制

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

### 4.6 Trace 子集约束（MVP）

为了保持可预测延迟并控制实现复杂度，`trace()` 在 MVP 仅支持受限子集：
- model 函数调用
- `nerva.cond()` 与 `nerva.parallel()`
- 作用于可追踪值的简单 Python 表达式

MVP 暂不支持：
- tracing 期间任意 Python 副作用
- trip count 依赖数据的动态循环
- 运行时图结构变更

---

## 5. Backend 抽象层

### 5.1 抽象接口

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

### 5.2 MVP 后端

**PyTorchBackend：**
- 加载用户提供的 `torch.nn.Module`
- 使用 `torch.inference_mode()` 优化
- 支持用户定义前后处理

**VLLMBackend：**
- 封装 vLLM 的 `AsyncLLMEngine`
- 原生 continuous batching（vLLM 自行管理 batching；Nerva 的 Batcher 在该节点旁路）
- 通过 vLLM 流式 API 输出
- 协调 Nerva Worker 进程与 vLLM 内部进程

### 5.3 注册机制

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

## 6. 动态批处理与多 GPU 调度

### 6.1 动态批处理

Master 进程中每个 Model 节点一个 Batcher 实例。

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

**参数**（通过 `batch()` transform 配置）：
```python
graph = batch(graph, targets=["llm"],
    max_size=32,         # max batch size
    max_delay_ms=10,     # max wait time
    pad_to_multiple=8,   # pad to multiples of 8 (GPU tensor core alignment)
)
```

**实现：**
- Master 中使用有界 `asyncio.Queue` + 定时器
- 请求到达后入队；队列满或定时器到期即触发 batch
- 变长输入做 padding，并记录原始长度用于 unpadding
- 结果按请求拆分并分发回对应 `asyncio.Future`
- 入队前执行准入控制，检查剩余 deadline
- 队列溢出立即返回过载错误（不做无界等待）

**vLLM 特殊处理：**vLLM 内部自带 continuous batching。Nerva 的 Batcher 在 vLLM 节点旁路。

### 6.2 多 GPU 调度

**模型放置** — 用户显式指定 device：
```python
llm = model("llm", LLMImpl, backend="vllm", device="cuda:0")
encoder = model("encoder", EncoderImpl, backend="pytorch", device="cuda:1")
```

**Worker 分配：**
- 每个 `model()` 声明对应一个 Worker 进程
- Worker 启动时通过 `CUDA_VISIBLE_DEVICES` 绑定 GPU
- 多个 Worker 可共享同一 GPU（用于 CPU 模型或显存允许时）

**负载均衡（多副本，后续迭代）：**
```python
llm_group = model_group("llm", LLMImpl, backend="vllm",
    devices=["cuda:0", "cuda:1"],
    strategy="round_robin",  # or "least_pending"
)
```

### 6.3 延迟优先调度（MVP）

- 请求调度优先级：earliest deadline first（EDF）
- cancellation 从入口传播到排队中和执行中的节点
- 按模型队列隔离，避免跨模型 HOL 阻塞
- 流式请求预留有界输出缓冲，保护全局延迟

---

## 7. 二进制流式 RPC 服务层（ASGI）

### 7.1 技术栈

- **数据面框架**：Starlette（原生 ASGI 请求/响应流式路径）
- **ASGI 服务器**：Uvicorn（单 Nerva master 进程）
- **管理 API**：FastAPI（模型控制、健康检查、指标）
- `serve()` transform 将 Graph 编译为 ASGI 二进制 RPC 应用
- 进程级并行由 Nerva Worker 进程提供，而非 Uvicorn worker 模式

### 7.2 API 端点

```
POST /rpc/{pipeline_name}               # Unary 或 server-streaming 二进制 RPC
POST /rpc/{request_id}/cancel           # 显式取消（可选）

GET  /v1/models                         # 已加载模型列表
GET  /v1/models/{model_name}            # 模型详情与状态
POST /v1/models/{model_name}/load       # 加载模型
POST /v1/models/{model_name}/unload     # 卸载模型

GET  /v1/health                         # 健康检查
GET  /v1/metrics                        # Prometheus 指标
```

### 7.3 二进制线协议（MVP）

请求头：
- `Content-Type: application/x-nerva-rpc`
- `Accept: application/x-nerva-rpc`
- `x-nerva-request-id`（可选；缺省由服务端生成）
- `x-nerva-deadline-ms`（延迟感知调度必填）
- `x-nerva-stream`（`0` unary，`1` server-streaming）

帧格式：

| 字段 | 大小 | 说明 |
|---|---:|---|
| `magic` | 2B | `0x4E56`（`NV`） |
| `version` | 1B | 协议版本（`1`） |
| `type` | 1B | `OPEN=1`、`DATA=2`、`END=3`、`ERROR=4`、`HEARTBEAT=5` |
| `flags` | 2B | 位标记（`compressed`、`eos`） |
| `reserved` | 2B | 保留 |
| `request_id` | 8B | 请求关联 ID |
| `stream_id` | 4B | 流 ID（MVP 固定为 `1`） |
| `payload_len` | 4B | payload 长度 |
| `crc32` | 4B | payload 校验（MVP 可选） |
| `header_ext_len` | 4B | 扩展头长度（MVP 为 `0`） |

Payload：
- `OPEN`：方法/pipeline 元数据
- `DATA`：二进制输入/输出分片
- `END`：最终元数据（延迟、token 数等）
- `ERROR`：错误码 + 错误消息 + retryable 标志

### 7.4 RPC 生命周期

Unary：
1. 客户端发送 `OPEN + DATA + END`
2. 服务端执行 DAG 并返回 `DATA + END`

Server-streaming：
1. 客户端发送 `OPEN + DATA + END`
2. 服务端返回多个 `DATA` 帧
3. 服务端以 `END` 结束

Cancel：
- 客户端断开连接，或显式调用 `/rpc/{request_id}/cancel`
- Master 将取消信号传播到队列和 Worker 执行路径

---

## 8. 可观测性与错误处理

### 8.1 指标（Prometheus）

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

### 8.2 日志

- 使用 `structlog` 输出结构化 JSON 日志
- 每个请求的 `request_id` 在完整 pipeline 链路中透传
- 级别：ERROR（异常）、WARNING（deadline/cancel/overload）、INFO（请求摘要）、DEBUG（节点级细节）

### 8.3 错误处理

| 失败类型 | 处理策略 |
|---|---|
| 非法帧 / 协议不匹配 | 返回 `ERROR(INVALID_ARGUMENT)` 并关闭流 |
| 队列过载 | 快速拒绝并返回 `ERROR(RESOURCE_EXHAUSTED)` |
| 请求超出 deadline | 取消下游工作并返回 `ERROR(DEADLINE_EXCEEDED)` |
| 客户端主动取消 | 传播取消信号，并将请求标记为 `ABORTED` |
| Worker 进程崩溃 | Worker Manager 自动重启；在途请求返回 `UNAVAILABLE` |
| GPU OOM | 隔离失败请求并记录指标，返回 `RESOURCE_EXHAUSTED` |
| 模型加载失败 | 模型标记为 `error` 状态；健康检查端点上报异常 |

---

## 9. 项目结构

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

## 10. 技术栈总览

| 组件 | 选型 | 理由 |
|---|---|---|
| Data Plane Server | Starlette + Uvicorn (ASGI) | 低开销二进制流式路径，兼容 Nerva Master-Worker 架构 |
| Management API | FastAPI | 运维友好：OpenAPI、校验、管理端点 |
| Async Runtime | asyncio | Python 标准、生态最广 |
| IPC | Unix Domain Socket + POSIX shm | 低延迟，大张量零拷贝 |
| RPC Protocol | Nerva Binary Streaming RPC | 支持 unary + server-streaming，支持 deadline/cancel |
| Serialization | msgpack（control）+ protobuf/msgpack（payload） | 降低热路径 JSON 开销 |
| Metrics | prometheus_client | 行业标准 |
| Logging | structlog | 结构化 JSON，性能良好 |
| Type Checking | mypy (strict) | 提前发现 bug |
| Linting | ruff | 快速、规则全面 |
| Testing | pytest + pytest-asyncio | 原生支持异步测试 |
| Package Management | uv | 快速、现代的 Python 工具链 |
