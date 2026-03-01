# Nerva MVP Implementation Roadmap

> 从设计文档到可执行代码的推进计划。

## 前置工作

### W1. 回应 Design Review Open Questions

澄清 [design-review.md](./2026-02-25-design-review.md) 中的架构决策，重点：
- P0: Model vs Backend 职责边界
- P0: Binary Protocol 的 MVP 选型
- P0: vLLM 集成进程模型

**产出：** design-review.md 中的 [设计回应] 填写完成

**状态：** ✅ 已完成 (2026-02-25)

**关键决策：**
- Model 是预测节点抽象，Backend 是执行后端抽象；调用链 Orchestrator → Backend → Model
- MVP 保留自研 Binary Protocol，先做音频数据传输场景
- vLLM 采用 Worker 内 thin wrapper + AsyncLLMEngine，Nerva 不接管内部调度
- 保留受限 trace + 显式 DAG 兜底
- IPC 考虑 ZeroMQ 替代原始 UDS
- serve() 从 transform 链剥离，Pipeline 可独立执行
- Batching 不做 padding，按 size 一致性组 batch

---

### W2. 定义 End-to-End Example

编写一个具体的、可执行的端到端示例作为实现目标：
- 定义 toy model（不依赖真实模型权重）
- 明确输入输出格式
- 写出完整的用户代码（从 model 声明到 serve）
- 包含单模型和多模型 Pipeline 两个示例

**产出：**
- `examples/01_single_model.py` — 单模型 unary 推理
- `examples/02_multi_model_pipeline.py` — 多模型 streaming pipeline（Tokenizer → LLM → Detokenizer）
- `examples/03_parallel_dag.py` — parallel + cond 控制流

**状态：** ✅ 已完成 (2026-02-25)

---

### W3. 搭建项目脚手架

- `pyproject.toml`：项目元数据、依赖声明、ruff/mypy/pytest 配置
- 目录结构创建（`src/nerva/` 布局）
- `uv lock` 锁定依赖

**产出：** 可运行 `uv run pytest` / `uv run ruff check` / `uv run mypy` 的项目骨架

**状态：** ✅ 已完成 (2026-02-25)

**技术栈：** Python 3.14, uv, pytest + pytest-asyncio, ruff, mypy (strict), hatchling

---

### W4. 高风险技术验证（Spikes）

| Spike | 验证目标 | 状态 | 报告 |
|---|---|---|---|
| S1: UDS + POSIX shm IPC | round-trip 延迟、吞吐量 | ✅ 已完成 | [`docs/spikes/s1-ipc-benchmark-report.md`](../spikes/s1-ipc-benchmark-report.md) |
| S2: Trace 机制原型 | Proxy + `__call__` 拦截 DAG 构建 | ✅ 已完成 | [`docs/spikes/s2-trace-prototype-report.md`](../spikes/s2-trace-prototype-report.md) |
| S3: vLLM AsyncLLMEngine 集成 | 嵌入子进程后行为 | ⬜ 跳过（需 GPU 环境） | - |
| S4: asyncio Dynamic Batching | Queue + timer batching 延迟 | ✅ 已完成 | [`docs/spikes/s4-async-batcher-report.md`](../spikes/s4-async-batcher-report.md) |

**关键结论：**
- S1: UDS+SHM 在 ≥64KB payload 有 5-55x 延迟优势；4MB p99 仅 355us
- S2: Proxy trace 机制可正确构建 DAG，包括 linear/parallel/cond/diamond 拓扑
- S4: Batching 有效提升吞吐 14.6x；asyncio 调度开销约 1.2ms；timer flush 在低并发下正确

---

### W5. 内部接口契约（Phase 0 部分）

Phase 0 所需的接口已在代码中定义：
- `nerva.Model` ABC — 用户模型基类（`core/model.py`）
- `nerva.Backend` ABC — 推理后端抽象（`backends/base.py`）
- `ModelConfig` / `InferContext` / `BatchMeta` — 数据契约
- Backend registry — 注册/查找机制（`backends/registry.py`）

后续 Phase 的接口（Graph IR、IPC 消息 schema 等）随各 Phase 设计文档定义。

**状态：** ✅ Phase 0 部分已完成 (2026-02-25)

---

### W6. 测试策略

Phase 0 已建立测试模式：
- 单元测试：直接调用，assert 行为（`test_model.py`, `test_backend_registry.py`）
- 异步测试：`pytest-asyncio` auto mode（`test_pytorch_backend.py`）
- 隔离：Backend registry 测试使用 setup/teardown 保存恢复全局状态
- 无 GPU：所有 Phase 0 测试在 CPU 环境运行

Phase 1 测试策略定义在 [Phase 1 设计文档](./2026-02-25-phase1-design.md) Section 13。

**状态：** ✅ 基础策略已建立 (2026-02-25)

---

## 实现阶段

### Phase 0 — 单模型 In-Process 调用

**目标：** 验证 Model / Backend 接口设计。不涉及 IPC、不启动 Worker 进程。

**范围：**
- `nerva.Model` 基类
- `Backend` ABC + `PyTorchBackend` 实现
- Backend registry 机制
- `model()` 声明 API
- 直接 in-process 同步/异步调用 model

**验证结果：** ruff 0 errors, mypy 0 issues, 25 tests passed (0.07s)

**状态：** ✅ 已完成 (2026-02-25)

**产出文件：**
```
src/nerva/__init__.py              # 公共 API 导出
src/nerva/core/model.py            # Model ABC, ModelHandle, model()
src/nerva/backends/base.py         # Backend ABC, ModelConfig, InferContext, BatchMeta
src/nerva/backends/registry.py     # register_backend / get_backend / list_backends
src/nerva/backends/pytorch.py      # PyTorchBackend
tests/test_model.py                # 11 tests
tests/test_backend_registry.py     # 6 tests
tests/test_pytorch_backend.py      # 8 tests
```

---

### Phase 1 — Master-Worker 进程分离 + IPC

**目标：** 验证进程模型和通信机制。

**设计文档：** [`2026-02-25-phase1-design.md`](./2026-02-25-phase1-design.md)

**范围：**
- IPC 控制通道：ZeroMQ PAIR over `ipc://`，msgpack 序列化
- IPC 数据通道：POSIX shared memory pool（slab + size classes），≤8KB inline 优化
- Descriptor payload codec：`msgpack_dict_v1`（默认）与 `raw_bytes_v1`（单字段 bytes 快速路径）
- Worker 进程 main loop（`worker/process.py`）
- Worker Manager 生命周期管理（`worker/manager.py`）
- Master 侧 WorkerProxy 异步 RPC 封装
- 单模型通过 Master → Worker IPC 完成推理

**关键设计决策：**
- 控制通道选用 ZeroMQ PAIR（降低实现复杂度，省去帧解析和断线处理）
- ≤8KB payload inline 到控制消息，>8KB 走 SHM（基于 S1 spike 数据）
- 单字段 bytes 输入支持 `raw_bytes_v1`，跳过 dict 级 `msgpack.packb` 降低序列化开销
- SHM + msgpack 解码优先 `memoryview -> msgpack.unpackb`，减少 `bytes(buf[slice])` 中间副本
- model_class 通过 import path 字符串跨进程传递（避免 pickle 限制）
- Phase 0 代码零修改，Backend/Model 原样运行在 Worker 进程中

**验证标准：**
- 单模型通过 IPC 完成推理，结果与 Phase 0 in-process 一致
- 测量 IPC overhead（对比 Phase 0 直接调用 vs Phase 1 跨进程调用）
- Worker 崩溃后 Master 能检测并重启
- SHM 分配/回收无泄漏

**验证结果：** ruff 0 errors, mypy 0 issues, 150 tests passed (28s)

**状态：** ✅ 已完成 (2026-02-26)

**产出文件：**
```
src/nerva/worker/process.py        # Worker 进程 main loop
src/nerva/worker/proxy.py          # WorkerProxy 异步 RPC 封装
src/nerva/worker/manager.py        # WorkerManager 生命周期管理
src/nerva/worker/ipc.py            # IPC 消息编解码
src/nerva/engine/shm_pool.py       # Shared Memory Pool
tests/test_worker_process.py       # Worker 进程测试
tests/test_worker_proxy.py         # WorkerProxy 测试
tests/test_worker_manager.py       # WorkerManager 测试
tests/test_ipc.py                  # IPC 消息测试
tests/test_shm_pool.py             # SHM pool 测试
tests/test_phase1_e2e.py           # 端到端集成测试
```

---

### Phase 2 — DAG Pipeline 编排

**目标：** 验证多模型串联 / 并联执行。

**设计文档：** [`2026-02-26-phase2-design.md`](./2026-02-26-phase2-design.md)

**范围：**
- Graph IR 数据结构（`core/graph.py`）— Edge 含 field_path 和 dst_input_key
- Proxy 对象 + `__getitem__` 路径追踪 + `trace()` 函数追踪（`core/proxy.py`）
- `nerva.cond()` / `nerva.parallel()` 控制流原语（`core/primitives.py`）
- 事件驱动 DAG Executor（`engine/executor.py`）— in-degree table + done_queue
- ModelHandle 双态行为（trace 模式 / 运行时模式）
- Dict 输入映射支持多模型融合场景

**关键设计决策：**
- Proxy.__getitem__ 记录 field_path tuple，执行时逐层取值
- trace 上下文使用 contextvars（并发安全）
- parallel 输出映射通过初始 field_path=(str(i),) 与 __getitem__ 统一
- Executor 为每个节点生成唯一 request_id，避免 WorkerProxy 冲突
- InferableProxy Protocol 解耦 Executor 与 WorkerProxy

**验证结果：** ruff 0 errors, mypy 0 issues, 140 tests passed (21s)

**状态：** ✅ 已完成 (2026-02-26)

**产出文件：**
```
src/nerva/core/graph.py            # Edge, Node, Graph IR
src/nerva/core/proxy.py            # Proxy, TraceContext, trace()
src/nerva/core/primitives.py       # cond(), parallel()
src/nerva/core/model.py            # ModelHandle.__call__ 双态（修改）
src/nerva/engine/executor.py       # Executor, resolve_field_path
src/nerva/__init__.py              # 新增导出（修改）
tests/test_graph.py                # 10 tests
tests/test_proxy.py                # 15 tests
tests/test_primitives.py           # 5 tests
tests/test_executor.py             # 13 tests (mock)
tests/test_phase2_e2e.py           # 5 tests (real Worker)
```

---

### Phase 3 — Dynamic Batching + Deadline 调度

**目标：** 验证批处理和延迟感知调度。

**范围：**
- Dynamic Batcher（`engine/batcher.py`）
- `batch()` transform
- Deadline-aware scheduler（`engine/scheduler.py`）
- Backpressure / queue overflow 拒绝
- Cancellation propagation（部分：deadline 过期过滤 + stop() drain；主动取消推迟至 Phase 4+）

**关键设计决策：**
- DynamicBatcher 作为 InferableProxy wrapper，对 Executor 完全透明
- Deadline 准入 + 批次聚合合并进 Batcher，不单独做 Scheduler
- 组图时透明：BatchConfig 通过 model() 参数声明，不影响 trace()
- Backpressure：等待 + 超时拒绝（queue_timeout_ms），RESOURCE_EXHAUSTED
- 批次并发执行：asyncio.gather 并发 N 个单请求（MVP，不改 Model 接口）

**验证结果：** ruff 0 errors, mypy 0 issues, 176 tests passed (23.84s)

**状态：** ✅ 已完成 (2026-02-27)

**产出文件：**
```
src/nerva/engine/batcher.py        # DynamicBatcher, BatchConfig, _PendingRequest
src/nerva/core/model.py            # model() + ModelHandle 新增 batch_config（修改）
tests/test_batcher.py              # 单元测试（mock inner proxy）13 tests
tests/test_phase3_e2e.py           # 集成测试（真实 Worker）2 tests
```

---

### Phase 4 — Binary RPC Serving Layer (Unary)

**目标：** 通过 HTTP Binary RPC 协议暴露 Nerva pipelines，自动管理 Worker 进程。

**设计文档：** [`2026-02-28-phase4-design.md`](./2026-02-28-phase4-design.md)

**范围：**
- Binary frame 编解码（`server/protocol.py`）— 32-byte fixed header + msgpack payload
- Binary RPC handler（`server/rpc.py`）— OPEN+DATA+END 帧解析，错误映射，deadline passthrough
- ASGI 应用组装（`server/app.py`）— RPC route + `/v1/health` + `/v1/models` management routes
- `serve()` 顶层 API（`server/serve.py`）— 自动扫描 Graph、查找 ModelHandle、spawn Worker、启动 uvicorn
- Model handle registry（`core/model.py`）— `model()` 注册、`get_model_handle()` / `list_model_handles()` 查找
- Public API export — `nerva.serve`, `nerva.get_model_handle`, `nerva.list_model_handles`

**关键设计决策：**
- serve() 从 transform 链剥离，Pipeline 可独立执行
- Binary Protocol v1（magic 0x4E56, version 1）用于音频等二进制数据场景
- Frame header 32 bytes big-endian，MVP 固定 stream_id=1, crc32=0, ext_hdr_len=0
- 绝对 deadline epoch ms 通过 `x-nerva-deadline-ms` header 传入，RPC handler 转换为相对 TTL
- 错误映射：pipeline not found → INVALID_ARGUMENT(3), deadline expired → DEADLINE_EXCEEDED(4), 内部错误 → INTERNAL(13)
- `_PipelineExecutor` 在每次请求创建新的 `Executor` + `InferContext`，保证请求隔离
- 后续迭代增加 Streaming RPC 和 HTTP+JSON 协议支持

**验证结果：** ruff 0 errors, mypy 0 issues, 200 tests passed (23s)

**状态：** ✅ 已完成 (2026-02-28)

**产出文件：**
```
src/nerva/server/protocol.py       # FrameType, Frame, encode/decode, ProtocolError
src/nerva/server/rpc.py            # RpcHandler, ErrorCode, build_rpc_app
src/nerva/server/app.py            # build_app() — ASGI 应用组装
src/nerva/server/serve.py          # serve(), _collect_model_names(), _build_pipelines(), _PipelineExecutor
src/nerva/server/__init__.py       # server 包初始化
src/nerva/core/model.py            # _model_registry, get_model_handle(), list_model_handles()（修改）
src/nerva/__init__.py              # 新增 serve, get_model_handle, list_model_handles 导出（修改）
tests/test_protocol.py             # 15 tests
tests/test_rpc.py                  # 8 tests
tests/test_app.py                  # 4 tests
tests/test_serve.py                # 5 tests
tests/test_phase4_e2e.py           # 5 tests (real Worker + httpx AsyncClient)
tests/test_model.py                # 13 tests (含 4 个新增 registry tests)
```

---

### Phase 5 — vLLM Backend + 观测性

**目标：** 集成真实 LLM backend，补全生产级观测。

**范围：**
- `VLLMBackend` 实现（`backends/vllm.py`）
- Prometheus metrics 全量接入（`observability/metrics.py`）
- structlog 结构化日志（`observability/logging.py`）
- 端到端 example 跑通（tokenizer → LLM → detokenizer pipeline）

**关键决策（来自 Design Review #3）：**
- vLLM 节点采用 Worker 内 thin wrapper + AsyncLLMEngine
- 旁路 Nerva batcher，Nerva 仅控制入队与预算
- SHM 仅用于 Nerva 进程间传递，不替代 vLLM 显存管理

**验证标准：** 使用 vLLM 跑通 text generation pipeline，Prometheus 指标可观测，日志包含完整 request_id 链路。

**状态：** ⬜ 待设计

---

## 执行依赖关系

```
W1 (design review) ──→ W2 (e2e example) ──→ W5 (interface contracts) ──→ Phase 0
         ✅                   ✅                    ✅ (Phase 0)            ✅
W3 (scaffolding) ─────────────────────────────────────────────────────→ Phase 0
         ✅                                                                ✅
W4 (spikes) ──────────→ W5 (interface contracts)
    ✅ (S1/S2/S4)               ✅ (Phase 0)
                            W6 (test strategy) ───────────────────────→ Phase 0
                                    ✅                                     ✅

Phase 0 ──→ Phase 1 ──→ Phase 2 ──→ Phase 3 ──→ Phase 4 ──→ Phase 5
  ✅           ✅           ✅           ✅           ✅        待设计
```

---

## 相关文档索引

| 文档 | 用途 |
|---|---|
| [`2026-02-24-nerva-design.md`](./2026-02-24-nerva-design.md) | 总体设计文档 |
| [`2026-02-25-design-review.md`](./2026-02-25-design-review.md) | 设计审阅 + 决策回应 |
| [`protocol-v1.md`](./protocol-v1.md) | Binary Streaming RPC 协议规范 |
| [`ipc-contract.md`](./ipc-contract.md) | Master-Worker IPC 契约 |
| [`mvp-defaults.md`](./mvp-defaults.md) | MVP 默认参数表 |
| [`mvp-benchmark-plan.md`](./mvp-benchmark-plan.md) | 性能基准测试计划 |
| [`2026-02-25-phase1-design.md`](./2026-02-25-phase1-design.md) | Phase 1 详细设计 |
| [`2026-02-26-phase2-design.md`](./2026-02-26-phase2-design.md) | Phase 2 详细设计 |
| [`docs/spikes/s1-*.md`](../spikes/s1-ipc-benchmark-report.md) | S1 IPC 延迟报告 |
| [`docs/spikes/s2-*.md`](../spikes/s2-trace-prototype-report.md) | S2 Trace 原型报告 |
| [`docs/spikes/s4-*.md`](../spikes/s4-async-batcher-report.md) | S4 Batching 报告 |

---

## Changelog

| 日期 | 变更 |
|---|---|
| 2026-02-25 | 初始版本 |
| 2026-02-25 | 更新全部前置工作和 Phase 0 为已完成；补充关键决策、产出文件、spike 结论；更新 Phase 1-5 加入设计决策；新增文档索引 |
| 2026-02-26 | Phase 1 和 Phase 2 标记为已完成；补充产出文件清单和关键设计决策；新增 Phase 2 设计文档链接 |
| 2026-02-27 | Phase 3 标记为已完成；补充产出文件和关键设计决策 |
