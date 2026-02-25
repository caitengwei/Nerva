# Phase 1 Design — Master-Worker 进程分离 + IPC

## 1. 目标与范围

### 1.1 目标

将 Phase 0 的 in-process 单模型推理扩展为 Master-Worker 多进程架构：
- Master 进程接收请求，通过 IPC 分发给 Worker
- Worker 进程独立加载 Model，执行推理，返回结果
- 验证进程分离后的 round-trip 延迟开销

### 1.2 范围

**包含：**
- Worker 进程 main loop
- Worker Manager（spawn、health check、restart）
- IPC 控制通道（消息收发）
- IPC 数据通道（shared memory pool，基础版）
- Master 侧 WorkerProxy（异步 RPC 封装）
- 单模型通过 Master → Worker IPC 完成推理的端到端链路

**不包含（后续 Phase）：**
- Pipeline DAG 编排（Phase 2）
- Dynamic Batching（Phase 3）
- HTTP 服务层（Phase 4）
- 多 Worker 负载均衡

### 1.3 验证标准

- 单模型通过 IPC 完成推理，结果与 Phase 0 in-process 一致
- 测量 IPC overhead：对比 Phase 0 直接调用 vs Phase 1 跨进程调用的延迟差
- Worker 崩溃后 Master 能检测并重启 Worker
- shm 分配/回收无泄漏（在途请求超时后全部回收）

---

## 2. 架构概览

```
Master Process                              Worker Process
┌──────────────────────────────┐           ┌──────────────────────────┐
│                              │           │                          │
│  user code                   │           │  WorkerMain              │
│    │                         │           │    │                     │
│    ▼                         │           │    ▼                     │
│  WorkerProxy                 │    IPC    │  IPC Endpoint            │
│    │                         │◄─────────►│    │                     │
│    ├─ send INFER_SUBMIT ─────┤  control  │    ├─ recv INFER_SUBMIT  │
│    │                         │  channel  │    │                     │
│    │  [alloc shm, write]     │  (ZeroMQ  │    │  [read shm]         │
│    │                         │   PAIR)   │    ▼                     │
│    │                         │           │  Backend.infer()         │
│    │                         │           │    │                     │
│    ├─ recv INFER_ACK ◄───────┤           │    ├─ [write shm]        │
│    │                         │           │    ├─ send INFER_ACK     │
│    ▼                         │           │    ▼                     │
│  return result               │           │  loop                    │
│                              │           │                          │
│  WorkerManager               │           └──────────────────────────┘
│    ├─ spawn worker process   │
│    ├─ health check (timer)   │
│    └─ restart on crash       │
│                              │
│  ShmPool                     │  data channel (POSIX shared memory)
│    ├─ alloc(size) → slot     │  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─►
│    └─ free(slot)             │
└──────────────────────────────┘
```

---

## 3. IPC 控制通道：ZeroMQ vs 原始 UDS

### 3.1 选型分析

Design Review #5 提出考虑 ZeroMQ 替代原始 UDS。对比：

| 维度 | 原始 UDS (AF_UNIX) | ZeroMQ (PAIR over ipc://) |
|---|---|---|
| 消息帧定界 | 需自行实现 length-prefix | 内建消息帧，零拷贝语义 |
| 重连 / 心跳 | 手动实现 | 内建 monitor + heartbeat |
| 多消息模式 | 手动状态机 | REQ/REP, PUSH/PULL, PAIR 等 |
| 序列化 | 自行选择（msgpack） | 同上，payload 无关 |
| 依赖 | 无（stdlib socket） | pyzmq (~500KB wheel) |
| 延迟 (S1 spike) | p99 ~30us @1KB | 预期相当（同底层 UDS） |
| 实现复杂度 | 中高（帧解析、断线处理） | 低（zmq 封装） |

### 3.2 决策

**Phase 1 采用 ZeroMQ PAIR 模式**，理由：

1. **降低实现复杂度**：省去手写帧解析、断线重连、partial read 处理
2. **消息语义天然对齐**：一条 ZeroMQ 消息 = 一条 IPC 控制消息，无需 length-prefix
3. **延迟开销可接受**：ZeroMQ over `ipc://` 底层仍是 UDS，S1 spike 验证了 UDS 延迟足够低
4. **依赖可控**：pyzmq 是成熟库，单一额外依赖

**PAIR 模式选择**：Master-Worker 是 1:1 关系，每个 Worker 一个 ZeroMQ PAIR socket。PAIR 提供双向异步通信，最简单直接。

**序列化**：控制消息用 msgpack，与 ipc-contract.md 保持一致。

### 3.3 连接拓扑

```
Master                          Worker 0
  zmq.PAIR ◄──ipc:///tmp/nerva-worker-0.sock──► zmq.PAIR

Master                          Worker 1
  zmq.PAIR ◄──ipc:///tmp/nerva-worker-1.sock──► zmq.PAIR
```

- Master 侧 bind，Worker 侧 connect
- socket 路径：`/tmp/nerva-{worker_id}.sock`（可配置前缀）
- 使用 `asyncio` + `zmq.asyncio.Context` 实现非阻塞收发

---

## 4. IPC 数据通道：Shared Memory Pool

### 4.1 概述

基于 ipc-contract.md Section 5 和 S1 spike 结论：
- 64KB 以上 payload，UDS+SHM 比纯 UDS 有 5-55x 延迟优势
- 小 payload（<8KB）直接通过控制通道 inline 传输，跳过 SHM

### 4.2 ShmPool 设计

```python
class ShmPool:
    """Master 侧管理的 shared memory 池。"""

    def alloc(self, size: int) -> ShmSlot:
        """分配一个 slot。向上取整匹配 size class。

        Raises ShmPoolExhausted if no slot available.
        """

    def free(self, slot: ShmSlot) -> None:
        """回收 slot。"""

    def write(self, slot: ShmSlot, data: bytes) -> None:
        """写入数据到 slot。"""

    def read(self, slot: ShmSlot) -> bytes:
        """从 slot 读取数据。"""
```

**Size classes**（来自 mvp-defaults.md）：
`4KB / 16KB / 64KB / 256KB / 1MB / 4MB`

**分配策略**：
- 每个 size class 预分配固定数量的 slot
- 请求 size 向上取整匹配最小 size class
- 所有 slot 用完时返回 `RESOURCE_EXHAUSTED`，不阻塞

**Phase 1 简化**：
- 使用 `multiprocessing.shared_memory.SharedMemory` 实现（跨平台）
- 每个 size class 对应一个 SharedMemory 块，内部按 slot 大小等分
- 用 bitmap 追踪 slot 占用状态（Master 单进程，无需跨进程锁）

### 4.3 Inline 优化

当 payload <= `IPC_CONTROL_INLINE_MAX_BYTES` (8KB) 时：
- 数据直接嵌入控制消息，不使用 SHM
- descriptor 中 `shm_id = None`，`inline_data = <bytes>`
- 避免小 payload 的 SHM alloc/free 开销

### 4.4 Descriptor

沿用 ipc-contract.md Section 4 的 schema，增加 inline 字段：

```python
@dataclass
class Descriptor:
    schema_version: int = 1
    request_id: int
    node_id: int
    # SHM 路径（inline 时为 None）
    shm_name: str | None = None
    offset: int = 0
    length: int = 0
    # Inline 数据（SHM 时为 None）
    inline_data: bytes | None = None
    # 元信息
    dtype: str = "bytes"
    shape: list[int] = field(default_factory=list)
    lifetime_token: int = 0
```

---

## 5. Worker 进程设计

### 5.1 WorkerMain

Worker 是一个独立进程，职责单一：

```
WorkerMain
  │
  ├─ 1. 初始化 zmq.PAIR socket，connect 到 Master
  ├─ 2. 等待 LOAD_MODEL 命令
  ├─ 3. 创建 Backend 实例，调用 backend.load_model()
  ├─ 4. 发送 LOAD_MODEL_ACK
  └─ 5. 进入主循环：
       ├─ recv INFER_SUBMIT
       ├─ 从 SHM 或 inline_data 读取输入
       ├─ 调用 backend.infer()
       ├─ 将输出写入 SHM 或 inline
       ├─ send INFER_ACK
       └─ loop

       ├─ recv CANCEL → 标记 cancelled（best-effort）
       ├─ recv HEALTH_CHECK → send HEALTH_STATUS
       └─ recv SHUTDOWN → graceful exit
```

### 5.2 进程启动

- 使用 `multiprocessing.Process(target=worker_main, args=(...))` 启动
- Worker 进程设置 `CUDA_VISIBLE_DEVICES` 限制 GPU 可见性
- Worker 进程独立的 asyncio event loop

### 5.3 关键设计点

**阻塞推理处理**：`Backend.infer()` 是 async 的，但底层 PyTorch 推理是 CPU/GPU bound。Phase 1 暂时允许阻塞 Worker 的 event loop（Worker 只服务一个 Model，不需要高并发）。后续可用 `run_in_executor` 优化。

**取消传播**：Worker 收到 CANCEL 后设置 `InferContext.cancelled = True`。当前 Backend 实现会在下次检查时抛出异常。对于长时间推理，需要 Backend 内部配合 checkpoint cancel（Phase 1 不强制，best-effort）。

---

## 6. Worker Manager 设计

### 6.1 职责

Master 进程中的 WorkerManager 负责：
1. **启动 Worker**：根据 ModelHandle 创建 Worker 进程
2. **加载 Model**：通过 IPC 发送 LOAD_MODEL 命令
3. **健康检查**：定时发送 HEALTH_CHECK，检测超时
4. **崩溃恢复**：检测 Worker 进程退出，按退避策略重启
5. **优雅关闭**：发送 SHUTDOWN，等待 Worker 退出

### 6.2 生命周期状态

```
STARTING → LOADING → READY → RUNNING
                               │
              RESTARTING ◄─────┤ (crash detected)
                  │            │
                  └──► LOADING STOPPING → STOPPED
```

### 6.3 健康检查

- 间隔：`5s`（可配置）
- 超时：`3s`（连续 3 次超时视为不可用）
- 实现：HEALTH_CHECK / HEALTH_STATUS 消息对

### 6.4 崩溃恢复

- 通过 `Process.is_alive()` 检测
- 退避策略：`1s → 2s → 4s → 8s → 16s`（最大 5 次重试）
- 重启后重新 LOAD_MODEL
- 在途请求返回 `UNAVAILABLE`

### 6.5 优雅关闭

1. Master 收到 SIGTERM
2. WorkerManager 停止接收新请求
3. 发送 SHUTDOWN 给所有 Worker
4. 等待 Worker 退出（最长 `30s`）
5. 超时则 SIGKILL

---

## 7. Master 侧 WorkerProxy

### 7.1 概述

WorkerProxy 是 Master 进程中对单个 Worker 的异步 RPC 封装。上层调用者（未来的 Orchestrator / Batcher）通过 WorkerProxy 提交推理请求。

```python
class WorkerProxy:
    """Master-side async proxy for a single Worker."""

    async def load_model(self, config: ModelConfig) -> None:
        """Send LOAD_MODEL and wait for ACK."""

    async def infer(
        self, inputs: dict[str, Any], context: InferContext
    ) -> dict[str, Any]:
        """Send INFER_SUBMIT, wait for INFER_ACK, return outputs."""

    async def cancel(self, request_id: str) -> None:
        """Send CANCEL for in-flight request."""

    async def health_check(self) -> bool:
        """Send HEALTH_CHECK, return True if healthy."""

    async def shutdown(self) -> None:
        """Send SHUTDOWN and wait for worker exit."""
```

### 7.2 请求生命周期（Unary）

```
WorkerProxy.infer(inputs, context)
  │
  ├─ 1. 序列化 inputs → bytes (msgpack)
  ├─ 2. 判断 inline vs SHM
  │     ├─ <= 8KB → inline_data = bytes
  │     └─ > 8KB  → ShmPool.alloc() → ShmPool.write() → descriptor
  ├─ 3. 构建 INFER_SUBMIT 消息
  ├─ 4. zmq send
  ├─ 5. await INFER_ACK（带 submit_timeout）
  │     ├─ 超时 → 标记 EXPIRED，返回 UNAVAILABLE
  │     ├─ ACK(OK) → 读取 out_descriptor → 反序列化 → return
  │     └─ ACK(ERROR) → raise 对应异常
  └─ 6. ShmPool.free(input_slot)
```

### 7.3 并发管理

- 每个 WorkerProxy 维护一个 `dict[str, asyncio.Future]` 跟踪 in-flight 请求
- zmq recv loop 作为后台 asyncio Task 持续运行
- 收到 INFER_ACK 时根据 `request_id` 匹配 Future 并 set_result

---

## 8. 消息格式

沿用 ipc-contract.md，具体 msgpack 编码：

### INFER_SUBMIT

```python
{
    "type": "INFER_SUBMIT",
    "request_id": <uint64>,
    "node_id": <uint32>,
    "deadline_ms": <uint64>,
    "descriptor": {
        "shm_name": <str | None>,
        "offset": <uint32>,
        "length": <uint32>,
        "inline_data": <bytes | None>,
        "dtype": <str>,
        "shape": [<uint32>, ...],
        "lifetime_token": <uint64>,
    },
    "batch_meta": None,  # Phase 1 不启用 batching
}
```

### INFER_ACK

```python
{
    "type": "INFER_ACK",
    "request_id": <uint64>,
    "node_id": <uint32>,
    "status": "OK" | "INTERNAL" | "DEADLINE_EXCEEDED" | ...,
    "out_descriptor": { ... } | None,
    "error": <str | None>,
}
```

### LOAD_MODEL

```python
{
    "type": "LOAD_MODEL",
    "model_name": <str>,
    "model_class_path": <str>,  # e.g. "myapp.models:SentimentClassifier"
    "backend": <str>,
    "device": <str>,
    "options": { ... },
}
```

### LOAD_MODEL_ACK

```python
{
    "type": "LOAD_MODEL_ACK",
    "model_name": <str>,
    "status": "OK" | "INTERNAL",
    "error": <str | None>,
}
```

### CANCEL / HEALTH_CHECK / HEALTH_STATUS / SHUTDOWN

```python
{"type": "CANCEL", "request_id": <uint64>, "reason": <str>}
{"type": "HEALTH_CHECK", "worker_id": <str>}
{"type": "HEALTH_STATUS", "worker_id": <str>, "ok": <bool>, "detail": <str>}
{"type": "SHUTDOWN"}
```

---

## 9. SHM 生命周期与回收

遵循 ipc-contract.md Section 6 的状态机，Phase 1 实现简化版：

```
alloc → ALLOCATED
write → WRITTEN
send INFER_SUBMIT → SUBMITTED
worker reads → CONSUMED (worker 侧状态，master 不感知)
recv INFER_ACK → ACKED
free → RECLAIMED
```

**回收规则：**
- 正常路径：收到 INFER_ACK 后立即 free input descriptor
- 超时路径：`IPC_SUBMIT_TIMEOUT_MS` (5s) 后仍无 ACK → EXPIRED → free
- Worker 崩溃路径：WorkerManager 检测到崩溃 → 批量 free 所有该 Worker 的 in-flight descriptors

**Phase 1 简化：** 不实现 lifetime_token + TTL GC。依赖 submit timeout 和崩溃检测来回收。

---

## 10. 文件结构

```
src/nerva/
├── worker/
│   ├── __init__.py
│   ├── process.py          # WorkerMain — worker 进程入口和主循环
│   ├── manager.py          # WorkerManager — 生命周期管理
│   └── ipc.py              # IPC 消息编解码、ZeroMQ 封装
├── engine/
│   └── shm_pool.py         # ShmPool — shared memory 池管理
├── (existing)
│   ├── core/model.py       # 无修改
│   ├── backends/base.py    # 无修改
│   ├── backends/pytorch.py # 无修改
│   └── backends/registry.py # 无修改
tests/
├── test_ipc.py             # IPC 消息编解码单元测试
├── test_shm_pool.py        # ShmPool alloc/free/read/write 单元测试
├── test_worker_process.py  # Worker 进程启动、推理、关闭集成测试
├── test_worker_manager.py  # WorkerManager spawn/health/restart 测试
└── test_worker_proxy.py    # WorkerProxy 端到端测试（含崩溃场景）
```

---

## 11. 与 Phase 0 的关系

Phase 0 的 Model / Backend / Registry 代码**不做修改**。Phase 1 在其上层新增：

```
Phase 0:  user code → Backend.infer() → Model.infer()  (in-process)

Phase 1:  user code → WorkerProxy.infer()
            → [IPC: INFER_SUBMIT]
            → WorkerMain → Backend.infer() → Model.infer()
            → [IPC: INFER_ACK]
            → WorkerProxy → return result
```

Backend/Model 运行在 Worker 进程中，接口不变。WorkerProxy 提供与 Backend 相似的 `infer()` 签名，上层调用者无需感知进程边界。

---

## 12. 依赖变更

```toml
# pyproject.toml 新增
dependencies = [
    ...,
    "pyzmq>=26.0",
]
```

---

## 13. 测试策略

### 单元测试（不依赖子进程）

| 测试 | 方法 |
|---|---|
| IPC 消息编解码 | 直接调用 encode/decode 函数 |
| ShmPool | alloc → write → read → free 循环；pool 耗尽测试 |
| Descriptor 序列化 | msgpack round-trip |

### 集成测试（真实子进程）

| 测试 | 方法 |
|---|---|
| Worker 启动 + LOAD_MODEL | spawn worker → send LOAD_MODEL → verify ACK |
| Worker 推理 | load model → send INFER_SUBMIT → verify ACK + output |
| Worker 健康检查 | send HEALTH_CHECK → verify HEALTH_STATUS |
| Worker 崩溃恢复 | kill worker → verify Manager 重启 → verify 推理恢复 |
| Worker 优雅关闭 | send SHUTDOWN → verify process exit |
| SHM 回收 | submit → kill worker → verify SHM slot 回收 |

### 性能测试

- 对比 Phase 0 in-process vs Phase 1 IPC 的 round-trip 延迟
- 不同 payload 大小下的 IPC overhead（预期与 S1 spike 数据一致）

---

## 14. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| ZeroMQ `ipc://` 在 macOS 上行为差异 | 开发环境不可用 | S1 spike 已验证 macOS UDS 正常；ZeroMQ ipc 底层相同 |
| Worker 进程 CUDA 初始化时间长 | 首次推理延迟高 | warmup 机制（LOAD_MODEL 后调用 backend.warmup()） |
| SHM `multiprocessing.shared_memory` 跨平台问题 | macOS/Linux 行为不同 | Phase 1 以功能正确性为主，性能调优放到 Linux 环境 |
| model_class 跨进程传递 | pickle 限制 | 传递 import path 字符串，Worker 侧 importlib 动态加载 |

---

## Changelog

| 日期 | 变更 |
|---|---|
| 2026-02-25 | 初始版本 |
