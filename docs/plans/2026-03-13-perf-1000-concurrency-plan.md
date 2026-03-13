# Nerva 1000 并发性能优化计划

**日期:** 2026-03-13
**基于:** `docs/bench/2026-03-12-framework-overhead-analysis.md` 第七章优化结论
**目标:** 单实例支撑 1000 并发推理请求，无 error，p99 尾延迟可控

---

## 一、当前架构在 1000 并发下的瓶颈总览

### 1.1 架构约束

当前每个模型对应 1 个 Worker 进程，通过 1 个 ZMQ PAIR socket 与 Master 通信。

```
HTTP Client ×1000
    ↓
[Master 进程 — 单 asyncio event loop]
    ↓  WorkerProxy._send_lock (asyncio.Lock)
    ↓  ZMQ PAIR socket (1:1)
    ↓
[Worker 进程 — 单 asyncio event loop + ThreadPoolExecutor]
    ↓  recv loop 串行分发
    ↓  asyncio.to_thread → asyncio.run(backend.infer())
    ↓  _send_lock (asyncio.Lock)
    ↓
[Backend — 用户 Python 模型代码]
```

### 1.2 瓶颈清单与量化

| 瓶颈 | 位置 | 耗时/影响 | 严重度 |
|---|---|---|---|
| **ThreadPool 容量** | `process.py:208` `asyncio.to_thread()` 默认 `min(32, cpu+4)` 线程 | 8 核机器 = 12 线程，吞吐硬上限 12/0.128s = **94 req/s** | **致命** |
| **`asyncio.run()` 双重包装** | `process.py:280` 每次推理创建+销毁临时 event loop | ~200-500μs GIL-held/次，上限 **~2000 req/s** | **高** |
| **recv loop 串行分发** | `process.py:93-112` 单协程 recv → decode → dispatch | 突发 1000 消息排队 ≈ **25ms** | **中** |
| **send_lock 串行化** | `process.py:362-365`, `proxy.py:305-309` | 100 个 ACK 同时完成 → 最大等待 **4ms** | **中** |
| **SHM Pool 容量** | `shm_pool.py` 默认 16 slots/size class | 1000 并发 × 大 payload → `ShmPoolExhausted` | **高** |
| **Master 侧 4 次重复序列化** | rpc.py + proxy.py 全链路 | 同一份数据 msgpack 编解码 4 次 | **中** |
| **单 uvicorn event loop** | 默认单进程 | 1000 HTTP handler + 1000 executor + ZMQ I/O 共享一个 loop | **中** |

---

## 二、单 Worker 进程吞吐上限推导

假设 backend.infer() 耗时 128ms，完全释放 GIL（最优情况）：

```
限制因素 1: ThreadPool 容量
  默认 12 线程 × (1000ms / 128ms) = 93.75 req/s
  增大到 128 线程 → 1000 req/s

限制因素 2: asyncio.run() GIL 串行化
  每次 ~500μs GIL-held
  上限 = 1 / 0.0005 = 2,000 req/s（即使线程池无限大）

限制因素 3: send_lock 串行化
  每次 ~40μs → 上限 25,000 send/s → 非瓶颈

限制因素 4: recv loop 串行分发
  每条 ~25μs → 上限 40,000 msg/s → 非瓶颈

结论: 单 Worker 理论上限 = min(ThreadPool 容量, asyncio.run 上限) = ~2,000 req/s
```

---

## 三、`asyncio.to_thread` + `asyncio.run` 问题分析

### 3.1 当前执行路径

```python
# Worker event loop 中
output = await asyncio.to_thread(self._run_infer_sync, inputs, context)

# 在 ThreadPool 线程中
def _run_infer_sync(self, inputs, context):
    return asyncio.run(self._backend.infer(inputs, context))
```

路径：`协程 → ThreadPool 线程 → 新 event loop → 协程`

### 3.2 开销拆解

| 操作 | 耗时 | GIL 状态 |
|---|---|---|
| ThreadPool 调度（排队 + 线程唤醒） | 10-50μs | held |
| `asyncio.run()` 创建 event loop | 100-500μs | held |
| `asyncio.run()` 销毁 event loop | 50-200μs | held |
| **合计 overhead / 调用** | **~200-750μs** | |

### 3.3 根本矛盾

`backend.infer()` 定义为 `async def`，但需要在线程中执行以避免阻塞 Worker event loop。
`asyncio.to_thread` 需要 sync callable，`asyncio.run()` 是从 sync 调 async 的唯一标准方式。

### 3.4 解法

- **方案 A（推荐）**: Backend 新增 `infer_sync()` 方法，直接在 `to_thread` 中调用，省去 `asyncio.run()` → 消除 200-500μs/次
- **方案 B**: 真正 non-blocking async 的 backend（如 vLLM wrapper），直接在 Worker event loop 中 `await`，不走 `to_thread`
- **方案 C**: Worker Pool 每个 Worker 只处理一个请求（类似 Triton model instance），用进程隔离替代线程并发

---

## 四、GIL-held 操作全链路清单

### 4.1 Master 进程（每请求）

| 操作 | 代码位置 | GIL 耗时 (μs) | payload 敏感 |
|---|---|---|---|
| `_parse_frames(body)` | rpc.py:57-64 | 5-20 | 是 |
| `_classify_frames(frames)` | rpc.py:67-81 | 1-2 | 否 |
| `msgpack.unpackb(open_frame)` | rpc.py:210 | 1-5 | 否 |
| `msgpack.unpackb(inputs)` | rpc.py:247 | 10-5000 | **是** |
| `msgpack.packb(inputs)` IPC | proxy.py:174 | 10-5000 | **是** |
| IPC 信封 encode | proxy.py:214-218 | 10-30 | 否 |
| IPC 信封 decode (ACK) | proxy.py:325 | 5-15 | 否 |
| `msgpack.unpackb(output)` | proxy.py:252-254 | 10-5000 | **是** |
| `msgpack.packb(result)` 响应 | rpc.py:293-298 | 10-5000 | **是** |
| structlog + Prometheus | rpc.py:260-285 | 10-30 | 否 |
| **合计（1KB payload）** | | **~80-150μs** | |
| **合计（1MB payload）** | | **~10-20ms** | |

**注意**：用户数据在 Master 中被 msgpack 编解码 4 次（M4 解包 HTTP body、M5 打包发 IPC、M8 解包 IPC 返回、M9 打包 HTTP 响应）。

### 4.2 Worker 进程（每请求，不含 backend.infer）

| 操作 | 代码位置 | GIL 耗时 (μs) | payload 敏感 |
|---|---|---|---|
| `decode_message(raw)` | process.py:100 | 5-15 | 否 |
| `msgpack.unpackb(inputs)` | process.py:186-188 | 10-5000 | **是** |
| `asyncio.to_thread()` 调度 | process.py:208 | 10-50 | 否 |
| **`asyncio.run()` 创建+销毁** | process.py:280 | **200-500** | 否 |
| `msgpack.packb(output)` | process.py:232 | 10-5000 | **是** |
| `encode_message(ack)` | process.py:235-240 | 10-30 | 否 |
| structlog bind/clear | process.py:180, 270 | 5-10 | 否 |
| **合计（1KB payload）** | | **~260-620μs** | |
| **合计（1MB payload）** | | **~10-20ms** | |

---

## 五、Rust 高性能模块分层分析

### 5.1 各层级方案

#### L1: ormsgpack Drop-in 替换（1 小时）

```python
import ormsgpack as msgpack  # Rust 实现，API 兼容
```

- 编解码速度 3-5×，但仍持有 GIL
- **不解决 asyncio.run() 问题**

| payload | 当前 GIL | ormsgpack GIL | 改善 |
|---|---|---|---|
| 1KB | ~120μs (Master) | ~90μs | -25% |
| 1MB | ~22ms (Master) | ~6ms | -73% |

#### L2: Rust PyO3 扩展模块——IPC codec + frame protocol（1-2 周）

```python
from nerva_native import encode_ipc_message, decode_ipc_message, parse_frames
```

- 合并多次编解码为一次 Rust 调用
- **大 payload 可释放 GIL**（`py.allow_threads`）——ormsgpack 做不到
- 零拷贝 memoryview 返回

| payload | 当前 GIL | Rust codec GIL | 改善 |
|---|---|---|---|
| 1KB | ~120μs (Master) | ~50μs | -58% |
| 1MB | ~22ms (Master) | ~2ms (含 ~3ms non-GIL Rust 计算) | -91% |

#### L3: Rust Worker Loop — PyO3 + tokio（3-6 周）

用 Rust 重写整个 `_WorkerLoop`：
- ZMQ recv/send、消息编解码、任务分发全在 tokio 中运行，不涉及 GIL
- **唯一需要 GIL 的操作**：调用 Python `backend.infer()`
- 彻底消除 asyncio.run()、to_thread 调度、Python asyncio 协程切换

| payload | 当前 framework GIL | Rust Worker GIL | 改善 |
|---|---|---|---|
| 1KB | ~300μs (Worker) | ~50μs | -83% |
| 1MB | ~12ms (Worker) | ~200μs | -98% |

**关键约束**：`backend.infer()` 是 `async def`，PyO3 调用仍需某种 event loop 桥接（pyo3-asyncio）。若 Backend 改为 sync 接口则可彻底消除开销。

#### L4: Rust Worker + 零拷贝 SHM（2-3 个月）

- 输入/输出数据留在 SHM，Rust 解析元数据，Python 仅拿 tensor view
- framework GIL → ~0
- 需要 Backend API 根本性变更

### 5.2 各层级对瓶颈的消解能力

| 瓶颈 | L1 ormsgpack | L2 Rust codec | L3 Rust Worker | L4 零拷贝 |
|---|---|---|---|---|
| asyncio.run() 200-500μs | ❌ | ❌ | ✅（需 sync Backend） | ✅ |
| ThreadPool 12 线程 | ❌ | ❌ | ✅ tokio 替代 | ✅ |
| msgpack 小 payload | ⚠️ -25% | ✅ -60% | ✅ -95% | ✅ ~0 |
| msgpack 大 payload | ✅ -73% | ✅ -91% | ✅ -95% | ✅ ~0 |
| recv loop 串行分发 | ❌ | ⚠️ decode 加速 | ✅ Rust native | ✅ |
| send_lock 串行化 | ❌ | ⚠️ encode 加速 | ✅ lock-free | ✅ |
| 4 次重复序列化 | ⚠️ 加速但次数不减 | ✅ 可合并 | ✅ | ✅ 消除 |
| backend.infer() GIL | ❌ | ❌ | ❌ | ❌ |

### 5.3 单 Worker 框架吞吐上限对比

```
                    小 payload (1KB)     大 payload (1MB)
当前 Python          ~2,000 req/s        ~50 req/s
L1 ormsgpack         ~2,000 req/s        ~170 req/s (+240%)
L2 Rust codec        ~2,000 req/s        ~500 req/s (+900%)
L3 Rust Worker       ~20,000 req/s       ~5,000 req/s (+9900%)
```

小 payload 时 L1/L2 无法突破 asyncio.run() 的 2000 req/s 天花板。

---

## 六、优化实施路线（按优先级）

### P0: Worker Pool — 多 Worker 进程/模型（最高优先级）

**目标**: 解锁水平扩展，消除单进程吞吐天花板。

**设计**:
```
当前:  ASGI → WorkerProxy(PAIR) → Worker(1 process)
目标:  ASGI → WorkerPoolProxy(DEALER) → [Worker₁, Worker₂, ..., Workerₙ]
```

- 替换 ZMQ PAIR → DEALER/DEALER，ZMQ 原生支持多 worker 分发
- Worker 数量可配置：`model(name="llm", replicas=4)`
- 消除 send_lock 竞争（每个 Worker 独立 socket）

**预期**: N 个 Worker → 吞吐线性扩展至 N × 单 Worker 上限

### P1: Backend sync 接口 + asyncio.run 消除

**目标**: 消除单 Worker 最大框架开销项（200-500μs/次）。

**设计**:
```python
# Backend 新增 sync 方法
class Backend(ABC):
    async def infer(self, inputs, context, batch_meta=None) -> dict:  # 保留
        ...
    def infer_sync(self, inputs, context, batch_meta=None) -> dict:   # 新增
        return asyncio.run(self.infer(inputs, context, batch_meta))   # 默认实现

# Worker 中：
def _run_infer_sync(self, inputs, context):
    return self._backend.infer_sync(inputs, context)  # 不再 asyncio.run()
```

PyTorchBackend 等具体实现 override `infer_sync()` 为真正的 sync 调用。
真正 async 的 backend（vLLM）在 Worker event loop 中直接 `await`，不走 `to_thread`。

**纯 Python 改动，1-2 天。单 Worker 小 payload 吞吐从 ~2000 提升至 ~10000+ req/s。**

### P2: SHM Pool 扩容

**目标**: 避免 1000 并发下资源耗尽。

- `slots_per_class` 可配置，默认根据 `replicas` 自动计算
- bitmap 从 `int` 位操作改为 `bytearray` 支持 >64 slots
- 或改为动态扩容策略

### P3: ormsgpack Drop-in（L1）

**目标**: 零风险拿到大 payload 序列化加速。

```toml
# pyproject.toml
[project.optional-dependencies]
perf = ["httptools>=0.6", "ormsgpack>=1.0"]
```

运行时自动检测：
```python
try:
    import ormsgpack as msgpack
except ImportError:
    import msgpack
```

### P4: 原生 ASGI handler 替换 Starlette RPC 路径

**目标**: 消除 rpc_body_read 的 Starlette 抽象开销。

- RPC 热路径直接 `await receive()` 读 body
- 保留 Starlette 仅用于 `/v1/health`、`/v1/models` 等管理接口

### P5: Rust codec 模块（L2）（大 payload 场景按需）

仅在大 payload 场景成为瓶颈时实施。P3 (ormsgpack) 通常已足够。

### P6: Linux uvloop 验证

macOS kqueue 下 uvloop 与 ZMQ ipc:// 不兼容（已验证），但 Linux epoll 可能无此问题。需在 Linux 环境单独压测验证。

---

## 七、验证计划

### 7.1 压测矩阵

```
并发级别: 1, 4, 16, 64, 256, 1000
Worker replicas: 1, 2, 4, 8
Backend: CPU mock (MOCK_TOKEN_LATENCY_MS=0.5, 256 tokens = 128ms)
Payload: 1KB (small dict), 1MB (image-like)
```

### 7.2 关键指标

| 指标 | 1000 并发目标 |
|---|---|
| error_rate | 0 |
| e2e p50 | < 150ms（mock 128ms + 框架 < 22ms） |
| e2e p99 | < 300ms |
| QPS | > 5000（需 Worker Pool） |
| framework overhead p50 | < 5ms |
| framework overhead p99 | < 15ms |

### 7.3 验证命令

```bash
# 带 timing 埋点启动
NERVA_TIMING_LOG_DIR=/tmp/nerva_timing MOCK_TOKEN_LATENCY_MS=0.5 \
  uv run uvicorn examples.mm_vllm_cpu_mock_server:app \
    --host 127.0.0.1 --port 8080

# 压测
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  uv run python scripts/bench/run_bench.py \
    --target nerva --concurrency-levels 1,4,16,64,256,1000 \
    --warmup-seconds 5 --sample-seconds 15

# 分析 timing
uv run python scripts/bench/analyze_timing_log.py --log-dir /tmp/nerva_timing
```
