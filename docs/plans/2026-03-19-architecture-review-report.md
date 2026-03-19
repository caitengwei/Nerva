# 架构 Review 报告

- 日期：2026-03-19
- 范围：全仓库代码，对照《架构原则指导手册》逐模块 Review
- 工具：5 个并行 code-reviewer subagent，覆盖 Core/DSL、Engine/执行、Worker/进程、Backends+Observability、Server/Serving 五层
- 优先级标准：P1 = 影响性能或压测结论；P2 = 正确性/合规问题但不影响性能；P3 = 改进建议

---

## 总览

| 模块 | P1 | P2 | P3 |
|------|----|----|-----|
| Core/DSL | 0 | 4 | 6 |
| Engine/执行 | 2 | 2 | 5 |
| Worker/进程 | 1 | 5 | 5 |
| Backends+Observability | 2 | 6 | 6 |
| Server/Serving | 3 | 5 | 4 |
| **合计** | **8** | **22** | **26** |

---

## P1 — 必须修复（8 项，均与性能/压测相关）

### 压测结论准确性（2 项）

#### P1-S1: DEADLINE_EXCEEDED 被标记为 retryable=True

- **文件**：`src/nerva/server/rpc.py:38`
- **性能影响**：压测客户端若依赖 `retryable` 字段自动重试，超时请求会被再次发出，导致 QPS 虚高、p99 偏低、error_rate 统计失真，压测结论不可信。
- **问题**：架构原则 §7.5 规定 `DEADLINE_EXCEEDED` 不重试（已超时），当前将其与 `RESOURCE_EXHAUSTED` 同等标记为 retryable。
- **建议**：`retryable = code == ErrorCode.RESOURCE_EXHAUSTED`

#### P1-S2: ErrorCode 缺少 UNAVAILABLE

- **文件**：`src/nerva/server/rpc.py:28-34`
- **性能影响**：Worker 崩溃恢复期间的请求被计为 INTERNAL（不重试）而非 UNAVAILABLE（可重试），压测中 Worker 重启窗口的可用性指标偏差，掩盖真实恢复能力。
- **问题**：Worker 层 `AckStatus.UNAVAILABLE` 已存在（`src/nerva/worker/ipc.py:45`），但 `ErrorCode` 枚举无此值，`_map_exception` 也无对应分支，Worker 崩溃被映射为 INTERNAL(13)。
- **建议**：`ErrorCode` 增加 `UNAVAILABLE = 14`；`_map_exception` 增加 `"UNAVAILABLE" in msg` 分支；retryable 逻辑增加 `UNAVAILABLE`。

### 热路径性能（4 项）

#### P1-PERF-1: _in_flight_futures list.remove() O(N)

- **文件**：`src/nerva/engine/batcher.py:70,188,223`
- **性能影响**：每个 batch dispatch 后，对每条 valid 请求调用 `list.remove()`，复杂度 O(batch_size × in_flight_count)。高 QPS + 大 batch_size 场景下退化明显，是 Batcher 热路径的隐藏瓶颈。
- **建议**：`_in_flight_futures` 改为 `set[asyncio.Future]`，deregister 改用 `discard()`（O(1)）。

#### P1-PERF-2: timing.write() 在调用方线程执行 json.dumps

- **文件**：`src/nerva/observability/timing.py:85`
- **性能影响**：`json.dumps` 在 event loop 线程同步执行（~2-5 us/次），每请求最多 4 次 timing.write（rpc + executor + proxy + infer），累计 ~8-20 us/req，与 per-request 对象分配开销（~6 us）同量级，高 QPS 下不可忽略。
- **建议**：将 dict 直接入队，`json.dumps` 移到 writer thread 的 `_writer_loop` 中执行，热路径只剩 `put_nowait`（~0.1 us）。

#### P1-PERF-3: _parse_frames 中 data[offset:] 每次循环内存拷贝

- **文件**：`src/nerva/server/rpc.py:56-64`
- **性能影响**：每次循环 `data[offset:]` 创建新 bytes 对象。三帧请求（OPEN+DATA+END）产生约 2/3 + 1/3 的额外拷贝。payload 接近 4MiB 上限时额外拷贝量 ~6.7 MiB，大 payload 场景下直接影响吞吐。
- **建议**：`decode_frame` 增加 `offset` 参数，内部用 `struct.unpack_from(fmt, data, offset)` + `memoryview`，`_parse_frames` 循环传入递增 offset，消除所有中间拷贝。

#### P1-PERF-4: _run_infer_sync 每次请求创建新 event loop

- **文件**：`src/nerva/worker/process.py:357-365`
- **性能影响**：`asyncio.run()` 每次创建 + 销毁完整 event loop，耗时数十微秒。仅影响 `async_infer=False` 的 sync 模式，但如果压测使用 sync backend（如纯 CPU 推理或 mock），此开销在 GPU 计算时间很短时比例显著。
- **建议**：在 thread pool 线程中维护长驻 event loop（`loop.run_until_complete()`）；或为 sync backend 提供 `infer_sync()` 可选方法，完全跳过 event loop。

### 压测稳定性（2 项）

#### P1-PERF-5: ShmPool 匹配 size class 满时直接报错不 fallback

- **文件**：`src/nerva/engine/shm_pool.py:159-164`
- **性能影响**：突发流量导致某个 size class 瞬时耗尽时，即使更大 class 有空余 slot，请求也立即失败（`ShmPoolExhausted`）。压测中 error_rate 偏高，但实际资源未耗尽，导致压测结论偏悲观。
- **建议**：增加 `allow_promotion: bool = False` 选项，启用后 fallback 到最近的更大 size class；默认行为不变。

#### P1-PERF-6: AsyncTimingSink._queue 无容量上限

- **文件**：`src/nerva/observability/timing.py:48`
- **性能影响**：长时压测（数小时）中若写盘 IO 卡住，timing 数据无限积压在内存，可能导致进程 OOM，压测中途崩溃，无法取得完整结果。
- **建议**：改用 `queue.Queue(maxsize=100_000)`，`write()` 改用 `put_nowait()` + 捕获 `queue.Full` 丢弃并限频 warning。

---

## P2 — 应当修复（22 项）

### Core/DSL（4 项）

| # | 文件:行 | 问题 |
|---|---------|------|
| D1 | `primitives.py:61-71,120-140` | `cond`/`parallel` 中 `swap_graph` 无 try/finally，分支函数抛异常时 `ctx.graph` 不恢复到 parent_graph |
| D2 | `model.py:116` | `_model_registry` 无公开 `clear_model_registry()` API，测试直接操作私有变量 |
| D3 | `proxy.py:97-125` | `_extract_proxy_edges` 对 list/tuple 含 Proxy 输入静默返回空 edges，应抛 `TypeError`（违反 §2.5 明确错误原则） |
| D4 | `graph.py:30-31,52-53` | `Node`/`Graph` 是可变 dataclass，trace 完成后未冻结；`Node` 可改为 `frozen=True` |

### Engine/执行（2 项）

| # | 文件:行 | 问题 |
|---|---------|------|
| E1 | `executor.py:373,399` | `_execute_parallel`/`_execute_cond` 创建子 Executor 时未传递 `_precomputed`，每请求重建 PrecomputedGraph（含拓扑排序），违反 §6.4 摊销原则 |
| E2 | `executor.py:73-79` | PrecomputedGraph dict 字段未真正冻结，`frozen=True` 只防字段重赋值不防 dict 内容 mutation；建议 `MappingProxyType` 或文档标注约定 |

### Worker/进程（5 项）

| # | 文件:行 | 问题 |
|---|---------|------|
| W-OPS1 | `process.py` 全文件 | 孤儿进程 watchdog 缺失：Master 被 kill -9 后 Worker 永久残留，占用 GPU/端口，需实现 ppid 周期检查 |
| W-OPS2 | `manager.py:328-381` | 重启退避策略未实现：只有 MAX_RESTARTS=5 上限，无 1s→2s→4s→16s 延迟，Worker 快速崩溃循环 |
| W1 | `manager.py:205-227` | lock file 与 socket file 之间存在 TOCTOU 窗口，多 uvicorn worker 同时启动时可能出现双 spawner |
| W2 | `manager.py:45-57,437-439` | `_refcount_decr` 双重 shutdown 竞态：两个 uvicorn worker 几乎同时降 refcount 到 0，各自执行 shutdown |
| W4 | `proxy.py:66-67` | `_health_future`/`_load_model_future` 单 slot 不支持并发等待，并发调用抛 RuntimeError |

### Backends+Observability（6 项）

| # | 文件:行 | 问题 |
|---|---------|------|
| B-VL1 | `vllm.py:110-131` | VLLMBackend 缺少 `context.cancelled` 检查（违反 §5.5 协作式取消），LLM 生成持续数秒期间 cancel 不生效 |
| B-VL2 | `vllm.py:133-138` | `infer_stream()` 签名多了未使用的 `batch_meta` 参数，与 Backend ABC 不一致，破坏接口契约 |
| B1 | `vllm.py:102-108` | `unload_model()` 只设 `None`，vLLM AsyncLLMEngine 的 GPU 显存依赖 GC 回收，Worker 重启时可能 OOM |
| B2 | `pytorch.py:49` | `load_model()` 直接赋值 `instance._options`（私有属性），语义不清晰 |
| B4 | `metrics.py:82-87` | `get_metrics()` lazy singleton 非线程安全，并发调用可能重复注册 Prometheus collectors |
| B5 | `timing.py` 整体 | 缺少独立单元测试（多线程 writer、sentinel、stop 超时等关键路径，违反 §8.1） |

### Server/Serving（5 项）

| # | 文件:行 | 问题 |
|---|---------|------|
| S-STR | `rpc.py:46-53` | `_map_exception` 基于字符串匹配错误码：脆弱、有优先级隐患、缺 UNAVAILABLE 分支。短期补 UNAVAILABLE + docstring 约束；中期改 typed exception |
| S2 | `rpc.py:176-190` | body read 完成后无 deadline 二次检查；大 payload + 低 deadline 场景下请求进入 pipeline 时已超时 |
| S3 | `rpc.py:101+` | RPC handler 未校验 Content-Type，非法请求在帧解析时报 ProtocolError，错误信息不友好 |
| S4 | `serve.py:300-319` | `live_executors` 闭包变量必须 `.update()` 而非赋值，缺少注释说明，重构时易断引用 |
| S5 | `serve.py:332-360` | `serve()` 硬编码 `workers=1`，未暴露 uvicorn workers 参数，无法通过 API 利用多核 |

---

## P3 — 观察项（26 项，简要列出）

| 模块 | 项目 |
|------|------|
| Engine | SlotBitmap O(N) 扫描（默认 16 slots 无影响）；deadline 计算基准点偏乐观；Executor 多节点并发失败只记录第一个；cond 空分支返回 predicate 语义不直观；ShmPool `_try_unlink` 只捕获 `FileNotFoundError` |
| Core/DSL | `Graph.node_map` property 每次重建 dict；`trace()` 不验证函数签名；parallel 输出用 str(i) 作为 key 不够直观；`_model_registry` 全局状态跨 pipeline 共享；`trace()` 忽略 pipeline 函数返回值 |
| Worker | `itertools.count()` 模运算随时间增长；IPC 无协议版本验证；`import_path_to_class` 不支持嵌套类；`_WorkerLoop._send_lock` 中 assert 在 -O 模式被移除；`daemon=False` 无注释说明原因 |
| Backend | VLLMBackend 每次 infer 都 `import vllm`（有缓存，但非必要）；`ModelConfig.model_class` 类型为 `type[Any]` 失去类型安全；`Backend.health_check()` 默认返回 True 语义存疑；`register_backend` 返回类型标注为 `Any`；`PyTorchBackend` 使用 stdlib logging 而非 structlog |
| Server | `_parse_frames` 静默忽略重复帧类型；`_NervaASGIApp.__del__` GC 清理静默失败无日志；`_NervaASGIApp._get_lock()` 的 None 检查假设单线程；metrics `/metrics` 端点使用全局 registry 影响测试隔离；request_id 在 RPC 帧为 u64 但内部透传为 str |

---

## 残余风险汇总

| 风险 | 来源 | 影响 |
|------|------|------|
| 错误码语义错误 | P1-S1/S2 | 压测结论失真；客户端重试策略混乱 |
| Batcher O(N) 退化 | P1-PERF-1 | 高 QPS + 大 batch 场景热路径性能下降 |
| timing json.dumps 热路径 | P1-PERF-2 | 高 QPS 下累计 8-20 us/req 不必要序列化开销 |
| timing 队列无上限 | P1-PERF-6 | 长时压测内存积压，可能 OOM 中途崩溃 |
| 孤儿进程 | W-OPS1 | 生产环境 Master crash 后 GPU/端口资源泄漏 |
| VLLMBackend cancel 无效 | B-VL1 | 长生成请求 cancel 后持续占用 GPU |

## 测试盲区汇总

1. 孤儿进程：无 Master crash 后 Worker 自行退出测试
2. 退避延迟：无重启间隔验证
3. VLLMBackend cancel：未实现，也未测试
4. cond 空分支 / parallel 部分失败：无测试覆盖
5. timing 独立单元测试缺失
6. 多 uvicorn worker 并发 spawn 竞争：无覆盖
7. body read 期间 deadline 漂移：无测试

---

## 建议修复顺序

**优先修复（影响性能/压测）**

1. **P1-S1**（rpc.py:38）：一行改动，消除 DEADLINE_EXCEEDED retryable 误标，压测结论立即可信
2. **P1-S2**（rpc.py:28-34）：补 UNAVAILABLE 错误码，Worker 崩溃场景正确统计
3. **P1-PERF-1**（batcher.py）：list → set，一行改动消除 O(N) 热路径风险
4. **P1-PERF-2**（timing.py:85）：json.dumps 移到 writer thread，热路径减少 ~15 us/req
5. **P1-PERF-6**（timing.py:48）：加 `maxsize`，防长时压测 OOM
6. **P1-PERF-3**（rpc.py:56-64）：`decode_frame` 加 offset 参数，消除拷贝
7. **P1-PERF-4**（process.py:357-365）：sync 模式长驻 event loop（如压测场景用 sync mode）
8. **P1-PERF-5**（shm_pool.py）：SHM size class fallback 降低突发场景 error_rate

**次优先（正确性/合规）**

9. **B-VL1/B-VL2**（vllm.py）：cancel 合规 + 接口一致性
10. **W-OPS1**（process.py）：孤儿进程 watchdog
11. **W-OPS2**（manager.py）：退避延迟
12. **S-STR**（rpc.py:46-53）：补 UNAVAILABLE 分支 + docstring 约束
13. 其余 P2 根据优先级排期迭代
