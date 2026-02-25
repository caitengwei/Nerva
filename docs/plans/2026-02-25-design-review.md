# Nerva v0.1 Design Review — Open Questions & Decision Points

> 针对 [nerva-design.md](./2026-02-24-nerva-design.md) 的审阅意见。
> 每个问题下方留有 **[设计回应]** 占位区域，请直接在文档中编辑填写设计理由。

---

## P0 — 架构层面

### 1. Model 与 Backend 的职责边界

**问题：** 文档定义了两个都有推理方法的抽象：

- `nerva.Model`（2.1 节）：用户子类化，实现 `load()` + `infer()`
- `Backend`（5.1 节）：也有 `load_model()` + `infer()`

两者的调用关系不明确：
- 用户的 `Model.infer()` 在什么上下文被调用？直接调用？被 batcher 批量调用？
- Backend 的 `infer(inputs: dict[str, Any])` 接收的是单条请求还是已 batched 的输入？`batch_meta: BatchMeta | None` 的可选性暗示同一接口要处理两种情况。
- Model 内部持有 Backend？还是 Backend 包装 Model？

**需要确认：** Model 和 Backend 的组合关系、调用链路、各自的职责边界。

**[设计回应]**

> nerva.Model 是模型预测节点的抽象，Backend 是模型执行后端的抽象。在infer的行为上，两者有一定的重合，但不能完全等价。
特别是在使用torch的场景中，用户可以自己通过 Model 类来实现自定义的 load 和 infer 代码。torch或者python backend 此时需要解决的是框架和用户 model 高效交互、分布式、异常处理等问题。

在调用关系上，Master Process 通过 Orchestrator 先调用 Backend 的API，然后由 Backend 去指定 Model 的 API。

---

### 2. MVP 阶段自研 Binary Protocol 的 ROI

**问题：** 自研二进制协议在 MVP 阶段的成本与收益不匹配：

- 没有现成工具可调试（curl、grpcurl、Postman 均不可用）
- 需要 client SDK 才能使用，但 client SDK 被排除在 MVP 外
- 增加大量 frame parsing、状态机、错误恢复的实现和测试工作
- 推理服务的延迟瓶颈在 GPU 计算和排队，协议序列化开销相对可忽略

**备选方案：** gRPC (HTTP/2 + protobuf) 或 HTTP + msgpack，已有成熟的 streaming 支持和工具链。

**需要确认：** 是否有具体的性能或功能需求使得 MVP 阶段必须自研协议？还是可以先用现成方案，后续再替换数据面？

**[设计回应]**

> 我觉得可以接受当前mvp自研二进制协议的复杂度，先按照最简单的一个音频数据的传输来设计，后续再考虑复杂的场景。后续一些纯文本场景我觉得可以增加http + json的协议。

---

### 3. vLLM 集成的进程模型

**问题：** vLLM 有自己的进程模型（`AsyncLLMEngine` 内部管理 GPU workers），与 Nerva 的 "一个 Worker 进程 = 一个 Model = 一个 GPU" 模型存在冲突：

- vLLM Worker 嵌套在 Nerva Worker 中？还是 Nerva Worker 只是 thin wrapper？
- Nerva 的 Batcher 对 vLLM 节点被绕过，Scheduler 的控制粒度如何？
- vLLM 内部的内存管理（PagedAttention）和 Nerva 的 shm pool 如何共存？

**需要确认：** Nerva 对 vLLM 节点的实际控制边界在哪里？multi-model 编排场景中 Nerva 对 vLLM 的调度能力有多大？

**[设计回应]**

> vLLM 节点采用“Worker 内 thin wrapper + AsyncLLMEngine”的模式。Nerva 不接管 vLLM 内部调度，仅负责请求准入、deadline/cancel 传播和跨节点编排。
vLLM 节点默认旁路 Nerva batcher；Nerva scheduler 只控制该节点的入队与预算，不控制其内部 token-level 调度。
shm 仅用于 Nerva 进程间数据传递，不替代 vLLM 内部显存管理。

---

## P1 — 设计决策

### 4. Trace 机制 vs 显式 DAG 定义

**问题：** JAX-style tracing 通过 Proxy 对象记录模型调用来构建 DAG。但：

- 与 JAX trace 的类比不准确：JAX trace 捕获数值运算用于编译/微分，Nerva 只记录粗粒度模型调用依赖
- 产生 leaky abstraction：声称 "Pipeline is just a function"，但 4.6 节承认不支持任意 Python 副作用和动态循环
- 用户遇到不支持的 Python 构造时会产生难以理解的错误
- `serve(stream(batch(trace(fn))))` 的嵌套读序违反直觉

**备选方案：** 显式 DAG 定义（类似 Airflow 的 `node >> node`），更简单、更可预测、更易调试。Trace 机制可作为后续迭代的语法糖。

**需要确认：** trace 机制在 MVP 阶段的核心价值是什么？是否值得承担 Proxy 对象、受限子集等复杂度？

**[设计回应]**

> MVP 保留受限 trace 作为默认易用路径，同时提供显式 DAG 兜底路径。
> 对于 trace 不支持的语法，直接抛出可解释错误（含位置和替代建议），避免隐式降级。
> 结论：trace 在 MVP 的价值是降低样板代码；显式 DAG 是可预测性保障。

---

### 5. IPC Shared Memory 的内存管理策略

**问题：** 文档中 IPC 设计缺少关键细节：

- shm pool 的大小如何确定？静态预分配还是动态增长？
- 碎片化如何处理？是否采用 slab/arena 分配器？
- `lifetime_token` "ref-count style reclaim" 过于模糊——跨进程引用计数在进程崩溃时如何回收？
- shm 满时的策略：阻塞等待？拒绝请求？spill to socket？
- 是否考虑内存对齐和 NUMA 亲和性？

**需要确认：** shm pool 的分配策略、容量规划、故障回收机制。

**[设计回应]**

> MVP 采用固定 slab + size classes，启动时预分配 shm 池，不做运行时动态扩容。
> shm 满时快速失败返回 RESOURCE_EXHAUSTED，不阻塞等待。
> 回收机制使用 lifetime_token + timeout GC，并定义回收上界窗口，进程崩溃后按窗口批量回收。
> 另外IPC方案我觉得可以考虑用zeromq替代，我自己测试性能相当不错。

---

### 6. 优雅关闭与故障恢复协议

**问题：** 文档未涉及以下场景：

- Master 收到 SIGTERM 后如何处理 in-flight 请求？是否有 drain 阶段？
- Worker 崩溃后 shared memory 中的残留数据如何清理？
- 多模型 Pipeline 中某个 Worker 不可用时，整个 Pipeline 全部失败还是有 partial failure 语义？
- Worker Manager "auto-restarts" 的具体策略（重试次数、backoff、健康检查间隔）？

**需要确认：** 关闭流程、崩溃恢复、以及 Pipeline 级别的故障语义。即使 MVP 不完整实现，也应明确设计意图。

**[设计回应]**

> Master 收到 SIGTERM 后进入 drain：停止接收新请求，等待在途请求到 deadline 或超时后退出。
> Worker 崩溃时，Worker Manager 按退避策略重启；在途请求统一返回 UNAVAILABLE。
> Pipeline 语义在 MVP 为 fail-fast：任一关键节点失败则整条请求失败，不做 partial success 语义。

---

## P1 — 补充设计

### 7. Master 单进程的性能边界

**问题：** Master 进程承担所有调度、batching、DAG 执行和 HTTP 服务。单个 Python 进程 + asyncio 的潜在瓶颈：

- DAG 执行中如有 CPU-bound 操作（序列化、padding、数据拼接）会阻塞 event loop
- 高并发下 asyncio 回调调度本身可能成为瓶颈
- 无水平扩展 Master 的方案

**需要确认：** 是否分析过 Master 的吞吐量上限？CPU-bound 工作是否需要 offload 到线程池？

**[设计回应]**

> cpu_bound 的操作，比如序列化、padding、数据拼接，会阻塞 event loop，需要 offload 到一个rust/C++模块的线程池中执行。这部分目前可以轻量化实现。

---

### 8. Dynamic Batching 的 padding 与 deadline 交互

**问题：**

- 不同长度输入的 padding 策略不明确：按最大长度？分桶(bucketing)？
- `pad_to_multiple=8` 只解决 tensor core 对齐，没解决 bucket 效率问题
- `max_delay_ms`（batcher 等待时间）和单个请求的 `deadline_ms` 冲突时如何权衡？
- Deadline-aware admission 在 enqueue 时检查，但 batch 中等待时间是否计入 deadline 消耗？

**需要确认：** padding 策略细节，以及 batcher delay 和 per-request deadline 的优先级关系。

**[设计回应]**

> 先不做padding，在调度时挑选size一致能组成batch的，否则不做batch。打开了dynamic batching的用户默认不需要极致低延迟，只是需要延迟和吞吐的平衡点，可以交给用户配置，默认值10ms。batch 等待时间需要计入，实际时间从请求进入服务就开始计算。

---

## P2 — API 与工程细节

### 9. `serve()` 作为 Transform 的定位

**问题：** `serve` 的返回类型（ASGI App）不同于其他 transform（Graph → Graph），打破了 composable transforms 的一致性。此外：

- 无法在不启动 HTTP 服务的情况下直接调用 Pipeline（测试、嵌入式使用）
- 一个 Server 挂载多个 Pipeline 的方式不明确

**建议：** `serve` 应作为独立的 deployment 层，而非 transform。Pipeline 本身应可独立执行。

**需要确认：** 是否同意将 `serve` 从 transform 链中分离？Pipeline 的非 HTTP 调用方式如何支持？

**[设计回应]**

> 同意将serve从transform链中剥离。pipeline本身应该可以独立执行。

---

### 10. Backend.infer() 的类型安全

**问题：** 核心推理接口使用 `dict[str, Any]`，与项目 "Python-first" + mypy strict 的目标矛盾。

**建议：** 使用 TypeVar/Generic 或 typed Protocol 提供类型约束，至少在用户侧 API 层面保证类型安全。

**需要确认：** 是否接受改进类型签名？对 Backend 接口的类型化程度有何预期？

**[设计回应]**

> 接受改进类型签名，

---

### 11. `model()` 全局声明的生命周期

**问题：**

- `model()` 在模块级调用，何时实际加载模型？import 时？`serve()` 启动时？
- 名称参数 `"tokenizer"` 与变量名 `tokenizer` 冗余
- 多个 Pipeline 引用同一 model 声明时，是共享 Worker 还是各创建一个？

**需要确认：** model 声明 → 加载 → 绑定 Worker 的完整生命周期。

**[设计回应]**

> 在model.load()时加载；多pipeline引用同一model声明时，分别单独创建。 

---

### 12. 配置与部署方案

**问题：** 模型路径、设备映射、batch 参数都硬编码在 Python 代码中：

- 没有环境变量 / 配置文件覆盖机制
- 生产环境部署方式未提及（Dockerfile、systemd、k8s 等）
- 不同环境（dev / staging / prod）的配置切换方式不明确

**需要确认：** MVP 阶段是否需要配置管理？还是纯代码配置即可？

**[设计回应]**

> MVP 阶段纯代码配置即可，后续可以考虑支持环境变量 / 配置文件覆盖机制

---

### 13. 其他小项

以下为较小的确认点，可简短回应：

**a) Binary Protocol 的 `crc32` 字段：** 标注为 "MVP optional" 但在固定头部中占 4B。不使用时填 0？还是从头部移除？

**[设计回应]** > 留固定头字段；MVP 未启用时填 0。后续启用校验无需改头布局

**b) `transforms.py` 的拆分：** `trace`, `batch`, `stream`, `serve` 职责差异大，是否应拆分为独立模块？

**[设计回应]** > 同意拆分为 trace/batch/stream/serve 子模块，保留统一导出。

**c) Device 验证：** 用户手动指定 `device="cuda:0"`，GPU 不存在或 VRAM 不足时的行为？

**[设计回应]** > 启动期做 fail-fast 校验（设备存在、可见性、基本容量）；运行期 OOM 返回 RESOURCE_EXHAUSTED 并记录指标。

**d) Protocol 版本兼容性：** version 字段为 1，client/server 版本不匹配时的行为？

**[设计回应]** > version 不匹配直接返回 INVALID_ARGUMENT 并关闭流；不做隐式兼容。

---

## Changelog

| 日期 | 变更 |
|---|---|
| 2026-02-25 | 初始审阅意见 |
| 2026-02-25 | 意见回复 |