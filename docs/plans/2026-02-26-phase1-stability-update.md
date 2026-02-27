# 2026-02-26 Phase 1 稳定性收口改动总结

## 背景

本次改动聚焦 Phase 1 的三条关键稳定性链路：

1. `deadline` 语义落地（请求超时返回可区分状态码）
2. `cancel` 协议落地（执行中取消返回 `ABORTED`）
3. 大输出数据路径落地（`output > 8KB` 走 SHM，而非仅 inline）

目标是将现有 Master-Worker IPC 从“可运行”推进到“可作为 Phase 2 基座”。

## 改动范围

### 1) IPC 协议扩展

- 新增消息类型：
  - `SHM_ALLOC_REQUEST`
  - `SHM_ALLOC_RESPONSE`
- 文件：
  - `src/nerva/worker/ipc.py`

### 2) Master 侧 WorkerProxy 增强

- 为每个请求维护 `shm_pool` 上下文，支持 Worker 申请 output SHM 槽位。
- 接收并处理 `SHM_ALLOC_REQUEST`，返回 `SHM_ALLOC_RESPONSE`。
- `infer()` 状态分支细化：
  - `DEADLINE_EXCEEDED` -> 明确报超时
  - `ABORTED` -> 明确报取消
- submit 超时时会 best-effort 发送 `CANCEL`。
- 输出解析支持两条路径：
  - inline descriptor
  - SHM descriptor（并确保 slot 回收）
- 文件：
  - `src/nerva/worker/proxy.py`

### 3) Worker 侧执行语义增强

- 为 in-flight 请求维护 `InferContext` 映射，用于 cancel 标记传播。
- `deadline_ms <= 0`：直接返回 `DEADLINE_EXCEEDED`。
- 推理执行超时：返回 `DEADLINE_EXCEEDED`。
- 接收 `CANCEL`：
  - 标记 `context.cancelled = True`
  - 取消 in-flight task
  - 返回 `ABORTED`（best-effort）
- 大输出路径：
  - Worker 向 Master 发送 `SHM_ALLOC_REQUEST`
  - 收到 `SHM_ALLOC_RESPONSE` 后写入共享内存
  - `INFER_ACK` 携带 SHM descriptor
- 文件：
  - `src/nerva/worker/process.py`

### 4) 测试补充

- 新增测试模型：
  - `BigOutputModel`（用于触发 output SHM 路径）
- 新增/更新测试：
  - `tests/test_worker_proxy.py`
    - deadline exceeded
    - cancel aborted
    - large output via SHM + slot 回收
  - `tests/test_ipc.py`
    - MessageType 枚举覆盖新增消息类型
- 文件：
  - `tests/helpers.py`
  - `tests/test_worker_proxy.py`
  - `tests/test_ipc.py`

## 验证结果

### 静态检查

- `uv run ruff check ...`（改动文件）：通过
- `uv run mypy`：通过（`Success: no issues found in 16 source files`）

### 测试

- 定向回归：
  - `uv run pytest tests/test_ipc.py tests/test_worker_proxy.py tests/test_worker_process.py tests/test_phase1_e2e.py -q`
  - 结果：`38 passed`
- 全量测试：
  - `uv run pytest -q`
  - 结果：`89 passed`

## 已知边界

当前取消/超时仍为 **best-effort**：

- `asyncio.to_thread()` 中已启动的底层推理任务无法被强制硬中断；
- 协议层会尽快返回 `ABORTED` / `DEADLINE_EXCEEDED`，但底层计算可能在短时间内继续执行直到自然结束。

该行为符合当前 MVP 约束，但后续若要做更强语义，需要 backend 侧提供 cooperative checkpoint cancel 或更细粒度中断机制。

## 建议的下一步

1. 在文档层同步更新 `docs/plans/2026-02-25-mvp-roadmap.md` 的 Phase 1 状态，避免“实现已完成但 roadmap 仍显示待实现”的漂移。
2. 将 `deadline/cancel/output SHM` 的关键 e2e 用例纳入 CI 必跑集（非 slow）。
3. 进入 Phase 2 时优先做显式 DAG Executor，先复用当前 Worker IPC 基座，再引入 trace 子集。

---

## 2026-02-26 第二轮收口（测试与可维护性）

基于代码 review 反馈，本轮补充了测试工程化与边界路径覆盖：

### 1) 测试去重（fixture 化）

- 将 `tests/test_worker_proxy.py` 中重复的 worker 启停与 model load 逻辑抽为 `started_worker` fixture。
- 效果：
  - 测试可读性提升
  - 生命周期清理逻辑集中，减少重复 teardown 漏改风险

### 2) 边界/错误路径补测

- 新增 `deadline_ms <= 0` 直接拒绝路径测试（期望 `DEADLINE_EXCEEDED`）。
- 新增 output SHM 资源不足路径测试（期望 `RESOURCE_EXHAUSTED`）。
- 新增 SHM 槽位竞争测试（两个请求竞争单槽位，第二个请求返回 `RESOURCE_EXHAUSTED`）。

### 3) 行为细化

- Worker output SHM 分配失败支持携带上游状态码（不再一律折叠为 `INTERNAL`）。
- output SHM alloc 超时从硬编码改为参数化：
  - `DEFAULT_SHM_ALLOC_TIMEOUT_S`
  - `_WorkerLoop(..., shm_alloc_timeout_s=...)`
- 降低 cancel/timeout 后“晚到 ACK”日志噪声：
  - 对已完成请求的晚到 `INFER_ACK` 由 warning 降为 debug。

### 4) 回归结果（第二轮）

- `uv run pytest -q`：`92 passed`
- `uv run ruff check ...`：通过
- `uv run mypy`：通过

---

## 2026-02-26 第三轮收口（拷贝路径优化 + 协议补充）

本轮目标是减少 IPC 热路径中的不必要中间拷贝，并将行为明确写入设计文档。

### 1) Descriptor 扩展（兼容 schema v1）

- 新增字段：
  - `payload_codec`（默认 `msgpack_dict_v1`）
  - `input_key`（`raw_bytes_v1` 场景必填）
- 语义：
  - `msgpack_dict_v1`：沿用原有 dict msgpack 编解码
  - `raw_bytes_v1`：单字段 bytes 输入快速路径，跳过 dict 级 `msgpack.packb`

### 2) 输入快速路径（raw bytes）

- `WorkerProxy.infer()` 在输入为单字段 bytes 时走 `raw_bytes_v1`：
  - 不再对整个 `inputs` 做 `msgpack.packb`
  - descriptor 仅携带 raw bytes 与 `input_key`
- Worker 侧按 codec 还原为 `{input_key: bytes}` 后进入 model infer。

### 3) SHM 解码中间副本优化

- SHM + `msgpack_dict_v1` 路径改为 `memoryview -> msgpack.unpackb`，
  避免 `bytes(buf[slice])` 额外临时副本。
- 说明：`raw_bytes_v1` 的 SHM 路径为保持 Python `bytes` 语义，当前仍保留一次 materialization。

### 4) 测试补充

- `tests/test_worker_process.py`
  - 新增 `raw_bytes_v1` descriptor 推理用例。
- `tests/test_worker_proxy.py`
  - 新增“单字段 bytes 不触发输入 dict msgpack 序列化”用例。
- `tests/test_shm_pool.py`
  - 新增 `read_view()` roundtrip 用例。
- `tests/test_ipc.py`
  - 补充 `payload_codec/input_key` 默认值与 roundtrip 断言。

### 5) 文档同步（设计变更落库）

- `docs/plans/ipc-contract.md`
  - Descriptor schema 增加 `payload_codec/input_key`
  - 新增 codec 规则与 `memoryview` 约定
- `docs/plans/2026-02-25-phase1-design.md`
  - 更新请求生命周期、消息格式、测试矩阵和 changelog

### 6) 回归结果（第三轮）

- `uv run pytest tests/ -v`：`150 passed`
- `uv run mypy`：通过
- `uv run ruff check`（改动文件）：通过
