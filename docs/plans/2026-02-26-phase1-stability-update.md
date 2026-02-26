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
