# Nerva IPC Contract (MVP)

## 1. Goal

定义 Master 与 Worker 的低延迟 IPC 契约，确保：
- 大 payload 不经 UDS 复制
- 生命周期可回收、可观测、可恢复
- 进程异常时不会长期泄漏 shm

默认参数来源：[`mvp-defaults.md`](./mvp-defaults.md)

## 2. Channels

控制通道（UDS + msgpack）：
- 用途：命令、descriptor、ack/error、健康检查
- 数据规模：小消息（建议 `<= IPC_CONTROL_INLINE_MAX_BYTES`）

数据通道（POSIX Shared Memory）：
- 用途：输入/中间/输出 payload 与 tensor bytes
- 数据规模：中大消息（`> IPC_CONTROL_INLINE_MAX_BYTES`）

## 2.1 End-To-End Sequence (Unary)

```text
Client            Master                     Worker
  |                 |                          |
  |   RPC request   |                          |
  |---------------> |                          |
  |                 | alloc shm slot           |
  |                 | write payload to shm     |
  |                 |--INFER_SUBMIT(desc)----> |
  |                 |                          | read shm
  |                 |                          | run backend infer
  |                 | <-INFER_ACK(out_desc)--- |
  |                 | reclaim input desc       |
  |  RPC response   |                          |
  | <-------------- |                          |
```

## 3. Message Types (Control Plane)

`LOAD_MODEL`
- `model_name`, `backend`, `device`, `options`

`INFER_SUBMIT`
- `request_id`, `node_id`, `deadline_ms`, `descriptor`, `batch_meta` (optional, None before Phase 3)

`INFER_ACK`
- `request_id`, `node_id`, `status`, `out_descriptor?`, `error?`

`INFER_ACK.status` 枚举：
- `OK`
- `INVALID_ARGUMENT`
- `DEADLINE_EXCEEDED`
- `ABORTED`
- `RESOURCE_EXHAUSTED`
- `UNAVAILABLE`
- `INTERNAL`

`CANCEL`
- `request_id`, `reason`

`HEALTH_CHECK`
- `worker_id`

`HEALTH_STATUS`
- `worker_id`, `ok`, `detail`

## 4. Descriptor Schema

```text
descriptor = {
  schema_version: u16,        # fixed: 1
  request_id: str,            # unique request identifier (e.g. UUID)
  node_id: u32,
  # SHM path (None when inline)
  shm_id: str | None,         # POSIX shm name; None for inline payload
  offset: u32,
  length: u32,
  # Inline payload (None when using SHM)
  inline_data: bytes | None,  # payload bytes when <= IPC_CONTROL_INLINE_MAX_BYTES
  payload_codec: str,         # "msgpack_dict_v1" | "raw_bytes_v1"
  input_key: str | None,      # raw_bytes_v1 场景下表示 bytes 应映射到 inputs 的哪个 key
  dtype: str,                 # e.g. "float16", "int64", "bytes"
  shape: list[u32],           # bytes payload may use [length]
  device: str,                # "cpu" / "cuda:0" ... (reserved, Phase 1 不使用)
  lifetime_token: u64,        # reclaim token (reserved, Phase 1 不启用 TTL GC)
  checksum: u32,              # optional, 0 = disabled (reserved, Phase 1 不启用)
}
```

Inline 优化规则：
- 当 payload <= `IPC_CONTROL_INLINE_MAX_BYTES` (8KB) 时，`shm_id = None`，数据放入 `inline_data`
- 当 payload > 8KB 时，`inline_data = None`，数据通过 SHM 传输

Payload codec 规则：
- `payload_codec = "msgpack_dict_v1"`：默认路径，`inline_data/SHM` 承载的是 `msgpack.packb(dict)` 结果。
- `payload_codec = "raw_bytes_v1"`：输入为单字段 bytes 时的快速路径，不再对该字段做 `msgpack.packb(dict)`；`input_key` 必填。
- 未识别 `payload_codec` 或 `raw_bytes_v1` 缺少 `input_key` 时，返回 `INFER_ACK(status=INVALID_ARGUMENT)`。

兼容策略：
- `schema_version` 不匹配时，返回 `INFER_ACK(status=INVALID_ARGUMENT)` 并拒绝执行

## 5. Shared Memory Pool Design

默认方案：`slab + size classes`
- 典型 size class：`IPC_SHM_SIZE_CLASSES`
- 每类固定数量槽位，启动时预分配
- 分配策略：向上取整匹配最小可容纳 class

回收策略：
- 正常路径：Worker `ACK` 后回收
- 异常路径：基于 `lifetime_token` + TTL 的惰性 GC

实现约定（减少中间拷贝）：
- SHM + `msgpack_dict_v1` 解码时，优先直接使用 `memoryview` 喂给 `msgpack.unpackb`，避免 `bytes(buf[slice])` 临时副本。
- `raw_bytes_v1` 在 SHM 路径下仍需产出 Python `bytes`（语义兼容），当前实现保留这一次必要复制。

配置项（建议默认）：
- `ipc_shm_total_bytes`: `IPC_SHM_TOTAL_BYTES`
- `ipc_alloc_timeout_ms`: `IPC_ALLOC_TIMEOUT_MS`
- `ipc_reclaim_ttl_ms`: `IPC_RECLAIM_TTL_MS`

## 6. Lifecycle State Machine

单个 descriptor 的状态：
1. `ALLOCATED`
2. `WRITTEN`
3. `SUBMITTED`
4. `CONSUMED`
5. `ACKED`
6. `RECLAIMED`

失败状态：
- `ALLOC_FAILED`
- `EXPIRED`
- `ABORTED`

关键规则：
- 仅 `ACKED/EXPIRED/ABORTED` 可进入 `RECLAIMED`
- 任意状态若进程崩溃，恢复流程必须可判定是否可回收

## 7. Timeout And Failure Handling

故障窗口参数（统一引用默认值）：

| Window | Key | Description |
|---|---|---|
| alloc window | `IPC_ALLOC_TIMEOUT_MS` | shm 分配等待上限 |
| submit window | `IPC_SUBMIT_TIMEOUT_MS` | `INFER_SUBMIT` 到 `INFER_ACK` 最大等待 |
| reclaim window | `IPC_RECLAIM_TTL_MS` | 失联 descriptor 的最长惰性回收窗口 |

提交超时：
- `SUBMITTED` 超过 `ipc_submit_timeout_ms` 未收到 ACK
- `ipc_submit_timeout_ms` 默认值为 `IPC_SUBMIT_TIMEOUT_MS`
- 动作：请求标记 `UNAVAILABLE`，descriptor 转 `EXPIRED`，后台回收

Worker 崩溃：
- Master 检测到 worker 退出后批量扫描其在途 descriptors
- 超过最短安全窗口后统一 `EXPIRED` + 回收
- 在途请求返回 `UNAVAILABLE`

资源最终回收上界：
- `T_reclaim_upper_bound = IPC_SUBMIT_TIMEOUT_MS + IPC_RECLAIM_TTL_MS`
- 该上界用于压测时验证“无长期泄漏”

Master 重启（MVP）：
- 不保证在途请求恢复
- 启动时清理遗留 shm 命名空间（按前缀）

## 8. Cancellation Contract

- 收到 `CANCEL(request_id)` 后：
  - 队列中任务：直接移除并回收 descriptor
  - 执行中任务：通知 Worker 尽快中断；最终统一 `ABORTED`
- 取消是 best-effort，但必须保证最终资源回收

取消与超时冲突规则：
- cancel 先生效 -> 返回 `ABORTED`
- deadline 先触发 -> 返回 `DEADLINE_EXCEEDED`

## 9. Scheduling Integration

`INFER_SUBMIT` 必填：
- `deadline_ms`
- `batch_meta`：Phase 3 之前为 `None`；Phase 3+ 必填 `batch_meta.batch_size`，`batch_meta.pad_to_multiple` 可选

调度器必须在提交前执行：
- deadline 准入检查
- 队列容量检查
- shm 可用容量检查

## 10. Metrics And Logging

必备指标：
- `nerva_ipc_control_latency_seconds{worker}`
- `nerva_ipc_data_bytes_total{worker,direction}`
- `nerva_ipc_shm_alloc_fail_total{worker}`
- `nerva_ipc_descriptor_inflight{worker}`
- `nerva_ipc_reclaim_seconds{worker}`

必备日志字段：
- `request_id`, `node_id`, `worker_id`
- `descriptor.lifetime_token`
- `ipc_state_from`, `ipc_state_to`
- `error_code`, `error_reason`

## 11. Test Matrix (Contract Level)

功能测试：
- 正常提交/ACK/回收
- 大 payload 分配与回收
- cancel 队列内任务
- cancel 执行中任务

故障测试：
- worker 崩溃时回收正确性
- shm 池耗尽时错误返回
- ACK 丢失时超时回收

性能测试：
- 控制通道 p99 延迟
- shm 分配 p99 延迟
- 回收吞吐（descriptors/s）

---

## Changelog

| 日期 | 变更 |
|---|---|
| 2026-02-25 | 初始版本 |
| 2026-02-25 | 修订：Descriptor schema 增加 inline_data 支持和 shm_id nullable；request_id 统一为 str；batch_meta 标注为 Phase 3 前 optional；补充 reserved 字段说明 |
| 2026-02-26 | 修订：Descriptor 增加 `payload_codec/input_key`；新增 `raw_bytes_v1` 快速路径；补充 SHM 解码 `memoryview` 约定以减少中间拷贝 |
