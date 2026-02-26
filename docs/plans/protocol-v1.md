# Nerva Binary Streaming RPC Protocol v1

## 1. Scope

本协议定义 Nerva 数据面的二进制流式 RPC 规范，目标是支持单机多 GPU、低延迟多模型编排。

默认参数来源：[`mvp-defaults.md`](./mvp-defaults.md)
相关决策来源：[`ADR-001`](./adr/ADR-001-payload-encoding.md), [`ADR-003`](./adr/ADR-003-cancel-propagation.md)

- 适用范围：`POST /rpc/{pipeline_name}` 与 `POST /rpc/{request_id}/cancel`
- 支持模式：`Unary`、`Server-Streaming`
- 不包含：双向流式、跨机鉴权、协议加密扩展（依赖上层网关或 mTLS）

## 2. Versioning And Compatibility

- 协议版本字段：`version=1`
- 兼容策略：仅保证同主版本兼容，次版本通过 `header_ext` 扩展
- 不兼容变更：提升主版本号（`version=2`）

## 3. Transport Binding

- 传输层：HTTP/1.1 or HTTP/2 over ASGI（实现为 Starlette + Uvicorn）
- `Content-Type`: `application/x-nerva-rpc`
- `Accept`: `application/x-nerva-rpc`
- 连接建议：keep-alive

请求头：
- `x-nerva-request-id`（可选）：`uint64` 十进制字符串；缺失时服务端生成
- `x-nerva-deadline-ms`（必选）：Unix epoch 毫秒时间戳
- `x-nerva-stream`（必选）：`0`（Unary）或 `1`（Server-Streaming）
- `x-nerva-trace-id`（可选）：透传链路追踪标识

请求头解析规则：
- `x-nerva-request-id` 非法（非 `u64`）时，返回 `ERROR(INVALID_ARGUMENT)`
- `x-nerva-stream` 非法值时，返回 `ERROR(INVALID_ARGUMENT)`
- `x-nerva-deadline-ms` 缺失或非法时，返回 `ERROR(INVALID_ARGUMENT)`

## 4. Frame Layout (Little-Endian)

固定头长度：`PROTOCOL_FIXED_HEADER_BYTES`（见 `mvp-defaults.md`）

| Field | Size | Type | Description |
|---|---:|---|---|
| `magic` | 2B | `u16` | 固定 `0x4E56`（`NV`） |
| `version` | 1B | `u8` | 协议版本，当前为 `1` |
| `type` | 1B | `u8` | `OPEN=1`,`DATA=2`,`END=3`,`ERROR=4`,`HEARTBEAT=5` |
| `flags` | 2B | `u16` | bit0=`compressed`, bit1=`eos` |
| `reserved` | 2B | `u16` | 预留，必须为 `0` |
| `request_id` | 8B | `u64` | 请求 ID |
| `stream_id` | 4B | `u32` | MVP 固定为 `PROTOCOL_STREAM_ID_DEFAULT` |
| `payload_len` | 4B | `u32` | Payload 长度 |
| `crc32` | 4B | `u32` | Payload 校验；MVP 可置 `0` |
| `header_ext_len` | 4B | `u32` | 扩展头长度；MVP 为 `0` |

校验规则：
- `magic`、`version`、`payload_len`、`header_ext_len` 非法即返回 `ERROR(INVALID_ARGUMENT)` 并关闭流
- `payload_len` 必须 `<= PROTOCOL_MAX_FRAME_PAYLOAD_BYTES`
- `request_id`（帧头）必须与请求头中的 `x-nerva-request-id` 一致（若请求头存在）

字段类型约束：
- 整数字段统一 little-endian 无符号整型
- `reserved` 必须为 `0`
- `header_ext_len > 0` 时，扩展头必须能被识别，否则返回 `ERROR(INVALID_ARGUMENT)`

## 5. Payload Semantics

`OPEN`：
- 编码：protobuf（canonical）
- 内容：`pipeline_name`、调用参数 schema 版本、可选 trace 元数据
- 限制：每个请求只允许一个 `OPEN`，且必须为首帧

`DATA`：
- 编码：protobuf envelope（业务数据放入 `bytes`）
- 内容：业务二进制数据 + 序号/分片信息
- 允许多帧分片

`END`：
- 编码：protobuf（canonical）
- 内容：终止元数据（`latency_ms`、`tokens_generated`、`status` 等）
- 请求方向和响应方向都必须以 `END` 收尾（异常可用 `ERROR` 替代响应 `END`）

`ERROR`：
- 编码：protobuf（canonical）
- 内容：`code`、`message`、`retryable`、可选 `details`
- 行为：发送后流立即结束

## 6. Size Limits And Fragmentation

- `MAX_FRAME_PAYLOAD_BYTES`: `PROTOCOL_MAX_FRAME_PAYLOAD_BYTES`
- `MAX_REQUEST_BYTES`: `PROTOCOL_MAX_REQUEST_BYTES`
- 超限行为：返回 `ERROR(RESOURCE_EXHAUSTED)` 并关闭流
- 分片规则：
  - 大于 `PROTOCOL_MAX_FRAME_PAYLOAD_BYTES` 的逻辑消息必须分为多个 `DATA`
  - 按到达顺序重组，不支持乱序

## 7. RPC State Machine

接收侧状态定义：
- `INIT`: 尚未接收任何帧
- `OPEN_RECV`: 已接收 `OPEN`
- `DATA_RECV`: 已接收一个或多个 `DATA`
- `END_RECV`: 已接收请求 `END`
- `RUNNING`: 正在执行 DAG
- `RESP_STREAMING`: 正在发送响应 `DATA`
- `HALT`: 请求结束

状态迁移表（服务端）：

| Current | Input | Next | Action |
|---|---|---|---|
| `INIT` | `OPEN` | `OPEN_RECV` | 记录上下文 |
| `INIT` | 其他 | `HALT` | `ERROR(INVALID_ARGUMENT)` |
| `OPEN_RECV` | `DATA` | `DATA_RECV` | 累积输入 |
| `OPEN_RECV` | `END` | `END_RECV` | 空输入结束（允许） |
| `OPEN_RECV` | `OPEN` | `HALT` | `ERROR(INVALID_ARGUMENT)` |
| `DATA_RECV` | `DATA` | `DATA_RECV` | 继续累积 |
| `DATA_RECV` | `END` | `END_RECV` | 请求输入完成 |
| `DATA_RECV` | `OPEN` | `HALT` | `ERROR(INVALID_ARGUMENT)` |
| `END_RECV` | 内部事件 | `RUNNING` | 提交执行 |
| `RUNNING` | unary 输出 | `RESP_STREAMING` | 发送单个 `DATA` |
| `RUNNING` | streaming 输出 | `RESP_STREAMING` | 发送多个 `DATA` |
| `RESP_STREAMING` | 响应结束 | `HALT` | 发送 `END` |
| `*` | 协议错误 | `HALT` | 发送 `ERROR` 并关闭 |

额外规则：
- 任何阶段收到重复 `END` 均视为 `INVALID_ARGUMENT`
- `ERROR` 帧发送后必须立即关闭响应流

## 8. Deadline, Cancel, Backpressure

Deadline：
- 入口校验 `x-nerva-deadline-ms`
- 当前时间超出 deadline：直接返回 `ERROR(DEADLINE_EXCEEDED)`
- 运行中超时：取消下游执行并返回 `ERROR(DEADLINE_EXCEEDED)`

Cancel：
- 隐式取消：客户端断连
- 显式取消：`POST /rpc/{request_id}/cancel`
- 取消结果：请求标记 `ABORTED`，尽快释放队列与执行资源

Backpressure：
- 每 pipeline/model/worker 采用有界队列
- 队列满时快速失败：`ERROR(RESOURCE_EXHAUSTED)`

优先级规则（终态判定）：
1. 客户端显式 cancel 先到达并生效 -> `ABORTED`
2. deadline 先触发 -> `DEADLINE_EXCEEDED`
3. 执行中 worker 异常 -> `UNAVAILABLE`

## 9. Error Code Contract

| Code | Name | Retryable | Meaning |
|---:|---|---|---|
| 0 | `OK` | No | 成功 |
| 3 | `INVALID_ARGUMENT` | No | 请求或帧格式错误 |
| 4 | `DEADLINE_EXCEEDED` | Maybe | 超过 deadline |
| 8 | `RESOURCE_EXHAUSTED` | Yes | 队列满/内存不足/限流 |
| 10 | `ABORTED` | Maybe | 请求被取消 |
| 13 | `INTERNAL` | Maybe | 服务端内部错误 |
| 14 | `UNAVAILABLE` | Yes | worker/服务暂不可用 |

## 10. Idempotency And Retry

- 协议层不保证幂等
- 客户端重试条件：
  - 仅在 `retryable=true` 且业务允许时重试
  - 建议使用新的 `request_id`
- 服务端不做去重语义保证（后续版本可扩展）

## 11. Security Notes (MVP)

- 内网部署前提下不在协议层加密
- 生产建议通过 mTLS 网关或 Service Mesh 提供传输安全
- 推荐在入口增加最大请求体限制与速率限制

## 12. Observability Mapping

协议事件需映射到以下指标：
- `nerva_request_total{pipeline,status}`
- `nerva_request_deadline_exceeded_total{pipeline}`
- `nerva_request_cancel_total{pipeline,source}`
- `nerva_queue_drop_total{model,reason}`

日志关键字段：
- `request_id`
- `pipeline`
- `deadline_ms`
- `frame_type`
- `error_code`
