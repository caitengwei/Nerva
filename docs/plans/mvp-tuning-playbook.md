# Nerva MVP 参数调优顺序指南

## 1. 目标与边界

本指南用于 **首版 MVP 实现后** 的性能收敛，不用于编码前拍脑袋改参数。

目标：
- 在固定环境下稳定降低 `p95/p99` 与 TTFT
- 保持可解释性（每次只改一类参数）
- 避免“吞吐提升但尾延迟雪崩”的误优化

前置条件：
- 已完成 `mvp-benchmark-plan.md` 的基线跑测
- 统一默认值来自 `mvp-defaults.md`

## 2. 基本原则

1. 单变量原则：每轮只改 1 个参数，其他参数固定。  
2. 固定口径：并发、输入集、采样时长与版本必须一致。  
3. 先稳后快：先控制拒绝/超时/队列，再优化吞吐。  
4. 回退机制：每轮都要有“未达标即回滚”条件。

## 3. 推荐调优顺序（必须按序）

### 阶段 A：准入与背压（先保系统稳定）

调参对象：
- `SCHED_QUEUE_CAPACITY_PER_MODEL`
- `SCHED_ADMISSION_MIN_REMAINING_DEADLINE_MS`
- `SCHED_STREAM_OUTBUF_BYTES_PER_REQ`

观察指标：
- `nerva_queue_depth`
- `nerva_queue_drop_total`
- deadline exceeded rate
- `p99 latency`

通过条件：
- 高并发下无无界排队
- 错误主要为可解释的 `RESOURCE_EXHAUSTED`，且比例稳定

### 阶段 B：Batch 策略（再提吞吐）

调参对象（实现参数）：
- `batch.max_size`
- `batch.max_delay_ms`
- `batch.pad_to_multiple`

观察指标：
- `nerva_batch_size`
- `nerva_batch_wait_seconds`
- `p95/p99 latency`
- GPU 利用率

通过条件：
- 吞吐提升且 `p99` 不恶化（或在 SLO 范围内）

### 阶段 C：IPC 路径（压缩跨进程开销）

调参对象：
- `IPC_CONTROL_INLINE_MAX_BYTES`
- `IPC_SHM_SIZE_CLASSES`
- `IPC_SHM_TOTAL_BYTES`
- `IPC_ALLOC_TIMEOUT_MS`
- `IPC_SUBMIT_TIMEOUT_MS`
- `IPC_RECLAIM_TTL_MS`

观察指标：
- `nerva_ipc_control_latency_seconds`
- `nerva_ipc_shm_alloc_fail_total`
- `nerva_ipc_data_bytes_total`
- worker RSS / shm 占用

通过条件：
- 分配失败率可接受，回收稳定，无长期内存膨胀

### 阶段 D：协议帧与请求上限（避免协议层放大尾延迟）

调参对象：
- `PROTOCOL_MAX_FRAME_PAYLOAD_BYTES`
- `PROTOCOL_MAX_REQUEST_BYTES`

观察指标：
- 请求拒绝率（超限）
- 分片数量分布
- 大请求下 `p99` 与 CPU 开销

通过条件：
- 大 payload 不引发全局尾延迟劣化

### 阶段 E：SLO Gate 收敛（最终验收）

使用门槛：
- `SLO_W3_C3_P99_LATENCY_MS`
- `SLO_W2_C3_P99_TTFT_MS`
- `SLO_ERROR_RATE_MAX_PERCENT`
- `SLO_STREAM_ABORT_RATE_MAX_PERCENT`

判定：
- 全部关键 workload 满足 SLO，且故障注入场景不退化

## 4. 每轮实验模板

每轮必须记录：
- 假设：为什么这个参数可能改善瓶颈
- 变更：`key: old -> new`
- 固定项：并发档位、输入集、采样时长、版本
- 结果：QPS / `p50/p95/p99` / 错误率 / 关键内部指标
- 结论：保留、回滚、继续细化

建议每个参数最多试 3 档：
- `baseline`
- `conservative+`
- `aggressive+`

## 5. 常见反模式（禁止）

1. 同时改队列和 batch 参数后宣称优化成功。  
2. 只看平均延迟，不看 `p99`。  
3. 只看吞吐，不看拒绝率和超时率。  
4. 在模型版本变更后复用旧结论。  
5. 不做回滚对照。

## 6. 快速决策树

- 症状：`queue_depth` 高且 `p99` 抖动  
  操作：先收紧 `SCHED_QUEUE_CAPACITY_PER_MODEL`，再调 `SCHED_ADMISSION_MIN_REMAINING_DEADLINE_MS`

- 症状：GPU 利用率低、batch_size 长期偏小  
  操作：增加 `batch.max_delay_ms` 或 `batch.max_size`，观察 `p99`

- 症状：`ipc_shm_alloc_fail_total` 上升  
  操作：提高 `IPC_SHM_TOTAL_BYTES` 或重配 `IPC_SHM_SIZE_CLASSES`

- 症状：大请求拖慢全局  
  操作：降低 `PROTOCOL_MAX_REQUEST_BYTES` 或降低 `PROTOCOL_MAX_FRAME_PAYLOAD_BYTES` 并加强限流

## 7. 输出物要求

调优结束后必须产出：
- 参数变更表（最终值 vs 默认值）
- 各阶段关键实验记录
- 未解决风险与下一轮建议

