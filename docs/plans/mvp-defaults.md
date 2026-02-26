# Nerva MVP 默认参数表（Single Source Of Truth）

## 1. 说明

本文档定义 PoC/MVP 阶段的统一默认参数，作为以下文档的唯一默认值来源：
- `protocol-v1.md`
- `ipc-contract.md`
- `mvp-benchmark-plan.md`
- `config-mapping.md`（默认值到运行时配置/环境变量映射）

约定：
- 参数名称使用全大写 snake_case。
- 各文档引用参数名，不重复定义不同默认值。
- 首轮基准后仅允许在本文件调整默认值，再同步实现配置。

## 2. Protocol Defaults

| Key | Default | Unit | 说明 |
|---|---:|---|---|
| `PROTOCOL_FIXED_HEADER_BYTES` | `32` | bytes | 二进制固定帧头长度 |
| `PROTOCOL_STREAM_ID_DEFAULT` | `1` | - | MVP 单流 ID |
| `PROTOCOL_MAX_FRAME_PAYLOAD_BYTES` | `4` | MiB | 单帧 payload 上限 |
| `PROTOCOL_MAX_REQUEST_BYTES` | `64` | MiB | 单请求总大小上限 |

## 3. IPC Defaults

| Key | Default | Unit | 说明 |
|---|---:|---|---|
| `IPC_CONTROL_INLINE_MAX_BYTES` | `8` | KiB | 控制通道小消息阈值 |
| `IPC_SHM_SIZE_CLASSES` | `4/16/64/256/1024/4096` | KiB | shm slab size classes |
| `IPC_SHM_TOTAL_BYTES` | `4` | GiB | shm 池总大小 |
| `IPC_ALLOC_TIMEOUT_MS` | `2` | ms | shm 分配超时 |
| `IPC_SUBMIT_TIMEOUT_MS` | `5000` | ms | `INFER_SUBMIT` 等待 ACK 超时 |
| `IPC_RECLAIM_TTL_MS` | `30000` | ms | descriptor 惰性回收 TTL |

## 4. Scheduler And Backpressure Defaults

| Key | Default | Unit | 说明 |
|---|---:|---|---|
| `SCHED_QUEUE_CAPACITY_PER_MODEL` | `2048` | requests | 每模型队列上限 |
| `SCHED_STREAM_OUTBUF_BYTES_PER_REQ` | `1024` | KiB | 单流请求输出缓冲上限 |
| `SCHED_ADMISSION_MIN_REMAINING_DEADLINE_MS` | `5` | ms | 准入时最小剩余 deadline |

## 5. Benchmark Defaults

| Key | Default | Unit | 说明 |
|---|---:|---|---|
| `BENCH_CONCURRENCY_LEVELS` | `1,32,128,512,1000` | - | 标准并发档位 |
| `BENCH_WARMUP_SECONDS` | `60` | s | 每档预热时长 |
| `BENCH_SAMPLE_SECONDS` | `300` | s | 每档采样时长 |
| `BENCH_MEDIA_PAYLOAD_SMALL` | `64` | KiB | 多媒体小负载 |
| `BENCH_MEDIA_PAYLOAD_MEDIUM` | `1` | MiB | 多媒体中负载 |
| `BENCH_MEDIA_PAYLOAD_LARGE` | `8` | MiB | 多媒体大负载 |

## 6. SLO Gate Defaults

| Key | Default | Unit | 说明 |
|---|---:|---|---|
| `SLO_W3_C3_P99_LATENCY_MS` | `120` | ms | `W3@C3` p99 延迟上限 |
| `SLO_W2_C3_P99_TTFT_MS` | `200` | ms | `W2@C3` TTFT p99 上限 |
| `SLO_ERROR_RATE_MAX_PERCENT` | `0.5` | % | 请求错误率上限 |
| `SLO_STREAM_ABORT_RATE_MAX_PERCENT` | `0.5` | % | 流式中止率上限 |
