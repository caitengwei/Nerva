# Nerva MVP 配置映射（Defaults -> Runtime Config -> Env）

## 1. 目的

将 `mvp-defaults.md` 的参数键映射到运行时配置路径与环境变量，作为实现阶段唯一配置映射规范。

优先级：
1. 环境变量（`NERVA_*`）
2. 配置文件（YAML/TOML，后续实现）
3. `mvp-defaults.md` 默认值

## 2. 映射表

| Default Key | Runtime Path | Env Var | Type | Unit |
|---|---|---|---|---|
| `PROTOCOL_FIXED_HEADER_BYTES` | `protocol.fixed_header_bytes` | `NERVA_PROTOCOL_FIXED_HEADER_BYTES` | int | bytes |
| `PROTOCOL_STREAM_ID_DEFAULT` | `protocol.stream_id_default` | `NERVA_PROTOCOL_STREAM_ID_DEFAULT` | int | - |
| `PROTOCOL_MAX_FRAME_PAYLOAD_BYTES` | `protocol.max_frame_payload_bytes` | `NERVA_PROTOCOL_MAX_FRAME_PAYLOAD_BYTES` | int | MiB |
| `PROTOCOL_MAX_REQUEST_BYTES` | `protocol.max_request_bytes` | `NERVA_PROTOCOL_MAX_REQUEST_BYTES` | int | MiB |
| `IPC_CONTROL_INLINE_MAX_BYTES` | `ipc.control_inline_max_bytes` | `NERVA_IPC_CONTROL_INLINE_MAX_BYTES` | int | KiB |
| `IPC_SHM_SIZE_CLASSES` | `ipc.shm_size_classes` | `NERVA_IPC_SHM_SIZE_CLASSES` | list[int] | KiB |
| `IPC_SHM_TOTAL_BYTES` | `ipc.shm_total_bytes` | `NERVA_IPC_SHM_TOTAL_BYTES` | int | GiB |
| `IPC_ALLOC_TIMEOUT_MS` | `ipc.alloc_timeout_ms` | `NERVA_IPC_ALLOC_TIMEOUT_MS` | int | ms |
| `IPC_SUBMIT_TIMEOUT_MS` | `ipc.submit_timeout_ms` | `NERVA_IPC_SUBMIT_TIMEOUT_MS` | int | ms |
| `IPC_RECLAIM_TTL_MS` | `ipc.reclaim_ttl_ms` | `NERVA_IPC_RECLAIM_TTL_MS` | int | ms |
| `SCHED_QUEUE_CAPACITY_PER_MODEL` | `scheduler.queue_capacity_per_model` | `NERVA_SCHED_QUEUE_CAPACITY_PER_MODEL` | int | requests |
| `SCHED_STREAM_OUTBUF_BYTES_PER_REQ` | `scheduler.stream_outbuf_bytes_per_req` | `NERVA_SCHED_STREAM_OUTBUF_BYTES_PER_REQ` | int | KiB |
| `SCHED_ADMISSION_MIN_REMAINING_DEADLINE_MS` | `scheduler.admission_min_remaining_deadline_ms` | `NERVA_SCHED_ADMISSION_MIN_REMAINING_DEADLINE_MS` | int | ms |
| `BENCH_CONCURRENCY_LEVELS` | `bench.concurrency_levels` | `NERVA_BENCH_CONCURRENCY_LEVELS` | list[int] | - |
| `BENCH_WARMUP_SECONDS` | `bench.warmup_seconds` | `NERVA_BENCH_WARMUP_SECONDS` | int | s |
| `BENCH_SAMPLE_SECONDS` | `bench.sample_seconds` | `NERVA_BENCH_SAMPLE_SECONDS` | int | s |
| `BENCH_MEDIA_PAYLOAD_SMALL` | `bench.media_payload_small` | `NERVA_BENCH_MEDIA_PAYLOAD_SMALL` | int | KiB |
| `BENCH_MEDIA_PAYLOAD_MEDIUM` | `bench.media_payload_medium` | `NERVA_BENCH_MEDIA_PAYLOAD_MEDIUM` | int | MiB |
| `BENCH_MEDIA_PAYLOAD_LARGE` | `bench.media_payload_large` | `NERVA_BENCH_MEDIA_PAYLOAD_LARGE` | int | MiB |
| `SLO_W3_C3_P99_LATENCY_MS` | `slo.w3_c3_p99_latency_ms` | `NERVA_SLO_W3_C3_P99_LATENCY_MS` | int | ms |
| `SLO_W2_C3_P99_TTFT_MS` | `slo.w2_c3_p99_ttft_ms` | `NERVA_SLO_W2_C3_P99_TTFT_MS` | int | ms |
| `SLO_ERROR_RATE_MAX_PERCENT` | `slo.error_rate_max_percent` | `NERVA_SLO_ERROR_RATE_MAX_PERCENT` | float | % |
| `SLO_STREAM_ABORT_RATE_MAX_PERCENT` | `slo.stream_abort_rate_max_percent` | `NERVA_SLO_STREAM_ABORT_RATE_MAX_PERCENT` | float | % |

## 3. 单位转换规则

- 文档中 `MiB/KiB/GiB` 在实现中统一转换为 bytes 存储。
- 环境变量传入时允许纯数字（按表单位解释）或带后缀（如 `4MiB`），解析失败即启动报错。
- list 类型环境变量采用逗号分隔（示例：`4,16,64,256,1024,4096`）。

## 4. 配置样例（建议）

```toml
[protocol]
fixed_header_bytes = 32
stream_id_default = 1
max_frame_payload_bytes = 4   # MiB
max_request_bytes = 64         # MiB

[ipc]
control_inline_max_bytes = 8
shm_size_classes = [4, 16, 64, 256, 1024, 4096] # KiB
shm_total_bytes = 4            # GiB
alloc_timeout_ms = 2
submit_timeout_ms = 5000
reclaim_ttl_ms = 30000
```

## 5. 变更流程

1. 先改 `mvp-defaults.md`。  
2. 更新本映射文档。  
3. 在代码配置模块同步默认值与校验规则。  
4. 运行 benchmark 验证改动影响。  

