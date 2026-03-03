# Nerva 基准测试报告模板（MVP）

## 1. 元信息

- 日期：
- Commit SHA：
- 执行人：
- 环境 ID：

## 2. 环境信息

### 硬件
- CPU：
- GPU：
- 内存：
- 存储：

### 软件
- OS/Kernel：
- Python：
- CUDA/Driver：
- Uvicorn/Starlette/FastAPI：
- PyTorch/vLLM：

## 3. 配置快照

- 相关环境变量/配置覆盖：
- 默认基线版本：
- ADR 集版本：

## 4. 工作负载矩阵结果

| 工作负载 | 并发 | 输入集 | QPS | p50 | p95 | p99 | 错误率 | TTFT p99 | 备注 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| W1 | 1 | short |  |  |  |  |  | N/A |  |
| W1 | 32 | short |  |  |  |  |  | N/A |  |
| W2 | 128 | medium |  |  |  |  |  |  |  |
| W3 | 128 | medium |  |  |  |  |  | N/A |  |
| W4 | 512 | mixed |  |  |  |  |  | N/A |  |

## 5. 内部指标汇总

- Queue：
  - `nerva_queue_depth` 峰值：
  - `nerva_queue_drop_total`：
- Scheduler：
  - `nerva_scheduler_wait_seconds` p99：
- IPC：
  - `nerva_ipc_control_latency_seconds` p99：
  - `nerva_ipc_shm_alloc_fail_total`：
- Worker：
  - GPU 利用率/显存：

## 6. 故障注入结果

| 场景 | 预期 | 实测 | 通过/失败 | 备注 |
|---|---|---|---|---|
| F1 Worker Crash | `UNAVAILABLE` + recovery |  |  |  |
| F2 GPU OOM | isolated `RESOURCE_EXHAUSTED` |  |  |  |
| F3 Cancel Storm | bounded inflight + reclaim |  |  |  |
| F4 Deadline Tight | stable `DEADLINE_EXCEEDED` |  |  |  |

## 7. SLO 门禁评估

- `W3@C3 p99 <= SLO_W3_C3_P99_LATENCY_MS`：PASS/FAIL
- `W2@C3 TTFT p99 <= SLO_W2_C3_P99_TTFT_MS`：PASS/FAIL
- `Error rate <= SLO_ERROR_RATE_MAX_PERCENT`：PASS/FAIL
- `Stream abort rate <= SLO_STREAM_ABORT_RATE_MAX_PERCENT`：PASS/FAIL

总体结论：GO / NO-GO

## 8. 瓶颈归因

- 主要瓶颈：
- 次要瓶颈：
- 证据：

## 9. 参数变更（如有）

| 参数 | 旧值 | 新值 | 变更原因 | 影响 |
|---|---|---|---|---|
|  |  |  |  |  |

## 10. 后续动作

1.
2.
3.
