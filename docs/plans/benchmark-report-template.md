# Nerva Benchmark Report Template（MVP）

## 1. Metadata

- Date:
- Commit SHA:
- Operator:
- Environment ID:

## 2. Environment

### Hardware
- CPU:
- GPU:
- Memory:
- Storage:

### Software
- OS/Kernel:
- Python:
- CUDA/Driver:
- Uvicorn/Starlette/FastAPI:
- PyTorch/vLLM:

## 3. Configuration Snapshot

- Relevant env/config overrides:
- Default baseline version:
- ADR set version:

## 4. Workload Matrix Results

| Workload | Concurrency | Input Set | QPS | p50 | p95 | p99 | Error Rate | TTFT p99 | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| W1 | 1 | short |  |  |  |  |  | N/A |  |
| W1 | 32 | short |  |  |  |  |  | N/A |  |
| W2 | 128 | medium |  |  |  |  |  |  |  |
| W3 | 128 | medium |  |  |  |  |  | N/A |  |
| W4 | 512 | mixed |  |  |  |  |  | N/A |  |

## 5. Internal Metrics Summary

- Queue:
  - peak `nerva_queue_depth`:
  - `nerva_queue_drop_total`:
- Scheduler:
  - `nerva_scheduler_wait_seconds` p99:
- IPC:
  - `nerva_ipc_control_latency_seconds` p99:
  - `nerva_ipc_shm_alloc_fail_total`:
- Worker:
  - GPU util / memory:

## 6. Fault Injection Results

| Scenario | Expected | Observed | Pass/Fail | Notes |
|---|---|---|---|---|
| F1 Worker Crash | `UNAVAILABLE` + recovery |  |  |  |
| F2 GPU OOM | isolated `RESOURCE_EXHAUSTED` |  |  |  |
| F3 Cancel Storm | bounded inflight + reclaim |  |  |  |
| F4 Deadline Tight | stable `DEADLINE_EXCEEDED` |  |  |  |

## 7. SLO Gate Evaluation

- `W3@C3 p99 <= SLO_W3_C3_P99_LATENCY_MS`: PASS/FAIL
- `W2@C3 TTFT p99 <= SLO_W2_C3_P99_TTFT_MS`: PASS/FAIL
- `Error rate <= SLO_ERROR_RATE_MAX_PERCENT`: PASS/FAIL
- `Stream abort rate <= SLO_STREAM_ABORT_RATE_MAX_PERCENT`: PASS/FAIL

Overall: GO / NO-GO

## 8. Bottleneck Attribution

- Primary bottleneck:
- Secondary bottleneck:
- Evidence:

## 9. Parameter Changes (if any)

| Key | Old | New | Reason | Impact |
|---|---|---|---|---|
|  |  |  |  |  |

## 10. Next Actions

1.
2.
3.

