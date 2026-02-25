# Nerva MVP Benchmark Plan

## 1. Objective

在编码实现前定义统一评测口径，验证三件事：
- 多模型编排可用
- 低延迟目标可达
- 瓶颈可定位（协议 / 调度 / IPC / 计算）

默认参数来源：[`mvp-defaults.md`](./mvp-defaults.md)

## 2. Benchmark Environment

硬件记录（必须固化到报告）：
- CPU 型号、核数、NUMA
- GPU 型号、数量、显存
- 内存大小、存储类型
- 网络环境（本机回环 / 内网）

软件记录（必须固化到报告）：
- OS / Kernel
- Python 版本
- Uvicorn/Starlette/FastAPI 版本
- PyTorch/vLLM/CUDA 驱动版本
- Commit SHA

## 3. Workload Profiles

`W1`: 单模型 Unary（PyTorch）
- 目标：建立最小基线

`W2`: 单模型 Streaming（vLLM）
- 目标：验证 TTFT 与 token 吞吐

`W3`: 多模型串联 DAG（Tokenizer -> LLM -> Detokenizer）
- 目标：验证编排路径延迟

`W4`: 多模型并行 DAG（ImageEncoder || TextEncoder -> Fusion）
- 目标：验证 `nerva.parallel()` 调度与队列隔离

## 4. Concurrency Matrix

每个 workload 运行统一并发档位：
- `BENCH_CONCURRENCY_LEVELS`（默认：`1,32,128,512,1000`）

每个并发档位至少执行：
- 预热：`BENCH_WARMUP_SECONDS`（默认：`60s`）
- 正式采样：`BENCH_SAMPLE_SECONDS`（默认：`300s`）

## 5. Input Sets

文本请求：
- 短文本（~32 tokens）
- 中文本（~256 tokens）
- 长文本（~1024 tokens）

多媒体请求（如适用）：
- 小 payload（`BENCH_MEDIA_PAYLOAD_SMALL`）
- 中 payload（`BENCH_MEDIA_PAYLOAD_MEDIUM`）
- 大 payload（`BENCH_MEDIA_PAYLOAD_LARGE`，验证分片与 backpressure）

## 6. Metrics

E2E：
- QPS
- latency `p50/p95/p99`
- error rate

流式：
- TTFT `p50/p95/p99`
- tokens/s

排队与调度：
- `nerva_queue_depth`
- `nerva_scheduler_wait_seconds`
- deadline exceed rate

IPC：
- `nerva_ipc_control_latency_seconds`
- `nerva_ipc_data_bytes_total`
- `nerva_ipc_shm_alloc_fail_total`

资源：
- CPU 利用率
- GPU 利用率/显存
- 进程 RSS

## 7. SLO Gates (Initial Draft)

以下门槛用于 MVP go/no-go（可按首轮结果微调）：

- `W3@C3`（多模型串联，128 并发）：
  - `p99 latency <= SLO_W3_C3_P99_LATENCY_MS`
  - `error rate < SLO_ERROR_RATE_MAX_PERCENT`

- `W2@C3`（流式，128 并发）：
  - `TTFT p99 <= SLO_W2_C3_P99_TTFT_MS`
  - `stream abort rate < SLO_STREAM_ABORT_RATE_MAX_PERCENT`

- `W3@C5`（1000 并发压力）：
  - 系统无雪崩（可受控拒绝）
  - `RESOURCE_EXHAUSTED` 占比可解释，进程无持续内存泄漏

## 8. Fault Injection Scenarios

`F1` Worker 崩溃：
- 注入方式：随机 kill worker 进程
- 验证：在途请求返回 `UNAVAILABLE`，worker 可拉起，shm 可回收

`F2` GPU OOM：
- 注入方式：构造超大 batch / 输入
- 验证：错误隔离，系统可继续服务其他请求

`F3` Cancel Storm：
- 注入方式：高并发后立即大量 cancel
- 验证：队列和执行资源可回收，不出现持续高 inflight

`F4` Deadline Tight：
- 注入方式：设置极短 deadline
- 验证：快速失败，`DEADLINE_EXCEEDED` 行为稳定

## 9. Experiment Procedure

1. 固化环境与配置（写入 report metadata）
2. 预热并校验健康状态
3. 按 workload/concurrency 矩阵运行
4. 每组输出原始结果与聚合指标
5. 执行 fault injection 并记录恢复时间
6. 汇总 Go/No-Go 结论和瓶颈归因

## 10. Result Template And Artifacts

每组结果必须包含：
- workload + 并发 + 输入集
- QPS / latency p50/p95/p99 / error rate
- TTFT（如流式）
- queue/scheduler/ipc 关键指标
- CPU/GPU/RSS
- 失败样本（前 5 条）

建议报告模板：
- [`benchmark-report-template.md`](./benchmark-report-template.md)

建议产物目录：
- `bench-results/<date>/<commit>/<workload>/<concurrency>/`

建议产物文件：
- `summary.json`
- `raw-latency.csv`
- `metrics-snapshot.prom`
- `report.md`

## 11. Execution Command Templates

说明：以下命令是执行模板，实际脚本路径可在实现阶段调整，但参数语义必须保持一致。

环境准备（示例）：
```bash
export BASE_URL="http://127.0.0.1:8000"
export DEADLINE_MS="1893456000000"  # 示例 epoch ms
export REQ_BIN="/abs/path/to/unary_req.bin"
export STREAM_REQ_BIN="/abs/path/to/stream_req.bin"
```

Unary 冒烟（单请求）：
```bash
curl -sS -X POST "${BASE_URL}/rpc/text_generation" \
  -H "Content-Type: application/x-nerva-rpc" \
  -H "Accept: application/x-nerva-rpc" \
  -H "x-nerva-stream: 0" \
  -H "x-nerva-deadline-ms: ${DEADLINE_MS}" \
  --data-binary @"${REQ_BIN}" \
  -o /tmp/nerva_unary_resp.bin
```

Streaming 冒烟（观察首包时间）：
```bash
curl -N -X POST "${BASE_URL}/rpc/text_generation" \
  -H "Content-Type: application/x-nerva-rpc" \
  -H "Accept: application/x-nerva-rpc" \
  -H "x-nerva-stream: 1" \
  -H "x-nerva-deadline-ms: ${DEADLINE_MS}" \
  --data-binary @"${STREAM_REQ_BIN}" \
  -o /tmp/nerva_stream_resp.bin
```

负载模板（k6）：
```bash
k6 run \
  -e BASE_URL="${BASE_URL}" \
  -e DEADLINE_MS="${DEADLINE_MS}" \
  -e CONCURRENCY="128" \
  -e DURATION="300s" \
  docs/plans/templates/k6-unary-template.js
```

## 12. Tooling Suggestions

- 压测：`k6` 或 `ghz`（按协议实现选择）
- 指标：Prometheus + Grafana
- 剖析：`py-spy` + `nsys`（需要时）

## 13. Bottleneck Attribution Path

1. `p99` 高且 `queue_depth` 高：先看调度/背压参数与 admission。  
2. `p99` 高但 queue 正常：看 IPC（control latency / shm alloc fail）。  
3. IPC 正常但 TTFT 高：看 backend 首 token 生成与 streaming flush。  
4. 错误率高且 `RESOURCE_EXHAUSTED` 激增：看队列上限与请求大小限制。  
5. 错误率高且 `UNAVAILABLE` 激增：看 worker 稳定性与重启行为。  

## 14. Exit Criteria For Coding Phase

进入功能扩展开发前，必须满足：
- 协议与 IPC 两份契约文档冻结为 `v1`
- Benchmark 基线跑通（`W1~W4`, `C1~C5`）
- 至少通过 `F1~F4` 故障注入中的 3 项

## 15. Tuning Sequence Reference

参数调优顺序与实验记录模板见：
- [`mvp-tuning-playbook.md`](./mvp-tuning-playbook.md)
