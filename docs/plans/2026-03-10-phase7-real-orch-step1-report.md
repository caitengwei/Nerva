# 2026-03-10 Phase7 Real-Orch Step1 压测报告

## 1. 目标与口径
- 目标: 先打通 Nerva vs Triton 的可执行对比链路（Step1）。
- 日期: 2026-03-10。
- commit: `607e8bc`（来自 `run-meta.json`）。
- workload: `mm_vllm`，`contract=full-e2e`。
- 参数: `warmup_seconds=2`，`sample_seconds=6`，`deadline_ms=30000`，`max_tokens=256`，`temperature=1.0`，`top_p=1.0`。
- 对比对象: `nerva` 与 `triton`。
- 本轮口径: 真实编排链路对比（Nerva DAG vs Triton ensemble）；按任务要求忽略 GPU 缺失，允许上游推理服务使用 mock fallback。

## 2. 链路对齐说明
- Nerva: `preprocess -> http_vllm -> postprocess`（HTTP 调上游推理服务）。
- Triton: `mm_preprocess -> mm_infer -> mm_postprocess`（ensemble 调度）。
- 两侧均为 `max_batch_size: 0` 标量请求口径，可用于对标编排、排队、协议与序列化开销。
- 结论边界: 本结果不代表真实 GPU 模型算子性能上限。

## 3. 结果汇总

### 3.1 修复后首轮（fixed）
- 结果目录: `bench-results/e2e-real-orch-compare-fixed-20260310-194905`

| Target | 并发 | QPS | p50(ms) | p95(ms) | p99(ms) | 错误率 | 错误数/总请求 |
|---|---:|---:|---:|---:|---:|---:|---:|
| nerva | 1 | 74.70 | 11.35 | 14.09 | 33.12 | 0.000% | 0/449 |
| nerva | 32 | 447.04 | 71.50 | 109.94 | 168.06 | 0.000% | 0/2711 |
| triton | 1 | 196.19 | 4.70 | 7.43 | 10.25 | 0.000% | 0/1178 |
| triton | 32 | 22.10 | 81.49 | 178.41 | 30003.45 | 4.552% | 32/703 |

### 3.2 `infer_instance_count=8`（inst8）
- 结果目录: `bench-results/e2e-real-orch-compare-inst8-20260310-202615`

| Target | 并发 | QPS | p50(ms) | p95(ms) | p99(ms) | 错误率 | 错误数/总请求 |
|---|---:|---:|---:|---:|---:|---:|---:|
| nerva | 1 | 82.64 | 11.19 | 12.57 | 14.37 | 0.000% | 0/496 |
| nerva | 32 | 474.77 | 64.48 | 107.16 | 179.50 | 0.000% | 0/2863 |
| nerva | 128 | 346.66 | 360.09 | 436.58 | 567.86 | 0.000% | 0/2149 |
| triton | 1 | 203.82 | 4.57 | 6.91 | 8.83 | 0.000% | 0/1223 |
| triton | 32 | 13.61 | 78.09 | 29777.61 | 30014.78 | 1.869% | 8/428 |
| triton | 128 | 51.41 | 308.48 | 29611.14 | 29638.99 | 0.457% | 8/1750 |

## 4. Step1 结论
- Step1 已达成: Nerva 与 Triton 的真实编排链路均可稳定跑完并产出结构化结果。
- Triton 在高并发仍有 30s 级尾延迟与超时；`infer_instance_count` 从 1 提到 8 后，错误数明显下降（例如 C32 从 32 降到 8），但未根治 tail latency。
- Nerva 在当前口径下未出现请求错误，C32 下吞吐与延迟相对稳定。

## 5. 本次代码变更（配套 Step1）
- `scripts/bench/infra/prepare_triton_repo.py`
  - 修复 `config.pbtxt` 中 input/output 列表项缺逗号导致的解析问题。
  - 为 ensemble 模型补齐版本目录 `mm_vllm/1`，满足 Triton 加载要求。
  - 新增 `infer_instance_count` 参数及 CLI `--infer-instance-count`，可调 `mm_infer` 实例数。
- `tests/test_bench_infra.py`
  - 增强 `prepare_triton_repo` 回归覆盖（列表格式、ensemble 版本目录）。
  - 新增 `infer_instance_count` 正常/非法值测试。
