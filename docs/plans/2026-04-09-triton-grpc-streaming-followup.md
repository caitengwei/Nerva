# Triton 26.03 gRPC streaming 对比验证下一步计划（2026-04-09）

## 背景与现状（文档口径）

### 已完成测试与结论摘要

- 2026-04-09：Triton 26.03 gRPC streaming（`STREAM=true`）与 Nerva CPU mock 对比压测完成，Triton QPS 明显低于 Nerva。
- 2026-04-09：控制实验 `STREAM=false`（仍为 gRPC streaming）后，Triton QPS 显著提升，表明性能下降主要与 per-token chunk 发送开销相关，而非 gRPC 本身。
- 2026-04-07：旧版 HTTP 非流式基线仍显著高于 26.03 `STREAM=true`。
- CPU mock 的 decoupled 代码整体符合规范，但尚未处理 `deadline` 与取消信号。

### 样例结果（同口径节选）

| 日期 | 方案 | C=32 QPS | p50 |
| --- | --- | --- | --- |
| 2026-04-09 | Triton 26.03 gRPC streaming `STREAM=true` | 14.75 | 2159.62 ms |
| 2026-04-09 | Triton 26.03 gRPC streaming `STREAM=false` | 242.57 | 131.35 ms |
| 2026-04-07 | Triton 24.08 HTTP 非流式 | 153.86 | 222.71 ms |
| 2026-04-09 | Nerva CPU mock | 195.73 | 166.86 ms |

> 数据来源（路径）：  
> - `/home/tengwei.ctw/Nerva/bench-results/remote-compare-triton2603-nerva-grpc-18003/mm_vllm/2026-04-09/c0e22d2/`  
> - `/home/tengwei.ctw/Nerva/bench-results/remote-triton2603-grpc-stream-false/mm_vllm/2026-04-09/c0e22d2/triton/`  
> - `/home/tengwei.ctw/Nerva/bench-results/remote-triton-cpu-mock-full/mm_vllm/2026-04-07/c0e22d2/triton/`

## 下一步工作（主目标）

### 1) 排查 26.03 + decoupled 模式性能低于旧基线

- 基于 `STREAM=true` 的结果，定位与旧版 HTTP 基线差异的核心瓶颈。
- 重点关注：decoupled 发送路径、chunk 数量与发送频率、server 侧调度/实例数、客户端聚合逻辑。

### 2) 问题解决后重新对比 Nerva

- 在确认性能回归问题解决后，重新进行 Triton 26.03 vs Nerva 对比测试。
- 对比口径保持一致（并发矩阵、输入规模、warmup/sample 参数一致）。
