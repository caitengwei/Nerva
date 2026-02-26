# ADR-002: MVP Worker 粒度（`model() -> worker`）

- Status: Accepted
- Date: 2026-02-25
- Decision Makers: Nerva MVP maintainers

## Context

当前设计有两条候选路径：
1. 每个 `model()` 声明对应一个 Worker（当前设计）。
2. 每 GPU 一个 Worker，Worker 内承载多个模型节点（更复杂调度）。

在 MVP 目标下，需要优先保证：
- 多进程隔离稳定性
- 故障可定位
- 研发复杂度可控

## Decision

1. MVP 固定采用 **`model() -> worker`** 粒度。  
2. Worker 与模型实例一一对应，生命周期由 Worker Manager 管理。  
3. 同 GPU 多 Worker 允许但不默认启用智能合并。  
4. 多副本负载均衡（`model_group`）保持未来迭代，不进入 MVP。

## Consequences

正向影响：
- 进程边界清晰，问题定位和容错更简单。
- 实现复杂度低，便于先打通端到端。

负向影响：
- 同 GPU 多节点 pipeline 会产生更多 IPC hop。
- 资源利用率与拓扑最优性在 MVP 阶段可能不是最优。

## Revisit Triggers

出现以下任一条件时可考虑 supersede：
- IPC 开销在多模型 pipeline 中成为主要瓶颈（`>20%` e2e）。
- 需要显著提升单机多卡吞吐并且隔离模型/进程成本不可接受。

