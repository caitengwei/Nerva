# ADR-003: 取消传播语义与完成状态边界

- Status: Accepted
- Date: 2026-02-25
- Decision Makers: Nerva MVP maintainers

## Context

MVP 需支持请求级 cancel / deadline，且跨层传播：
- RPC ingress
- 调度队列
- Worker 执行
- Backend 推理（含 streaming）

如果不固定语义边界，会导致：
- 客户端感知状态不一致
- 资源回收不可预测
- 压测结果不可解释

## Decision

1. 取消语义采用 **best-effort + 最终可回收保证**。  
2. 队列中请求：收到 cancel 后必须立即移除并回收资源。  
3. 执行中请求：采用协作式取消（节点边界/stream chunk 边界检查）。  
4. 状态返回优先级：
   - 客户端显式 cancel 生效：`ABORTED`
   - deadline 先触发：`DEADLINE_EXCEEDED`
   - worker 崩溃或通道异常：`UNAVAILABLE`
5. 资源回收上界：`IPC_SUBMIT_TIMEOUT_MS + IPC_RECLAIM_TTL_MS` 内完成回收。

## Consequences

正向影响：
- 状态语义一致，便于客户端重试和告警归因。
- 压测时可验证 cancel 行为与资源回收闭环。

负向影响：
- 对 backend 取消点设计提出要求（需要周期性检查 cancel token）。
- 无法保证“硬实时中断”，仅能保证最终一致回收。

## Revisit Triggers

出现以下任一条件时可考虑 supersede：
- 某核心 backend 不支持协作式取消且造成显著尾延迟抖动。
- 业务要求硬中断语义（非 best-effort）并可接受更高复杂度。

