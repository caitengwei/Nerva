# Nerva ADR Index（MVP）

## 1. 目的

集中记录 MVP 阶段所有架构决策，避免规范与实现漂移。

规则：
- `Status=Accepted` 的 ADR 视为冻结决策。
- 修改冻结决策必须新增 superseding ADR，不直接改旧 ADR 结论。
- 主设计文档仅引用 ADR，不重复解释取舍细节。

## 2. 决策清单

| ADR | 标题 | 状态 | 日期 | 影响范围 |
|---|---|---|---|---|
| [ADR-001](./adr/ADR-001-payload-encoding.md) | 数据面 payload 编码与 schema 规范 | Accepted | 2026-02-25 | `protocol-v1`, `server/rpc`, client |
| [ADR-002](./adr/ADR-002-worker-granularity.md) | MVP Worker 粒度（`model() -> worker`） | Accepted | 2026-02-25 | `worker/manager`, scheduler, IPC 路径 |
| [ADR-003](./adr/ADR-003-cancel-propagation.md) | 取消传播语义与完成状态边界 | Accepted | 2026-02-25 | protocol, scheduler, backend contract |

## 3. 变更流程

1. 提出变更原因与指标证据（至少包含 `p99` 或错误率影响）。
2. 新建 ADR（`Proposed`）。
3. 评审通过后标记为 `Accepted` 并更新 `adr-index.md`。
4. 同步更新受影响规范与实现配置。

