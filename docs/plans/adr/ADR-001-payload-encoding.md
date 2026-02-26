# ADR-001: 数据面 Payload 编码与 Schema 规范

- Status: Accepted
- Date: 2026-02-25
- Decision Makers: Nerva MVP maintainers

## Context

MVP 需要在 Starlette + ASGI 数据面支持低延迟二进制 RPC。  
此前文档允许 `protobuf/msgpack` 并存，存在实现与客户端语义漂移风险。

核心矛盾：
- 灵活性（多编码） vs 兼容性与可维护性（单一 canonical 编码）
- 热路径性能 vs 调试便利性

## Decision

1. **数据面 canonical 编码固定为 protobuf**。  
2. `OPEN/END/ERROR` 元数据帧 payload 必须采用 protobuf 消息。  
3. `DATA` 帧 payload 采用 protobuf envelope，业务数据放入 `bytes` 字段。  
4. `msgpack` 仅保留在 IPC 控制通道，不作为 RPC 数据面标准编码。  
5. 协议实现必须以 schema 版本进行兼容检查（不匹配返回 `INVALID_ARGUMENT`）。

## Consequences

正向影响：
- 客户端与服务端契约单一，降低灰度与回归风险。
- 更适合内部高吞吐场景（稳定 schema + 二进制编码）。

负向影响：
- 调试时可读性较 JSON/msgpack 弱。
- 需要维护 protobuf schema 与版本升级策略。

## Revisit Triggers

出现以下任一条件时可考虑 supersede：
- 发现 protobuf 在目标负载下产生不可接受的 CPU 或尾延迟开销。
- 多语言客户端演进要求 envelope 无法满足的数据表达能力。

