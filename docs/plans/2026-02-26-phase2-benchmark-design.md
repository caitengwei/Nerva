# Phase 2 DAG Benchmark 设计

## 1. 目标

量化 Phase 2 DAG 编排层的性能特征，建立 baseline：
- **框架开销**：trace 构图、Executor 调度、field_path 解析等编排层自身的开销
- **IPC 综合性能**：贴近 multimodal 场景的 payload 大小下，inline vs SHM 路径的端到端表现
- **并行效率**：parallel() 原语的实际并发增益

## 2. Benchmark Model 设计

4 个贴近 multimodal 场景的可控模型，通过 `asyncio.sleep` 模拟计算耗时，通过构造 realistic 大小的输出模拟 IPC 传输：

```python
class BenchImageEncoder(Model):
    # options: dim (int), delay_ms (float)
    # 输入: {"image_bytes": bytes}
    # 输出: {"features": [float] * dim}

class BenchTextEncoder(Model):
    # options: dim (int), delay_ms (float)
    # 输入: {"text": str}
    # 输出: {"features": [float] * dim}

class BenchFusionModel(Model):
    # options: dim (int), delay_ms (float)
    # 输入: {"img_features": list, "txt_features": list}
    # 输出: {"fused_features": [float] * dim}

class BenchClassifier(Model):
    # options: delay_ms (float)
    # 输入: {"fused_features": list}
    # 输出: {"label": str, "score": float}
```

所有模型通过 `options` 传入 `dim`（特征维度）和 `delay_ms`（模拟计算延迟），不硬编码。

## 3. Payload 档位

| 档位 | image_bytes | text | feature dim | 单 feature 大小 | IPC 路径 |
|------|------------|------|------------|----------------|---------|
| Small | 4KB | 100B | 768 | ~3KB | inline（<8KB） |
| Large | 256KB | 1KB | 1280 | ~5KB | 输入 SHM，输出 inline 边界 |

现实参考：CLIP ViT-L 输出 768-dim float32 ≈ 3KB，ViT-G 输出 1280-dim ≈ 5KB。

## 4. DAG 拓扑

```
ImageEncoder ──┐
               ├── FusionModel ── Classifier
TextEncoder  ──┘
```

与 `examples/03_parallel_dag.py` 一致。手动构建 Graph 控制精确的 edge field_path 映射，同时测一次 trace() 构图验证一致性。

## 5. Benchmark 矩阵

| 测试 | 测什么 | 方法 |
|------|--------|------|
| B1: trace overhead | trace() 构图耗时 | 1000 次取平均，纯 CPU 不启动 Worker |
| B2: executor overhead | DAG 调度层开销 | delay=0 的模型，executor 总耗时 - 节点 infer 总耗时 |
| B3: parallel speedup | parallel 并发增益 | delay=10ms 的模型，parallel vs sequential 耗时比 |
| B4: e2e pipeline | 端到端 pipeline p50/p95/p99 | Small + Large 两档，各 100 次 |

## 6. 指标

| 层级 | 指标 | 说明 |
|------|------|------|
| trace 层 | trace() 构图耗时 (μs) | 纯 CPU，与 payload 无关 |
| executor 调度层 | executor overhead (μs) | DAG 执行总耗时 - 各节点 infer 耗时之和 |
| parallel 效率 | parallel speedup ratio | parallel(a,b) 耗时 / max(a,b) 耗时，理想值 ≈ 1.0 |
| 端到端 | pipeline latency p50/p95/p99 (ms) | 多次执行的分位数 |
| IPC 路径 | inline vs SHM 延迟差异 | Small 档 vs Large 档对比体现 |

## 7. 输出格式

- pytest 用例，标记 `@pytest.mark.slow`
- 每个 benchmark 输出 JSON 到 `bench-results/phase2/`
- JSON 包含：指标值、环境信息（Python 版本、commit SHA、timestamp）
- 终端 print summary 供人查看
- assert sanity gate（executor overhead < 5ms，parallel speedup > 1.5x）

## 8. Sanity Gates

| 指标 | 门槛 | 说明 |
|------|------|------|
| trace() 单次 | < 100μs | 纯 CPU 构图不应成为瓶颈 |
| executor overhead | < 5ms | 调度层开销相对 IPC 应该很小 |
| parallel speedup | > 1.5x | 并行应有明显增益（理想 2x） |
| e2e error rate | 0% | 功能正确性 |

## 9. 文件结构

```
tests/
  helpers.py              # 新增 BenchImageEncoder 等 4 个 benchmark model
  test_phase2_bench.py    # B1-B4 benchmark 用例
bench-results/
  phase2/                 # JSON 产物目录（gitignore）
```

## Changelog

| 日期 | 变更 |
|------|------|
| 2026-02-26 | 初始版本 |
