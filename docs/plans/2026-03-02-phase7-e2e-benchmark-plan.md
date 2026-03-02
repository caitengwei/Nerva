# Phase 7 E2E 压测与对照实验 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 构建可复现的端到端压测体系，验证 `client -> server` 在多模态前后处理 + vLLM 场景下达到 `1000` 并发，并完成与开源 vLLM、Triton Inference Server 的同口径对照实验。

**Architecture:** 新建独立的 `Phase 7` 多模态 DAG（不复用 `examples/03_parallel_dag.py`），并实现统一 loadgen + target adapter 架构：`Nerva Binary RPC`、`Native vLLM(OpenAI API)`、`Triton HTTP/gRPC`。三套目标共享同一输入集、并发矩阵、统计口径与产物目录。

**Tech Stack:** Python 3.11+, asyncio, httpx, msgpack, Starlette/uvicorn, vLLM, Triton Inference Server, pytest

---

### Task 1: 建立 Phase 7 benchmark 配置契约（含 1000 并发硬约束）

**Files:**
- Create: `scripts/bench/config.py`
- Create: `scripts/bench/configs/phase7_baseline.json`
- Create: `tests/test_phase7_bench_config.py`

**Step 1: 写失败测试（配置解析 + 边界校验）**

```python
from scripts.bench.config import load_bench_config


def test_default_concurrency_contains_1000(tmp_path):
    cfg = {
        "concurrency_levels": [1, 32, 128, 512, 1000],
        "warmup_seconds": 60,
        "sample_seconds": 300,
    }
    p = tmp_path / "cfg.json"
    p.write_text(__import__("json").dumps(cfg))

    loaded = load_bench_config(p)
    assert 1000 in loaded.concurrency_levels
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_phase7_bench_config.py -v`
Expected: `ModuleNotFoundError: No module named 'scripts.bench.config'`

**Step 3: 最小实现配置 dataclass + 校验器**

```python
@dataclass(frozen=True)
class BenchConfig:
    concurrency_levels: list[int]
    warmup_seconds: int
    sample_seconds: int


def load_bench_config(path: Path) -> BenchConfig:
    ...
```

校验规则：
- `concurrency_levels` 非空且全为正整数
- 必须包含 `1000`
- `warmup_seconds > 0`、`sample_seconds > 0`

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_phase7_bench_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/bench/config.py scripts/bench/configs/phase7_baseline.json tests/test_phase7_bench_config.py
git commit -m "feat(bench): add phase7 benchmark config contract with c1000 gate"
```

---

### Task 2: 新建独立多模态 + vLLM DAG 服务（禁止复用 03_parallel_dag.py）

**Files:**
- Create: `examples/phase7_multimodal_vllm_server.py`
- Create: `tests/test_phase7_multimodal_graph.py`

**Step 1: 写失败测试（图结构验证）**

```python
from examples.phase7_multimodal_vllm_server import build_phase7_graph


def test_phase7_graph_nodes():
    graph = build_phase7_graph()
    model_names = [n.model_name for n in graph.nodes if n.node_type == "call"]
    assert model_names == ["mm_preprocess", "mm_vllm", "mm_postprocess"]
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_phase7_multimodal_graph.py -v`
Expected: `ModuleNotFoundError: No module named 'examples.phase7_multimodal_vllm_server'`

**Step 3: 最小实现新 DAG**

要求：
- 文件内提供 `build_phase7_graph()` 与 `app = build_nerva_app({"phase7_mm_vllm": graph})`
- 节点链路固定为：
  1. `mm_preprocess`（图像/文本前处理）
  2. `mm_vllm`（backend=`vllm`，输入 `prompt`）
  3. `mm_postprocess`（后处理与标准化输出）
- 不引用或导入 `examples/03_parallel_dag.py`

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_phase7_multimodal_graph.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add examples/phase7_multimodal_vllm_server.py tests/test_phase7_multimodal_graph.py
git commit -m "feat(examples): add phase7 multimodal + vllm DAG server"
```

---

### Task 3: 实现统一 target adapter（Nerva / vLLM / Triton）

**Files:**
- Create: `scripts/bench/targets/base.py`
- Create: `scripts/bench/targets/nerva_binary_rpc.py`
- Create: `scripts/bench/targets/vllm_openai_api.py`
- Create: `scripts/bench/targets/triton_infer.py`
- Create: `tests/test_phase7_targets.py`

**Step 1: 写失败测试（统一返回结构）**

```python
from scripts.bench.targets.base import TargetResponse


def test_target_response_schema():
    r = TargetResponse(ok=True, latency_ms=12.3, ttft_ms=None, error="")
    assert r.ok is True
    assert r.latency_ms > 0
```

另加 3 个 adapter 的 mock 测试：
- `nerva_binary_rpc` 正常解析 DATA/ERROR 帧
- `vllm_openai_api` 解析 OpenAI 响应中的文本字段
- `triton_infer` 解析 Triton `infer` JSON 输出

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_phase7_targets.py -v`
Expected: `ModuleNotFoundError`

**Step 3: 最小实现 adapter 协议**

```python
class BenchTarget(Protocol):
    async def infer(self, payload: dict[str, Any], *, deadline_ms: int) -> TargetResponse: ...
```

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_phase7_targets.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/bench/targets/base.py scripts/bench/targets/nerva_binary_rpc.py scripts/bench/targets/vllm_openai_api.py scripts/bench/targets/triton_infer.py tests/test_phase7_targets.py
git commit -m "feat(bench): add phase7 target adapters for nerva vllm triton"
```

---

### Task 4: 实现 1000 并发 loadgen 引擎（闭环并发）

**Files:**
- Create: `scripts/bench/loadgen.py`
- Create: `tests/test_phase7_loadgen.py`

**Step 1: 写失败测试（并发上限与统计）**

```python
from scripts.bench.loadgen import run_closed_loop


async def test_never_exceeds_concurrency_limit():
    result = await run_closed_loop(..., concurrency=1000, duration_s=2)
    assert result.max_in_flight <= 1000
```

再加一个统计测试：
- `p50 <= p95 <= p99`
- `error_rate` 计算正确

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_phase7_loadgen.py -v`
Expected: `ModuleNotFoundError`

**Step 3: 最小实现 loadgen**

实现点：
- `asyncio.Semaphore(concurrency)` 控制在飞请求
- 每请求记录：开始时间、结束时间、是否成功、错误类型、TTFT（可选）
- 输出结构至少包含：`qps/p50/p95/p99/error_rate/max_in_flight`

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_phase7_loadgen.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/bench/loadgen.py tests/test_phase7_loadgen.py
git commit -m "feat(bench): add phase7 loadgen engine for up to 1000 concurrency"
```

---

### Task 5: 实现 Phase 7 统一 runner（矩阵执行 + 产物落盘）

**Files:**
- Create: `scripts/bench/run_phase7.py`
- Create: `tests/test_phase7_runner.py`

**Step 1: 写失败测试（dry-run + 结果目录）**

```python
from scripts.bench.run_phase7 import build_matrix


def test_build_matrix_contains_c1000():
    matrix = build_matrix([1, 32, 128, 512, 1000], ["nerva", "vllm", "triton"])
    assert any(item.concurrency == 1000 for item in matrix)
```

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_phase7_runner.py -v`
Expected: `ModuleNotFoundError`

**Step 3: 最小实现 runner CLI**

功能要求：
- `--target {nerva,vllm,triton}`（支持多次）
- `--workload phase7_mm_vllm`
- `--concurrency-levels 1,32,128,512,1000`
- `--warmup-seconds` / `--sample-seconds`
- 结果落盘：`bench-results/phase7/<date>/<commit>/<target>/<concurrency>/`

每目录写入：
- `summary.json`
- `raw-latency.csv`
- `run-meta.json`

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_phase7_runner.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/bench/run_phase7.py tests/test_phase7_runner.py
git commit -m "feat(bench): add phase7 matrix runner and artifact writer"
```

---

### Task 6: 实现 vLLM / Triton 可启动胶水层（供 runbook 与 runner 复用）

**Files:**
- Create: `scripts/bench/infra/start_vllm_server.py`
- Create: `scripts/bench/infra/start_triton_server.py`
- Create: `scripts/bench/infra/prepare_triton_repo.py`
- Create: `scripts/bench/infra/wait_service_ready.py`
- Create: `tests/test_phase7_infra_launchers.py`

**Step 1: 写失败测试（启动命令与就绪探针）**

在 `tests/test_phase7_infra_launchers.py` 添加断言：
- `start_vllm_server.py --dry-run` 输出完整可执行命令（含 `vllm serve`、host、port、model）
- `start_triton_server.py --dry-run` 输出完整可执行命令（含 `tritonserver --model-repository`）
- `wait_service_ready.py` 能对 vLLM `/health` 与 Triton `/v2/health/ready` 做重试探测

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_phase7_infra_launchers.py -v`
Expected: `ModuleNotFoundError`

**Step 3: 最小实现胶水脚本**

实现要求：
1. `start_vllm_server.py`
   - 参数：`--model`, `--host`, `--port`, `--dtype`, `--tensor-parallel-size`, `--gpu-memory-utilization`
   - 支持 `--dry-run` 与真实启动
2. `prepare_triton_repo.py`
   - 生成最小可跑的 Triton model repository 目录结构（含 `config.pbtxt`）
   - 默认产出 `phase7_mm_vllm` 服务所需模型/ensemble 配置模板
3. `start_triton_server.py`
   - 参数：`--model-repo`, `--http-port`, `--grpc-port`, `--metrics-port`
   - 支持 `--dry-run` 与真实启动
4. `wait_service_ready.py`
   - 轮询健康接口，超时后返回非 0 退出码

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_phase7_infra_launchers.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add scripts/bench/infra/start_vllm_server.py scripts/bench/infra/start_triton_server.py scripts/bench/infra/prepare_triton_repo.py scripts/bench/infra/wait_service_ready.py tests/test_phase7_infra_launchers.py
git commit -m "feat(bench): add phase7 infra launchers for vllm and triton"
```

---

### Task 7: 产出对照实验 runbook（引用可执行胶水命令）

**Files:**
- Create: `docs/plans/2026-03-02-phase7-e2e-benchmark-runbook.md`
- Modify: `README.md`（新增 benchmark 快速入口）
- Create: `tests/test_phase7_runbook_refs.py`

**Step 1: 写文档测试（命令可落地执行）**

在 `tests/test_phase7_runbook_refs.py` 添加最小断言：
- runbook 中包含 `scripts/bench/infra/start_vllm_server.py` 启动命令
- runbook 中包含 `scripts/bench/infra/start_triton_server.py` 启动命令
- runbook 中包含 `scripts/bench/infra/wait_service_ready.py` 就绪检查命令
- 包含 `1000` 并发档位描述

**Step 2: 运行测试验证失败**

Run: `uv run pytest tests/test_phase7_runbook_refs.py -v`
Expected: FAIL（文件不存在）

**Step 3: 填写 runbook**

至少包含：
1. Nerva 启动（Phase 7 server）
2. Native vLLM 启动（通过 `start_vllm_server.py`）
3. Triton 启动（`prepare_triton_repo.py` + `start_triton_server.py`）
4. 服务就绪检查（`wait_service_ready.py`）
5. 统一压测命令示例
6. 常见故障排查（client 打满、server 饱和、GPU OOM）

**Step 4: 运行测试验证通过**

Run: `uv run pytest tests/test_phase7_runbook_refs.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add docs/plans/2026-03-02-phase7-e2e-benchmark-runbook.md README.md tests/test_phase7_runbook_refs.py
git commit -m "docs(bench): add phase7 e2e benchmark runbook with executable launchers"
```

---

### Task 8: 集成验证与首轮基线采集

**Files:**
- Modify: `docs/plans/benchmark-report-template.md`（已中文化，如需补充字段在本任务调整）
- Create: `bench-results/phase7/<date>/<commit>/...`（运行产物）

**Step 1: 代码质量检查**

```bash
export PATH="$HOME/.local/bin:$PATH"
uv run ruff check src/ tests/ examples/ scripts/
uv run mypy
uv run pytest tests/test_phase7_bench_config.py tests/test_phase7_multimodal_graph.py tests/test_phase7_targets.py tests/test_phase7_loadgen.py tests/test_phase7_runner.py tests/test_phase7_infra_launchers.py tests/test_phase7_runbook_refs.py -v
```

Expected: 全部通过

**Step 2: 启动对照服务并做就绪检查**

```bash
# vLLM
uv run python scripts/bench/infra/start_vllm_server.py --model <MODEL_PATH> --host 127.0.0.1 --port 8001
uv run python scripts/bench/infra/wait_service_ready.py --kind vllm --url http://127.0.0.1:8001/health --timeout-seconds 120

# Triton
uv run python scripts/bench/infra/prepare_triton_repo.py --output /tmp/phase7-triton-repo
uv run python scripts/bench/infra/start_triton_server.py --model-repo /tmp/phase7-triton-repo --http-port 8002 --grpc-port 8003 --metrics-port 8004
uv run python scripts/bench/infra/wait_service_ready.py --kind triton --url http://127.0.0.1:8002/v2/health/ready --timeout-seconds 120
```

Expected: vLLM 与 Triton 都可用

**Step 3: 小流量冒烟（C=1,32）**

```bash
uv run python scripts/bench/run_phase7.py --target nerva --target vllm --target triton --concurrency-levels 1,32 --warmup-seconds 10 --sample-seconds 30
```

Expected: 三目标均输出 `summary.json`，错误率可解释

**Step 4: 全矩阵压测（含 C=1000）**

```bash
uv run python scripts/bench/run_phase7.py --target nerva --target vllm --target triton --concurrency-levels 1,32,128,512,1000 --warmup-seconds 60 --sample-seconds 300
```

Expected: 产物完整，无进程泄漏；若出现拒绝/超时，需在报告中给出归因

**Step 5: 生成对照报告**

```bash
uv run python scripts/bench/report_phase7.py --input-root bench-results/phase7 --output docs/plans/phase7-benchmark-report.md
```

Expected: 输出中文版对照报告，包含三系统横向比较

**Step 6: Commit**

```bash
git add docs/plans/phase7-benchmark-report.md bench-results/phase7
# bench-results 如被 .gitignore 忽略，则仅提交报告与聚合摘要
git commit -m "perf(phase7): run e2e benchmark matrix up to c1000 with nerva/vllm/triton baselines"
```

---

## 验收标准（Phase 7）

1. 代码层：`scripts/bench/` 与 `examples/phase7_multimodal_vllm_server.py` 完整可运行。  
2. 工作负载层：新 DAG 明确为“多模态前处理 + vLLM + 后处理”，且不依赖 `03_parallel_dag.py`。  
3. 压测层：同一硬件、同一输入集下完成 `Nerva / Native vLLM / Triton` 三方对照。  
4. 并发层：至少一次稳定执行 `C=1000` 采样窗口，产出可复现数据。  
5. 报告层：使用中文版模板输出结论，包含 QPS、p50/p95/p99、错误率、TTFT（如适用）、瓶颈归因与下一步动作。  

## 风险与兜底

- 若单机 loadgen 无法真实打到 1000 并发：增加多进程/多机分片模式并在报告中标注 client 瓶颈。  
- 若 Triton 无法在同等链路承载“前后处理 + LLM”完整 DAG：先保持 LLM 核心路径对照，前后处理外置并明确口径差异。  
- 若 vLLM 或 Triton 版本差异影响 API：在 runbook 固化版本并记录镜像/包版本。  
