# Remote Bench Skill 设计文档

**日期**: 2026-04-03
**分支**: feature/linux-perf-auto
**状态**: Draft

## 概述

设计一个 Claude Code skill (`/remote-bench`)，用于在远程 Linux 开发机上自动化执行 Nerva 性能测试。采用"远程编排脚本 + Skill 调度"架构：远程新增一组 Python 脚本封装原子操作，Skill 通过 SSH 调度这些脚本，关键节点人工确认。

## 约束与上下文

- **远程机器**: `ssh tengwei.ctw@11.160.41.175`，NVIDIA P100×2，Docker 可用
- **代码**: 远程已有 `git clone`，通过 `git pull` 更新
- **压测目标**: Nerva vs vLLM vs Triton 全对比
- **运行模式**: Mock（测框架开销）+ Real（真实小模型端到端）
- **交互模型**: 自动为主，关键节点确认（服务就绪后确认开始压测、异常时暂停）
- **结果处理**: 拉原始数据（summary.json + raw-latency.csv + timing logs + profiling）回本地，本地生成报告

## 整体架构

```
本地 (macOS)                          远程 (Linux P100×2)
┌──────────────┐    SSH commands     ┌──────────────────────────┐
│ Claude Code  │ ──────────────────→ │ scripts/bench/remote/    │
│ + Skill      │                     │   env_setup.py           │
│              │                     │   service_ctl.py         │
│              │    scp results      │   run_perf.py            │
│              │ ←────────────────── │   profiler_ctl.py        │
│ report_bench │                     │   collect_results.py     │
│ analyze_timing│                    │                          │
└──────────────┘                     │ 现有基础设施:            │
                                     │   run_bench.py           │
                                     │   loadgen.py             │
                                     │   targets/               │
                                     └──────────────────────────┘
```

**调度流程**:
1. `env_setup.py` — git pull、uv sync、docker pull 镜像
2. `service_ctl.py start` — 启动 Nerva / vLLM / Triton 服务
3. **确认点**: 服务就绪，用户确认开始压测
4. `run_perf.py` — 调用 run_bench.py 执行压测矩阵
5. `profiler_ctl.py` — 压测过程中按需启动 profiling
6. `collect_results.py` — 打包所有产物为 tar.gz
7. 本地 scp 拉回 → `report_bench.py` + `analyze_timing_log.py` 生成报告

**异常处理**: 任何阶段非零退出码 → Skill 暂停，展示错误日志，等用户决定。

## 远程脚本设计 (`scripts/bench/remote/`)

### env_setup.py

```bash
ssh remote "cd ~/workspace/Nerva && python scripts/bench/remote/env_setup.py \
    [--pull] [--sync-deps] [--pull-images]"
```

**职责**:
- `--pull`: `git pull --ff-only`，失败则报错（不自动 merge）
- `--sync-deps`: `uv sync --all-extras`
- `--pull-images`: 根据 bench 目标拉取 Docker 镜像
  - vLLM: `vllm/vllm-openai:latest`（P100 需要 compute capability 6.0 兼容版本；首次 `env_setup.py` 运行时校验 `nvidia-smi` 输出并警告不兼容）
  - Triton: `nvcr.io/nvidia/tritonserver:24.05-py3`（固定已知稳定版本，避免 latest 漂移）

**输出**: JSON status
```json
{
  "git_commit": "abc123de",
  "git_branch": "main",
  "python_version": "3.14.0",
  "gpus": [
    {"name": "Tesla P100-PCIE-16GB", "memory_mb": 16384},
    {"name": "Tesla P100-PCIE-16GB", "memory_mb": 16384}
  ],
  "docker_images": ["vllm/vllm-openai:latest", "nvcr.io/nvidia/tritonserver:24.05-py3"],
  "uv_sync": "ok"
}
```

### service_ctl.py

```bash
ssh remote "python scripts/bench/remote/service_ctl.py start \
    --targets nerva,vllm,triton [--mode mock|real] [--model-path ...] \
    [--nerva-instances 'pre=2,infer=1,post=2']"
ssh remote "python scripts/bench/remote/service_ctl.py stop [--targets nerva,vllm,triton]"
ssh remote "python scripts/bench/remote/service_ctl.py status"
```

**职责**:
- `start`: 按 target 启动服务，等待健康检查通过
  - Nerva: `uv run python -m nerva.server ...` + wait_service_ready
  - vLLM: `docker run -d --gpus all vllm/vllm-openai ...` + wait
  - Triton: `prepare_triton_repo.py` → `docker run -d --gpus all nvcr.io/...` + wait
- `stop`: graceful shutdown（SIGTERM → 超时 SIGKILL），docker stop
- `status`: 报告各服务 PID / container ID / 端口 / 健康状态
- **Mock 模式**: Nerva 用 mock backend，Triton 用 `--cpu-mock`，vLLM 跳过或用 dummy model
- **Real 模式**: 指定 `--model-path`，自动检测 GPU 显存是否够用

**输出**: JSON
```json
{
  "nerva": {"status": "running", "pid": 12345, "endpoint": "http://127.0.0.1:8080", "health": "ok"},
  "vllm":  {"status": "running", "container_id": "abc123", "endpoint": "http://127.0.0.1:8001", "health": "ok"},
  "triton": {"status": "running", "container_id": "def456", "endpoint": "http://127.0.0.1:8002", "health": "ok"}
}
```

### run_perf.py

```bash
ssh remote "python scripts/bench/remote/run_perf.py \
    --targets nerva,vllm,triton --concurrency 1,32,128,512 \
    [--warmup 60] [--sample 300] [--mode mock|real] [--output-root bench-results]"
```

**职责**:
- 薄封装，核心调用现有 `run_bench.py`
- 自动设置 `NERVA_TIMING_LOG_DIR=/tmp/nerva_timing`
- 压测前自动清理旧 timing logs
- 首轮（C=1）结果校验 — error_rate=0、p50 合理性检查，不通过则 exit 1

**输出**: JSON 进度报告
```json
{
  "completed": [
    {"target": "nerva", "concurrency": 1, "qps": 7.8, "p50_ms": 128.3, "error_rate": 0.0},
    {"target": "nerva", "concurrency": 32, "qps": 245.1, "p50_ms": 130.5, "error_rate": 0.0}
  ],
  "remaining": [
    {"target": "nerva", "concurrency": 128},
    {"target": "vllm", "concurrency": 1}
  ],
  "validation": {"c1_check": "pass"}
}
```

### profiler_ctl.py

```bash
ssh remote "python scripts/bench/remote/profiler_ctl.py start \
    --type py-spy|nvidia-smi|nsys|perf-stat --pid <PID> \
    [--duration 30] [--interval 1] [--output /tmp/profile/]"
ssh remote "python scripts/bench/remote/profiler_ctl.py stop [--all]"
ssh remote "python scripts/bench/remote/profiler_ctl.py list"
```

**支持的 profiler**:

| Type | 命令 | 产物 |
|------|------|------|
| py-spy | `py-spy record -o flamegraph.svg --pid <PID> --duration <N>` | flamegraph.svg |
| nvidia-smi | `nvidia-smi dmon -s pucvmet -d <interval> -f gpu_monitor.csv` | gpu_monitor.csv |
| nsys | `nsys profile --trace=cuda,nvtx --duration=<N> -o timeline -p <PID>` | timeline.nsys-rep |
| perf-stat | `perf stat -p <PID> sleep <N>` | stdout → perf_stat.txt |

**职责**:
- 统一接口管理各种 profiler 的生命周期
- 每个 profiler 启动后记录 PID，`stop` 时 SIGINT/SIGTERM 优雅停止
- `list` 显示当前运行中的 profiler

**输出**: JSON
```json
{
  "profilers": [
    {"type": "py-spy", "pid": 23456, "target_pid": 12345, "output": "/tmp/profile/flamegraph.svg", "status": "running"},
    {"type": "nvidia-smi", "pid": 23457, "output": "/tmp/profile/gpu_monitor.csv", "status": "running"}
  ]
}
```

### collect_results.py

```bash
ssh remote "python scripts/bench/remote/collect_results.py \
    --output-root bench-results --timing-dir /tmp/nerva_timing \
    --profile-dir /tmp/profile/ --archive /tmp/bench-archive-<timestamp>.tar.gz"
```

**职责**:
- 收集三类产物:
  - bench-results/（summary.json + raw-latency.csv + run-meta.json）
  - timing logs（/tmp/nerva_timing/）
  - profiling 产物（flamegraph.svg、gpu_monitor.csv、nsys-rep、perf_stat.txt）
- 打包为 tar.gz

**输出**: JSON
```json
{
  "archive_path": "/tmp/bench-archive-20260403-143021.tar.gz",
  "size_mb": 12.3,
  "manifest": {
    "bench_results": 24,
    "timing_logs": 8,
    "profiling": 3
  },
  "git_commit": "abc123de"
}
```

## Skill 文件设计 (`~/.claude/skills/remote-bench/SKILL.md`)

### 元数据

```yaml
---
name: remote-bench
description: 在远程 Linux 开发机上执行 Nerva 性能测试。自动化环境准备、服务管理、压测执行、profiling 采集、结果收集和报告生成。
---
```

### 阶段定义

**Phase 0: 连接验证**
- `ssh -o ConnectTimeout=5 <remote> "echo ok"` 验证连通性
- 失败则提示用户检查网络 / SSH key

**Phase 1: 环境准备**
- `ssh remote "cd ~/workspace/Nerva && python scripts/bench/remote/env_setup.py --pull --sync-deps --pull-images"`
- 展示 JSON：commit SHA、GPU 信息、镜像状态
- git pull 失败（有未提交修改）→ 暂停告知用户

**Phase 2: 启动服务**
- `ssh remote "python scripts/bench/remote/service_ctl.py start --targets <targets> --mode <mode>"`
- 展示各服务 endpoint 和健康状态
- **确认点**: 「服务已就绪，准备开始压测 [targets] × [concurrency levels]，确认？」

**Phase 3: 执行压测**
- `ssh remote "python scripts/bench/remote/run_perf.py --targets <targets> --concurrency <levels>"`
- 首轮 C=1 自动校验，不通过则暂停
- 压测进行中可并行执行 Phase 3.5

**Phase 3.5: 按需 Profiling（可选）**
- 用户可在压测运行中随时请求：
  - 「抓火焰图」→ `profiler_ctl.py start --type py-spy --pid <PID> --duration 30`
  - 「看 GPU 利用率」→ `profiler_ctl.py start --type nvidia-smi --interval 1`
  - 「CUDA timeline」→ `profiler_ctl.py start --type nsys --pid <PID> --duration 30`
  - 「系统 perf」→ `profiler_ctl.py start --type perf-stat --pid <PID> --duration 30`
- 压测结束后自动 `profiler_ctl.py stop --all`

**Phase 4: 收集结果**
- `ssh remote "python scripts/bench/remote/collect_results.py --archive /tmp/bench-archive-<ts>.tar.gz"`
- `scp remote:/tmp/bench-archive-<ts>.tar.gz bench-results/remote/`
- 本地解压到 `bench-results/remote/<date>/<commit>/`

**Phase 5: 生成报告**
- `python scripts/bench/report_bench.py --input-root bench-results/remote/<date>/<commit>/ --output bench-results/remote/<date>/<commit>/report.md`
- `python scripts/bench/analyze_timing_log.py --log-dir bench-results/remote/<date>/<commit>/timing/`
- 展示报告摘要（QPS 对比表、关键延迟指标）
- 如有 profiling 产物，列出文件路径

**Phase 6: 清理**
- `ssh remote "python scripts/bench/remote/service_ctl.py stop"`
- 确认是否清理远程临时文件

### Skill 参数化

```
/remote-bench full                    # 全量：Phase 0-6
/remote-bench profile                 # 只进入 profiling 模式（假设服务已启动）
/remote-bench report                  # 只拉结果生成报告（Phase 4-5）
/remote-bench status                  # 检查远程服务状态
/remote-bench stop                    # 停止远程服务并清理
```

### 内置 Troubleshooting 表

| 症状 | 原因 | 处理 |
|------|------|------|
| error_rate=1.0, latency 异常低 (~17ms) | 代理变量 (all_proxy) 污染 httpx | 压测命令加 `env -u all_proxy -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY` |
| p50 远高于预期 | GPU throttling / 显存不足 | `nvidia-smi` 检查温度和显存占用 |
| vLLM 启动失败 | P100 不支持 FP16 Tensor Core | 加 `--dtype float32` 或 `--dtype auto` |
| SSH 命令超时 | 长时间压测 SSH 断连 | 远程脚本使用 nohup + 后台 PID 模式（见 SSH 命令模板），Skill 通过轮询日志末行获取进度，不依赖 SSH session 保活 |
| Triton container 端口不可达 | `--network host` 问题 | Linux 上用 `--network host`；macOS 上用 `-p` |
| uv sync 失败 | vllm 包 macOS 不兼容 | 远程是 Linux，应正常；检查 Python 版本 |
| 首轮 C=1 校验失败 | 服务未完全就绪 / 端口冲突 | `service_ctl.py status` 检查，必要时 stop + restart |

## 远程脚本通用规范

### 输出格式

所有远程脚本遵循统一输出协议:
- **stdout**: 仅 JSON（最后一行为完整 JSON 对象）
- **stderr**: 人类可读的进度日志（`structlog` 格式）
- **exit code**: 0=成功, 1=可恢复错误, 2=不可恢复错误

Skill 解析逻辑: 读 stdout 最后一行作为结构化结果，stderr 作为调试信息在异常时展示。

### SSH 命令模板

远程主机配置:
```
REMOTE=tengwei.ctw@11.160.41.175
REMOTE_DIR=~/workspace/Nerva
```

标准调用:
```bash
ssh $REMOTE "cd $REMOTE_DIR && uv run python scripts/bench/remote/<script>.py <args>"
```

长时间任务（防 SSH 断连）:
```bash
ssh $REMOTE "cd $REMOTE_DIR && nohup uv run python scripts/bench/remote/<script>.py <args> > /tmp/bench-<phase>.log 2>&1 & echo \$!"
# 后续轮询:
ssh $REMOTE "tail -1 /tmp/bench-<phase>.log"
```

### 文件组织

```
scripts/bench/remote/
├── __init__.py
├── env_setup.py          # ~100 行
├── service_ctl.py        # ~250 行（最复杂，管理三种服务生命周期）
├── run_perf.py           # ~120 行（薄封装 run_bench.py）
├── profiler_ctl.py       # ~200 行（统一 profiler 管理）
├── collect_results.py    # ~80 行（打包产物）
└── _common.py            # ~50 行（JSON 输出、日志初始化、GPU 检测）
```

预估总代码量: ~800 行 Python。

## 成功标准

1. `/remote-bench full` 一键完成：环境准备 → 三服务启动 → 压测矩阵 → 结果拉回 → 报告生成
2. 压测过程中能随时 `/remote-bench profile` 抓取火焰图、GPU 监控等
3. 首轮 C=1 自动校验，error_rate=0，p50 在合理范围内
4. 所有远程脚本可独立通过 SSH 手动调用，不依赖 Claude Code
5. 报告格式与现有 `report_bench.py` 输出一致
