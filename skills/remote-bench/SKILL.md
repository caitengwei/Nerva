---
name: remote-bench
description: 在远程 Linux 开发机上执行 Nerva 性能测试。自动化环境准备、服务管理、压测执行、profiling 采集、结果收集和报告生成。支持 /remote-bench full|profile|report|status|stop。
---

# Remote Bench Skill

## 远程主机配置

```
REMOTE=tengwei.ctw@11.160.41.175
REMOTE_DIR=/home/tengwei.ctw/Nerva
```

## 调用格式

```
/remote-bench full         # Phase 0-6 全量：环境准备 → 服务启动 → 压测 → 收集 → 报告
/remote-bench profile      # 仅 profiling（假设服务已启动）
/remote-bench report       # 仅拉结果并生成报告（Phase 4-5）
/remote-bench status       # 检查远程服务状态
/remote-bench stop         # 停止远程服务并清理
```

## 阶段定义

### Phase 0 — 连接验证

```bash
ssh -o ConnectTimeout=5 tengwei.ctw@11.160.41.175 "echo ok"
```

失败则停止，提示用户检查网络 / SSH key。

### Phase 1 — 环境准备

```bash
ssh tengwei.ctw@11.160.41.175 "cd /home/tengwei.ctw/Nerva && uv run python scripts/bench/remote/env_setup.py --pull --sync-deps --pull-images --clean-timing"
```

解析 stdout 最后一行 JSON，展示：commit SHA、GPU 信息、镜像状态。
git pull 返回 error → 暂停，告知用户有未提交修改。

### Phase 2 — 启动服务

**Mock 模式**（默认，测框架开销）:
```bash
ssh tengwei.ctw@11.160.41.175 "cd /home/tengwei.ctw/Nerva && uv run python scripts/bench/remote/service_ctl.py start --targets nerva,triton --mode mock"
```

**Real 模式**（真实模型端到端）:
```bash
ssh tengwei.ctw@11.160.41.175 "cd /home/tengwei.ctw/Nerva && uv run python scripts/bench/remote/service_ctl.py start --targets nerva,vllm,triton --mode real --model-path <MODEL_PATH>"
```

解析 stdout JSON，展示各服务 endpoint 和健康状态。

**[确认点]** 展示：「服务已就绪 — 目标: [targets]，并发级别: [levels]。确认开始压测？」
等待用户明确确认后再继续。

### Phase 3 — C=1 校验轮

```bash
ssh tengwei.ctw@11.160.41.175 "cd /home/tengwei.ctw/Nerva && env -u all_proxy -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY NERVA_TIMING_LOG_DIR=/tmp/nerva_timing uv run python scripts/bench/run_bench.py --target nerva --target triton --concurrency-levels 1 --warmup-seconds 10 --sample-seconds 30"
```

（real 模式时加 `--target vllm`）

压测完成后，读取校验结果：
```bash
ssh tengwei.ctw@11.160.41.175 "cat /home/tengwei.ctw/Nerva/bench-results/mm_vllm/$(date +%Y-%m-%d)/*/nerva/1/summary.json"
```

检查条件：
- `error_rate == 0`（否则停止，排查代理/连接问题）
- `p50_ms < 5000`（否则警告，可能服务未就绪或配置有误）

校验通过后继续 Phase 3.5（完整矩阵）。

### Phase 3.5 — 完整压测矩阵

**长时间任务使用 nohup 防断连：**
```bash
ssh tengwei.ctw@11.160.41.175 "cd /home/tengwei.ctw/Nerva && nohup env -u all_proxy -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY NERVA_TIMING_LOG_DIR=/tmp/nerva_timing uv run python scripts/bench/run_bench.py --target nerva --target triton --concurrency-levels 1,32,128,512 --warmup-seconds 60 --sample-seconds 300 > /tmp/bench-run.log 2>&1 & echo $!"
```

轮询进度（每 30s）：
```bash
ssh tengwei.ctw@11.160.41.175 "tail -5 /tmp/bench-run.log"
```

### Phase 3.5 — 按需 Profiling（压测运行中）

用户请求「火焰图」时：
```bash
NERVA_PID=$(ssh tengwei.ctw@11.160.41.175 "pgrep -f 'nerva.server' | head -1")
ssh tengwei.ctw@11.160.41.175 "cd /home/tengwei.ctw/Nerva && uv run python scripts/bench/remote/profiler_ctl.py start --type py-spy --pid $NERVA_PID --duration 30"
```

用户请求「GPU 利用率」时：
```bash
ssh tengwei.ctw@11.160.41.175 "cd /home/tengwei.ctw/Nerva && uv run python scripts/bench/remote/profiler_ctl.py start --type nvidia-smi --interval 1"
```

用户请求「CUDA timeline」时：
```bash
NERVA_PID=$(ssh tengwei.ctw@11.160.41.175 "pgrep -f 'nerva.server' | head -1")
ssh tengwei.ctw@11.160.41.175 "cd /home/tengwei.ctw/Nerva && uv run python scripts/bench/remote/profiler_ctl.py start --type nsys --pid $NERVA_PID --duration 30"
```

用户请求「系统 perf」时：
```bash
NERVA_PID=$(ssh tengwei.ctw@11.160.41.175 "pgrep -f 'nerva.server' | head -1")
ssh tengwei.ctw@11.160.41.175 "cd /home/tengwei.ctw/Nerva && uv run python scripts/bench/remote/profiler_ctl.py start --type perf-stat --pid $NERVA_PID --duration 30"
```

压测结束后自动停止所有 profiler：
```bash
ssh tengwei.ctw@11.160.41.175 "cd /home/tengwei.ctw/Nerva && uv run python scripts/bench/remote/profiler_ctl.py stop --all"
```

### Phase 4 — 收集结果

```bash
ssh tengwei.ctw@11.160.41.175 "cd /home/tengwei.ctw/Nerva && uv run python scripts/bench/remote/collect_results.py --archive /tmp/bench-archive-$(date +%Y%m%d-%H%M%S).tar.gz"
```

读取 stdout JSON，获取 `archive_path`，然后 scp：
```bash
mkdir -p bench-results/remote
scp tengwei.ctw@11.160.41.175:<archive_path> bench-results/remote/
```

本地解压：
```bash
tar -xzf bench-results/remote/bench-archive-*.tar.gz -C bench-results/remote/
```

### Phase 5 — 生成报告

获取 commit SHA（从 Phase 1 env_setup 的 JSON 输出），然后：

```bash
uv run python scripts/bench/report_bench.py \
    --input-root bench-results/remote/bench-results/ \
    --output bench-results/remote/report.md
```

```bash
uv run python scripts/bench/analyze_timing_log.py \
    --log-dir bench-results/remote/timing/
```

展示报告摘要（QPS 对比表、p50/p95/p99）。如有 profiling 产物，列出文件路径。

### Phase 6 — 清理

```bash
ssh tengwei.ctw@11.160.41.175 "cd /home/tengwei.ctw/Nerva && uv run python scripts/bench/remote/service_ctl.py stop"
```

询问用户是否清理远程 `/tmp/bench-*.tar.gz` 和 `/tmp/bench-run.log`：
```bash
ssh tengwei.ctw@11.160.41.175 "rm -f /tmp/bench-*.tar.gz /tmp/bench-run.log"
```

## 异常处理原则

任何 SSH 命令返回非零 exit code → 停止，展示 stderr，等用户决定。

## Troubleshooting

| 症状 | 原因 | 处理 |
|------|------|------|
| error_rate=1.0, latency ~17ms | all_proxy 污染 httpx | 已在压测命令中 `env -u ...` 清除；若仍出现检查 `/etc/environment` |
| p50 远高于预期 | GPU throttling / 显存不足 | `ssh remote "nvidia-smi"` 检查温度和显存 |
| vLLM 启动失败 (P100) | P100 不支持 FP16 Tensor Core | service_ctl.py real 模式加 `--dtype float32` |
| SSH 命令超时 | 长时间压测断连 | Phase 3.5 已用 nohup + 轮询；重连后 `tail /tmp/bench-run.log` 查进度 |
| Triton 端口不可达 | Docker 网络配置 | Linux 用 `--network host`；macOS 用 `-p` |
| 首轮 C=1 校验失败 | 服务未就绪 / 端口冲突 | `service_ctl.py status` 检查，stop + restart |
