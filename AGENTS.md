# AGENTS.md

本文件是本仓库内的协作规范。若 `AGENTS.md` 与 `CLAUDE.md` 有冲突，以 `AGENTS.md` 为准。

## 0. 个人开发偏好（仓库内继承）

- 简单编辑或执行型任务默认直接处理，不进入 plan mode，不先做澄清提问。
- 主要语言与栈：Python、CUDA/C++（kernel dev）、Markdown、Shell。
- 沟通风格：回复简洁直接；仅在 ML/AI Infra/HPC 复杂问题上做必要展开。
- 代码修改偏好：宁可保守，也不要引入 bug。

## 1. 沟通与输出

- 始终使用中文回复（代码、命令、专有名词除外）。
- 回答结构优先“结论 -> 证据 -> 建议动作”。
- 涉及时间描述时，优先使用绝对日期，避免歧义。
- 若结论依赖文档而非实时执行，必须明确标注“文档口径”。

## 2. 技术关注点（默认视角）

- 以 ML/AI Infra/HPC 领域问题为核心：延迟、吞吐、并发、安全隔离、可观测性、故障恢复。
- 优先详细分析，避免未经验证的快速猜测。
- 对关键判断尽量提供可复现依据（代码路径、测试结果、最小复现步骤）。
- 涉及并发、IPC、调度、资源回收的改动，默认要求补回归测试或最小复现脚本。

## 3. 开发与修改规范

- 保守改动，优先避免引入 bug。
- 代码风格与项目保持一致（typing、async 模式、日志与错误处理风格、测试组织方式）。
- 不改动无关文件；若发现无关脏改动，保持原样并单独说明。
- 行为变更应配套测试；修复问题时优先补回归测试。
- 未经执行验证，不得声称“已修复”“已通过”。

## 4. 检查与质量门槛

### 4.1 快速检查（局部改动）

```bash
export PATH="$HOME/.local/bin:$PATH"
uv run ruff check src/ tests/
uv run pytest tests/<target_test>.py -v
```

### 4.2 提交前检查（默认）

```bash
export PATH="$HOME/.local/bin:$PATH"
uv run ruff check src/ tests/ examples/ scripts/
uv run mypy
uv run pytest tests/ -v
```

### 4.3 里程碑检查（性能与可观测性相关改动）

- 至少提供一项可复现实测数据（延迟或吞吐，对比基线）。
- 验证 `/metrics` 可访问，且关键日志链路包含 `request_id`。

## 5. Code Review 规范

- 重点关注：功能正确性、并发/进程边界、资源泄漏、IPC/协议兼容性、错误处理、测试覆盖缺口。
- findings 按严重级别排序（如 P1/P2/P3）。
- 每条问题必须附文件与行号（`path:line`）。
- 若未发现明显问题，明确说明“未发现阻断问题”，并给出残余风险与测试盲区。

## 6. 仓库状态更新机制

- 权威来源：`README.md`、`docs/plans/2026-02-25-mvp-roadmap.md`、最新 phase 计划/实现文档。
- 触发条件：phase 状态变化、对外 API 或 endpoint 变化、质量数据（lint/type/test）变化、已知风险状态变化。
- 更新步骤：
1. 先更新 roadmap 与对应实现文档。
2. 若对外行为变化，同步更新 README。
3. 回填本文件第 7 节（状态）与第 8 节（风险）。
4. 在状态描述中写明“更新时间（绝对日期）+ 口径来源”。

## 7. 仓库状态核验（文档口径，更新于 2026-03-02）

- 口径来源：`README.md`、`docs/plans/2026-02-25-mvp-roadmap.md`、`docs/plans/2026-03-02-phase6-e2e-runnable-server.md`。
- roadmap 状态：W1-W6 已完成；Phase 0 / Phase 1 / Phase 2 / Phase 3 / Phase 4 / Phase 5 均标记为已完成（Phase 5 完成日期 2026-03-02）。
- 当前对外能力（README）：支持 `build_nerva_app(...)` 与 `serve(...)` 两种服务入口；内置 `POST /rpc/{pipeline_name}`、`GET /v1/health`、`GET /v1/models`、`GET /metrics`。
- 文档记录的质量数据（roadmap）：`ruff` 0 errors、`mypy` 0 issues、测试 278 passed（Phase 5 完成口径）。
- 说明：本节是文档执行状态摘要，不等价于实时运行结果；实时结论以最新本地回归与 CI 为准。

## 8. 已知风险与回归必测项

### 8.1 风险登记模板（新增风险时按此记录）

- 风险 ID：
- 首次记录日期：
- 触发条件：
- 影响范围：
- 必测用例：
- 关闭条件：
- 最后核验日期：

### 8.2 当前风险：R-PH2-PROXY-CAPTURE（首次记录 2026-02-26）

- 触发条件：`cond()` / `parallel()` 分支捕获上游 `Proxy`（非仅 pipeline 原始输入）。
- 风险表现：可能出现执行卡住（`done_queue` 无事件）或分支输入语义错误。
- 必测用例 1：`out = a(x); cond(out["flag"], lambda: b(out), lambda: c(out))`。
- 必测用例 2：`out = a(x); parallel(lambda: b(out), lambda: c(out))`。
- 测试约束：上述用例必须加超时断言，并校验分支节点实收输入符合业务语义（不是 predicate 值或错误透传值）。
- 关闭条件：新增回归测试稳定通过，且多轮压力/超时场景无阻塞。
- 最后核验日期：2026-03-02（文档口径）。

## 9. 文档同步清单

- 改公共 API（如 `build_nerva_app`、`serve`）或 HTTP 路由：同步 `README.md`、roadmap、本文件第 7 节。
- 改 phase 完成状态：同步 roadmap 的状态与 Changelog，并更新本文件第 7 节日期与摘要。
- 改质量数据（lint/type/test）：同步 roadmap 或阶段实现文档，并更新本文件第 7 节。
- 新增或关闭风险：同步本文件第 8 节，并附对应测试文件路径。

## 10. 变更提交建议

- 提交前至少保证相关测试通过。
- commit message 建议使用语义化前缀（如 `fix(...)`、`feat(...)`、`docs(...)`）。
- 避免将本地工具配置或无关文件混入功能提交。
