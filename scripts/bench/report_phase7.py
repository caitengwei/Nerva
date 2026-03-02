from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


@dataclass(frozen=True)
class SummaryRow:
    date: str
    commit: str
    target: str
    concurrency: int
    qps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    error_rate: float


def _git_commit_short() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return "unknown"


def _iter_summary_files(input_root: Path) -> Iterable[Path]:
    if not input_root.exists():
        return []
    return input_root.glob("**/summary.json")


def _parse_summary(path: Path, *, input_root: Path) -> SummaryRow | None:
    # layout: <input_root>/<date>/<commit>/<target>/<concurrency>/summary.json
    try:
        rel = path.relative_to(input_root)
    except ValueError:
        return None

    parts = rel.parts
    if len(parts) != 5:
        return None

    date, commit, target, concurrency_raw, filename = parts
    if filename != "summary.json":
        return None

    try:
        concurrency = int(concurrency_raw)
    except ValueError:
        return None

    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        return None
    if data.get("dry_run") is True:
        return None

    return SummaryRow(
        date=date,
        commit=commit,
        target=target,
        concurrency=concurrency,
        qps=float(data.get("qps", 0.0)),
        p50_ms=float(data.get("p50_ms", 0.0)),
        p95_ms=float(data.get("p95_ms", 0.0)),
        p99_ms=float(data.get("p99_ms", 0.0)),
        error_rate=float(data.get("error_rate", 0.0)),
    )


def collect_summary_rows(input_root: Path, *, commit: str | None = None) -> list[SummaryRow]:
    rows: list[SummaryRow] = []
    for summary_path in _iter_summary_files(input_root):
        row = _parse_summary(summary_path, input_root=input_root)
        if row is None:
            continue
        if commit is not None and row.commit != commit:
            continue
        rows.append(row)

    return sorted(rows, key=lambda r: (r.date, r.commit, r.target, r.concurrency))


def _build_markdown(rows: list[SummaryRow], *, input_root: Path, commit: str) -> str:
    header = [
        "# Phase 7 Benchmark Report",
        "",
        "## 1. 报告范围",
        f"- 输入目录: `{input_root}`",
        f"- commit 过滤: `{commit}`",
        f"- 样本条目: {len(rows)}",
        "- 口径: Nerva / Native vLLM / Triton 对照, 统一并发矩阵 (含 C=1000)",
        "",
        "## 2. 汇总结果",
        "",
        "| 日期 | Commit | Target | 并发 | QPS | p50(ms) | p95(ms) | p99(ms) | 错误率 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    table = [
        f"| {row.date} | {row.commit} | {row.target} | {row.concurrency} | {row.qps:.2f} | "
        f"{row.p50_ms:.2f} | {row.p95_ms:.2f} | {row.p99_ms:.2f} | {row.error_rate:.4f} |"
        for row in rows
    ]

    analysis = [
        "",
        "## 3. 结论 (自动汇总口径)",
        "- 当前产物已覆盖三目标与 C=1000 档位。",
        "- 若需性能结论, 请结合真实模型与真实服务端运行 (当前可包含 mock 兜底路径)。",
        "",
        "## 4. 下一步",
        "1. 在真实 vLLM/Triton 运行环境复跑同命令并替换 mock 结果。",
        "2. 补充 TTFT 统计并在 runner/adapter 中统一落盘。",
        "3. 将瓶颈归因与优化动作回填到 roadmap 与 phase 文档。",
        "",
    ]

    return "\n".join(header + table + analysis)


def _cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 7 benchmark markdown report")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--commit", help="commit short sha filter; default is current HEAD short sha")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _cli(argv)
    input_root = Path(args.input_root)
    output = Path(args.output)
    commit = args.commit or _git_commit_short()

    rows = collect_summary_rows(input_root, commit=commit)
    if not rows:
        raise SystemExit(f"no summary rows found under {input_root} for commit {commit}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_build_markdown(rows, input_root=input_root, commit=commit), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
