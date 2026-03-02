from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkRun:
    target: str
    concurrency: int
    workload: str
    warmup_seconds: int
    sample_seconds: int


def _parse_concurrency_levels(levels: str) -> list[int]:
    parsed: list[int] = []
    for raw in levels.split(","):
        val = int(raw.strip())
        if val <= 0:
            raise ValueError("concurrency must be positive")
        parsed.append(val)
    return parsed


def _git_commit_short() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return "unknown"


def build_matrix(concurrency_levels: list[int], targets: list[str]) -> list[BenchmarkRun]:
    matrix: list[BenchmarkRun] = []
    for target in targets:
        for concurrency in concurrency_levels:
            matrix.append(
                BenchmarkRun(
                    target=target,
                    concurrency=concurrency,
                    workload="phase7_mm_vllm",
                    warmup_seconds=60,
                    sample_seconds=300,
                )
            )
    return matrix


def build_artifact_dir(root: Path, *, date: dt.date, commit: str, run: BenchmarkRun) -> Path:
    return root / "phase7" / date.isoformat() / commit / run.target / str(run.concurrency)


def write_artifacts(
    artifact_dir: Path,
    *,
    summary: dict[str, float | int | str],
    latencies_ms: list[float],
    meta: dict[str, object],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    (artifact_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    with (artifact_dir / "raw-latency.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["latency_ms"])
        for latency in latencies_ms:
            writer.writerow([latency])

    (artifact_dir / "run-meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 7 benchmark matrix")
    parser.add_argument("--target", action="append", choices=["nerva", "vllm", "triton"], required=True)
    parser.add_argument("--workload", default="phase7_mm_vllm")
    parser.add_argument("--concurrency-levels", default="1,32,128,512,1000")
    parser.add_argument("--warmup-seconds", type=int, default=60)
    parser.add_argument("--sample-seconds", type=int, default=300)
    parser.add_argument("--output-root", default="bench-results")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _cli()
    concurrency_levels = _parse_concurrency_levels(args.concurrency_levels)
    commit = _git_commit_short()
    today = dt.date.today()
    root = Path(args.output_root)

    matrix = [
        BenchmarkRun(
            target=target,
            concurrency=concurrency,
            workload=args.workload,
            warmup_seconds=args.warmup_seconds,
            sample_seconds=args.sample_seconds,
        )
        for target in args.target
        for concurrency in concurrency_levels
    ]

    for run in matrix:
        artifact_dir = build_artifact_dir(root, date=today, commit=commit, run=run)
        summary = {
            "target": run.target,
            "concurrency": run.concurrency,
            "workload": run.workload,
            "qps": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "error_rate": 0.0,
        }
        latencies: list[float] = []
        meta = {
            "target": run.target,
            "concurrency": run.concurrency,
            "workload": run.workload,
            "warmup_seconds": run.warmup_seconds,
            "sample_seconds": run.sample_seconds,
            "date": today.isoformat(),
            "commit": commit,
            "dry_run": args.dry_run,
        }

        write_artifacts(artifact_dir, summary=summary, latencies_ms=latencies, meta=meta)
        print(f"[phase7] wrote artifacts: {artifact_dir}")


if __name__ == "__main__":
    main()
