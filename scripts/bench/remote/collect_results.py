# scripts/bench/remote/collect_results.py
from __future__ import annotations

import argparse
import datetime
import subprocess
import tarfile
from pathlib import Path
from typing import Any

from scripts.bench.remote._common import emit_json, init_logging


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def collect_archive(
    *,
    output_root: Path,
    timing_dir: Path,
    profile_dir: Path,
    archive_path: Path,
) -> dict[str, Any]:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, int] = {}

    with tarfile.open(archive_path, "w:gz") as tar:
        if output_root.exists():
            tar.add(output_root, arcname="bench-results")
            manifest["bench_results"] = sum(1 for _ in output_root.rglob("*") if _.is_file())

        if timing_dir.exists():
            tar.add(timing_dir, arcname="timing")
            manifest["timing_logs"] = sum(1 for _ in timing_dir.rglob("*") if _.is_file())

        if profile_dir.exists():
            tar.add(profile_dir, arcname="profiling")
            manifest["profiling"] = sum(1 for _ in profile_dir.rglob("*") if _.is_file())

    size_mb = round(archive_path.stat().st_size / (1024 * 1024), 6)
    return {
        "archive_path": str(archive_path),
        "size_mb": size_mb,
        "manifest": manifest,
        "git_commit": _git_commit(),
    }


def _cli(argv: list[str] | None = None) -> argparse.Namespace:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description="Package bench results into archive")
    parser.add_argument("--output-root", default="bench-results")
    parser.add_argument("--timing-dir", default="/tmp/nerva_timing")
    parser.add_argument("--profile-dir", default="/tmp/profile")
    parser.add_argument("--archive", default=f"/tmp/bench-archive-{ts}.tar.gz")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    init_logging()
    args = _cli(argv)
    try:
        result = collect_archive(
            output_root=Path(args.output_root),
            timing_dir=Path(args.timing_dir),
            profile_dir=Path(args.profile_dir),
            archive_path=Path(args.archive),
        )
    except Exception as e:
        emit_json({"error": str(e), "step": "collect_archive"})
        return 1
    emit_json(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
