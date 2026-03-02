from __future__ import annotations

import datetime as dt
import json
from typing import TYPE_CHECKING

from scripts.bench.run_phase7 import (
    BenchmarkRun,
    build_artifact_dir,
    build_matrix,
    write_artifacts,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_build_matrix_contains_c1000() -> None:
    matrix = build_matrix([1, 32, 128, 512, 1000], ["nerva", "vllm", "triton"])
    assert any(item.concurrency == 1000 for item in matrix)


def test_write_artifacts_creates_expected_files(tmp_path: Path) -> None:
    run = BenchmarkRun(
        target="nerva",
        concurrency=32,
        workload="phase7_mm_vllm",
        warmup_seconds=10,
        sample_seconds=30,
    )
    root = tmp_path / "bench-results"
    date = dt.date(2026, 3, 2)
    path = build_artifact_dir(root, date=date, commit="abc123", run=run)

    summary = {"qps": 123.4, "error_rate": 0.01}
    latencies = [1.0, 2.0, 3.0]
    meta = {"target": "nerva", "concurrency": 32}

    write_artifacts(path, summary=summary, latencies_ms=latencies, meta=meta)

    assert (path / "summary.json").exists()
    assert (path / "raw-latency.csv").exists()
    assert (path / "run-meta.json").exists()

    loaded = json.loads((path / "summary.json").read_text())
    assert loaded["qps"] == 123.4
