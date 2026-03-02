from __future__ import annotations

import datetime as dt
import json
from typing import TYPE_CHECKING, Any

from scripts.bench.run_phase7 import (
    BenchmarkRun,
    build_artifact_dir,
    build_matrix,
    execute_benchmark_run,
    write_artifacts,
)
from scripts.bench.targets.base import TargetResponse

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


async def test_execute_benchmark_run_generates_non_zero_metrics() -> None:
    class _DummyTarget:
        def __init__(self) -> None:
            self.calls = 0

        async def infer(self, payload: dict[str, Any], *, deadline_ms: int) -> TargetResponse:
            assert deadline_ms == 100
            assert "prompt" in payload
            self.calls += 1
            return TargetResponse(ok=True, latency_ms=1.0, ttft_ms=None, error="", output_text="ok")

    run = BenchmarkRun(
        target="vllm",
        concurrency=8,
        workload="phase7_mm_vllm",
        warmup_seconds=0,
        sample_seconds=1,
    )
    dummy_target = _DummyTarget()
    summary, latencies, meta = await execute_benchmark_run(
        run,
        target=dummy_target,
        deadline_ms=100,
    )

    assert dummy_target.calls > 0
    assert summary["qps"] > 0
    assert summary["p50_ms"] > 0
    assert summary["total_requests"] == len(latencies)
    assert meta["deadline_ms"] == 100
