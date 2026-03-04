from __future__ import annotations

import datetime as dt
import json
from typing import TYPE_CHECKING, Any

from scripts.bench.run_phase7 import (
    BenchmarkRun,
    _payload_for_target,
    _phase7_postprocess,
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


def test_payload_for_targets_uses_real_newline_and_binary_bytes() -> None:
    nerva_payload = _payload_for_target("nerva", seq=7, workload="phase7_mm_vllm")
    assert nerva_payload["text"] == "phase7 benchmark sample #7"
    assert nerva_payload["image_bytes"] == b"\x00" * 16
    assert len(nerva_payload["image_bytes"]) == 16

    vllm_payload = _payload_for_target("vllm", seq=7, workload="phase7_mm_vllm")
    assert vllm_payload["prompt"] == "[image_bytes=16]\nphase7 benchmark sample #7"

    triton_payload = _payload_for_target("triton", seq=7, workload="phase7_mm_vllm")
    assert triton_payload["text"] == "phase7 benchmark sample #7"
    assert triton_payload["image_bytes"] == b"\x00" * 16


def test_phase7_postprocess_matches_server_schema() -> None:
    normalized = _phase7_postprocess("  hello world  ")
    assert normalized == {"output_text": "hello world", "raw": "hello world"}


async def test_execute_benchmark_run_counts_postprocess_missing_output_as_error() -> None:
    class _DummyTarget:
        async def infer(self, payload: dict[str, Any], *, deadline_ms: int) -> TargetResponse:
            assert deadline_ms == 100
            assert "prompt" in payload
            return TargetResponse(ok=True, latency_ms=1.0, ttft_ms=None, error="", output_text=None)

    run = BenchmarkRun(
        target="vllm",
        concurrency=2,
        workload="phase7_mm_vllm",
        warmup_seconds=0,
        sample_seconds=1,
    )
    summary, _latencies, _meta = await execute_benchmark_run(
        run,
        target=_DummyTarget(),
        deadline_ms=100,
    )

    assert summary["total_requests"] > 0
    assert summary["error_count"] == summary["total_requests"]
    assert summary["error_rate"] == 1.0
