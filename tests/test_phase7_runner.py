from __future__ import annotations

import datetime as dt
import json
from typing import TYPE_CHECKING, Any

from scripts.bench.run_phase7 import (
    BenchmarkRun,
    _detect_backend_mode,
    _payload_for_target,
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
            assert "text" in payload
            assert "image_bytes" in payload
            assert payload["max_tokens"] == 256
            assert payload["temperature"] == 1.0
            assert payload["top_p"] == 1.0
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
    nerva_payload = _payload_for_target(seq=7, workload="phase7_mm_vllm")
    assert nerva_payload["text"] == "phase7 benchmark sample #7"
    assert nerva_payload["image_bytes"] == b"\x00" * 16
    assert len(nerva_payload["image_bytes"]) == 16
    assert nerva_payload["max_tokens"] == 256
    assert nerva_payload["temperature"] == 1.0
    assert nerva_payload["top_p"] == 1.0

    vllm_payload = _payload_for_target(seq=7, workload="phase7_mm_vllm")
    assert vllm_payload["text"] == "phase7 benchmark sample #7"
    assert vllm_payload["image_bytes"] == b"\x00" * 16
    assert vllm_payload["max_tokens"] == 256
    assert vllm_payload["temperature"] == 1.0
    assert vllm_payload["top_p"] == 1.0

    triton_payload = _payload_for_target(seq=7, workload="phase7_mm_vllm")
    assert triton_payload["text"] == "phase7 benchmark sample #7"
    assert triton_payload["image_bytes"] == b"\x00" * 16
    assert triton_payload["max_tokens"] == 256
    assert triton_payload["temperature"] == 1.0
    assert triton_payload["top_p"] == 1.0


def test_payload_for_targets_accepts_custom_sampling_params() -> None:
    payload = _payload_for_target(
        seq=1,
        workload="phase7_mm_vllm",
        max_tokens=128,
        temperature=0.2,
        top_p=0.9,
    )
    assert payload["max_tokens"] == 128
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.9


async def test_execute_benchmark_run_counts_target_errors() -> None:
    class _DummyTarget:
        async def infer(self, payload: dict[str, Any], *, deadline_ms: int) -> TargetResponse:
            assert deadline_ms == 100
            assert "text" in payload
            return TargetResponse(ok=False, latency_ms=1.0, ttft_ms=None, error="boom", output_text=None)

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


async def test_detect_backend_mode_marks_mock_from_health_payload() -> None:
    class _Args:
        deadline_ms = 1234
        vllm_url = "http://127.0.0.1:8001"
        triton_url = "http://127.0.0.1:8002"

    async def getter(url: str, timeout_ms: int) -> tuple[int, dict[str, Any] | None]:
        assert timeout_ms == 1234
        if url.endswith("/health"):
            return 200, {"backend": "mock_vllm"}
        return 200, {"backend": "mock_triton"}

    assert await _detect_backend_mode(_Args(), "vllm", health_getter=getter) == "mock"
    assert await _detect_backend_mode(_Args(), "triton", health_getter=getter) == "mock"
