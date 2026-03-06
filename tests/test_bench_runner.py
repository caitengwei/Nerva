from __future__ import annotations

import datetime as dt
import json
from typing import TYPE_CHECKING, Any

import pytest
import scripts.bench.run_bench as run_bench_module
from scripts.bench.run_bench import (
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
        workload="mm_vllm",
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
        workload="mm_vllm",
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


def test_payload_for_target_uses_real_newline_and_binary_bytes() -> None:
    payload = _payload_for_target(seq=7, workload="mm_vllm")
    assert payload["text"] == "mm_vllm benchmark sample #7"
    assert payload["image_bytes"] == b"\x00" * 16
    assert len(payload["image_bytes"]) == 16
    assert payload["max_tokens"] == 256
    assert payload["temperature"] == 1.0
    assert payload["top_p"] == 1.0


def test_payload_for_targets_accepts_custom_sampling_params() -> None:
    payload = _payload_for_target(
        seq=1,
        workload="mm_vllm",
        max_tokens=128,
        temperature=0.2,
        top_p=0.9,
    )
    assert payload["max_tokens"] == 128
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 0.9


@pytest.mark.parametrize("invalid_top_p", [0.0, -0.1, 1.1, float("inf"), float("nan")])
def test_payload_for_targets_rejects_invalid_top_p(invalid_top_p: float) -> None:
    with pytest.raises(ValueError, match="top_p must be finite and in \\(0, 1\\]"):
        _payload_for_target(seq=1, workload="mm_vllm", top_p=invalid_top_p)


@pytest.mark.parametrize("invalid_temperature", [-0.1, float("inf"), float("nan")])
def test_payload_for_targets_rejects_invalid_temperature(invalid_temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature must be finite and >= 0"):
        _payload_for_target(
            seq=1,
            workload="mm_vllm",
            temperature=invalid_temperature,
        )


async def test_execute_benchmark_run_counts_target_errors() -> None:
    class _DummyTarget:
        async def infer(self, payload: dict[str, Any], *, deadline_ms: int) -> TargetResponse:
            assert deadline_ms == 100
            assert "text" in payload
            return TargetResponse(ok=False, latency_ms=1.0, ttft_ms=None, error="boom", output_text=None)

    run = BenchmarkRun(
        target="vllm",
        concurrency=2,
        workload="mm_vllm",
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
        nerva_url = "http://127.0.0.1:8080"
        vllm_url = "http://127.0.0.1:8001"
        triton_url = "http://127.0.0.1:8002"

    async def getter(url: str, timeout_ms: int) -> tuple[int, dict[str, Any] | None]:
        assert timeout_ms == 1234
        if url.endswith("/v1/health"):
            return 200, {"backend": "mock_nerva"}
        if url.endswith("/health"):
            return 200, {"backend": "mock_vllm"}
        return 200, {"backend": "mock_triton"}

    assert await _detect_backend_mode(_Args(), "nerva", health_getter=getter) == "mock"
    assert await _detect_backend_mode(_Args(), "vllm", health_getter=getter) == "mock"
    assert await _detect_backend_mode(_Args(), "triton", health_getter=getter) == "mock"


async def test_detect_backend_mode_marks_nerva_unknown_when_health_not_ready() -> None:
    class _Args:
        deadline_ms = 2000
        nerva_url = "http://127.0.0.1:8080"
        vllm_url = "http://127.0.0.1:8001"
        triton_url = "http://127.0.0.1:8002"

    async def getter(url: str, timeout_ms: int) -> tuple[int, dict[str, Any] | None]:
        del timeout_ms
        assert url.endswith("/v1/health")
        return 503, {"status": "starting"}

    assert await _detect_backend_mode(_Args(), "nerva", health_getter=getter) == "unknown"


async def test_default_health_getter_reuses_and_closes_module_client(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeResponse:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {"ready": True}

    class _FakeClient:
        def __init__(self) -> None:
            self.get_calls = 0
            self.closed = False

        async def get(self, url: str, *, timeout: float) -> _FakeResponse:
            assert url == "http://127.0.0.1:8001/health"
            assert timeout == 0.5
            self.get_calls += 1
            return _FakeResponse()

        async def aclose(self) -> None:
            self.closed = True

    created_clients: list[_FakeClient] = []

    def _make_client(*_args: object, **_kwargs: object) -> _FakeClient:
        client = _FakeClient()
        created_clients.append(client)
        return client

    monkeypatch.setattr(run_bench_module.httpx, "AsyncClient", _make_client)
    await run_bench_module._close_health_client()
    status1, payload1 = await run_bench_module._default_health_getter(
        "http://127.0.0.1:8001/health",
        500,
    )
    status2, payload2 = await run_bench_module._default_health_getter(
        "http://127.0.0.1:8001/health",
        500,
    )
    await run_bench_module._close_health_client()

    assert status1 == 200
    assert payload1 == {"ready": True}
    assert status2 == 200
    assert payload2 == {"ready": True}
    assert len(created_clients) == 1
    assert created_clients[0].get_calls == 2
    assert created_clients[0].closed is True
