from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import pytest
import scripts.bench.run_phase7 as run_phase7
from scripts.bench.infra.perf_compare_scenario import build_linux_gpu_perf_compare_scenario
from scripts.bench.run_phase7 import _amain, _cli

if TYPE_CHECKING:
    from pathlib import Path


async def test_phase7_perf_compare_dry_run_writes_all_targets(tmp_path: Path) -> None:
    args = _cli(
        [
            "--target",
            "nerva",
            "--target",
            "vllm",
            "--target",
            "triton",
            "--workload",
            "phase7_mm_vllm",
            "--concurrency-levels",
            "1",
            "--warmup-seconds",
            "1",
            "--sample-seconds",
            "1",
            "--output-root",
            str(tmp_path),
            "--dry-run",
        ]
    )
    await _amain(args)

    summary_files = sorted(tmp_path.glob("phase7/*/*/*/*/summary.json"))
    assert len(summary_files) == 3

    targets: set[str] = set()
    for file in summary_files:
        summary = json.loads(file.read_text())
        targets.add(summary["target"])
        assert summary["dry_run"] is True
    assert targets == {"nerva", "vllm", "triton"}


def test_linux_gpu_perf_compare_scenario_uses_nerdctl() -> None:
    scenario = build_linux_gpu_perf_compare_scenario(
        model_path="/models/Qwen/Qwen2.5-7B-Instruct",
        triton_repo="/tmp/phase7-triton-repo",
    )

    nerva_cmd = " ".join(scenario.nerva_server_cmd)
    assert "PHASE7_VLLM_MODEL_PATH=/models/Qwen/Qwen2.5-7B-Instruct" in nerva_cmd

    assert scenario.vllm_container_cmd[:2] == ["nerdctl", "run"]
    assert "--gpus" in scenario.vllm_container_cmd
    assert "all" in scenario.vllm_container_cmd

    assert scenario.triton_container_cmd[:2] == ["nerdctl", "run"]
    assert "--gpus" in scenario.triton_container_cmd
    assert "all" in scenario.triton_container_cmd

    prepare_cmd = " ".join(scenario.triton_prepare_cmd)
    assert "--vllm-url http://127.0.0.1:8001" in prepare_cmd
    assert "--vllm-model /models" in prepare_cmd

    nerva_bench_cmd = " ".join(scenario.benchmark_cmd_by_target["nerva"])
    assert "--target nerva" in nerva_bench_cmd
    assert "--target vllm" not in nerva_bench_cmd
    assert "--target triton" not in nerva_bench_cmd

    vllm_bench_cmd = " ".join(scenario.benchmark_cmd_by_target["vllm"])
    assert "--target vllm" in vllm_bench_cmd
    assert "--target nerva" not in vllm_bench_cmd
    assert "--target triton" not in vllm_bench_cmd

    triton_bench_cmd = " ".join(scenario.benchmark_cmd_by_target["triton"])
    assert "--target triton" in triton_bench_cmd
    assert "--target nerva" not in triton_bench_cmd
    assert "--target vllm" not in triton_bench_cmd
    assert "--vllm-url http://127.0.0.1:8001" in triton_bench_cmd
    assert "--require-real-backend" in triton_bench_cmd
    assert "--max-tokens 256" in triton_bench_cmd
    assert "--temperature 1.0" in triton_bench_cmd
    assert "--top-p 1.0" in triton_bench_cmd
    assert "--vllm-model /models" in triton_bench_cmd

    assert "vllm/vllm-openai:v0.6.0" in " ".join(scenario.vllm_container_cmd)


def test_linux_gpu_perf_compare_scenario_rejects_empty_concurrency_levels() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        build_linux_gpu_perf_compare_scenario(
            model_path="/models/Qwen/Qwen2.5-7B-Instruct",
            triton_repo="/tmp/phase7-triton-repo",
            concurrency_levels=[],
        )


@pytest.mark.parametrize("invalid_top_p", [0.0, -0.1, 1.1, float("inf"), float("nan")])
def test_linux_gpu_perf_compare_scenario_rejects_invalid_top_p(invalid_top_p: float) -> None:
    with pytest.raises(ValueError, match="top_p must be finite and in \\(0, 1\\]"):
        build_linux_gpu_perf_compare_scenario(
            model_path="/models/Qwen/Qwen2.5-7B-Instruct",
            triton_repo="/tmp/phase7-triton-repo",
            top_p=invalid_top_p,
        )


@pytest.mark.parametrize("invalid_temperature", [-0.1, float("inf"), float("nan")])
def test_linux_gpu_perf_compare_scenario_rejects_invalid_temperature(
    invalid_temperature: float,
) -> None:
    with pytest.raises(ValueError, match="temperature must be finite and >= 0"):
        build_linux_gpu_perf_compare_scenario(
            model_path="/models/Qwen/Qwen2.5-7B-Instruct",
            triton_repo="/tmp/phase7-triton-repo",
            temperature=invalid_temperature,
        )


async def test_phase7_perf_compare_non_dry_run_executes_all_targets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    built_targets: dict[str, _FakeTarget] = {}
    called_targets: list[str] = []

    def fake_build_target(_args: object, target_name: str) -> _FakeTarget:
        target = _FakeTarget(target_name)
        built_targets[target_name] = target
        return target

    async def fake_execute(
        run: run_phase7.BenchmarkRun,
        *,
        target: _FakeTarget,
        deadline_ms: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[dict[str, object], list[float], dict[str, object]]:
        called_targets.append(run.target)
        assert target.name == run.target
        assert max_tokens == 256
        assert temperature == 1.0
        assert top_p == 1.0
        return (
            {
                "target": run.target,
                "concurrency": run.concurrency,
                "workload": run.workload,
                "qps": 1.0,
                "p50_ms": 1.0,
                "p95_ms": 1.0,
                "p99_ms": 1.0,
                "error_rate": 0.0,
                "max_in_flight": 1,
                "total_requests": 1,
                "error_count": 0,
            },
            [1.0],
            {
                "target": run.target,
                "concurrency": run.concurrency,
                "workload": run.workload,
                "warmup_seconds": run.warmup_seconds,
                "sample_seconds": run.sample_seconds,
                "deadline_ms": deadline_ms,
            },
        )

    monkeypatch.setattr(run_phase7, "_build_target_from_args", fake_build_target)
    monkeypatch.setattr(run_phase7, "execute_benchmark_run", fake_execute)

    async def fake_backend_mode(*_args: object, **_kwargs: object) -> str:
        return "real"

    monkeypatch.setattr(run_phase7, "_detect_backend_mode", fake_backend_mode)

    args = _cli(
        [
            "--target",
            "nerva",
            "--target",
            "vllm",
            "--target",
            "triton",
            "--workload",
            "phase7_mm_vllm",
            "--concurrency-levels",
            "1",
            "--warmup-seconds",
            "1",
            "--sample-seconds",
            "1",
            "--deadline-ms",
            "99",
            "--output-root",
            str(tmp_path),
        ]
    )
    await _amain(args)

    assert called_targets == ["nerva", "vllm", "triton"]
    assert set(built_targets.keys()) == {"nerva", "vllm", "triton"}
    assert all(target.closed for target in built_targets.values())

    for target_name in ("nerva", "vllm", "triton"):
        summaries = list(tmp_path.glob(f"phase7/*/*/{target_name}/1/summary.json"))
        metas = list(tmp_path.glob(f"phase7/*/*/{target_name}/1/run-meta.json"))
        assert len(summaries) == 1
        assert len(metas) == 1

        summary = json.loads(summaries[0].read_text())
        assert summary["target"] == target_name
        assert summary["qps"] == 1.0
        assert "dry_run" not in summary

        meta = json.loads(metas[0].read_text())
        assert meta["target"] == target_name
        assert meta["dry_run"] is False
        assert meta["backend_mode"] == "real"
        assert meta["deadline_ms"] == 99


async def test_phase7_perf_compare_requires_real_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def fake_backend_mode(*_args: object, **_kwargs: object) -> str:
        return "mock"

    monkeypatch.setattr(run_phase7, "_detect_backend_mode", fake_backend_mode)
    args = _cli(
        [
            "--target",
            "vllm",
            "--workload",
            "phase7_mm_vllm",
            "--concurrency-levels",
            "1",
            "--warmup-seconds",
            "1",
            "--sample-seconds",
            "1",
            "--output-root",
            str(tmp_path),
            "--require-real-backend",
        ]
    )
    with pytest.raises(RuntimeError, match="not running in real mode"):
        await _amain(args)


class _FakeTarget:
    def __init__(self, name: str) -> None:
        self.name = name
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.gpu
async def test_phase7_perf_compare_real_services_smoke(tmp_path: Path) -> None:
    if os.getenv("NERVA_PHASE7_E2E_COMPARE") != "1":
        pytest.skip(
            "set NERVA_PHASE7_E2E_COMPARE=1 and start nerva/vllm/triton services before running"
        )

    args = _cli(
        [
            "--target",
            "nerva",
            "--target",
            "vllm",
            "--target",
            "triton",
            "--workload",
            "phase7_mm_vllm",
            "--vllm-model",
            os.getenv("NERVA_PHASE7_VLLM_MODEL", "phase7_mm_vllm"),
            "--concurrency-levels",
            "1",
            "--warmup-seconds",
            "1",
            "--sample-seconds",
            "1",
            "--nerva-url",
            os.getenv("NERVA_PHASE7_NERVA_URL", "http://127.0.0.1:8080"),
            "--vllm-url",
            os.getenv("NERVA_PHASE7_VLLM_URL", "http://127.0.0.1:8001"),
            "--triton-url",
            os.getenv("NERVA_PHASE7_TRITON_URL", "http://127.0.0.1:8002"),
            "--output-root",
            str(tmp_path),
        ]
    )

    await _amain(args)
    summary_files = sorted(tmp_path.glob("phase7/*/*/*/*/summary.json"))
    assert len(summary_files) == 3
