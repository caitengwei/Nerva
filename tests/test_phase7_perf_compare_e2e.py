from __future__ import annotations

import json
from typing import TYPE_CHECKING

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

    assert scenario.vllm_container_cmd[:2] == ["nerdctl", "run"]
    assert "--gpus" in scenario.vllm_container_cmd
    assert "all" in scenario.vllm_container_cmd

    assert scenario.triton_container_cmd[:2] == ["nerdctl", "run"]
    assert "--gpus" in scenario.triton_container_cmd
    assert "all" in scenario.triton_container_cmd

    bench_cmd = " ".join(scenario.benchmark_cmd)
    assert "--target nerva" in bench_cmd
    assert "--target vllm" in bench_cmd
    assert "--target triton" in bench_cmd
