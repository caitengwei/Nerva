from __future__ import annotations

from pathlib import Path

RUNBOOK = Path("docs/plans/2026-03-02-phase7-e2e-benchmark-runbook.md")
PLAN = Path("docs/plans/2026-03-02-phase7-e2e-benchmark-plan.md")


def test_runbook_includes_launcher_commands() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "examples.mm_vllm_server:app" in text
    assert "scripts/bench/infra/start_vllm_server.py" in text
    assert "scripts/bench/infra/start_triton_server.py" in text
    assert "scripts/bench/infra/wait_service_ready.py" in text


def test_runbook_mentions_c1000() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "1000" in text


def test_runbook_mentions_full_e2e_contract() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "full-e2e" in text
    assert "不再区分 infer-only" in text
    assert "MM_VLLM_MODEL_PATH=<MODEL_PATH>" in text
    assert "ensemble" in text
    assert "--max-tokens" in text
    assert "--temperature" in text
    assert "--top-p" in text
    assert "按目标分三次执行" in text


def test_runbook_mentions_nerdctl_gpu_commands() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    assert text.count("nerdctl run --rm --gpus all") >= 2
    assert "--network host --ipc host" in text
    assert "vllm/vllm-openai:v0.6.0" in text
    assert "nvcr.io/nvidia/tritonserver:24.08-py3" in text
    assert "--vllm-model /models" in text
    assert "--require-real-backend" in text


def test_bench_plan_has_no_agent_specific_instructions() -> None:
    text = PLAN.read_text(encoding="utf-8")
    assert "For Claude" not in text
    assert "REQUIRED SUB-SKILL" not in text
