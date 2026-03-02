from __future__ import annotations

from pathlib import Path

RUNBOOK = Path("docs/plans/2026-03-02-phase7-e2e-benchmark-runbook.md")


def test_runbook_includes_launcher_commands() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "scripts/bench/infra/start_vllm_server.py" in text
    assert "scripts/bench/infra/start_triton_server.py" in text
    assert "scripts/bench/infra/wait_service_ready.py" in text


def test_runbook_mentions_c1000() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    assert "1000" in text
