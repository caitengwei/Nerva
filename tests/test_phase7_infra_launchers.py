from __future__ import annotations

import argparse

import pytest
from scripts.bench.infra.start_triton_server import build_triton_command
from scripts.bench.infra.start_triton_server import main as triton_main
from scripts.bench.infra.start_triton_server import resolve_launch_mode as resolve_triton_mode
from scripts.bench.infra.start_vllm_server import build_vllm_command
from scripts.bench.infra.start_vllm_server import resolve_launch_mode as resolve_vllm_mode
from scripts.bench.infra.wait_service_ready import wait_service_ready


def test_start_vllm_server_dry_run_command() -> None:
    cmd = build_vllm_command(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        host="127.0.0.1",
        port=8001,
        dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
    )
    joined = " ".join(cmd)
    assert "vllm" in joined
    assert "serve" in joined
    assert "--host" in joined
    assert "--port" in joined


def test_start_triton_server_dry_run_command() -> None:
    cmd = build_triton_command(
        model_repo="/tmp/phase7-triton-repo",
        http_port=8002,
        grpc_port=8003,
        metrics_port=8004,
    )
    joined = " ".join(cmd)
    assert "tritonserver" in joined
    assert "--model-repository" in joined


def test_vllm_launch_mode_requires_explicit_mock_opt_in() -> None:
    assert resolve_vllm_mode(binary_exists=True, allow_mock=False) == "real"
    assert resolve_vllm_mode(binary_exists=False, allow_mock=True) == "mock"
    with pytest.raises(RuntimeError, match="vllm executable not found"):
        resolve_vllm_mode(binary_exists=False, allow_mock=False)


def test_triton_launch_mode_requires_explicit_mock_opt_in() -> None:
    assert resolve_triton_mode(binary_exists=True, allow_mock=False) == "real"
    assert resolve_triton_mode(binary_exists=False, allow_mock=True) == "mock"
    with pytest.raises(RuntimeError, match="tritonserver executable not found"):
        resolve_triton_mode(binary_exists=False, allow_mock=False)


async def test_wait_service_ready_retries_for_vllm() -> None:
    attempts = {"n": 0}

    async def getter(url: str) -> int:
        del url
        attempts["n"] += 1
        if attempts["n"] < 3:
            return 503
        return 200

    ok = await wait_service_ready(
        kind="vllm",
        url="http://127.0.0.1:8001/health",
        timeout_seconds=1.0,
        interval_seconds=0.01,
        getter=getter,
    )
    assert ok is True


async def test_wait_service_ready_retries_for_triton() -> None:
    attempts = {"n": 0}

    async def getter(url: str) -> int:
        del url
        attempts["n"] += 1
        if attempts["n"] < 2:
            return 503
        return 200

    ok = await wait_service_ready(
        kind="triton",
        url="http://127.0.0.1:8002/v2/health/ready",
        timeout_seconds=1.0,
        interval_seconds=0.01,
        getter=getter,
    )
    assert ok is True


def test_triton_mock_mode_uses_loopback_host(monkeypatch: pytest.MonkeyPatch) -> None:
    args = argparse.Namespace(
        model_repo="/tmp/model-repo",
        http_port=8002,
        grpc_port=8003,
        metrics_port=8004,
        allow_mock=True,
        dry_run=False,
    )

    called: dict[str, object] = {}

    def fake_cli(argv: list[str] | None = None) -> argparse.Namespace:
        del argv
        return args

    def fake_run_mock_server(*, host: str, port: int, model_repo: str) -> int:
        called["host"] = host
        called["port"] = port
        called["model_repo"] = model_repo
        return 0

    monkeypatch.setattr("scripts.bench.infra.start_triton_server._cli", fake_cli)
    monkeypatch.setattr("scripts.bench.infra.start_triton_server.shutil.which", lambda _binary: None)
    monkeypatch.setattr(
        "scripts.bench.infra.start_triton_server._run_mock_server",
        fake_run_mock_server,
    )

    assert triton_main([]) == 0
    assert called["host"] == "127.0.0.1"
