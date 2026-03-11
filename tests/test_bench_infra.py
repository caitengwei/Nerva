from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import pytest
from scripts.bench.infra.prepare_triton_repo import prepare_triton_repo
from scripts.bench.infra.start_triton_server import _resolve_model_name, build_triton_command
from scripts.bench.infra.start_triton_server import main as triton_main
from scripts.bench.infra.start_triton_server import resolve_launch_mode as resolve_triton_mode
from scripts.bench.infra.start_vllm_server import build_vllm_command
from scripts.bench.infra.start_vllm_server import resolve_launch_mode as resolve_vllm_mode
from scripts.bench.infra.wait_service_ready import wait_service_ready

if TYPE_CHECKING:
    from pathlib import Path


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
        model_repo="/tmp/mm_vllm-triton-repo",
        http_port=8002,
        grpc_port=8003,
        metrics_port=8004,
    )
    joined = " ".join(cmd)
    assert "tritonserver" in joined
    assert "--model-repository" in joined


def test_resolve_model_name_prefers_ensemble_model(tmp_path: Path) -> None:
    (tmp_path / "mm_vllm").mkdir()
    (tmp_path / "mm_preprocess").mkdir()
    assert _resolve_model_name(str(tmp_path)) == "mm_vllm"


def test_resolve_model_name_prefers_legacy_ensemble_model(tmp_path: Path) -> None:
    (tmp_path / "phase7_mm_vllm").mkdir()
    (tmp_path / "phase7_preprocess").mkdir()
    assert _resolve_model_name(str(tmp_path)) == "phase7_mm_vllm"


def test_resolve_model_name_skips_stage_models_in_fallback(tmp_path: Path) -> None:
    (tmp_path / "mm_preprocess").mkdir()
    (tmp_path / "mm_infer").mkdir()
    (tmp_path / "mm_postprocess").mkdir()
    (tmp_path / "custom_ensemble_model").mkdir()
    assert _resolve_model_name(str(tmp_path)) == "custom_ensemble_model"


def test_resolve_model_name_skips_legacy_stage_models_in_fallback(tmp_path: Path) -> None:
    (tmp_path / "phase7_preprocess").mkdir()
    (tmp_path / "phase7_infer").mkdir()
    (tmp_path / "phase7_postprocess").mkdir()
    (tmp_path / "legacy_ensemble_model").mkdir()
    assert _resolve_model_name(str(tmp_path)) == "legacy_ensemble_model"


def test_prepare_triton_repo_builds_ensemble_pipeline(tmp_path: Path) -> None:
    repo = prepare_triton_repo(
        tmp_path,
        model_name="mm_vllm",
        vllm_model_name="/models",
    )

    assert (repo / "mm_preprocess" / "config.pbtxt").exists()
    assert (repo / "mm_preprocess" / "1" / "model.py").exists()
    assert (repo / "mm_infer" / "config.pbtxt").exists()
    assert (repo / "mm_infer" / "1" / "model.py").exists()
    assert (repo / "mm_postprocess" / "config.pbtxt").exists()
    assert (repo / "mm_postprocess" / "1" / "model.py").exists()
    assert (repo / "mm_vllm" / "1").is_dir()

    preprocess_cfg = (repo / "mm_preprocess" / "config.pbtxt").read_text()
    assert "}\n  {" not in preprocess_cfg
    assert "max_batch_size: 0" in preprocess_cfg
    assert 'name: "MAX_TOKENS"' in preprocess_cfg
    assert 'name: "TEMPERATURE"' in preprocess_cfg
    assert 'name: "TOP_P"' in preprocess_cfg
    assert 'name: "DEADLINE_MS"' in preprocess_cfg

    infer_cfg = (repo / "mm_infer" / "config.pbtxt").read_text()
    assert "max_batch_size: 0" in infer_cfg
    assert "decoupled: true" in infer_cfg
    assert 'name: "PROMPT"' in infer_cfg
    assert 'name: "STREAM"' in infer_cfg

    infer_py = (repo / "mm_infer" / "1" / "model.py").read_text()
    assert "AsyncLLMEngine" in infer_py
    assert "AsyncEngineArgs" in infer_py
    assert "/v1/completions" not in infer_py
    assert "/models" in infer_py
    assert "deadline_ms" in infer_py

    ensemble_cfg = (repo / "mm_vllm" / "config.pbtxt").read_text()
    assert "}\n  {" not in ensemble_cfg
    assert "max_batch_size: 0" in ensemble_cfg
    assert "decoupled: true" in ensemble_cfg
    assert 'platform: "ensemble"' in ensemble_cfg
    assert 'model_name: "mm_preprocess"' in ensemble_cfg
    assert 'model_name: "mm_infer"' in ensemble_cfg
    assert 'model_name: "mm_postprocess"' in ensemble_cfg

    postprocess_py = (repo / "mm_postprocess" / "1" / "model.py").read_text()
    assert "raw_text = _to_str(text_raw)" in postprocess_py


def test_prepare_triton_repo_escapes_vllm_model_name_with_repr(tmp_path: Path) -> None:
    repo = prepare_triton_repo(
        tmp_path / "repo-escaped",
        model_name="mm_vllm",
        vllm_model_name="my'model\ninjected",
    )
    infer_py = (repo / "mm_infer" / "1" / "model.py").read_text()
    compile(infer_py, "<mm_infer_model>", "exec")
    literal_lines = [
        line for line in infer_py.splitlines() if "AsyncEngineArgs" in line
    ]
    assert len(literal_lines) == 1
    assert "\\n" in literal_lines[0]


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
