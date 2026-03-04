from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CONCURRENCY_LEVELS = [1, 32, 128, 512, 1000]
DEFAULT_VLLM_IMAGE = "vllm/vllm-openai:v0.6.0"
DEFAULT_TRITON_IMAGE = "nvcr.io/nvidia/tritonserver:24.08-py3"
DEFAULT_VLLM_MODEL_NAME = "/models"


@dataclass(frozen=True)
class PerfCompareScenario:
    nerva_server_cmd: list[str]
    vllm_container_cmd: list[str]
    triton_prepare_cmd: list[str]
    triton_container_cmd: list[str]
    benchmark_cmd: list[str]


def _validate_positive_levels(levels: list[int]) -> None:
    if not levels:
        raise ValueError("concurrency_levels must not be empty")
    if any(level <= 0 for level in levels):
        raise ValueError("concurrency_levels must be positive integers")


def build_linux_gpu_perf_compare_scenario(
    *,
    model_path: str,
    triton_repo: str,
    container_cli: str = "nerdctl",
    vllm_image: str = DEFAULT_VLLM_IMAGE,
    vllm_model_name: str = DEFAULT_VLLM_MODEL_NAME,
    triton_image: str = DEFAULT_TRITON_IMAGE,
    workload: str = "phase7_mm_vllm",
    concurrency_levels: list[int] | None = None,
    warmup_seconds: int = 60,
    sample_seconds: int = 300,
) -> PerfCompareScenario:
    if not model_path:
        raise ValueError("model_path must not be empty")
    if not triton_repo:
        raise ValueError("triton_repo must not be empty")
    if not container_cli:
        raise ValueError("container_cli must not be empty")
    if not vllm_model_name:
        raise ValueError("vllm_model_name must not be empty")
    if warmup_seconds <= 0:
        raise ValueError("warmup_seconds must be > 0")
    if sample_seconds <= 0:
        raise ValueError("sample_seconds must be > 0")

    if concurrency_levels is None:
        levels = list(DEFAULT_CONCURRENCY_LEVELS)
    else:
        levels = list(concurrency_levels)
    _validate_positive_levels(levels)

    level_arg = ",".join(str(level) for level in levels)

    nerva_server_cmd = [
        "uv",
        "run",
        "uvicorn",
        "examples.phase7_multimodal_vllm_server:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
    ]
    vllm_container_cmd = [
        container_cli,
        "run",
        "--rm",
        "--gpus",
        "all",
        "--network",
        "host",
        "--ipc",
        "host",
        "-v",
        f"{model_path}:/models:ro",
        vllm_image,
        "--model",
        "/models",
        "--host",
        "0.0.0.0",
        "--port",
        "8001",
    ]
    triton_prepare_cmd = [
        "uv",
        "run",
        "python",
        "scripts/bench/infra/prepare_triton_repo.py",
        "--output",
        triton_repo,
        "--model-name",
        workload,
    ]
    triton_container_cmd = [
        container_cli,
        "run",
        "--rm",
        "--gpus",
        "all",
        "--network",
        "host",
        "--ipc",
        "host",
        "-v",
        f"{triton_repo}:/models",
        triton_image,
        "tritonserver",
        "--model-repository=/models",
        "--http-port=8002",
        "--grpc-port=8003",
        "--metrics-port=8004",
    ]
    benchmark_cmd = [
        "uv",
        "run",
        "python",
        "scripts/bench/run_phase7.py",
        "--target",
        "nerva",
        "--target",
        "vllm",
        "--target",
        "triton",
        "--vllm-model",
        vllm_model_name,
        "--workload",
        workload,
        "--concurrency-levels",
        level_arg,
        "--warmup-seconds",
        str(warmup_seconds),
        "--sample-seconds",
        str(sample_seconds),
    ]
    return PerfCompareScenario(
        nerva_server_cmd=nerva_server_cmd,
        vllm_container_cmd=vllm_container_cmd,
        triton_prepare_cmd=triton_prepare_cmd,
        triton_container_cmd=triton_container_cmd,
        benchmark_cmd=benchmark_cmd,
    )
