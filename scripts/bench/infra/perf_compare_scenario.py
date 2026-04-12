from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_CONCURRENCY_LEVELS = [1, 32, 128, 512, 1000]
DEFAULT_CPU_MOCK_CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32]
DEFAULT_CPU_MOCK_TOKEN_LATENCY_MS = 0.5
DEFAULT_VLLM_IMAGE = "vllm/vllm-openai:v0.6.0"
DEFAULT_TRITON_IMAGE = "nvcr.io/nvidia/tritonserver:26.03-py3"
DEFAULT_VLLM_URL = "http://127.0.0.1:8001"
DEFAULT_VLLM_MODEL_NAME = "/models"
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0


@dataclass(frozen=True)
class PerfCompareScenario:
    nerva_server_cmd: list[str]
    vllm_container_cmd: list[str]
    triton_prepare_cmd: list[str]
    triton_container_cmd: list[str]
    benchmark_cmd_by_target: dict[str, list[str]]


def _validate_positive_levels(levels: list[int]) -> None:
    if not levels:
        raise ValueError("concurrency_levels must not be empty")
    if any(level <= 0 for level in levels):
        raise ValueError("concurrency_levels must be positive integers")


def _build_benchmark_cmd(
    *,
    targets: list[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    vllm_model_name: str,
    vllm_url: str,
    workload: str,
    level_arg: str,
    warmup_seconds: int,
    sample_seconds: int,
    require_real_backend: bool,
    triton_transport: str = "http",
    triton_grpc_url: str = "127.0.0.1:8003",
) -> list[str]:
    cmd: list[str] = [
        "uv",
        "run",
        "python",
        "scripts/bench/run_bench.py",
    ]
    for target in targets:
        cmd.extend(["--target", target])
    cmd.extend(
        [
            "--max-tokens",
            str(max_tokens),
            "--temperature",
            str(temperature),
            "--top-p",
            str(top_p),
            "--vllm-model",
            vllm_model_name,
            "--vllm-url",
            vllm_url,
            "--workload",
            workload,
            "--concurrency-levels",
            level_arg,
            "--warmup-seconds",
            str(warmup_seconds),
            "--sample-seconds",
            str(sample_seconds),
        ]
    )
    if "triton" in targets:
        cmd.extend(["--triton-transport", triton_transport])
        if triton_transport == "grpc-streaming":
            cmd.extend(["--triton-grpc-url", triton_grpc_url])
    if require_real_backend:
        cmd.append("--require-real-backend")
    return cmd


def build_linux_gpu_perf_compare_scenario(
    *,
    model_path: str,
    triton_repo: str,
    container_cli: str = "nerdctl",
    vllm_image: str = DEFAULT_VLLM_IMAGE,
    vllm_url: str = DEFAULT_VLLM_URL,
    vllm_model_name: str = DEFAULT_VLLM_MODEL_NAME,
    triton_image: str = DEFAULT_TRITON_IMAGE,
    workload: str = "mm_vllm",
    concurrency_levels: list[int] | None = None,
    warmup_seconds: int = 60,
    sample_seconds: int = 300,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    require_real_backend: bool = True,
    triton_transport: str = "http",
    triton_grpc_url: str = "127.0.0.1:8003",
    pre_post_instance_count: int = 20,
) -> PerfCompareScenario:
    if not model_path:
        raise ValueError("model_path must not be empty")
    if not triton_repo:
        raise ValueError("triton_repo must not be empty")
    if not container_cli:
        raise ValueError("container_cli must not be empty")
    if not vllm_url:
        raise ValueError("vllm_url must not be empty")
    if not vllm_model_name:
        raise ValueError("vllm_model_name must not be empty")
    if warmup_seconds <= 0:
        raise ValueError("warmup_seconds must be > 0")
    if sample_seconds <= 0:
        raise ValueError("sample_seconds must be > 0")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and >= 0")
    if not math.isfinite(top_p) or top_p <= 0 or top_p > 1:
        raise ValueError("top_p must be finite and in (0, 1]")
    if triton_transport not in {"http", "grpc-streaming"}:
        raise ValueError("triton_transport must be 'http' or 'grpc-streaming'")
    if not triton_grpc_url:
        raise ValueError("triton_grpc_url must not be empty")

    if concurrency_levels is None:
        levels = list(DEFAULT_CONCURRENCY_LEVELS)
    else:
        levels = list(concurrency_levels)
    _validate_positive_levels(levels)

    level_arg = ",".join(str(level) for level in levels)

    nerva_server_cmd = [
        "env",
        f"MM_VLLM_MODEL_PATH={model_path}",
        "uv",
        "run",
        "uvicorn",
        "examples.mm_vllm_server:app",
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
        "--vllm-model",
        vllm_model_name,
        "--pre-post-instance-count",
        str(pre_post_instance_count),
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
    benchmark_cmd_by_target = {
        "nerva": _build_benchmark_cmd(
            targets=["nerva"],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            vllm_model_name=vllm_model_name,
            vllm_url=vllm_url,
            workload=workload,
            level_arg=level_arg,
            warmup_seconds=warmup_seconds,
            sample_seconds=sample_seconds,
            require_real_backend=require_real_backend,
            triton_transport=triton_transport,
            triton_grpc_url=triton_grpc_url,
        ),
        "vllm": _build_benchmark_cmd(
            targets=["vllm"],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            vllm_model_name=vllm_model_name,
            vllm_url=vllm_url,
            workload=workload,
            level_arg=level_arg,
            warmup_seconds=warmup_seconds,
            sample_seconds=sample_seconds,
            require_real_backend=require_real_backend,
            triton_transport=triton_transport,
            triton_grpc_url=triton_grpc_url,
        ),
        "triton": _build_benchmark_cmd(
            targets=["triton"],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            vllm_model_name=vllm_model_name,
            vllm_url=vllm_url,
            workload=workload,
            level_arg=level_arg,
            warmup_seconds=warmup_seconds,
            sample_seconds=sample_seconds,
            require_real_backend=require_real_backend,
            triton_transport=triton_transport,
            triton_grpc_url=triton_grpc_url,
        ),
    }
    return PerfCompareScenario(
        nerva_server_cmd=nerva_server_cmd,
        vllm_container_cmd=vllm_container_cmd,
        triton_prepare_cmd=triton_prepare_cmd,
        triton_container_cmd=triton_container_cmd,
        benchmark_cmd_by_target=benchmark_cmd_by_target,
    )


def build_cpu_mock_perf_compare_scenario(
    *,
    triton_model_repo: str = "/tmp/triton_cpu_mock_repo",
    workload: str = "mm_vllm",
    concurrency_levels: list[int] | None = None,
    warmup_seconds: int = 10,
    sample_seconds: int = 30,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    token_latency_ms: float = DEFAULT_CPU_MOCK_TOKEN_LATENCY_MS,
    triton_transport: str = "http",
    triton_grpc_url: str = "127.0.0.1:8003",
    pre_post_instance_count: int = 20,
) -> PerfCompareScenario:
    """Build a CPU-only mock scenario for Nerva vs Triton e2e comparison.

    Neither Nerva nor Triton requires GPU or vLLM.  Both sides simulate LLM
    token-generation latency via asyncio.sleep(max_tokens * token_latency_ms).

    Start servers manually before running the benchmark:

        # Terminal 1 — Nerva
        MOCK_TOKEN_LATENCY_MS=0.5 uv run uvicorn \\
            examples.mm_vllm_cpu_mock_server:app --host 127.0.0.1 --port 8080

        # Terminal 2 — Triton mock
        uv run python scripts/bench/infra/start_triton_server.py \\
            --allow-mock --model-repo /tmp/triton_cpu_mock_repo \\
            --mock-token-latency-ms 0.5

        # Terminal 3 — benchmark
        uv run python scripts/bench/run_bench.py \\
            --target nerva --target triton \\
            --config scripts/bench/configs/cpu_mock_bench.json
    """
    if warmup_seconds <= 0:
        raise ValueError("warmup_seconds must be > 0")
    if sample_seconds <= 0:
        raise ValueError("sample_seconds must be > 0")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and >= 0")
    if not math.isfinite(top_p) or top_p <= 0 or top_p > 1:
        raise ValueError("top_p must be finite and in (0, 1]")
    if not math.isfinite(token_latency_ms) or token_latency_ms < 0:
        raise ValueError("token_latency_ms must be finite and >= 0")
    if triton_transport not in {"http", "grpc-streaming"}:
        raise ValueError("triton_transport must be 'http' or 'grpc-streaming'")
    if not triton_grpc_url:
        raise ValueError("triton_grpc_url must not be empty")

    if concurrency_levels is None:
        levels = list(DEFAULT_CPU_MOCK_CONCURRENCY_LEVELS)
    else:
        levels = list(concurrency_levels)
    _validate_positive_levels(levels)

    level_arg = ",".join(str(level) for level in levels)

    nerva_server_cmd = [
        "env",
        f"MOCK_TOKEN_LATENCY_MS={token_latency_ms}",
        "uv",
        "run",
        "uvicorn",
        "examples.mm_vllm_cpu_mock_server:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
    ]
    triton_server_cmd = [
        "uv",
        "run",
        "python",
        "scripts/bench/infra/start_triton_server.py",
        "--allow-mock",
        "--model-repo",
        triton_model_repo,
        "--mock-token-latency-ms",
        str(token_latency_ms),
    ]
    benchmark_cmd_by_target = {
        "nerva": _build_benchmark_cmd(
            targets=["nerva"],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            vllm_model_name="",
            vllm_url="http://127.0.0.1:8001",
            workload=workload,
            level_arg=level_arg,
            warmup_seconds=warmup_seconds,
            sample_seconds=sample_seconds,
            require_real_backend=False,
            triton_transport=triton_transport,
            triton_grpc_url=triton_grpc_url,
        ),
        "triton": _build_benchmark_cmd(
            targets=["triton"],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            vllm_model_name="",
            vllm_url="http://127.0.0.1:8001",
            workload=workload,
            level_arg=level_arg,
            warmup_seconds=warmup_seconds,
            sample_seconds=sample_seconds,
            require_real_backend=False,
            triton_transport=triton_transport,
            triton_grpc_url=triton_grpc_url,
        ),
    }
    # vllm_container_cmd / triton_container_cmd not needed for CPU mock;
    # triton_prepare_cmd is replaced by triton_server_cmd.
    return PerfCompareScenario(
        nerva_server_cmd=nerva_server_cmd,
        vllm_container_cmd=[],
        triton_prepare_cmd=triton_server_cmd,
        triton_container_cmd=[],
        benchmark_cmd_by_target=benchmark_cmd_by_target,
    )
