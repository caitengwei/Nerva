from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import shutil
import subprocess
from typing import TYPE_CHECKING, Any, cast

import uvicorn
from scripts.bench.infra.perf_compare_scenario import DEFAULT_TRITON_IMAGE
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

if TYPE_CHECKING:
    from collections.abc import Sequence

    from starlette.requests import Request


def build_triton_command(
    *,
    model_repo: str,
    http_port: int,
    grpc_port: int,
    metrics_port: int,
) -> list[str]:
    return [
        "tritonserver",
        "--model-repository",
        model_repo,
        "--http-port",
        str(http_port),
        "--grpc-port",
        str(grpc_port),
        "--metrics-port",
        str(metrics_port),
    ]


def build_container_command(
    *,
    container_cli: str,
    triton_image: str,
    model_repo: str,
    http_port: int,
    grpc_port: int,
    metrics_port: int,
) -> list[str]:
    """Build command to run tritonserver inside a container (podman/docker/nerdctl).

    Uses port publishing (-p) instead of --network host for compatibility
    with macOS container runtimes (podman machine uses a Linux VM).
    """
    return [
        container_cli,
        "run",
        "--rm",
        "-p", f"{http_port}:{http_port}",
        "-p", f"{grpc_port}:{grpc_port}",
        "-p", f"{metrics_port}:{metrics_port}",
        "-v", f"{model_repo}:/models",
        triton_image,
        "tritonserver",
        "--model-repository=/models",
        f"--http-port={http_port}",
        f"--grpc-port={grpc_port}",
        f"--metrics-port={metrics_port}",
    ]


def _cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start Triton server for Phase 7 benchmark")
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--http-port", type=int, default=8002)
    parser.add_argument("--grpc-port", type=int, default=8003)
    parser.add_argument("--metrics-port", type=int, default=8004)
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="fallback to a mock Triton-compatible HTTP server when tritonserver is unavailable",
    )
    parser.add_argument(
        "--mock-token-latency-ms",
        type=float,
        default=0.0,
        help="per-token latency (ms) injected into mock infer responses to simulate LLM generation"
        " (default: 0.0, instant echo)",
    )
    parser.add_argument(
        "--container-cli",
        default="",
        help="container runtime to use for launching Triton (e.g. podman, docker, nerdctl);"
        " when set, runs tritonserver inside a container instead of as a local binary",
    )
    parser.add_argument(
        "--triton-image",
        default=DEFAULT_TRITON_IMAGE,
        help=f"Triton container image (default: {DEFAULT_TRITON_IMAGE})",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def resolve_launch_mode(*, binary_exists: bool, allow_mock: bool) -> str:
    if binary_exists:
        return "real"
    if allow_mock:
        return "mock"
    raise RuntimeError(
        "tritonserver executable not found; install Triton or rerun with --allow-mock for local smoke only"
    )


def _extract_max_tokens(inputs: list[Any]) -> int:
    """Parse MAX_TOKENS value from a Triton v2 inputs list."""
    for inp in inputs:
        if not isinstance(inp, dict):
            continue
        if str(inp.get("name", "")).upper() != "MAX_TOKENS":
            continue
        data = inp.get("data")
        if isinstance(data, list) and data:
            try:
                return max(int(data[0]), 1)
            except (TypeError, ValueError):
                pass
    return 256


def _run_mock_server(*, host: str, port: int, model_repo: str, token_latency_ms: float = 0.0) -> int:
    model_name = _resolve_model_name(model_repo)

    async def ready(_request: Request) -> JSONResponse:
        return JSONResponse({"ready": True, "backend": "mock_triton"})

    async def infer(request: Request) -> JSONResponse:
        body: dict[str, Any] = cast("dict[str, Any]", await request.json())
        prompt = ""
        inputs = body.get("inputs")
        if isinstance(inputs, list) and inputs:
            first = inputs[0]
            if isinstance(first, dict):
                data = first.get("data")
                if isinstance(data, list) and data:
                    prompt = str(data[0])
            if token_latency_ms > 0:
                max_tokens = _extract_max_tokens(inputs)
                await asyncio.sleep(max_tokens * token_latency_ms / 1000.0)
        payload = {
            "model_name": model_name,
            "outputs": [{"name": "text", "data": [f"[mock-triton] {prompt}"]}],
        }
        return JSONResponse(payload)

    app = Starlette(
        routes=[
            Route("/v2/health/ready", ready, methods=["GET"]),
            Route(f"/v2/models/{model_name}/infer", infer, methods=["POST"]),
        ]
    )
    print(
        json.dumps(
            {"mode": "mock_triton", "host": host, "port": port, "model_repo": model_repo}
        )
    )
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


def _resolve_model_name(model_repo: str) -> str:
    preferred_models = ("mm_vllm", "phase7_mm_vllm")
    for model_name in preferred_models:
        if os.path.isdir(os.path.join(model_repo, model_name)):
            return model_name

    stage_models = {
        "mm_preprocess",
        "mm_infer",
        "mm_postprocess",
        "phase7_preprocess",
        "phase7_infer",
        "phase7_postprocess",
    }
    candidates: list[str] = []
    with os.scandir(model_repo) as entries:
        for entry in entries:
            if entry.is_dir() and entry.name not in stage_models:
                candidates.append(entry.name)

    if candidates:
        return sorted(candidates)[0]

    return preferred_models[0]


def main(argv: Sequence[str] | None = None) -> int:
    args = _cli(argv)

    # Container mode: delegate entirely to the container runtime.
    if args.container_cli:
        container_cmd = build_container_command(
            container_cli=args.container_cli,
            triton_image=args.triton_image,
            model_repo=args.model_repo,
            http_port=args.http_port,
            grpc_port=args.grpc_port,
            metrics_port=args.metrics_port,
        )
        if args.dry_run:
            print(shlex.join(container_cmd))
            return 0
        completed = subprocess.run(container_cmd, check=False)
        return completed.returncode

    cmd = build_triton_command(
        model_repo=args.model_repo,
        http_port=args.http_port,
        grpc_port=args.grpc_port,
        metrics_port=args.metrics_port,
    )

    if args.dry_run:
        print(shlex.join(cmd))
        return 0

    try:
        launch_mode = resolve_launch_mode(
            binary_exists=shutil.which("tritonserver") is not None,
            allow_mock=args.allow_mock,
        )
    except RuntimeError as exc:
        print(exc)
        return 2

    if launch_mode == "mock":
        return _run_mock_server(
            host="127.0.0.1",
            port=args.http_port,
            model_repo=args.model_repo,
            token_latency_ms=args.mock_token_latency_ms,
        )

    completed = subprocess.run(cmd, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
