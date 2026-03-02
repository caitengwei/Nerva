from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from typing import TYPE_CHECKING, Any, cast

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

if TYPE_CHECKING:
    from collections.abc import Sequence

    from starlette.requests import Request


def build_vllm_command(
    *,
    model: str,
    host: str,
    port: int,
    dtype: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
) -> list[str]:
    return [
        "vllm",
        "serve",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--dtype",
        dtype,
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]


def _cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start vLLM server for Phase 7 benchmark")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="fallback to a mock OpenAI-compatible server when vllm binary is unavailable",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def resolve_launch_mode(*, binary_exists: bool, allow_mock: bool) -> str:
    if binary_exists:
        return "real"
    if allow_mock:
        return "mock"
    raise RuntimeError(
        "vllm executable not found; install vllm or rerun with --allow-mock for local smoke only"
    )


def _run_mock_server(*, host: str, port: int, model: str) -> int:
    async def health(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "backend": "mock_vllm"})

    async def completions(request: Request) -> JSONResponse:
        body: dict[str, Any] = cast("dict[str, Any]", await request.json())
        prompt = str(body.get("prompt", ""))
        payload = {
            "id": "mock-vllm",
            "object": "text_completion",
            "model": model,
            "choices": [{"text": f"[mock-vllm] {prompt}", "index": 0, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        return JSONResponse(payload)

    app = Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/v1/completions", completions, methods=["POST"]),
        ]
    )
    print(json.dumps({"mode": "mock_vllm", "host": host, "port": port, "model": model}))
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _cli(argv)
    cmd = build_vllm_command(
        model=args.model,
        host=args.host,
        port=args.port,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    if args.dry_run:
        print(shlex.join(cmd))
        return 0

    try:
        launch_mode = resolve_launch_mode(
            binary_exists=shutil.which("vllm") is not None,
            allow_mock=args.allow_mock,
        )
    except RuntimeError as exc:
        print(exc)
        return 2

    if launch_mode == "mock":
        return _run_mock_server(host=args.host, port=args.port, model=args.model)

    completed = subprocess.run(cmd, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
