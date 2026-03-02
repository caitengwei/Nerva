from __future__ import annotations

import argparse
import shlex
import subprocess
from collections.abc import Sequence


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
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


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

    completed = subprocess.run(cmd, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
