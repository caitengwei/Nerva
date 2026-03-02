from __future__ import annotations

import argparse
import shlex
import subprocess
from collections.abc import Sequence


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


def _cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start Triton server for Phase 7 benchmark")
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--http-port", type=int, default=8002)
    parser.add_argument("--grpc-port", type=int, default=8003)
    parser.add_argument("--metrics-port", type=int, default=8004)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _cli(argv)
    cmd = build_triton_command(
        model_repo=args.model_repo,
        http_port=args.http_port,
        grpc_port=args.grpc_port,
        metrics_port=args.metrics_port,
    )

    if args.dry_run:
        print(shlex.join(cmd))
        return 0

    completed = subprocess.run(cmd, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
