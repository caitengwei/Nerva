from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
import functools
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.bench.config import BenchConfig, load_bench_config
from scripts.bench.loadgen import LoadgenResult, run_closed_loop
from scripts.bench.targets.nerva_binary_rpc import NervaBinaryRPCTarget
from scripts.bench.targets.triton_infer import TritonInferTarget
from scripts.bench.targets.vllm_openai_api import VLLMOpenAIAPITarget

if TYPE_CHECKING:
    from scripts.bench.targets.base import BenchTarget

DEFAULT_CONCURRENCY_LEVELS = [1, 32, 128, 512, 1000]
DEFAULT_WARMUP_SECONDS = 60
DEFAULT_SAMPLE_SECONDS = 300
DEFAULT_DEADLINE_MS = 30_000
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
# Synthetic image payload size in bytes.  16 bytes is unrealistically small;
# 65536 (64 KB) better approximates a compressed thumbnail.  For real-world
# multimodal workloads this can be several MB (use --image-size-bytes).
DEFAULT_IMAGE_SIZE_BYTES = 65536
FULL_E2E_CONTRACT = "full-e2e"
# Process-global client reused within one script run; closed in _amain finally.
_HEALTH_CLIENT: httpx.AsyncClient | None = None


@dataclass(frozen=True)
class BenchmarkRun:
    target: str
    concurrency: int
    workload: str
    warmup_seconds: int
    sample_seconds: int


def _parse_concurrency_levels(levels: str) -> list[int]:
    parsed: list[int] = []
    for raw in levels.split(","):
        value = int(raw.strip())
        if value <= 0:
            raise ValueError("concurrency must be positive")
        parsed.append(value)
    return parsed


def _git_commit_short() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return "unknown"


def _resolve_runtime_config(args: argparse.Namespace) -> tuple[list[int], int, int]:
    config: BenchConfig | None = None
    if args.config is not None:
        config = load_bench_config(Path(args.config))

    if args.concurrency_levels is not None:
        levels = _parse_concurrency_levels(args.concurrency_levels)
    elif config is not None:
        levels = config.concurrency_levels
    else:
        levels = list(DEFAULT_CONCURRENCY_LEVELS)

    if args.warmup_seconds is not None:
        warmup_seconds = args.warmup_seconds
    elif config is not None:
        warmup_seconds = config.warmup_seconds
    else:
        warmup_seconds = DEFAULT_WARMUP_SECONDS

    if args.sample_seconds is not None:
        sample_seconds = args.sample_seconds
    elif config is not None:
        sample_seconds = config.sample_seconds
    else:
        sample_seconds = DEFAULT_SAMPLE_SECONDS

    if warmup_seconds <= 0:
        raise ValueError("warmup_seconds must be > 0")
    if sample_seconds <= 0:
        raise ValueError("sample_seconds must be > 0")

    return levels, warmup_seconds, sample_seconds


def build_matrix(
    concurrency_levels: list[int],
    targets: list[str],
    *,
    workload: str = "mm_vllm",
    warmup_seconds: int = DEFAULT_WARMUP_SECONDS,
    sample_seconds: int = DEFAULT_SAMPLE_SECONDS,
) -> list[BenchmarkRun]:
    return [
        BenchmarkRun(
            target=target,
            concurrency=concurrency,
            workload=workload,
            warmup_seconds=warmup_seconds,
            sample_seconds=sample_seconds,
        )
        for target in targets
        for concurrency in concurrency_levels
    ]


def build_artifact_dir(root: Path, *, date: dt.date, commit: str, run: BenchmarkRun) -> Path:
    return root / "mm_vllm" / date.isoformat() / commit / run.target / str(run.concurrency)


def write_artifacts(
    artifact_dir: Path,
    *,
    summary: dict[str, Any],
    latencies_ms: list[float],
    meta: dict[str, Any],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    (artifact_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    with (artifact_dir / "raw-latency.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["latency_ms"])
        for latency in latencies_ms:
            writer.writerow([latency])

    (artifact_dir / "run-meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def _payload_for_target(
    *,
    seq: int,
    workload: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    image_size_bytes: int = DEFAULT_IMAGE_SIZE_BYTES,
) -> dict[str, Any]:
    return _mm_vllm_source_input(
        seq=seq,
        workload=workload,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        image_size_bytes=image_size_bytes,
    )


@functools.lru_cache(maxsize=8)
def _synthetic_image(size: int) -> bytes:
    """Return a zero-filled synthetic image buffer, cached by size to avoid per-request allocation."""
    return b"\x00" * size


def _mm_vllm_source_input(
    *,
    seq: int,
    workload: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    image_size_bytes: int = DEFAULT_IMAGE_SIZE_BYTES,
) -> dict[str, Any]:
    if workload != "mm_vllm":
        raise ValueError(f"unsupported workload: {workload}")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if not math.isfinite(top_p) or top_p <= 0 or top_p > 1:
        raise ValueError("top_p must be finite and in (0, 1]")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and >= 0")
    if image_size_bytes <= 0:
        raise ValueError("image_size_bytes must be > 0")

    text = f"mm_vllm benchmark sample #{seq}"
    return {
        "text": text,
        "image_bytes": _synthetic_image(image_size_bytes),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }


def _get_health_client() -> httpx.AsyncClient:
    global _HEALTH_CLIENT
    if _HEALTH_CLIENT is None:
        _HEALTH_CLIENT = httpx.AsyncClient()
    return _HEALTH_CLIENT


async def _close_health_client() -> None:
    global _HEALTH_CLIENT
    if _HEALTH_CLIENT is not None:
        await _HEALTH_CLIENT.aclose()
        _HEALTH_CLIENT = None


async def _default_health_getter(
    url: str,
    timeout_ms: int,
) -> tuple[int, dict[str, Any] | None]:
    timeout_s = max(timeout_ms / 1000.0, 0.001)
    response = await _get_health_client().get(url, timeout=timeout_s)
    payload: dict[str, Any] | None = None
    try:
        body = response.json()
    except Exception:
        body = None
    if isinstance(body, dict):
        payload = body
    return response.status_code, payload


async def _detect_backend_mode(
    args: argparse.Namespace,
    target_name: str,
    *,
    health_getter: Any = _default_health_getter,
) -> str:
    # TODO(next-iteration): for target_name == "triton", also verify the upstream vLLM
    # dependency is running in real mode. Current check only validates Triton's own
    # health endpoint, which can miss "real triton + mock vllm" mixed deployments.
    if target_name == "nerva":
        url = f"{args.nerva_url.rstrip('/')}/v1/health"
    elif target_name == "vllm":
        url = f"{args.vllm_url.rstrip('/')}/health"
    elif target_name == "triton":
        url = f"{args.triton_url.rstrip('/')}/v2/health/ready"
    else:
        raise ValueError(f"unknown target for backend detection: {target_name}")

    try:
        status_code, payload = await health_getter(url, args.deadline_ms)
    except Exception:
        return "unknown"

    if status_code != 200:
        return "unknown"

    if isinstance(payload, dict):
        backend = str(payload.get("backend", "")).strip().lower()
        if backend == "mock" or backend.startswith("mock_"):
            return "mock"

    return "real"


def _build_target_from_args(args: argparse.Namespace, target_name: str) -> BenchTarget:
    if target_name == "nerva":
        return NervaBinaryRPCTarget(base_url=args.nerva_url, pipeline_name=args.nerva_pipeline)
    if target_name == "vllm":
        return VLLMOpenAIAPITarget(base_url=args.vllm_url, model_name=args.vllm_model)
    if target_name == "triton":
        return TritonInferTarget(base_url=args.triton_url, model_name=args.triton_model)
    raise ValueError(f"unknown target: {target_name}")


def _target_endpoint(args: argparse.Namespace, target_name: str) -> str:
    if target_name == "nerva":
        return f"{args.nerva_url.rstrip('/')}/rpc/{args.nerva_pipeline}"
    if target_name == "vllm":
        return f"{args.vllm_url.rstrip('/')}/v1/completions"
    if target_name == "triton":
        return f"{args.triton_url.rstrip('/')}/v2/models/{args.triton_model}/infer"
    raise ValueError(f"unknown target: {target_name}")


def _summary_from_result(run: BenchmarkRun, result: LoadgenResult) -> dict[str, Any]:
    return {
        "target": run.target,
        "concurrency": run.concurrency,
        "workload": run.workload,
        "qps": result.qps,
        "p50_ms": result.p50_ms,
        "p95_ms": result.p95_ms,
        "p99_ms": result.p99_ms,
        "error_rate": result.error_rate,
        "max_in_flight": result.max_in_flight,
        "total_requests": result.total_requests,
        "error_count": result.error_count,
    }


async def execute_benchmark_run(
    run: BenchmarkRun,
    *,
    target: BenchTarget,
    deadline_ms: int,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    image_size_bytes: int = DEFAULT_IMAGE_SIZE_BYTES,
) -> tuple[dict[str, Any], list[float], dict[str, Any]]:
    if deadline_ms <= 0:
        raise ValueError("deadline_ms must be > 0")

    async def _invoke(request_meta: dict[str, int], per_request_deadline_ms: int) -> tuple[bool, str]:
        seq = request_meta.get("seq", 0)
        payload = _payload_for_target(
            seq=seq,
            workload=run.workload,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            image_size_bytes=image_size_bytes,
        )
        response = await target.infer(payload, deadline_ms=min(deadline_ms, per_request_deadline_ms))
        return response.ok, response.error

    if run.warmup_seconds > 0:
        await run_closed_loop(
            _invoke,
            concurrency=run.concurrency,
            duration_s=float(run.warmup_seconds),
            deadline_ms=deadline_ms,
            record_latencies=False,
        )

    result = await run_closed_loop(
        _invoke,
        concurrency=run.concurrency,
        duration_s=float(run.sample_seconds),
        deadline_ms=deadline_ms,
    )

    summary = _summary_from_result(run, result)
    meta: dict[str, Any] = {
        "target": run.target,
        "concurrency": run.concurrency,
        "workload": run.workload,
        "contract": FULL_E2E_CONTRACT,
        "warmup_seconds": run.warmup_seconds,
        "sample_seconds": run.sample_seconds,
        "deadline_ms": deadline_ms,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "image_size_bytes": image_size_bytes,
    }
    return summary, result.latencies_ms, meta


def _cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mm_vllm benchmark matrix")
    parser.add_argument("--target", action="append", choices=["nerva", "vllm", "triton"], required=True)
    parser.add_argument("--workload", default="mm_vllm")
    parser.add_argument("--config", help="path to benchmark config json")
    parser.add_argument("--concurrency-levels", help="comma separated, e.g. 1,32,128,512,1000")
    parser.add_argument("--warmup-seconds", type=int)
    parser.add_argument("--sample-seconds", type=int)
    parser.add_argument("--deadline-ms", type=int, default=DEFAULT_DEADLINE_MS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument(
        "--image-size-bytes",
        type=int,
        default=DEFAULT_IMAGE_SIZE_BYTES,
        help="synthetic image payload size in bytes (default: 65536 / 64 KB)",
    )
    parser.add_argument(
        "--require-real-backend",
        action="store_true",
        help="fail if target health endpoint reports non-real backend mode",
    )

    parser.add_argument("--nerva-url", default="http://127.0.0.1:8080")
    parser.add_argument("--nerva-pipeline", default="mm_vllm")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8001")
    parser.add_argument("--vllm-model", default="mm_vllm")
    parser.add_argument("--triton-url", default="http://127.0.0.1:8002")
    parser.add_argument("--triton-model", default="mm_vllm")

    parser.add_argument("--output-root", default="bench-results")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


async def _amain(args: argparse.Namespace) -> None:
    concurrency_levels, warmup_seconds, sample_seconds = _resolve_runtime_config(args)
    matrix = build_matrix(
        concurrency_levels,
        args.target,
        workload=args.workload,
        warmup_seconds=warmup_seconds,
        sample_seconds=sample_seconds,
    )

    commit = _git_commit_short()
    today = dt.date.today()
    root = Path(args.output_root)
    backend_mode_by_target: dict[str, str] = {}

    try:
        for run in matrix:
            artifact_dir = build_artifact_dir(root, date=today, commit=commit, run=run)

            if args.dry_run:
                summary: dict[str, Any] = {
                    "target": run.target,
                    "concurrency": run.concurrency,
                    "workload": run.workload,
                    "qps": 0.0,
                    "p50_ms": 0.0,
                    "p95_ms": 0.0,
                    "p99_ms": 0.0,
                    "error_rate": 0.0,
                    "max_in_flight": 0,
                    "total_requests": 0,
                    "error_count": 0,
                    "dry_run": True,
                }
                latencies: list[float] = []
                meta: dict[str, Any] = {
                    "target": run.target,
                    "target_endpoint": _target_endpoint(args, run.target),
                    "concurrency": run.concurrency,
                    "workload": run.workload,
                    "warmup_seconds": run.warmup_seconds,
                    "sample_seconds": run.sample_seconds,
                    "deadline_ms": args.deadline_ms,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "image_size_bytes": args.image_size_bytes,
                    "date": today.isoformat(),
                    "commit": commit,
                    "dry_run": True,
                }
            else:
                backend_mode = backend_mode_by_target.get(run.target)
                if backend_mode is None:
                    backend_mode = await _detect_backend_mode(args, run.target)
                    backend_mode_by_target[run.target] = backend_mode
                if args.require_real_backend and backend_mode != "real":
                    raise RuntimeError(
                        f"target '{run.target}' is not running in real mode: {backend_mode}"
                    )

                target = _build_target_from_args(args, run.target)
                try:
                    summary, latencies, meta = await execute_benchmark_run(
                        run,
                        target=target,
                        deadline_ms=args.deadline_ms,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        image_size_bytes=args.image_size_bytes,
                    )
                finally:
                    close = getattr(target, "aclose", None)
                    if callable(close):
                        await close()
                meta.update(
                    {
                        "target_endpoint": _target_endpoint(args, run.target),
                        "date": today.isoformat(),
                        "commit": commit,
                        "backend_mode": backend_mode,
                        "dry_run": False,
                    }
                )

            write_artifacts(artifact_dir, summary=summary, latencies_ms=latencies, meta=meta)
            print(f"[bench] wrote artifacts: {artifact_dir}")
    finally:
        await _close_health_client()


def main() -> None:
    args = _cli()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
