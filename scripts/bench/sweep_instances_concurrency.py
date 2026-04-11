"""Instance x Concurrency sweep benchmark.

Sweeps pre/post instance_count (default 1-10) x concurrency levels for
Nerva and/or Triton, records results per run and aggregates to a master CSV.

Usage:
  uv run python scripts/bench/sweep_instances_concurrency.py \\
    --target nerva triton \\
    --instances 1 2 3 5 8 10 \\
    --concurrency-levels 1,4,8,16,32,64,100 \\
    --warmup-seconds 20 \\
    --sample-seconds 30

Results:
  bench-results-sweep/<target>/instances_<N>/mm_vllm/.../<C>/summary.json
  bench-results-sweep/summary.csv  (aggregated)
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from scripts.bench.infra.perf_compare_scenario import DEFAULT_TRITON_IMAGE

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INSTANCES = list(range(1, 11))
DEFAULT_CONCURRENCY = "1,4,8,16,32,64,100"
DEFAULT_WARMUP_S = 20
DEFAULT_SAMPLE_S = 30
DEFAULT_NERVA_WORKERS = 3
DEFAULT_NERVA_PORT = 8080
DEFAULT_TRITON_HTTP_PORT = 8001
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "bench-results-sweep"

CSV_FIELDS = [
    "target", "instances", "concurrency", "workload",
    "qps", "p50_ms", "p95_ms", "p99_ms",
    "error_rate", "total_requests", "error_count",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env_no_proxy() -> dict[str, str]:
    env = os.environ.copy()
    for k in ("all_proxy", "http_proxy", "https_proxy",
              "ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(k, None)
    return env


def _wait_ready(url: str, timeout_s: float = 120.0) -> bool:
    import urllib.request
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


# ---------------------------------------------------------------------------
# Nerva server lifecycle
# ---------------------------------------------------------------------------

class NervaServer:
    def __init__(self, *, instances: int, workers: int, port: int):
        self.instances = instances
        self.workers = workers
        self.port = port
        self._proc: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        env = _env_no_proxy()
        env["NERVA_PRE_POST_INSTANCES"] = str(self.instances)
        self._proc = subprocess.Popen(
            [
                "uv", "run", "uvicorn",
                "examples.mm_vllm_cpu_mock_server:app",
                "--host", "0.0.0.0",
                "--port", str(self.port),
                "--workers", str(self.workers),
            ],
            env=env,
            cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    def stop(self) -> None:
        if self._proc is None:
            return
        try:
            pgid = os.getpgid(self._proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            self._proc.wait(timeout=15)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
        finally:
            self._proc = None
        time.sleep(2)

    def health_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1/health"


# ---------------------------------------------------------------------------
# Triton server lifecycle
# ---------------------------------------------------------------------------

class TritonServer:
    def __init__(self, *, instances: int, http_port: int):
        self.instances = instances
        self.http_port = http_port
        self._name = f"triton_sweep_mi{instances}"
        self._repo = Path(f"/private/tmp/triton_sweep_repo_mi{instances}")

    def prepare_repo(self) -> None:
        subprocess.run(
            [
                "uv", "run", "python",
                "scripts/bench/infra/prepare_triton_repo.py",
                "--output", str(self._repo),
                "--cpu-mock",
                "--mock-token-latency-ms", "0.5",
                "--mock-preprocess-latency-ms", "5.0",
                "--mock-postprocess-latency-ms", "2.0",
                "--mock-latency-jitter-frac", "0.1",
                "--pre-post-instance-count", str(self.instances),
            ],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
        )

    def start(self) -> None:
        subprocess.run(["podman", "stop", self._name], capture_output=True)
        time.sleep(1)
        subprocess.run(
            [
                "podman", "run", "--rm", "-d",
                "--name", self._name,
                "-p", f"{self.http_port}:8000",
                "-p", f"{self.http_port + 1}:8001",
                "-p", f"{self.http_port + 2}:8002",
                "-v", f"{self._repo}:/models:ro",
                DEFAULT_TRITON_IMAGE,
                "tritonserver",
                "--model-repository=/models",
                "--http-thread-count=16",
            ],
            check=True,
            capture_output=True,
        )

    def stop(self) -> None:
        subprocess.run(["podman", "stop", self._name], capture_output=True)
        time.sleep(1)

    def health_url(self) -> str:
        return f"http://127.0.0.1:{self.http_port}/v2/health/ready"


# ---------------------------------------------------------------------------
# Benchmark invocation
# ---------------------------------------------------------------------------

def _run_bench(
    *,
    target: str,
    concurrency_levels: str,
    warmup_s: int,
    sample_s: int,
    output_root: Path,
    nerva_port: int,
    triton_port: int,
    deadline_ms: int,
) -> None:
    cmd = [
        "uv", "run", "python", "scripts/bench/run_bench.py",
        "--target", target,
        "--workload", "mm_vllm",
        "--concurrency-levels", concurrency_levels,
        "--warmup-seconds", str(warmup_s),
        "--sample-seconds", str(sample_s),
        "--deadline-ms", str(deadline_ms),
        "--max-tokens", "256",
        "--output-root", str(output_root),
    ]
    if target == "nerva":
        cmd += ["--nerva-url", f"http://127.0.0.1:{nerva_port}"]
    elif target == "triton":
        cmd += [
            "--triton-url", f"http://127.0.0.1:{triton_port}",
            "--triton-model", "mm_vllm",
        ]
    subprocess.run(cmd, env=_env_no_proxy(), cwd=str(REPO_ROOT), check=True)


# ---------------------------------------------------------------------------
# Result collection → master CSV
# ---------------------------------------------------------------------------

def _append_to_csv(csv_path: Path, run_root: Path, instances: int) -> None:
    rows: list[dict[str, Any]] = []
    for f in sorted(run_root.rglob("summary.json")):
        data = json.loads(f.read_text())
        data["instances"] = instances
        rows.append(data)
    if not rows:
        return
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Instance x Concurrency sweep benchmark")
    p.add_argument("--target", nargs="+", choices=["nerva", "triton"],
                   default=["nerva", "triton"])
    p.add_argument("--instances", nargs="+", type=int, default=DEFAULT_INSTANCES,
                   metavar="N", help="pre/post instance counts (default: 1-10)")
    p.add_argument("--concurrency-levels", default=DEFAULT_CONCURRENCY,
                   help="comma-separated list (default: 1,4,8,16,32,64,100)")
    p.add_argument("--warmup-seconds", type=int, default=DEFAULT_WARMUP_S)
    p.add_argument("--sample-seconds", type=int, default=DEFAULT_SAMPLE_S)
    p.add_argument("--nerva-workers", type=int, default=DEFAULT_NERVA_WORKERS)
    p.add_argument("--nerva-port", type=int, default=DEFAULT_NERVA_PORT)
    p.add_argument("--triton-http-port", type=int, default=DEFAULT_TRITON_HTTP_PORT)
    p.add_argument("--deadline-ms", type=int, default=30_000)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--skip-existing", action="store_true",
                   help="跳过已有结果的 (target, instances) 组合")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _cli()
    output_root: Path = args.output_root
    csv_path = output_root / "summary.csv"
    output_root.mkdir(parents=True, exist_ok=True)

    total = len(args.target) * len(args.instances)
    done = 0

    print(f"输出目录 : {output_root}")
    print(f"Target   : {args.target}")
    print(f"Instances: {args.instances}")
    print(f"Concurrency: {args.concurrency_levels}")
    print(f"Warmup/Sample: {args.warmup_seconds}s / {args.sample_seconds}s")
    print(f"总运行次数: {total} 个 (target, instances) 组合\n")

    for target in args.target:
        for instances in args.instances:
            done += 1
            run_root = output_root / target / f"instances_{instances}"
            tag = f"[{done}/{total}] {target} instances={instances}"

            if args.skip_existing and any(run_root.rglob("summary.json")):
                print(f"{tag}  已有结果，跳过")  # noqa: RUF001
                continue

            print(f"{tag}")

            # 实例化服务管理器
            if target == "nerva":
                server: NervaServer | TritonServer = NervaServer(
                    instances=instances,
                    workers=args.nerva_workers,
                    port=args.nerva_port,
                )
            else:
                server = TritonServer(
                    instances=instances,
                    http_port=args.triton_http_port,
                )
                print("  准备 Triton model repo ...")
                server.prepare_repo()

            print("  启动服务 ...")
            server.start()

            print("  等待就绪 ...")
            if not _wait_ready(server.health_url(), timeout_s=120):
                print("  [ERROR] 服务启动超时，跳过")  # noqa: RUF001
                server.stop()
                continue

            print(f"  运行压测 (C={args.concurrency_levels}) ...")
            try:
                _run_bench(
                    target=target,
                    concurrency_levels=args.concurrency_levels,
                    warmup_s=args.warmup_seconds,
                    sample_s=args.sample_seconds,
                    output_root=run_root,
                    nerva_port=args.nerva_port,
                    triton_port=args.triton_http_port,
                    deadline_ms=args.deadline_ms,
                )
            except subprocess.CalledProcessError as exc:
                print(f"  [ERROR] 压测失败: {exc}")
            finally:
                print("  停止服务 ...")
                server.stop()

            _append_to_csv(csv_path, run_root, instances)
            print("  完成\n")

    print(f"全部完成。汇总 CSV: {csv_path}")


if __name__ == "__main__":
    main()
