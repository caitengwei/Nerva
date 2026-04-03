# scripts/bench/remote/service_ctl.py
from __future__ import annotations

import argparse
import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib import request
from urllib.error import URLError

import structlog

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.bench.infra.perf_compare_scenario import (
    build_cpu_mock_perf_compare_scenario,
    build_linux_gpu_perf_compare_scenario,
)
from scripts.bench.remote._common import (
    STATE_FILE,
    emit_json,
    init_logging,
    load_state,
    save_state,
)

NERVA_LOG = Path("/tmp/bench-nerva.log")
NERVA_PORT = 8080
VLLM_PORT = 8001
TRITON_HTTP_PORT = 8002
HEALTH_TIMEOUT = 180  # seconds


def _wait_http_ok(url: str, timeout_s: float = HEALTH_TIMEOUT) -> bool:
    """Poll URL until HTTP 200 or timeout. Returns True if ok."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except (URLError, OSError):
            pass
        time.sleep(2)
    return False


def _env_no_proxy() -> dict[str, str]:
    env = os.environ.copy()
    for k in ("all_proxy", "http_proxy", "https_proxy",
              "ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(k, None)
    return env


def start_nerva(scenario_cmd: list[str], *, log_path: Path = NERVA_LOG) -> dict[str, Any]:
    """Start Nerva server in background, wait for health check."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_fh:
        proc = subprocess.Popen(
            scenario_cmd,
            stdout=log_fh,
            stderr=log_fh,
            env=_env_no_proxy(),
            start_new_session=True,
        )
    endpoint = f"http://127.0.0.1:{NERVA_PORT}"
    if not _wait_http_ok(f"{endpoint}/v1/health"):
        raise RuntimeError(
            f"Nerva health check timed out after {HEALTH_TIMEOUT}s. Log: {log_path}"
        )
    return {"pid": proc.pid, "endpoint": endpoint, "log": str(log_path)}


def start_vllm(container_cmd: list[str]) -> dict[str, Any]:
    """Start vLLM docker container, wait for health check."""
    result = subprocess.run(container_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"vLLM container start failed:\n{result.stderr}")
    container_id = result.stdout.strip()
    endpoint = f"http://127.0.0.1:{VLLM_PORT}"
    ready = subprocess.run(
        [
            sys.executable,
            "scripts/bench/infra/wait_service_ready.py",
            "--kind", "vllm",
            "--url", f"{endpoint}/health",
            "--timeout-seconds", str(HEALTH_TIMEOUT),
        ],
        capture_output=True, text=True,
    )
    if ready.returncode != 0:
        raise RuntimeError(f"vLLM health check timed out. Container: {container_id}")
    return {"container_id": container_id, "endpoint": endpoint}


def start_triton(prepare_cmd: list[str], container_cmd: list[str]) -> dict[str, Any]:
    """Prepare Triton model repo and start container, wait for health check."""
    prep = subprocess.run(prepare_cmd, capture_output=True, text=True)
    if prep.returncode != 0:
        raise RuntimeError(f"Triton prepare failed:\n{prep.stderr}")
    result = subprocess.run(container_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Triton container start failed:\n{result.stderr}")
    container_id = result.stdout.strip()
    endpoint = f"http://127.0.0.1:{TRITON_HTTP_PORT}"
    ready = subprocess.run(
        [
            sys.executable,
            "scripts/bench/infra/wait_service_ready.py",
            "--kind", "triton",
            "--url", f"{endpoint}/v2/health/ready",
            "--timeout-seconds", str(HEALTH_TIMEOUT),
        ],
        capture_output=True, text=True,
    )
    if ready.returncode != 0:
        raise RuntimeError(f"Triton health check timed out. Container: {container_id}")
    return {"container_id": container_id, "endpoint": endpoint}


def stop_services(targets: list[str], state: dict[str, Any]) -> None:
    """Stop requested services using persisted state."""
    for target in targets:
        info = state.get(target, {})
        if not info:
            continue
        if target == "nerva":
            pid = info.get("pid")
            if pid:
                try:
                    pgid = os.getpgid(pid)
                    os.killpg(pgid, signal.SIGTERM)
                    time.sleep(5)
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        else:
            cid = info.get("container_id")
            if cid:
                subprocess.run(["docker", "stop", cid], capture_output=True)


def service_status(state: dict[str, Any]) -> dict[str, Any]:
    """Check health of each service recorded in state."""
    result: dict[str, Any] = {}
    health_paths = {
        "nerva": "/v1/health",
        "vllm": "/health",
        "triton": "/v2/health/ready",
    }
    for target, info in state.items():
        if not info:
            result[target] = {"status": "stopped"}
            continue
        endpoint = info.get("endpoint", "")
        url = endpoint + health_paths.get(target, "/health")
        try:
            with request.urlopen(url, timeout=3) as r:
                health = "ok" if r.status == 200 else f"http_{r.status}"
        except (URLError, OSError):
            health = "unreachable"
        result[target] = {
            **info,
            "health": health,
            "status": "running" if health == "ok" else "unhealthy",
        }
    return result


def _parse_targets(targets_str: str) -> list[str]:
    valid = {"nerva", "vllm", "triton"}
    targets = [t.strip() for t in targets_str.split(",")]
    invalid = set(targets) - valid
    if invalid:
        raise ValueError(f"Unknown targets: {invalid}. Valid: {valid}")
    return targets


def _cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control bench services on remote machine")
    sub = parser.add_subparsers(dest="command", required=True)

    start_p = sub.add_parser("start")
    start_p.add_argument("--targets", default="nerva,vllm,triton")
    start_p.add_argument("--mode", choices=["mock", "real"], default="mock")
    start_p.add_argument("--model-path", default="")
    start_p.add_argument("--triton-repo", default="/tmp/triton_repo")

    stop_p = sub.add_parser("stop")
    stop_p.add_argument("--targets", default="nerva,vllm,triton")

    sub.add_parser("status")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    init_logging()
    args = _cli(argv)
    log = structlog.get_logger()

    if args.command == "start":
        targets = _parse_targets(args.targets)
        state = load_state(STATE_FILE)

        if args.mode == "mock":
            scenario = build_cpu_mock_perf_compare_scenario(
                triton_model_repo=args.triton_repo,
            )
        else:
            if not args.model_path:
                emit_json({"error": "--model-path required for real mode", "step": "start"})
                return 1
            scenario = build_linux_gpu_perf_compare_scenario(
                model_path=args.model_path,
                triton_repo=args.triton_repo,
                container_cli="docker",
            )

        out: dict[str, Any] = {}
        for target in targets:
            log.info("starting", target=target)
            try:
                if target == "nerva":
                    info = start_nerva(scenario.nerva_server_cmd)
                elif target == "vllm":
                    if args.mode == "mock":
                        log.info("skipping vllm in mock mode")
                        continue
                    info = start_vllm(scenario.vllm_container_cmd)
                elif target == "triton":
                    info = start_triton(scenario.triton_prepare_cmd, scenario.triton_container_cmd)
                else:
                    continue
                state[target] = info
                out[target] = {**info, "status": "running", "health": "ok"}
            except RuntimeError as e:
                emit_json({"error": str(e), "step": f"start_{target}"})
                return 1

        save_state(STATE_FILE, state)
        emit_json(out)
        return 0

    elif args.command == "stop":
        targets = _parse_targets(args.targets)
        state = load_state(STATE_FILE)
        stop_services(targets, state)
        for t in targets:
            state.pop(t, None)
        save_state(STATE_FILE, state)
        emit_json({"stopped": targets})
        return 0

    elif args.command == "status":
        state = load_state(STATE_FILE)
        emit_json(service_status(state))
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
