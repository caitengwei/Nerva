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


_NO_PROXY_OPENER = request.build_opener(request.ProxyHandler({}))


def _wait_http_ok(url: str, timeout_s: float = HEALTH_TIMEOUT) -> bool:
    """Poll URL until HTTP 200 or timeout. Uses a proxy-disabled opener to avoid
    all_proxy/http_proxy interfering with 127.0.0.1 health checks."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with _NO_PROXY_OPENER.open(url, timeout=3) as r:
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
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        raise RuntimeError(
            f"Nerva health check timed out after {HEALTH_TIMEOUT}s. "
            f"pid={proc.pid} Log: {log_path}"
        )
    return {"pid": proc.pid, "endpoint": endpoint, "log": str(log_path)}


def _inject_detach(cmd: list[str]) -> list[str]:
    """Ensure 'docker run' is detached (-d) so it returns immediately with the container ID."""
    if (
        len(cmd) >= 2
        and cmd[0] in ("docker", "podman")
        and cmd[1] == "run"
        and "-d" not in cmd
        and "--detach" not in cmd
    ):
        return [cmd[0], cmd[1], "-d", *cmd[2:]]
    return cmd


def start_vllm(container_cmd: list[str]) -> dict[str, Any]:
    """Start vLLM docker container in detached mode, wait for health check."""
    cmd = _inject_detach(container_cmd)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"vLLM container start failed:\n{result.stderr}")
    container_id = result.stdout.strip()
    if not container_id:
        raise RuntimeError("vLLM docker run returned no container ID — was it run without -d?")
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
        env=_env_no_proxy(),
    )
    if ready.returncode != 0:
        subprocess.run(["docker", "stop", container_id], capture_output=True)
        raise RuntimeError(f"vLLM health check timed out. Container: {container_id}")
    return {"container_id": container_id, "endpoint": endpoint}


TRITON_LOG = Path("/tmp/bench-triton-mock.log")


def start_triton(
    prepare_cmd: list[str],
    container_cmd: list[str],
    *,
    log_path: Path = TRITON_LOG,
) -> dict[str, Any]:
    """Prepare Triton model repo and start container (or mock server), wait for health check.

    Mock mode: container_cmd is empty; prepare_cmd is a Python process that serves Triton
    directly and must be started in the background via Popen.

    Real mode: prepare_cmd is a blocking repo-prep script; container_cmd launches a container.
    """
    endpoint = f"http://127.0.0.1:{TRITON_HTTP_PORT}"

    if not container_cmd:
        # Mock mode — prepare_cmd IS the server process; run it in the background.
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as log_fh:
            proc = subprocess.Popen(
                prepare_cmd,
                stdout=log_fh,
                stderr=log_fh,
                env=_env_no_proxy(),
                start_new_session=True,
            )
        if not _wait_http_ok(f"{endpoint}/v2/health/ready"):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            raise RuntimeError(
                f"Triton mock health check timed out after {HEALTH_TIMEOUT}s. "
                f"pid={proc.pid} Log: {log_path}"
            )
        return {"pid": proc.pid, "endpoint": endpoint, "log": str(log_path)}

    # Real mode — prepare repo (blocking), then launch container in detached mode.
    prep = subprocess.run(prepare_cmd, capture_output=True, text=True)
    if prep.returncode != 0:
        raise RuntimeError(f"Triton prepare failed:\n{prep.stderr}")
    cmd = _inject_detach(container_cmd)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Triton container start failed:\n{result.stderr}")
    container_id = result.stdout.strip()
    if not container_id:
        raise RuntimeError("Triton docker run returned no container ID — was it run without -d?")
    ready = subprocess.run(
        [
            sys.executable,
            "scripts/bench/infra/wait_service_ready.py",
            "--kind", "triton",
            "--url", f"{endpoint}/v2/health/ready",
            "--timeout-seconds", str(HEALTH_TIMEOUT),
        ],
        capture_output=True, text=True,
        env=_env_no_proxy(),
    )
    if ready.returncode != 0:
        subprocess.run(["docker", "stop", container_id], capture_output=True)
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
            else:
                # Mock mode: service was started as a background process (pid-based).
                pid = info.get("pid")
                if pid:
                    try:
                        pgid = os.getpgid(pid)
                        os.killpg(pgid, signal.SIGTERM)
                        time.sleep(2)
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


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
            with _NO_PROXY_OPENER.open(url, timeout=3) as r:
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
    seen: set[str] = set()
    targets = []
    for t in targets_str.split(","):
        t = t.strip()
        if t and t not in seen:
            targets.append(t)
            seen.add(t)
    invalid = seen - valid
    if invalid:
        raise ValueError(f"Unknown targets: {invalid}. Valid: {valid}")
    if not targets:
        raise ValueError("targets must not be empty")
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
        try:
            targets = _parse_targets(args.targets)
        except ValueError as e:
            emit_json({"error": str(e), "step": "parse_targets"})
            return 1
        state = load_state(STATE_FILE)

        try:
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
        except ValueError as e:
            emit_json({"error": str(e), "step": "build_scenario"})
            return 1

        out: dict[str, Any] = {}
        for target in targets:
            log.info("starting", target=target)
            try:
                if target == "nerva":
                    info = start_nerva(scenario.nerva_server_cmd)
                elif target == "vllm":
                    if args.mode == "mock":
                        log.info("skipping vllm in mock mode; clearing any stale state")
                        if state.pop("vllm", None) is not None:
                            save_state(STATE_FILE, state)
                        continue
                    info = start_vllm(scenario.vllm_container_cmd)
                elif target == "triton":
                    info = start_triton(scenario.triton_prepare_cmd, scenario.triton_container_cmd)
                else:
                    continue
                state[target] = info
                save_state(STATE_FILE, state)  # persist after each start so partial state survives failure
                out[target] = {**info, "status": "running", "health": "ok"}
            except RuntimeError as e:
                emit_json({"error": str(e), "step": f"start_{target}"})
                return 1


        emit_json(out)
        return 0

    elif args.command == "stop":
        try:
            targets = _parse_targets(args.targets)
        except ValueError as e:
            emit_json({"error": str(e), "step": "parse_targets"})
            return 1
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
