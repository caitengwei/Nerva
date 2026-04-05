# Remote Bench Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `scripts/bench/remote/` Python toolkit + `skills/remote-bench/SKILL.md` that lets Claude Code automate Nerva performance benchmarking on a remote Linux GPU machine via SSH.

**Architecture:** Four remote Python scripts (`_common`, `env_setup`, `service_ctl`, `profiler_ctl`, `collect_results`) handle atomic operations; the Skill orchestrates them via SSH with human confirmation checkpoints. Service state (PIDs/container IDs) persists in `/tmp/nerva-bench-state.json` so `stop`/`status` work across SSH sessions.

**Tech Stack:** Python 3.11+, structlog, subprocess, tarfile, argparse, json; existing `perf_compare_scenario.py`, `wait_service_ready.py`; pytest + unittest.mock for tests.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `scripts/bench/remote/__init__.py` | Create | Package marker |
| `scripts/bench/remote/_common.py` | Create | `emit_json()`, `init_logging()`, `gpu_info()`, state file helpers |
| `scripts/bench/remote/env_setup.py` | Create | git pull, uv sync, docker pull images, clean timing/profile dirs |
| `scripts/bench/remote/service_ctl.py` | Create | Start/stop/status Nerva + vLLM + Triton; reuses `perf_compare_scenario.py` |
| `scripts/bench/remote/profiler_ctl.py` | Create | Start/stop/list py-spy, nvidia-smi, nsys, perf-stat |
| `scripts/bench/remote/collect_results.py` | Create | Package bench-results + timing + profile into tar.gz |
| `skills/remote-bench/SKILL.md` | Create | Claude Code skill definition for `/remote-bench` |
| `tests/test_remote_common.py` | Create | Tests for `_common.py` |
| `tests/test_remote_env_setup.py` | Create | Tests for `env_setup.py` |
| `tests/test_remote_service_ctl.py` | Create | Tests for `service_ctl.py` |
| `tests/test_remote_profiler_ctl.py` | Create | Tests for `profiler_ctl.py` |
| `tests/test_remote_collect_results.py` | Create | Tests for `collect_results.py` |

---

## Task 1: Package init + `_common.py`

**Files:**
- Create: `scripts/bench/remote/__init__.py`
- Create: `scripts/bench/remote/_common.py`
- Create: `tests/test_remote_common.py`

### What `_common.py` provides

```python
# scripts/bench/remote/_common.py
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def init_logging() -> None:
    """Configure structlog to stderr, no stdout pollution."""
    import structlog
    structlog.configure(
        processors=[
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )


def emit_json(data: dict[str, Any]) -> None:
    """Print a single JSON line to stdout. This is the script's output contract."""
    print(json.dumps(data, ensure_ascii=False), flush=True)


def gpu_info() -> list[dict[str, Any]]:
    """Return list of GPUs via nvidia-smi. Returns [] if nvidia-smi unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            gpus.append({"name": parts[0], "memory_mb": int(parts[1])})
    return gpus


STATE_FILE = Path("/tmp/nerva-bench-state.json")
PROFILER_STATE_FILE = Path("/tmp/nerva-bench-profilers.json")


def load_state(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_state(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2))
```

- [ ] **Step 1: Create `scripts/bench/remote/__init__.py`** (empty file)

```bash
touch scripts/bench/remote/__init__.py
```

- [ ] **Step 2: Write failing tests for `_common.py`**

```python
# tests/test_remote_common.py
from __future__ import annotations
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from scripts.bench.remote._common import emit_json, gpu_info, load_state, save_state, STATE_FILE


def test_emit_json_outputs_valid_json(capsys):
    emit_json({"foo": "bar", "n": 42})
    out = capsys.readouterr().out
    assert json.loads(out.strip()) == {"foo": "bar", "n": 42}


def test_emit_json_goes_to_stdout_not_stderr(capsys):
    emit_json({"x": 1})
    captured = capsys.readouterr()
    assert captured.out.strip() != ""
    assert captured.err == ""


def test_gpu_info_parses_nvidia_smi_output():
    fake_output = "Tesla P100-PCIE-16GB, 16280\nTesla P100-PCIE-16GB, 16280\n"
    with patch("subprocess.check_output", return_value=fake_output):
        result = gpu_info()
    assert result == [
        {"name": "Tesla P100-PCIE-16GB", "memory_mb": 16280},
        {"name": "Tesla P100-PCIE-16GB", "memory_mb": 16280},
    ]


def test_gpu_info_returns_empty_when_nvidia_smi_missing():
    with patch("subprocess.check_output", side_effect=FileNotFoundError):
        result = gpu_info()
    assert result == []


def test_load_state_returns_empty_dict_when_file_missing(tmp_path):
    result = load_state(tmp_path / "nonexistent.json")
    assert result == {}


def test_save_and_load_state_roundtrip(tmp_path):
    p = tmp_path / "state.json"
    save_state(p, {"nerva": {"pid": 123}})
    assert load_state(p) == {"nerva": {"pid": 123}}
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_remote_common.py -v
```
Expected: ImportError or ModuleNotFoundError (file doesn't exist yet)

- [ ] **Step 4: Create `scripts/bench/remote/_common.py`**

Write the full content shown in the "What `_common.py` provides" section above.

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_remote_common.py -v
```
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add scripts/bench/remote/__init__.py scripts/bench/remote/_common.py tests/test_remote_common.py
git commit -m "feat(bench/remote): add remote package + _common utilities"
```

---

## Task 2: `env_setup.py`

**Files:**
- Create: `scripts/bench/remote/env_setup.py`
- Create: `tests/test_remote_env_setup.py`

### What `env_setup.py` does

```python
# scripts/bench/remote/env_setup.py
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.bench.remote._common import emit_json, gpu_info, init_logging

VLLM_IMAGE = "vllm/vllm-openai:latest"
TRITON_IMAGE = "nvcr.io/nvidia/tritonserver:24.05-py3"
TIMING_DIR = Path("/tmp/nerva_timing")
PROFILE_DIR = Path("/tmp/profile")


def git_pull() -> dict[str, str]:
    result = subprocess.run(
        ["git", "pull", "--ff-only"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git pull failed:\n{result.stderr}")
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    return {"git_commit": commit, "git_branch": branch}


def uv_sync() -> str:
    result = subprocess.run(
        ["uv", "sync", "--all-extras"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"uv sync failed:\n{result.stderr}")
    return "ok"


def docker_pull_images(images: list[str]) -> list[str]:
    pulled = []
    for image in images:
        result = subprocess.run(
            ["docker", "pull", image],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"docker pull {image} failed:\n{result.stderr}")
        pulled.append(image)
    return pulled


def clean_dirs() -> None:
    for d in [TIMING_DIR, PROFILE_DIR]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)


def _cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up remote bench environment")
    parser.add_argument("--pull", action="store_true", help="git pull --ff-only")
    parser.add_argument("--sync-deps", action="store_true", help="uv sync --all-extras")
    parser.add_argument("--pull-images", action="store_true", help="docker pull vllm + triton images")
    parser.add_argument("--clean-timing", action="store_true", help="rm -rf /tmp/nerva_timing /tmp/profile")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    init_logging()
    args = _cli(argv)
    import structlog
    log = structlog.get_logger()

    result: dict[str, Any] = {}

    if args.pull:
        log.info("git pull")
        try:
            result.update(git_pull())
        except RuntimeError as e:
            emit_json({"error": str(e), "step": "git_pull"})
            return 1
    else:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        result["git_commit"] = commit
        result["git_branch"] = branch

    if args.sync_deps:
        log.info("uv sync")
        try:
            result["uv_sync"] = uv_sync()
        except RuntimeError as e:
            emit_json({"error": str(e), "step": "uv_sync"})
            return 1

    if args.pull_images:
        log.info("docker pull images")
        try:
            result["docker_images"] = docker_pull_images([VLLM_IMAGE, TRITON_IMAGE])
        except RuntimeError as e:
            emit_json({"error": str(e), "step": "docker_pull"})
            return 1

    if args.clean_timing:
        log.info("clean timing + profile dirs")
        clean_dirs()
        result["clean_timing"] = "ok"

    result["gpus"] = gpu_info()
    result["python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    emit_json(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 1: Write failing tests**

```python
# tests/test_remote_env_setup.py
from __future__ import annotations
import json
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from scripts.bench.remote.env_setup import (
    git_pull, uv_sync, docker_pull_images, clean_dirs,
    TIMING_DIR, PROFILE_DIR, main,
)


def test_git_pull_returns_commit_and_branch():
    with patch("subprocess.run") as mock_run, \
         patch("subprocess.check_output", side_effect=["abc1234\n", "main\n"]):
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        result = git_pull()
    assert result == {"git_commit": "abc1234", "git_branch": "main"}


def test_git_pull_raises_on_failure():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="merge conflict")
        with pytest.raises(RuntimeError, match="git pull failed"):
            git_pull()


def test_uv_sync_returns_ok():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        assert uv_sync() == "ok"


def test_uv_sync_raises_on_failure():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="error")
        with pytest.raises(RuntimeError, match="uv sync failed"):
            uv_sync()


def test_docker_pull_images_returns_pulled_list():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr="")
        result = docker_pull_images(["img1:latest", "img2:latest"])
    assert result == ["img1:latest", "img2:latest"]


def test_docker_pull_raises_on_failure():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="not found")
        with pytest.raises(RuntimeError, match="docker pull img1:latest failed"):
            docker_pull_images(["img1:latest"])


def test_clean_dirs_creates_empty_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.bench.remote.env_setup.TIMING_DIR", tmp_path / "timing")
    monkeypatch.setattr("scripts.bench.remote.env_setup.PROFILE_DIR", tmp_path / "profile")
    (tmp_path / "timing").mkdir()
    (tmp_path / "timing" / "old.log").write_text("old data")
    clean_dirs()
    assert (tmp_path / "timing").exists()
    assert not list((tmp_path / "timing").iterdir())
    assert (tmp_path / "profile").exists()


def test_main_outputs_valid_json(capsys):
    with patch("subprocess.check_output", side_effect=["abc1234\n", "main\n"]), \
         patch("scripts.bench.remote.env_setup.gpu_info", return_value=[]):
        rc = main([])
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out.strip())
    assert "git_commit" in data
    assert "gpus" in data
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/test_remote_env_setup.py -v
```
Expected: ImportError

- [ ] **Step 3: Create `scripts/bench/remote/env_setup.py`**

Write the full content shown above.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_remote_env_setup.py -v
```
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/bench/remote/env_setup.py tests/test_remote_env_setup.py
git commit -m "feat(bench/remote): add env_setup.py for remote environment preparation"
```

---

## Task 3: `service_ctl.py`

**Files:**
- Create: `scripts/bench/remote/service_ctl.py`
- Create: `tests/test_remote_service_ctl.py`

Note: `wait_service_ready.py` only accepts `kind` in `{"vllm", "triton"}`. For Nerva we use an inline `_wait_http_ok()` helper. For vLLM/Triton we subprocess-call `wait_service_ready.py` CLI.

### State file schema (written to `/tmp/nerva-bench-state.json`)

```json
{
  "nerva": {"pid": 12345, "endpoint": "http://127.0.0.1:8080", "log": "/tmp/bench-nerva.log"},
  "vllm":  {"container_id": "abc123", "endpoint": "http://127.0.0.1:8001"},
  "triton": {"container_id": "def456", "endpoint": "http://127.0.0.1:8002"}
}
```

### What `service_ctl.py` does

```python
# scripts/bench/remote/service_ctl.py
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib import request
from urllib.error import URLError

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import structlog

from scripts.bench.remote._common import (
    STATE_FILE, emit_json, init_logging, load_state, save_state,
)
from scripts.bench.infra.perf_compare_scenario import (
    build_cpu_mock_perf_compare_scenario,
    build_linux_gpu_perf_compare_scenario,
)

NERVA_LOG = Path("/tmp/bench-nerva.log")
NERVA_PORT = 8080
VLLM_PORT = 8001
TRITON_HTTP_PORT = 8002
HEALTH_TIMEOUT = 180  # seconds


def _wait_http_ok(url: str, timeout_s: float = HEALTH_TIMEOUT) -> bool:
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
    """Start Nerva server, return state dict."""
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
        raise RuntimeError(f"Nerva health check timed out after {HEALTH_TIMEOUT}s. Log: {log_path}")
    return {"pid": proc.pid, "endpoint": endpoint, "log": str(log_path)}


def start_vllm(container_cmd: list[str]) -> dict[str, Any]:
    """Start vLLM docker container, return state dict."""
    result = subprocess.run(container_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"vLLM container start failed:\n{result.stderr}")
    container_id = result.stdout.strip()
    endpoint = f"http://127.0.0.1:{VLLM_PORT}"
    # Use wait_service_ready CLI for vLLM
    ready = subprocess.run(
        [sys.executable, "scripts/bench/infra/wait_service_ready.py",
         "--kind", "vllm", "--url", f"{endpoint}/health", "--timeout-seconds", str(HEALTH_TIMEOUT)],
        capture_output=True, text=True,
    )
    if ready.returncode != 0:
        raise RuntimeError(f"vLLM health check timed out. Container: {container_id}")
    return {"container_id": container_id, "endpoint": endpoint}


def start_triton(prepare_cmd: list[str], container_cmd: list[str]) -> dict[str, Any]:
    """Prepare Triton repo and start container, return state dict."""
    prep = subprocess.run(prepare_cmd, capture_output=True, text=True)
    if prep.returncode != 0:
        raise RuntimeError(f"Triton prepare failed:\n{prep.stderr}")
    result = subprocess.run(container_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Triton container start failed:\n{result.stderr}")
    container_id = result.stdout.strip()
    endpoint = f"http://127.0.0.1:{TRITON_HTTP_PORT}"
    ready = subprocess.run(
        [sys.executable, "scripts/bench/infra/wait_service_ready.py",
         "--kind", "triton", "--url", f"{endpoint}/v2/health/ready", "--timeout-seconds", str(HEALTH_TIMEOUT)],
        capture_output=True, text=True,
    )
    if ready.returncode != 0:
        raise RuntimeError(f"Triton health check timed out. Container: {container_id}")
    return {"container_id": container_id, "endpoint": endpoint}


def stop_services(targets: list[str], state: dict[str, Any]) -> None:
    """Stop requested services using state."""
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
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                except ProcessLookupError:
                    pass
        else:
            cid = info.get("container_id")
            if cid:
                subprocess.run(["docker", "stop", cid], capture_output=True)


def service_status(state: dict[str, Any]) -> dict[str, Any]:
    """Check health of each service in state."""
    result: dict[str, Any] = {}
    for target, info in state.items():
        if not info:
            result[target] = {"status": "stopped"}
            continue
        endpoint = info.get("endpoint", "")
        health_paths = {"nerva": "/v1/health", "vllm": "/health", "triton": "/v2/health/ready"}
        url = endpoint + health_paths.get(target, "/health")
        try:
            with request.urlopen(url, timeout=3) as r:
                health = "ok" if r.status == 200 else f"http_{r.status}"
        except (URLError, OSError):
            health = "unreachable"
        entry = {**info, "health": health,
                 "status": "running" if health == "ok" else "unhealthy"}
        result[target] = entry
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
```

- [ ] **Step 1: Write failing tests**

```python
# tests/test_remote_service_ctl.py
from __future__ import annotations
import json
import signal
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import pytest

from scripts.bench.remote.service_ctl import (
    _parse_targets, _wait_http_ok, stop_services, service_status, main,
)


def test_parse_targets_valid():
    assert _parse_targets("nerva,vllm") == ["nerva", "vllm"]
    assert _parse_targets("triton") == ["triton"]


def test_parse_targets_invalid_raises():
    with pytest.raises(ValueError, match="Unknown targets"):
        _parse_targets("nerva,badtarget")


def test_wait_http_ok_returns_true_on_200():
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.status = 200
    with patch("urllib.request.urlopen", return_value=mock_resp):
        assert _wait_http_ok("http://127.0.0.1:8080/health", timeout_s=5)


def test_wait_http_ok_returns_false_on_timeout():
    from urllib.error import URLError
    with patch("urllib.request.urlopen", side_effect=URLError("refused")), \
         patch("time.sleep"), \
         patch("time.monotonic", side_effect=[0, 0.5, 1.0, 6.0]):
        assert not _wait_http_ok("http://127.0.0.1:8080/health", timeout_s=5)


def test_stop_services_kills_nerva_process():
    state = {"nerva": {"pid": 9999, "endpoint": "http://127.0.0.1:8080"}}
    with patch("os.getpgid", return_value=9999), \
         patch("os.killpg") as mock_kill, \
         patch("time.sleep"):
        stop_services(["nerva"], state)
    mock_kill.assert_any_call(9999, signal.SIGTERM)


def test_stop_services_docker_stops_container():
    state = {"vllm": {"container_id": "abc123", "endpoint": "http://127.0.0.1:8001"}}
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        stop_services(["vllm"], state)
    mock_run.assert_called_once_with(["docker", "stop", "abc123"], capture_output=True)


def test_service_status_returns_unreachable_for_dead_service():
    from urllib.error import URLError
    state = {"nerva": {"pid": 9999, "endpoint": "http://127.0.0.1:8080"}}
    with patch("urllib.request.urlopen", side_effect=URLError("refused")):
        result = service_status(state)
    assert result["nerva"]["health"] == "unreachable"
    assert result["nerva"]["status"] == "unhealthy"


def test_main_status_emits_json_for_empty_state(capsys, tmp_path):
    state_file = tmp_path / "state.json"
    with patch("scripts.bench.remote.service_ctl.STATE_FILE", state_file):
        rc = main(["status"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert isinstance(out, dict)


def test_main_stop_clears_state(capsys, tmp_path):
    import json as js
    state_file = tmp_path / "state.json"
    state_file.write_text(js.dumps({"nerva": {"pid": 1234, "endpoint": "http://127.0.0.1:8080"}}))
    with patch("scripts.bench.remote.service_ctl.STATE_FILE", state_file), \
         patch("os.getpgid", side_effect=ProcessLookupError), \
         patch("time.sleep"):
        rc = main(["stop", "--targets", "nerva"])
    assert rc == 0
    remaining = js.loads(state_file.read_text())
    assert "nerva" not in remaining
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/test_remote_service_ctl.py -v
```
Expected: ImportError

- [ ] **Step 3: Create `scripts/bench/remote/service_ctl.py`**

Write the full content shown above.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_remote_service_ctl.py -v
```
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/bench/remote/service_ctl.py tests/test_remote_service_ctl.py
git commit -m "feat(bench/remote): add service_ctl.py — start/stop/status Nerva+vLLM+Triton"
```

---

## Task 4: `profiler_ctl.py`

**Files:**
- Create: `scripts/bench/remote/profiler_ctl.py`
- Create: `tests/test_remote_profiler_ctl.py`

### Profiler state schema (`/tmp/nerva-bench-profilers.json`)

```json
{
  "py-spy-12345": {"type": "py-spy", "pid": 23456, "target_pid": 12345, "output": "/tmp/profile/flamegraph.svg", "status": "running"},
  "nvidia-smi-0": {"type": "nvidia-smi", "pid": 23457, "output": "/tmp/profile/gpu_monitor.csv", "status": "running"}
}
```

### What `profiler_ctl.py` does

```python
# scripts/bench/remote/profiler_ctl.py
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from scripts.bench.remote._common import (
    PROFILER_STATE_FILE, emit_json, init_logging, load_state, save_state,
)

PROFILE_DIR = Path("/tmp/profile")


def _make_key(profiler_type: str, target_pid: int | None) -> str:
    return f"{profiler_type}-{target_pid or 0}"


def start_profiler(
    *,
    profiler_type: str,
    target_pid: int | None = None,
    duration: int = 30,
    interval: int = 1,
    output_dir: Path = PROFILE_DIR,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    if profiler_type == "py-spy":
        if target_pid is None:
            raise ValueError("--pid required for py-spy")
        output = output_dir / f"flamegraph-{target_pid}.svg"
        cmd = ["py-spy", "record", "-o", str(output),
               "--pid", str(target_pid), "--duration", str(duration)]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"type": "py-spy", "pid": proc.pid, "target_pid": target_pid,
                "output": str(output), "status": "running"}

    elif profiler_type == "nvidia-smi":
        output = output_dir / "gpu_monitor.csv"
        cmd = ["nvidia-smi", "dmon", "-s", "pucvmet", "-d", str(interval), "-f", str(output)]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"type": "nvidia-smi", "pid": proc.pid, "target_pid": None,
                "output": str(output), "status": "running"}

    elif profiler_type == "nsys":
        if target_pid is None:
            raise ValueError("--pid required for nsys")
        output = output_dir / f"timeline-{target_pid}"
        cmd = ["nsys", "profile", "--trace=cuda,nvtx",
               f"--duration={duration}", f"-o", str(output), "-p", str(target_pid)]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"type": "nsys", "pid": proc.pid, "target_pid": target_pid,
                "output": str(output) + ".nsys-rep", "status": "running"}

    elif profiler_type == "perf-stat":
        if target_pid is None:
            raise ValueError("--pid required for perf-stat")
        output = output_dir / f"perf_stat-{target_pid}.txt"
        with open(output, "w") as fh:
            cmd = ["perf", "stat", "-p", str(target_pid), "sleep", str(duration)]
            proc = subprocess.Popen(cmd, stdout=fh, stderr=fh)
        return {"type": "perf-stat", "pid": proc.pid, "target_pid": target_pid,
                "output": str(output), "status": "running"}

    else:
        raise ValueError(f"Unknown profiler type: {profiler_type}. "
                         f"Valid: py-spy, nvidia-smi, nsys, perf-stat")


def stop_profilers(state: dict[str, Any], *, all_: bool = False, key: str | None = None) -> list[str]:
    stopped = []
    keys = list(state.keys()) if all_ else ([key] if key else [])
    for k in keys:
        info = state.get(k)
        if not info:
            continue
        pid = info.get("pid")
        if pid:
            try:
                os.kill(pid, signal.SIGINT)
                time.sleep(1)
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            except ProcessLookupError:
                pass
        stopped.append(k)
    return stopped


def _cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control profilers on remote machine")
    sub = parser.add_subparsers(dest="command", required=True)

    start_p = sub.add_parser("start")
    start_p.add_argument("--type", required=True,
                          choices=["py-spy", "nvidia-smi", "nsys", "perf-stat"])
    start_p.add_argument("--pid", type=int, default=None)
    start_p.add_argument("--duration", type=int, default=30)
    start_p.add_argument("--interval", type=int, default=1)
    start_p.add_argument("--output", default=str(PROFILE_DIR))

    stop_p = sub.add_parser("stop")
    stop_p.add_argument("--all", action="store_true")

    sub.add_parser("list")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    init_logging()
    args = _cli(argv)

    if args.command == "start":
        state = load_state(PROFILER_STATE_FILE)
        try:
            info = start_profiler(
                profiler_type=args.type,
                target_pid=args.pid,
                duration=args.duration,
                interval=args.interval,
                output_dir=Path(args.output),
            )
        except (ValueError, FileNotFoundError) as e:
            emit_json({"error": str(e), "step": "start_profiler"})
            return 1
        key = _make_key(args.type, args.pid)
        state[key] = info
        save_state(PROFILER_STATE_FILE, state)
        emit_json({"profilers": [info]})
        return 0

    elif args.command == "stop":
        state = load_state(PROFILER_STATE_FILE)
        stopped = stop_profilers(state, all_=args.all)
        for k in stopped:
            state.pop(k, None)
        save_state(PROFILER_STATE_FILE, state)
        emit_json({"stopped": stopped})
        return 0

    elif args.command == "list":
        state = load_state(PROFILER_STATE_FILE)
        emit_json({"profilers": list(state.values())})
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 1: Write failing tests**

```python
# tests/test_remote_profiler_ctl.py
from __future__ import annotations
import json
import signal
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from scripts.bench.remote.profiler_ctl import (
    start_profiler, stop_profilers, _make_key, main,
)


def test_make_key_with_pid():
    assert _make_key("py-spy", 1234) == "py-spy-1234"


def test_make_key_without_pid():
    assert _make_key("nvidia-smi", None) == "nvidia-smi-0"


def test_start_profiler_pyspy_requires_pid(tmp_path):
    with pytest.raises(ValueError, match="--pid required for py-spy"):
        start_profiler(profiler_type="py-spy", target_pid=None, output_dir=tmp_path)


def test_start_profiler_unknown_type_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown profiler type"):
        start_profiler(profiler_type="badtype", output_dir=tmp_path)


def test_start_profiler_nvidia_smi_returns_info(tmp_path):
    mock_proc = MagicMock()
    mock_proc.pid = 5678
    with patch("subprocess.Popen", return_value=mock_proc):
        info = start_profiler(profiler_type="nvidia-smi", output_dir=tmp_path)
    assert info["type"] == "nvidia-smi"
    assert info["pid"] == 5678
    assert info["status"] == "running"
    assert "gpu_monitor.csv" in info["output"]


def test_start_profiler_pyspy_returns_info(tmp_path):
    mock_proc = MagicMock()
    mock_proc.pid = 7890
    with patch("subprocess.Popen", return_value=mock_proc):
        info = start_profiler(profiler_type="py-spy", target_pid=1234, output_dir=tmp_path)
    assert info["type"] == "py-spy"
    assert info["target_pid"] == 1234
    assert "flamegraph" in info["output"]


def test_stop_profilers_sends_sigint():
    state = {"py-spy-1234": {"type": "py-spy", "pid": 5678, "target_pid": 1234}}
    with patch("os.kill") as mock_kill, patch("time.sleep"):
        stopped = stop_profilers(state, all_=True)
    mock_kill.assert_any_call(5678, signal.SIGINT)
    assert stopped == ["py-spy-1234"]


def test_stop_profilers_handles_dead_process():
    state = {"nvidia-smi-0": {"type": "nvidia-smi", "pid": 9999}}
    with patch("os.kill", side_effect=ProcessLookupError), patch("time.sleep"):
        stopped = stop_profilers(state, all_=True)
    assert stopped == ["nvidia-smi-0"]


def test_main_list_emits_empty_profilers(capsys, tmp_path):
    with patch("scripts.bench.remote.profiler_ctl.PROFILER_STATE_FILE", tmp_path / "p.json"):
        rc = main(["list"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out == {"profilers": []}
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/test_remote_profiler_ctl.py -v
```
Expected: ImportError

- [ ] **Step 3: Create `scripts/bench/remote/profiler_ctl.py`**

Write the full content shown above.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_remote_profiler_ctl.py -v
```
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/bench/remote/profiler_ctl.py tests/test_remote_profiler_ctl.py
git commit -m "feat(bench/remote): add profiler_ctl.py — py-spy/nvidia-smi/nsys/perf-stat management"
```

---

## Task 5: `collect_results.py`

**Files:**
- Create: `scripts/bench/remote/collect_results.py`
- Create: `tests/test_remote_collect_results.py`

### What `collect_results.py` does

```python
# scripts/bench/remote/collect_results.py
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any

from scripts.bench.remote._common import emit_json, init_logging


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def collect_archive(
    *,
    output_root: Path,
    timing_dir: Path,
    profile_dir: Path,
    archive_path: Path,
) -> dict[str, Any]:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, int] = {}

    with tarfile.open(archive_path, "w:gz") as tar:
        if output_root.exists():
            tar.add(output_root, arcname="bench-results")
            manifest["bench_results"] = sum(1 for _ in output_root.rglob("*") if _.is_file())

        if timing_dir.exists():
            tar.add(timing_dir, arcname="timing")
            manifest["timing_logs"] = sum(1 for _ in timing_dir.rglob("*") if _.is_file())

        if profile_dir.exists():
            tar.add(profile_dir, arcname="profiling")
            manifest["profiling"] = sum(1 for _ in profile_dir.rglob("*") if _.is_file())

    size_mb = round(archive_path.stat().st_size / (1024 * 1024), 2)
    return {
        "archive_path": str(archive_path),
        "size_mb": size_mb,
        "manifest": manifest,
        "git_commit": _git_commit(),
    }


def _cli(argv: list[str] | None = None) -> argparse.Namespace:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description="Package bench results into archive")
    parser.add_argument("--output-root", default="bench-results")
    parser.add_argument("--timing-dir", default="/tmp/nerva_timing")
    parser.add_argument("--profile-dir", default="/tmp/profile")
    parser.add_argument("--archive", default=f"/tmp/bench-archive-{ts}.tar.gz")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    init_logging()
    args = _cli(argv)
    try:
        result = collect_archive(
            output_root=Path(args.output_root),
            timing_dir=Path(args.timing_dir),
            profile_dir=Path(args.profile_dir),
            archive_path=Path(args.archive),
        )
    except Exception as e:
        emit_json({"error": str(e), "step": "collect_archive"})
        return 1
    emit_json(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 1: Write failing tests**

```python
# tests/test_remote_collect_results.py
from __future__ import annotations
import json
import tarfile
from pathlib import Path
from unittest.mock import patch
import pytest

from scripts.bench.remote.collect_results import collect_archive, main


def test_collect_archive_creates_tarball(tmp_path):
    # Create fake bench-results
    bench_dir = tmp_path / "bench-results"
    bench_dir.mkdir()
    (bench_dir / "summary.json").write_text('{"qps": 100}')
    timing_dir = tmp_path / "timing"
    timing_dir.mkdir()
    (timing_dir / "nerva.log").write_text("timing data")
    profile_dir = tmp_path / "profile"  # empty / non-existent

    archive = tmp_path / "archive.tar.gz"
    with patch("scripts.bench.remote.collect_results._git_commit", return_value="abc1234"):
        result = collect_archive(
            output_root=bench_dir,
            timing_dir=timing_dir,
            profile_dir=profile_dir,
            archive_path=archive,
        )

    assert archive.exists()
    assert result["manifest"]["bench_results"] == 1
    assert result["manifest"]["timing_logs"] == 1
    assert result["git_commit"] == "abc1234"
    assert result["size_mb"] > 0


def test_collect_archive_skips_missing_dirs(tmp_path):
    archive = tmp_path / "archive.tar.gz"
    with patch("scripts.bench.remote.collect_results._git_commit", return_value="abc1234"):
        result = collect_archive(
            output_root=tmp_path / "missing-bench",
            timing_dir=tmp_path / "missing-timing",
            profile_dir=tmp_path / "missing-profile",
            archive_path=archive,
        )
    assert archive.exists()
    assert result["manifest"] == {}


def test_collect_archive_contents_are_readable(tmp_path):
    bench_dir = tmp_path / "bench-results"
    bench_dir.mkdir()
    (bench_dir / "summary.json").write_text('{"qps": 42}')
    archive = tmp_path / "archive.tar.gz"
    with patch("scripts.bench.remote.collect_results._git_commit", return_value="x"):
        collect_archive(
            output_root=bench_dir,
            timing_dir=tmp_path / "no-timing",
            profile_dir=tmp_path / "no-profile",
            archive_path=archive,
        )
    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()
    assert any("summary.json" in n for n in names)


def test_main_outputs_valid_json(capsys, tmp_path):
    bench_dir = tmp_path / "bench-results"
    bench_dir.mkdir()
    archive = tmp_path / "out.tar.gz"
    with patch("scripts.bench.remote.collect_results._git_commit", return_value="abc"):
        rc = main([
            "--output-root", str(bench_dir),
            "--timing-dir", str(tmp_path / "no-t"),
            "--profile-dir", str(tmp_path / "no-p"),
            "--archive", str(archive),
        ])
    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert "archive_path" in out
    assert "manifest" in out
```

- [ ] **Step 2: Run to verify they fail**

```bash
uv run pytest tests/test_remote_collect_results.py -v
```
Expected: ImportError

- [ ] **Step 3: Create `scripts/bench/remote/collect_results.py`**

Write the full content shown above.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_remote_collect_results.py -v
```
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/bench/remote/collect_results.py tests/test_remote_collect_results.py
git commit -m "feat(bench/remote): add collect_results.py — package bench+timing+profiling artifacts"
```

---

## Task 6: Full test suite run

- [ ] **Step 1: Run all remote tests together**

```bash
uv run pytest tests/test_remote_common.py tests/test_remote_env_setup.py \
    tests/test_remote_service_ctl.py tests/test_remote_profiler_ctl.py \
    tests/test_remote_collect_results.py -v
```
Expected: all passed, 0 failures

- [ ] **Step 2: Run ruff lint**

```bash
uv run ruff check scripts/bench/remote/ tests/test_remote_*.py
```
Expected: no errors

- [ ] **Step 3: Commit any lint fixes**

```bash
git add -p  # stage only lint fixes
git commit -m "fix(bench/remote): lint fixes"
```

---

## Task 7: `skills/remote-bench/SKILL.md`

**Files:**
- Create: `skills/remote-bench/SKILL.md`

No unit tests for the Skill file — it is verified by running `/remote-bench status` after deployment.

- [ ] **Step 1: Create `skills/remote-bench/SKILL.md`**

```markdown
---
name: remote-bench
description: 在远程 Linux 开发机上执行 Nerva 性能测试。自动化环境准备、服务管理、压测执行、profiling 采集、结果收集和报告生成。支持 /remote-bench full|profile|report|status|stop。
---

# Remote Bench Skill

## 远程主机配置

```
REMOTE=tengwei.ctw@11.160.41.175
REMOTE_DIR=/home/tengwei.ctw/Nerva
```

## 调用格式

```
/remote-bench full         # Phase 0-6 全量：环境准备 → 服务启动 → 压测 → 收集 → 报告
/remote-bench profile      # 仅 profiling（假设服务已启动）
/remote-bench report       # 仅拉结果并生成报告（Phase 4-5）
/remote-bench status       # 检查远程服务状态
/remote-bench stop         # 停止远程服务并清理
```

## 阶段定义

### Phase 0 — 连接验证

```bash
ssh -o ConnectTimeout=5 tengwei.ctw@11.160.41.175 "echo ok"
```

失败则停止，提示用户检查网络 / SSH key。

### Phase 1 — 环境准备

```bash
ssh tengwei.ctw@11.160.41.175 "cd /home/tengwei.ctw/Nerva && uv run python scripts/bench/remote/env_setup.py --pull --sync-deps --pull-images --clean-timing"
```

解析 stdout 最后一行 JSON，展示：commit SHA、GPU 信息、镜像状态。
git pull 返回 error → 暂停，告知用户有未提交修改。

### Phase 2 — 启动服务

**Mock 模式**（默认，测框架开销）:
```bash
ssh tengwei.ctw@11.160.41.175 "cd /home/tengwei.ctw/Nerva && uv run python scripts/bench/remote/service_ctl.py start --targets nerva,triton --mode mock"
```

**Real 模式**（真实模型端到端）:
```bash
ssh tengwei.ctw@11.160.41.175 "cd /home/tengwei.ctw/Nerva && uv run python scripts/bench/remote/service_ctl.py start --targets nerva,vllm,triton --mode real --model-path <MODEL_PATH>"
```

解析 stdout JSON，展示各服务 endpoint 和健康状态。

**[确认点]** 展示：「服务已就绪 — 目标: [targets]，并发级别: [levels]。确认开始压测？」
等待用户明确确认后再继续。

### Phase 3 — C=1 校验轮

```bash
ssh tengwei.ctw@11.160.41.175 "cd /home/tengwei.ctw/Nerva && env -u all_proxy -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY NERVA_TIMING_LOG_DIR=/tmp/nerva_timing uv run python scripts/bench/run_bench.py --target nerva --target triton --concurrency-levels 1 --warmup-seconds 10 --sample-seconds 30"
```

（real 模式时加 `--target vllm`）

压测完成后，读取校验结果：
```bash
ssh tengwei.ctw@11.160.41.175 "cat /home/tengwei.ctw/Nerva/bench-results/mm_vllm/$(date +%Y-%m-%d)/*/nerva/1/summary.json"
```

检查条件：
- `error_rate == 0`（否则停止，排查代理/连接问题）
- `p50_ms < 5000`（否则警告，可能服务未就绪或配置有误）

校验通过后继续 Phase 3.5（完整矩阵）。

### Phase 3.5 — 完整压测矩阵

**长时间任务使用 nohup 防断连：**
```bash
ssh tengwei.ctw@11.160.41.175 "cd /home/tengwei.ctw/Nerva && nohup env -u all_proxy -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY NERVA_TIMING_LOG_DIR=/tmp/nerva_timing uv run python scripts/bench/run_bench.py --target nerva --target triton --concurrency-levels 1,32,128,512 --warmup-seconds 60 --sample-seconds 300 > /tmp/bench-run.log 2>&1 & echo $!"
```

轮询进度（每 30s）：
```bash
ssh tengwei.ctw@11.160.41.175 "tail -5 /tmp/bench-run.log"
```

### Phase 3.5 — 按需 Profiling（压测运行中）

用户请求「火焰图」时：
```bash
NERVA_PID=$(ssh tengwei.ctw@11.160.41.175 "pgrep -f 'nerva.server' | head -1")
ssh tengwei.ctw@11.160.41.175 "cd /home/tengwei.ctw/Nerva && uv run python scripts/bench/remote/profiler_ctl.py start --type py-spy --pid $NERVA_PID --duration 30"
```

用户请求「GPU 利用率」时：
```bash
ssh tengwei.ctw@11.160.41.175 "cd /home/tengwei.ctw/Nerva && uv run python scripts/bench/remote/profiler_ctl.py start --type nvidia-smi --interval 1"
```

用户请求「CUDA timeline」时：
```bash
NERVA_PID=$(ssh tengwei.ctw@11.160.41.175 "pgrep -f 'nerva.server' | head -1")
ssh tengwei.ctw@11.160.41.175 "cd /home/tengwei.ctw/Nerva && uv run python scripts/bench/remote/profiler_ctl.py start --type nsys --pid $NERVA_PID --duration 30"
```

用户请求「系统 perf」时：
```bash
NERVA_PID=$(ssh tengwei.ctw@11.160.41.175 "pgrep -f 'nerva.server' | head -1")
ssh tengwei.ctw@11.160.41.175 "cd /home/tengwei.ctw/Nerva && uv run python scripts/bench/remote/profiler_ctl.py start --type perf-stat --pid $NERVA_PID --duration 30"
```

压测结束后自动停止所有 profiler：
```bash
ssh tengwei.ctw@11.160.41.175 "cd /home/tengwei.ctw/Nerva && uv run python scripts/bench/remote/profiler_ctl.py stop --all"
```

### Phase 4 — 收集结果

```bash
ssh tengwei.ctw@11.160.41.175 "cd /home/tengwei.ctw/Nerva && uv run python scripts/bench/remote/collect_results.py --archive /tmp/bench-archive-$(date +%Y%m%d-%H%M%S).tar.gz"
```

读取 stdout JSON，获取 `archive_path`，然后 scp：
```bash
mkdir -p bench-results/remote
scp tengwei.ctw@11.160.41.175:<archive_path> bench-results/remote/
```

本地解压：
```bash
tar -xzf bench-results/remote/bench-archive-*.tar.gz -C bench-results/remote/
```

### Phase 5 — 生成报告

获取 commit SHA（从 Phase 1 env_setup 的 JSON 输出），然后：

```bash
uv run python scripts/bench/report_bench.py \
    --input-root bench-results/remote/bench-results/ \
    --output bench-results/remote/report.md
```

```bash
uv run python scripts/bench/analyze_timing_log.py \
    --log-dir bench-results/remote/timing/
```

展示报告摘要（QPS 对比表、p50/p95/p99）。如有 profiling 产物，列出文件路径。

### Phase 6 — 清理

```bash
ssh tengwei.ctw@11.160.41.175 "cd /home/tengwei.ctw/Nerva && uv run python scripts/bench/remote/service_ctl.py stop"
```

询问用户是否清理远程 `/tmp/bench-*.tar.gz` 和 `/tmp/bench-run.log`：
```bash
ssh tengwei.ctw@11.160.41.175 "rm -f /tmp/bench-*.tar.gz /tmp/bench-run.log"
```

## 异常处理原则

任何 SSH 命令返回非零 exit code → 停止，展示 stderr，等用户决定。

## Troubleshooting

| 症状 | 原因 | 处理 |
|------|------|------|
| error_rate=1.0, latency ~17ms | all_proxy 污染 httpx | 已在压测命令中 `env -u ...` 清除；若仍出现检查 `/etc/environment` |
| p50 远高于预期 | GPU throttling / 显存不足 | `ssh remote "nvidia-smi"` 检查温度和显存 |
| vLLM 启动失败 (P100) | P100 不支持 FP16 Tensor Core | service_ctl.py real 模式加 `--dtype float32` |
| SSH 命令超时 | 长时间压测断连 | Phase 3.5 已用 nohup + 轮询；重连后 `tail /tmp/bench-run.log` 查进度 |
| Triton 端口不可达 | Docker 网络配置 | Linux 用 `--network host`；macOS 用 `-p` |
| 首轮 C=1 校验失败 | 服务未就绪 / 端口冲突 | `service_ctl.py status` 检查，stop + restart |
```

- [ ] **Step 2: Create `skills/` directory and write SKILL.md**

```bash
mkdir -p skills/remote-bench
```

Then write the full SKILL.md content above to `skills/remote-bench/SKILL.md`.

- [ ] **Step 3: Verify skill is loadable**

```bash
# Check the file exists and has valid frontmatter
head -5 skills/remote-bench/SKILL.md
```
Expected output:
```
---
name: remote-bench
description: ...
---
```

- [ ] **Step 4: Commit**

```bash
git add skills/remote-bench/SKILL.md
git commit -m "feat: add remote-bench Claude Code skill for automated remote perf testing"
```

---

## Task 8: Integration smoke test (local)

This task verifies each script's CLI works with `--help` and that the JSON output format is parseable. No SSH required.

- [ ] **Step 1: Verify all scripts run with `--help`**

```bash
uv run python scripts/bench/remote/env_setup.py --help
uv run python scripts/bench/remote/service_ctl.py --help
uv run python scripts/bench/remote/profiler_ctl.py --help
uv run python scripts/bench/remote/collect_results.py --help
```
Expected: usage lines printed, exit 0.

- [ ] **Step 2: Verify `env_setup.py` JSON output (no-op mode)**

```bash
uv run python scripts/bench/remote/env_setup.py 2>/dev/null | python -m json.tool
```
Expected: valid JSON with `git_commit`, `git_branch`, `gpus`, `python_version`.

- [ ] **Step 3: Verify `service_ctl.py status` JSON output (empty state)**

```bash
uv run python scripts/bench/remote/service_ctl.py status 2>/dev/null | python -m json.tool
```
Expected: `{}` (empty dict, state file doesn't exist).

- [ ] **Step 4: Verify `profiler_ctl.py list` JSON output (empty state)**

```bash
uv run python scripts/bench/remote/profiler_ctl.py list 2>/dev/null | python -m json.tool
```
Expected: `{"profilers": []}`.

- [ ] **Step 5: Verify `collect_results.py` JSON output**

```bash
uv run python scripts/bench/remote/collect_results.py \
    --output-root /tmp/nonexistent-bench \
    --timing-dir /tmp/nonexistent-timing \
    --profile-dir /tmp/nonexistent-profile \
    --archive /tmp/smoke-test-archive.tar.gz 2>/dev/null | python -m json.tool
rm -f /tmp/smoke-test-archive.tar.gz
```
Expected: valid JSON with `archive_path`, `manifest`, `git_commit`.

- [ ] **Step 6: Run full test suite one final time**

```bash
uv run pytest tests/test_remote_common.py tests/test_remote_env_setup.py \
    tests/test_remote_service_ctl.py tests/test_remote_profiler_ctl.py \
    tests/test_remote_collect_results.py -v
```
Expected: all passed.

- [ ] **Step 7: Final commit**

```bash
git add .
git commit -m "feat(bench/remote): complete remote bench toolkit — 5 scripts + skill + tests"
```
