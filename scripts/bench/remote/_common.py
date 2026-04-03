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
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
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
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_state(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2))
