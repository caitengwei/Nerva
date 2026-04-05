# scripts/bench/remote/profiler_ctl.py
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

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.bench.remote._common import (
    PROFILER_STATE_FILE,
    emit_json,
    init_logging,
    load_state,
    save_state,
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
        cmd = [
            "py-spy",
            "record",
            "-o",
            str(output),
            "--pid",
            str(target_pid),
            "--duration",
            str(duration),
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {
            "type": "py-spy",
            "pid": proc.pid,
            "target_pid": target_pid,
            "output": str(output),
            "status": "running",
        }

    elif profiler_type == "nvidia-smi":
        output = output_dir / "gpu_monitor.csv"
        cmd = [
            "nvidia-smi",
            "dmon",
            "-s",
            "pucvmet",
            "-d",
            str(interval),
            "-f",
            str(output),
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {
            "type": "nvidia-smi",
            "pid": proc.pid,
            "target_pid": None,
            "output": str(output),
            "status": "running",
        }

    elif profiler_type == "nsys":
        if target_pid is None:
            raise ValueError("--pid required for nsys")
        output = output_dir / f"timeline-{target_pid}"
        cmd = [
            "nsys",
            "profile",
            "--trace=cuda,nvtx",
            f"--duration={duration}",
            "-o",
            str(output),
            "-p",
            str(target_pid),
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {
            "type": "nsys",
            "pid": proc.pid,
            "target_pid": target_pid,
            "output": str(output) + ".nsys-rep",
            "status": "running",
        }

    elif profiler_type == "perf-stat":
        if target_pid is None:
            raise ValueError("--pid required for perf-stat")
        output = output_dir / f"perf_stat-{target_pid}.txt"
        out_fh = open(output, "w")  # noqa: SIM115 — fd intentionally left open for subprocess
        cmd = ["perf", "stat", "-p", str(target_pid), "sleep", str(duration)]
        proc = subprocess.Popen(cmd, stdout=out_fh, stderr=out_fh)
        out_fh.close()  # parent closes its copy; child inherits fd and keeps it open
        return {
            "type": "perf-stat",
            "pid": proc.pid,
            "target_pid": target_pid,
            "output": str(output),
            "status": "running",
        }

    else:
        raise ValueError(
            f"Unknown profiler type: {profiler_type}. "
            "Valid: py-spy, nvidia-smi, nsys, perf-stat"
        )


def stop_profilers(
    state: dict[str, Any],
    *,
    all_: bool = False,
    key: str | None = None,
) -> list[str]:
    stopped = []
    keys = list(state.keys()) if all_ else ([key] if key else [])
    for k in keys:
        info = state.get(k)
        if not info:
            continue
        pid = info.get("pid")
        if pid:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGINT)
                time.sleep(1)
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGTERM)
        stopped.append(k)
    return stopped


def _cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control profilers on remote machine")
    sub = parser.add_subparsers(dest="command", required=True)

    start_p = sub.add_parser("start")
    start_p.add_argument(
        "--type",
        required=True,
        choices=["py-spy", "nvidia-smi", "nsys", "perf-stat"],
    )
    start_p.add_argument("--pid", type=int, default=None)
    start_p.add_argument("--duration", type=int, default=30)
    start_p.add_argument("--interval", type=int, default=1)
    start_p.add_argument("--output", default=str(PROFILE_DIR))

    stop_p = sub.add_parser("stop")
    stop_p.add_argument("--all", action="store_true")
    stop_p.add_argument("--key", default=None, help="stop a single profiler by key (e.g. py-spy-1234)")

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
        if args.all == bool(args.key):
            emit_json({"error": "stop requires exactly one of --all or --key <key>", "step": "stop"})
            return 1
        state = load_state(PROFILER_STATE_FILE)
        stopped = stop_profilers(state, all_=args.all, key=args.key)
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
