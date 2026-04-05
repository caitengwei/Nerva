# scripts/bench/remote/env_setup.py
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import structlog
from scripts.bench.infra.perf_compare_scenario import DEFAULT_TRITON_IMAGE, DEFAULT_VLLM_IMAGE
from scripts.bench.remote._common import (
    emit_json,
    gpu_info,
    init_logging,
)

VLLM_IMAGE = DEFAULT_VLLM_IMAGE
TRITON_IMAGE = DEFAULT_TRITON_IMAGE
TIMING_DIR = Path("/tmp/nerva_timing")
PROFILE_DIR = Path("/tmp/profile")


def git_pull() -> dict[str, str]:
    result = subprocess.run(
        ["git", "pull", "--ff-only"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git pull failed:\n{result.stderr}")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"git rev-parse failed:\n{exc.output or str(exc)}") from exc
    return {"git_commit": commit, "git_branch": branch}


def uv_sync() -> str:
    result = subprocess.run(
        ["uv", "sync", "--all-extras"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"uv sync failed:\n{result.stderr}")
    return "ok"


def docker_pull_images(images: list[str]) -> list[str]:
    pulled = []
    for image in images:
        result = subprocess.run(
            ["docker", "pull", image],
            capture_output=True,
            text=True,
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
    parser.add_argument(
        "--pull-images", action="store_true", help="docker pull vllm + triton images"
    )
    parser.add_argument(
        "--clean-timing",
        action="store_true",
        help="rm -rf /tmp/nerva_timing /tmp/profile",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    init_logging()
    args = _cli(argv)
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
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], text=True
            ).strip()
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
            ).strip()
        except (subprocess.CalledProcessError, OSError) as e:
            emit_json({"error": str(e), "step": "git_rev_parse"})
            return 1
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
            result["docker_images"] = docker_pull_images(
                [VLLM_IMAGE, TRITON_IMAGE]
            )
        except RuntimeError as e:
            emit_json({"error": str(e), "step": "docker_pull"})
            return 1

    if args.clean_timing:
        log.info("clean timing + profile dirs")
        try:
            clean_dirs()
        except OSError as e:
            emit_json({"error": str(e), "step": "clean_dirs"})
            return 1
        result["clean_timing"] = "ok"

    result["gpus"] = gpu_info()
    result["python_version"] = (
        f"{sys.version_info.major}.{sys.version_info.minor}"
        f".{sys.version_info.micro}"
    )

    emit_json(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
