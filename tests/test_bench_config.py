from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from scripts.bench.config import load_bench_config

if TYPE_CHECKING:
    from pathlib import Path


def test_config_loader_does_not_require_c1000(tmp_path: Path) -> None:
    cfg = {
        "concurrency_levels": [1, 32],
        "warmup_seconds": 60,
        "sample_seconds": 300,
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))

    loaded = load_bench_config(p)
    assert loaded.concurrency_levels == [1, 32]


def test_rejects_empty_concurrency_levels(tmp_path: Path) -> None:
    cfg = {
        "concurrency_levels": [],
        "warmup_seconds": 60,
        "sample_seconds": 300,
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))

    with pytest.raises(ValueError, match="concurrency_levels"):
        load_bench_config(p)


def test_rejects_non_list_concurrency_levels(tmp_path: Path) -> None:
    cfg = {
        "concurrency_levels": 1000,
        "warmup_seconds": 60,
        "sample_seconds": 300,
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))

    with pytest.raises(ValueError, match="concurrency_levels must be a JSON array"):
        load_bench_config(p)


def test_rejects_non_positive_durations(tmp_path: Path) -> None:
    cfg = {
        "concurrency_levels": [1, 1000],
        "warmup_seconds": 0,
        "sample_seconds": 300,
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))

    with pytest.raises(ValueError, match="warmup_seconds"):
        load_bench_config(p)
