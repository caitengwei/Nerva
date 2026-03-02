from __future__ import annotations

import json

import pytest
from scripts.bench.config import load_bench_config


def test_default_concurrency_contains_1000(tmp_path: pytest.TempPathFactory) -> None:
    cfg = {
        "concurrency_levels": [1, 32, 128, 512, 1000],
        "warmup_seconds": 60,
        "sample_seconds": 300,
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))

    loaded = load_bench_config(p)
    assert 1000 in loaded.concurrency_levels


def test_rejects_empty_concurrency_levels(tmp_path: pytest.TempPathFactory) -> None:
    cfg = {
        "concurrency_levels": [],
        "warmup_seconds": 60,
        "sample_seconds": 300,
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))

    with pytest.raises(ValueError, match="concurrency_levels"):
        load_bench_config(p)


def test_rejects_non_positive_durations(tmp_path: pytest.TempPathFactory) -> None:
    cfg = {
        "concurrency_levels": [1, 1000],
        "warmup_seconds": 0,
        "sample_seconds": 300,
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))

    with pytest.raises(ValueError, match="warmup_seconds"):
        load_bench_config(p)
