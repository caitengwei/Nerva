from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchConfig:
    concurrency_levels: list[int]
    warmup_seconds: int
    sample_seconds: int


def _validate_positive_int_list(values: list[object], *, field_name: str) -> list[int]:
    if not values:
        raise ValueError(f"{field_name} must be non-empty")

    parsed: list[int] = []
    for value in values:
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field_name} must contain positive integers")
        parsed.append(value)
    return parsed


def _validate_positive_int(value: object, *, field_name: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return value


def load_bench_config(path: Path) -> BenchConfig:
    raw = json.loads(path.read_text())

    if not isinstance(raw, dict):
        raise ValueError("config root must be an object")

    concurrency_levels = _validate_positive_int_list(
        list(raw.get("concurrency_levels", [])),
        field_name="concurrency_levels",
    )
    if 1000 not in concurrency_levels:
        raise ValueError("concurrency_levels must include 1000")

    warmup_seconds = _validate_positive_int(raw.get("warmup_seconds"), field_name="warmup_seconds")
    sample_seconds = _validate_positive_int(raw.get("sample_seconds"), field_name="sample_seconds")

    return BenchConfig(
        concurrency_levels=concurrency_levels,
        warmup_seconds=warmup_seconds,
        sample_seconds=sample_seconds,
    )
