from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.bench.report_phase7 import collect_summary_rows

if TYPE_CHECKING:
    from pathlib import Path


def _write_summary(
    root: Path,
    *,
    date: str,
    commit: str,
    target: str,
    concurrency: int,
    qps: float,
) -> None:
    path = root / date / commit / target / str(concurrency)
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "qps": qps,
                "p50_ms": 1.0,
                "p95_ms": 2.0,
                "p99_ms": 3.0,
                "error_rate": 0.1,
            }
        )
    )


def test_collect_summary_rows_filters_commit(tmp_path: Path) -> None:
    _write_summary(
        tmp_path,
        date="2026-03-03",
        commit="abc123",
        target="nerva",
        concurrency=32,
        qps=10.0,
    )
    _write_summary(
        tmp_path,
        date="2026-03-03",
        commit="def456",
        target="nerva",
        concurrency=32,
        qps=20.0,
    )

    rows = collect_summary_rows(tmp_path, commit="abc123")
    assert len(rows) == 1
    assert rows[0].commit == "abc123"
    assert rows[0].qps == 10.0


def test_collect_summary_rows_sorted_by_target_and_concurrency(tmp_path: Path) -> None:
    _write_summary(
        tmp_path,
        date="2026-03-03",
        commit="abc123",
        target="triton",
        concurrency=1000,
        qps=30.0,
    )
    _write_summary(
        tmp_path,
        date="2026-03-03",
        commit="abc123",
        target="nerva",
        concurrency=1,
        qps=5.0,
    )

    rows = collect_summary_rows(tmp_path, commit="abc123")
    assert [row.target for row in rows] == ["nerva", "triton"]
    assert [row.concurrency for row in rows] == [1, 1000]
