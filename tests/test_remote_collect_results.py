# tests/test_remote_collect_results.py
from __future__ import annotations

import json
import tarfile
from unittest.mock import patch

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
