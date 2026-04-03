# tests/test_remote_env_setup.py
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from scripts.bench.remote.env_setup import (
    clean_dirs,
    docker_pull_images,
    git_pull,
    main,
    uv_sync,
)


def test_git_pull_returns_commit_and_branch():
    with patch("subprocess.run") as mock_run, patch(
        "subprocess.check_output", side_effect=["abc1234\n", "main\n"]
    ):
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
    with patch("subprocess.check_output", side_effect=["abc1234\n", "main\n"]), patch(
        "scripts.bench.remote.env_setup.gpu_info", return_value=[]
    ):
        rc = main([])
    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out.strip())
    assert "git_commit" in data
    assert "gpus" in data
