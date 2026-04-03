# tests/test_remote_profiler_ctl.py
from __future__ import annotations

import json
import signal
from unittest.mock import MagicMock, patch

import pytest
from scripts.bench.remote.profiler_ctl import (
    _make_key,
    main,
    start_profiler,
    stop_profilers,
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
