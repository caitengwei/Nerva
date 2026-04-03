# tests/test_remote_common.py
from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

from scripts.bench.remote._common import emit_json, gpu_info, load_state, save_state


def test_emit_json_outputs_valid_json(capsys):
    emit_json({"foo": "bar", "n": 42})
    out = capsys.readouterr().out
    assert json.loads(out.strip()) == {"foo": "bar", "n": 42}


def test_emit_json_goes_to_stdout_not_stderr(capsys):
    emit_json({"x": 1})
    captured = capsys.readouterr()
    assert captured.out.strip() != ""
    assert captured.err == ""


def test_gpu_info_parses_nvidia_smi_output():
    fake_output = "Tesla P100-PCIE-16GB, 16280\nTesla P100-PCIE-16GB, 16280\n"
    with patch("subprocess.check_output", return_value=fake_output):
        result = gpu_info()
    assert result == [
        {"name": "Tesla P100-PCIE-16GB", "memory_mb": 16280},
        {"name": "Tesla P100-PCIE-16GB", "memory_mb": 16280},
    ]


def test_gpu_info_returns_empty_when_nvidia_smi_missing():
    with patch("subprocess.check_output", side_effect=FileNotFoundError):
        result = gpu_info()
    assert result == []


def test_load_state_returns_empty_dict_when_file_missing(tmp_path):
    result = load_state(tmp_path / "nonexistent.json")
    assert result == {}


def test_save_and_load_state_roundtrip(tmp_path):
    p = tmp_path / "state.json"
    save_state(p, {"nerva": {"pid": 123}})
    assert load_state(p) == {"nerva": {"pid": 123}}


def test_gpu_info_returns_empty_when_nvidia_smi_fails():
    with patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "nvidia-smi")):
        result = gpu_info()
    assert result == []


def test_load_state_returns_empty_on_corrupt_json(tmp_path):
    p = tmp_path / "corrupt.json"
    p.write_text("{invalid json")
    assert load_state(p) == {}
