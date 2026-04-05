# tests/test_remote_service_ctl.py
from __future__ import annotations

import json
import signal
from unittest.mock import MagicMock, patch

import pytest
from scripts.bench.remote.service_ctl import (
    _parse_targets,
    _wait_http_ok,
    main,
    service_status,
    stop_services,
)


def test_parse_targets_valid():
    assert _parse_targets("nerva,vllm") == ["nerva", "vllm"]
    assert _parse_targets("triton") == ["triton"]


def test_parse_targets_invalid_raises():
    with pytest.raises(ValueError, match="Unknown targets"):
        _parse_targets("nerva,badtarget")


def test_wait_http_ok_returns_true_on_200():
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.status = 200
    with patch("scripts.bench.remote.service_ctl._NO_PROXY_OPENER") as mock_opener:
        mock_opener.open.return_value = mock_resp
        assert _wait_http_ok("http://127.0.0.1:8080/health", timeout_s=5)


def test_wait_http_ok_returns_false_on_timeout():
    from urllib.error import URLError
    with patch("scripts.bench.remote.service_ctl._NO_PROXY_OPENER") as mock_opener, \
         patch("time.sleep"), \
         patch("time.monotonic", side_effect=[0, 0.5, 1.0, 6.0]):
        mock_opener.open.side_effect = URLError("refused")
        assert not _wait_http_ok("http://127.0.0.1:8080/health", timeout_s=5)


def test_stop_services_kills_nerva_process():
    state = {"nerva": {"pid": 9999, "endpoint": "http://127.0.0.1:8080"}}
    with patch("os.getpgid", return_value=9999), \
         patch("os.killpg") as mock_kill, \
         patch("time.sleep"):
        stop_services(["nerva"], state)
    mock_kill.assert_any_call(9999, signal.SIGTERM)


def test_stop_services_docker_stops_container():
    state = {"vllm": {"container_id": "abc123", "endpoint": "http://127.0.0.1:8001"}}
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        stop_services(["vllm"], state)
    mock_run.assert_called_once_with(["docker", "stop", "abc123"], capture_output=True)


def test_service_status_returns_unreachable_for_dead_service():
    from urllib.error import URLError
    state = {"nerva": {"pid": 9999, "endpoint": "http://127.0.0.1:8080"}}
    with patch("urllib.request.urlopen", side_effect=URLError("refused")):
        result = service_status(state)
    assert result["nerva"]["health"] == "unreachable"
    assert result["nerva"]["status"] == "unhealthy"


def test_main_status_emits_json_for_empty_state(capsys, tmp_path):
    state_file = tmp_path / "state.json"
    with patch("scripts.bench.remote.service_ctl.STATE_FILE", state_file):
        rc = main(["status"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert isinstance(out, dict)


def test_main_stop_clears_state(capsys, tmp_path):
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"nerva": {"pid": 1234, "endpoint": "http://127.0.0.1:8080"}}))
    with patch("scripts.bench.remote.service_ctl.STATE_FILE", state_file), \
         patch("os.getpgid", side_effect=ProcessLookupError), \
         patch("time.sleep"):
        rc = main(["stop", "--targets", "nerva"])
    assert rc == 0
    remaining = json.loads(state_file.read_text())
    assert "nerva" not in remaining
