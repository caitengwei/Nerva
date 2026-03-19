"""Tests for nerva.observability — metrics and structured logging."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from prometheus_client import CollectorRegistry

if TYPE_CHECKING:
    import pathlib

from nerva.observability.metrics import NervaMetrics, get_metrics


class TestNervaMetricsConstruction:
    def test_custom_registry_no_collision(self) -> None:
        r1 = CollectorRegistry()
        r2 = CollectorRegistry()
        m1 = NervaMetrics(registry=r1)
        m2 = NervaMetrics(registry=r2)
        assert m1 is not m2

    def test_metrics_attributes_exist(self, metrics: NervaMetrics) -> None:
        for attr in [
            "request_total", "request_duration_seconds", "request_in_flight",
            "batch_size", "batch_wait_seconds", "queue_depth",
            "worker_status", "worker_infer_seconds",
        ]:
            assert hasattr(metrics, attr), f"missing: {attr}"


class TestRequestMetrics:
    def test_request_total_increment(self, metrics: NervaMetrics) -> None:
        metrics.request_total.labels(pipeline="test", status="ok").inc()
        val = metrics.request_total.labels(pipeline="test", status="ok")._value.get()
        assert val == 1.0

    def test_request_in_flight_gauge(self, metrics: NervaMetrics) -> None:
        g = metrics.request_in_flight.labels(pipeline="chat")
        g.inc()
        g.inc()
        g.dec()
        assert g._value.get() == 1.0

    def test_request_duration_observe(self, metrics: NervaMetrics) -> None:
        metrics.request_duration_seconds.labels(pipeline="chat").observe(0.05)
        h = metrics.request_duration_seconds.labels(pipeline="chat")
        assert h._sum.get() == pytest.approx(0.05)


class TestBatchMetrics:
    def test_batch_size_observe(self, metrics: NervaMetrics) -> None:
        metrics.batch_size.labels(model="llm").observe(8)
        assert metrics.batch_size.labels(model="llm")._sum.get() == pytest.approx(8.0)

    def test_queue_depth_gauge(self, metrics: NervaMetrics) -> None:
        metrics.queue_depth.labels(model="llm").set(5)
        assert metrics.queue_depth.labels(model="llm")._value.get() == 5.0


class TestWorkerMetrics:
    def test_worker_status_gauge(self, metrics: NervaMetrics) -> None:
        metrics.worker_status.labels(model="llm", device="cpu").set(1)
        assert metrics.worker_status.labels(model="llm", device="cpu")._value.get() == 1.0

    def test_worker_infer_seconds(self, metrics: NervaMetrics) -> None:
        metrics.worker_infer_seconds.labels(model="llm").observe(0.123)
        assert metrics.worker_infer_seconds.labels(model="llm")._sum.get() == pytest.approx(0.123)


class TestGetMetricsSingleton:
    def test_get_metrics_returns_same_instance(self) -> None:
        assert get_metrics() is get_metrics()

    def test_get_metrics_is_nerva_metrics(self) -> None:
        assert isinstance(get_metrics(), NervaMetrics)


# ============================================================
# Logging tests
# ============================================================

from nerva.observability.logging import configure_logging  # noqa: E402


class TestConfigureLogging:
    def test_configure_logging_does_not_raise(self) -> None:
        configure_logging(dev=True)

    def test_configure_logging_idempotent(self) -> None:
        configure_logging(dev=True)
        configure_logging(dev=True)

    def test_configure_logging_json_mode(self) -> None:
        configure_logging(dev=False)

    def test_contextvars_bind_and_clear(self) -> None:
        import structlog.contextvars
        configure_logging(dev=True)
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id="req-123")
        ctx = structlog.contextvars.get_contextvars()
        assert ctx.get("request_id") == "req-123"
        structlog.contextvars.clear_contextvars()

    def test_get_logger_returns_bound_logger(self) -> None:
        import structlog
        configure_logging(dev=True)
        log = structlog.get_logger("test.module")
        assert log is not None


# ============================================================
# AsyncTimingSink tests
# ============================================================

import json  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402

from nerva.observability.timing import AsyncTimingSink  # noqa: E402


class TestAsyncTimingSink:
    async def test_write_noop_before_start(self) -> None:
        """write() before start() is a no-op (no exception)."""
        sink = AsyncTimingSink()
        sink.write({"event": "test"})  # should not raise

    async def test_roundtrip(self, tmp_path: pathlib.Path) -> None:
        """Written dicts appear as JSON lines in the log file."""
        sink = AsyncTimingSink()
        await sink.start(str(tmp_path), "test.log")
        sink.write({"event": "a", "val": 1})
        sink.write({"event": "b", "val": 2})
        await sink.stop()
        lines = (tmp_path / "test.log").read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"event": "a", "val": 1}
        assert json.loads(lines[1]) == {"event": "b", "val": 2}

    async def test_queue_full_drops_silently(self) -> None:
        """When queue is full, write() drops entries without raising."""
        sink = AsyncTimingSink()
        # Use a tiny maxsize to test backpressure.
        import queue
        sink._queue = queue.Queue(maxsize=2)
        # Simulate started state with a fake thread ref.
        sink._thread = type("FakeThread", (), {"is_alive": lambda self: True})()  # type: ignore[assignment]
        sink._stopping = False
        # Fill the queue.
        sink.write({"a": 1})
        sink.write({"a": 2})
        # This should be silently dropped (queue full).
        sink.write({"a": 3})
        assert sink._queue.qsize() == 2

    async def test_write_noop_when_stopping(self) -> None:
        """write() is a no-op once stop has been initiated."""
        sink = AsyncTimingSink()
        with tempfile.TemporaryDirectory() as d:
            await sink.start(d, "stop_test.log")
            await sink.stop()
            # After stop, write should be no-op.
            sink.write({"event": "late"})
            with open(os.path.join(d, "stop_test.log")) as f:
                content = f.read()
            assert "late" not in content
