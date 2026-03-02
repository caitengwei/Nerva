"""Tests for nerva.observability — metrics and structured logging."""

from __future__ import annotations

import pytest
from prometheus_client import CollectorRegistry

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
