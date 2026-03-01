"""Nerva Prometheus metrics definitions."""

from __future__ import annotations

import prometheus_client
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

_DEFAULT_BUCKETS_SECONDS = (
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
)
_BATCH_SIZE_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


class NervaMetrics:
    """Container for all Nerva Prometheus metrics.

    Args:
        registry: CollectorRegistry. Pass CollectorRegistry() for test isolation.
                  Defaults to prometheus_client.REGISTRY (global).
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        reg = registry if registry is not None else prometheus_client.REGISTRY

        self.request_total = Counter(
            "nerva_request_total",
            "Total RPC requests by pipeline and final status.",
            ["pipeline", "status"],
            registry=reg,
        )
        self.request_duration_seconds = Histogram(
            "nerva_request_duration_seconds",
            "End-to-end RPC request duration in seconds.",
            ["pipeline"],
            buckets=_DEFAULT_BUCKETS_SECONDS,
            registry=reg,
        )
        self.request_in_flight = Gauge(
            "nerva_request_in_flight",
            "Currently in-flight RPC requests by pipeline.",
            ["pipeline"],
            registry=reg,
        )
        self.batch_size = Histogram(
            "nerva_batch_size_total",
            "Batch size distribution per model.",
            ["model"],
            buckets=_BATCH_SIZE_BUCKETS,
            registry=reg,
        )
        self.batch_wait_seconds = Histogram(
            "nerva_batch_wait_seconds",
            "Time a request waited in batcher queue before dispatch.",
            ["model"],
            buckets=_DEFAULT_BUCKETS_SECONDS,
            registry=reg,
        )
        self.queue_depth = Gauge(
            "nerva_queue_depth",
            "Current batcher queue depth per model.",
            ["model"],
            registry=reg,
        )
        self.worker_status = Gauge(
            "nerva_worker_status",
            "Worker health: 1=READY, 0=not ready.",
            ["model", "device"],
            registry=reg,
        )
        self.worker_infer_seconds = Histogram(
            "nerva_worker_infer_seconds",
            "Worker inference latency per model.",
            ["model"],
            buckets=_DEFAULT_BUCKETS_SECONDS,
            registry=reg,
        )


_global_metrics: NervaMetrics | None = None


def get_metrics() -> NervaMetrics:
    """Return the process-level global NervaMetrics singleton."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = NervaMetrics()
    return _global_metrics
