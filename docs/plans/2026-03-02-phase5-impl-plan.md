# Phase 5 — vLLM Backend + Observability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate vLLM as a production LLM backend and add full observability (structured logging + Prometheus metrics).

**Architecture:** VLLMBackend wraps vllm.AsyncLLMEngine as a thin layer inside the Worker process, bypassing Nerva's DynamicBatcher. Observability is cross-cutting: structlog replaces stdlib logging in hot paths, Prometheus metrics are collected at RPC handler and batcher boundaries. vllm is an optional dependency (GPU required); all tests run without it via mocks.

**Tech Stack:** structlog>=24.0, prometheus-client>=0.21, vllm (optional/GPU), pytest, mypy --strict

---

## Key Constraints

1. **prometheus_client metrics are global singletons** — tests use `NervaMetrics(registry=CollectorRegistry())` for isolation, never the global registry
2. **vllm is optional** — `import nerva.backends.vllm` succeeds without vllm; only `load_model()` raises `ImportError`
3. **Conservative changes** — metrics/logging are injected as optional parameters, no functional logic changed
4. **mypy strict** — all new modules require complete type annotations; vllm already in `[[tool.mypy.overrides]]`

---

## Task 1: Observability Package — metrics.py skeleton + conftest fixture

**Files:**
- Create: `src/nerva/observability/__init__.py` (update)
- Create: `src/nerva/observability/metrics.py`
- Modify: `tests/conftest.py`

**Step 1: Verify dependencies already in pyproject.toml**

```bash
grep -E "prometheus|structlog" pyproject.toml
# Expected: prometheus-client>=0.21 and structlog>=24.0 present
```

**Step 2: Create `src/nerva/observability/metrics.py`**

```python
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
```

**Step 3: Update `src/nerva/observability/__init__.py`**

```python
"""Nerva observability: metrics and structured logging."""

from nerva.observability.logging import configure_logging
from nerva.observability.metrics import NervaMetrics, get_metrics

__all__ = ["NervaMetrics", "configure_logging", "get_metrics"]
```

**Step 4: Add metrics fixture to `tests/conftest.py`**

Append after existing `_clean_model_registry` fixture:

```python
import pytest
from prometheus_client import CollectorRegistry
from nerva.observability.metrics import NervaMetrics


@pytest.fixture()
def metrics() -> NervaMetrics:
    """Isolated NervaMetrics for tests — avoids 'Duplicated timeseries' errors."""
    return NervaMetrics(registry=CollectorRegistry())
```

**Step 5: Verify imports work**

```bash
uv run python -c "from nerva.observability.metrics import NervaMetrics, get_metrics; print('ok')"
uv run ruff check src/nerva/observability/metrics.py
uv run mypy src/nerva/observability/metrics.py
```

**Step 6: Commit**

```bash
git add src/nerva/observability/ tests/conftest.py
git commit -m "feat(observability): add NervaMetrics with isolated registry support"
```

---

## Task 2: structlog Logging — logging.py + tests

**Files:**
- Create: `src/nerva/observability/logging.py`
- Create: `tests/test_observability.py`

**Step 1: Write failing tests first**

Create `tests/test_observability.py`:

```python
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
```

**Step 2: Run tests (expect failures)**

```bash
uv run pytest tests/test_observability.py -v
# Expected: ImportError on configure_logging (logging.py not yet created)
```

**Step 3: Create `src/nerva/observability/logging.py`**

```python
"""Structured logging configuration for Nerva using structlog.

Usage:
    from nerva.observability.logging import configure_logging
    configure_logging(dev=True)   # human-friendly (local)
    configure_logging(dev=False)  # JSON (production)

Per-request context binding (async-safe via contextvars):
    import structlog.contextvars
    structlog.contextvars.bind_contextvars(request_id="req-abc", pipeline="chat")
    structlog.contextvars.clear_contextvars()
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(
    *,
    dev: bool = False,
    level: int = logging.INFO,
) -> None:
    """Configure structlog for Nerva.

    Args:
        dev: If True, use ConsoleRenderer (human-friendly).
             If False, use JSONRenderer (production).
        level: stdlib logging level. Defaults to INFO.
    """
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
        force=True,
    )

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if dev:
        processors: list[structlog.types.Processor] = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(),
        ]
    else:
        processors = [
            *shared_processors,
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,  # False for test safety
    )
```

**Step 4: Run tests (expect pass)**

```bash
uv run pytest tests/test_observability.py -v
# Expected: all pass
uv run ruff check src/nerva/observability/
uv run mypy src/nerva/observability/
```

**Step 5: Commit**

```bash
git add src/nerva/observability/logging.py tests/test_observability.py
git commit -m "feat(observability): implement configure_logging with structlog; add tests"
```

---

## Task 3: Wire Metrics into DynamicBatcher

**Files:**
- Modify: `src/nerva/engine/batcher.py`
- Modify: `tests/test_batcher.py`

**Step 1: Read current batcher structure**

```bash
grep -n "def __init__\|def infer\|def _batch_loop\|_queue\|_model_name" src/nerva/engine/batcher.py | head -30
```

**Step 2: Add metrics tests to `tests/test_batcher.py`**

Append at end:

```python
from prometheus_client import CollectorRegistry
from nerva.observability.metrics import NervaMetrics


async def test_batcher_metrics_batch_size(mock_inner: Any) -> None:
    """Batcher records batch_size histogram when dispatching."""
    reg = CollectorRegistry()
    m = NervaMetrics(registry=reg)
    cfg = BatchConfig(max_batch_size=4, max_delay_ms=5.0)

    async with DynamicBatcher(mock_inner, cfg, model_name="test-model", metrics=m) as batcher:
        ctx = InferContext(request_id="r1", deadline_ms=5000)
        await batcher.infer({"x": 1}, ctx)

    assert m.batch_size.labels(model="test-model")._count.get() >= 1


async def test_batcher_without_metrics_no_error(mock_inner: Any) -> None:
    """Batcher with no metrics= arg works unchanged."""
    async with DynamicBatcher(mock_inner, BatchConfig()) as batcher:
        ctx = InferContext(request_id="r1", deadline_ms=5000)
        result = await batcher.infer({"x": 1}, ctx)
    assert result is not None
```

**Step 3: Modify `src/nerva/engine/batcher.py`**

Key changes (minimal — inject optional params, no functional changes):

1. Add import at top: `from nerva.observability.metrics import NervaMetrics`
2. Add `model_name: str = "unknown"` and `metrics: NervaMetrics | None = None` to `__init__`
3. Store as `self._model_name` and `self._metrics`
4. In `infer()`, before `put()`: `if self._metrics: self._metrics.queue_depth.labels(model=self._model_name).set(self._queue.qsize())`
5. Add `enqueue_time: float` field to `_PendingRequest` dataclass
6. In `_batch_loop`, after collecting valid batch: observe `batch_size` and per-item `batch_wait_seconds`

**Step 4: Run tests**

```bash
uv run pytest tests/test_batcher.py -v
# Expected: all existing + new tests pass
```

**Step 5: Commit**

```bash
git add src/nerva/engine/batcher.py tests/test_batcher.py
git commit -m "feat(batcher): wire optional metrics into DynamicBatcher"
```

---

## Task 4: /metrics Endpoint + RPC Handler Instrumentation

**Files:**
- Modify: `src/nerva/server/app.py`
- Modify: `src/nerva/server/rpc.py`
- Modify: `tests/test_app.py`
- Modify: `tests/test_rpc.py`

**Step 1: Add /metrics route tests to `tests/test_app.py`**

```python
class TestMetricsRoute:
    def test_metrics_endpoint_exists(self) -> None:
        from starlette.testclient import TestClient
        app = build_app(pipelines={}, model_info=[])
        client = TestClient(app)
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
```

**Step 2: Modify `src/nerva/server/app.py`** — add `/metrics` route

```python
import prometheus_client
from starlette.requests import Request
from starlette.responses import Response

# Inside build_app(), add route handler:
async def metrics_endpoint(request: Request) -> Response:
    data = prometheus_client.generate_latest()
    return Response(content=data, media_type=prometheus_client.CONTENT_TYPE_LATEST)

# Add to routes list:
Route("/metrics", metrics_endpoint, methods=["GET"]),
```

**Step 3: Add metrics + structlog to `src/nerva/server/rpc.py`**

Replace `logger = logging.getLogger(__name__)` with `import structlog; logger = structlog.get_logger(__name__)`

Add `metrics: NervaMetrics | None = None` to `RpcHandler.__init__`, store as `self._metrics = metrics or get_metrics()`

In `handle()`:
- After extracting `pipeline_name` and `request_id`: `structlog.contextvars.bind_contextvars(request_id=str(request_id), pipeline=pipeline_name)`
- Record `t0 = time.monotonic()` and `self._metrics.request_in_flight.labels(pipeline=pipeline_name).inc()`
- In `finally`: observe duration, dec in_flight, `structlog.contextvars.clear_contextvars()`
- On success: `self._metrics.request_total.labels(pipeline=pipeline_name, status="ok").inc()`
- On error: `self._metrics.request_total.labels(pipeline=pipeline_name, status=code.name.lower()).inc()`

**Step 4: Add RPC metrics tests to `tests/test_rpc.py`**

```python
async def test_request_total_incremented_on_success(...) -> None:
    # Use RpcHandler(..., metrics=NervaMetrics(registry=CollectorRegistry()))
    # Make a successful request, assert request_total == 1.0
    ...

async def test_request_in_flight_returns_to_zero(...) -> None:
    # After request completes, assert request_in_flight == 0.0
    ...
```

**Step 5: Run tests**

```bash
uv run pytest tests/test_app.py tests/test_rpc.py -v
```

**Step 6: Commit**

```bash
git add src/nerva/server/app.py src/nerva/server/rpc.py tests/test_app.py tests/test_rpc.py
git commit -m "feat(server): add /metrics endpoint; instrument RpcHandler with metrics + structlog"
```

---

## Task 5: Worker Instrumentation (structlog + worker_status metrics)

**Files:**
- Modify: `src/nerva/worker/process.py`
- Modify: `src/nerva/worker/manager.py`
- Modify: `tests/test_worker_manager.py`

**Step 1: Add structlog to `src/nerva/worker/process.py` hot path**

Replace `logger = logging.getLogger(__name__)` with `import structlog; logger = structlog.get_logger(__name__)`

In `_handle_infer()`:
- At start: `structlog.contextvars.bind_contextvars(request_id=request_id)`
- In `finally`: `structlog.contextvars.clear_contextvars()`

No functional changes — existing tests must pass unchanged.

**Step 2: Add worker_status metrics to `src/nerva/worker/manager.py`**

Add `metrics: NervaMetrics | None = None` to `WorkerManager.__init__`

In `start_worker()`, after state → READY:
```python
if self._metrics:
    self._metrics.worker_status.labels(model=handle.name, device=handle.device).set(1)
```

In `_close_worker()`, on state → STOPPING:
```python
if self._metrics:
    self._metrics.worker_status.labels(model=entry.handle.name, device=entry.handle.device).set(0)
```

**Step 3: Add test to `tests/test_worker_manager.py`**

```python
async def test_worker_manager_accepts_custom_metrics() -> None:
    reg = CollectorRegistry()
    m = NervaMetrics(registry=reg)
    manager = WorkerManager(metrics=m)
    assert manager._metrics is m
    await manager.shutdown_all()  # no-op, no workers started
```

**Step 4: Run tests**

```bash
uv run pytest tests/test_worker_process.py tests/test_worker_manager.py -v
# All existing tests must still pass
```

**Step 5: Commit**

```bash
git add src/nerva/worker/process.py src/nerva/worker/manager.py tests/test_worker_manager.py
git commit -m "feat(worker): add structlog to infer hot path; add worker_status metrics to manager"
```

---

## Task 6: VLLMBackend Implementation (TDD)

**Files:**
- Create: `tests/test_vllm_backend.py`
- Create: `src/nerva/backends/vllm.py`

**Step 1: Write tests first**

Create `tests/test_vllm_backend.py`:

```python
"""Tests for VLLMBackend — all tests run without real vllm (GPU not required).

Strategy: mock sys.modules["vllm"] with MagicMock to simulate vllm API.
VLLMBackend uses conditional import in load_model() only.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nerva.backends.base import InferContext, ModelConfig


def _ctx(request_id: str = "req-1", deadline_ms: int = 30000) -> InferContext:
    return InferContext(request_id=request_id, deadline_ms=deadline_ms)


def _config(model_path: str = "/models/llm", device: str = "cuda:0") -> ModelConfig:
    return ModelConfig(
        model_name="test-llm",
        model_class=object,
        device=device,
        backend_options={"model_path": model_path},
    )


def _make_vllm_mock(text_chunks: list[str] | None = None) -> Any:
    """Build a vllm mock that streams the given text chunks."""
    chunks = text_chunks or ["Hello", " world!"]

    async def _generate(prompt: str, sampling_params: Any, request_id: str):  # type: ignore[no-untyped-def]
        for i, text in enumerate(chunks):
            output = MagicMock()
            output.outputs = [MagicMock()]
            output.outputs[0].text = "".join(chunks[: i + 1])
            output.finished = i == len(chunks) - 1
            yield output

    engine = MagicMock()
    engine.generate = _generate

    vllm_mod = MagicMock()
    vllm_mod.AsyncLLMEngine.from_engine_args = MagicMock(return_value=engine)
    vllm_mod.AsyncEngineArgs = MagicMock(return_value=MagicMock())
    vllm_mod.SamplingParams = MagicMock(return_value=MagicMock())
    return vllm_mod


class TestVLLMBackendImport:
    def test_import_does_not_crash_without_vllm(self) -> None:
        """Importing VLLMBackend must not crash even if vllm is absent."""
        with patch.dict("sys.modules", {"vllm": None}):  # type: ignore[arg-type]
            mod_key = "nerva.backends.vllm"
            saved = sys.modules.pop(mod_key, None)
            try:
                import nerva.backends.vllm  # noqa: F401
            finally:
                if saved is not None:
                    sys.modules[mod_key] = saved

    def test_vllm_backend_class_accessible(self) -> None:
        from nerva.backends.vllm import VLLMBackend
        assert VLLMBackend is not None


class TestVLLMBackendLifecycle:
    async def test_is_loaded_false_before_load(self) -> None:
        from nerva.backends.vllm import VLLMBackend
        assert not VLLMBackend().is_loaded

    async def test_health_check_false_before_load(self) -> None:
        from nerva.backends.vllm import VLLMBackend
        assert not VLLMBackend().health_check()

    async def test_load_model_sets_is_loaded(self) -> None:
        from nerva.backends.vllm import VLLMBackend
        with patch.dict("sys.modules", {"vllm": _make_vllm_mock()}):
            backend = VLLMBackend()
            await backend.load_model(_config())
            assert backend.is_loaded

    async def test_load_model_without_vllm_raises_import_error(self) -> None:
        from nerva.backends.vllm import VLLMBackend
        with patch.dict("sys.modules", {"vllm": None}):  # type: ignore[arg-type]
            backend = VLLMBackend()
            with pytest.raises((ImportError, RuntimeError)):
                await backend.load_model(_config())

    async def test_unload_model_clears_is_loaded(self) -> None:
        from nerva.backends.vllm import VLLMBackend
        with patch.dict("sys.modules", {"vllm": _make_vllm_mock()}):
            backend = VLLMBackend()
            await backend.load_model(_config())
            await backend.unload_model()
            assert not backend.is_loaded

    async def test_load_model_missing_model_path_raises(self) -> None:
        from nerva.backends.vllm import VLLMBackend
        with patch.dict("sys.modules", {"vllm": _make_vllm_mock()}):
            backend = VLLMBackend()
            bad_config = ModelConfig(
                model_name="bad", model_class=object, device="cpu",
                backend_options={},  # no model_path
            )
            with pytest.raises(ValueError, match="model_path"):
                await backend.load_model(bad_config)


class TestVLLMBackendInfer:
    async def test_infer_returns_text_key(self) -> None:
        from nerva.backends.vllm import VLLMBackend
        with patch.dict("sys.modules", {"vllm": _make_vllm_mock(["Hello", " world!"])}):
            backend = VLLMBackend()
            await backend.load_model(_config())
            result = await backend.infer({"prompt": "Hi"}, _ctx())
            assert "text" in result
            assert isinstance(result["text"], str)

    async def test_infer_without_load_raises_runtime_error(self) -> None:
        from nerva.backends.vllm import VLLMBackend
        with pytest.raises(RuntimeError, match="No model loaded"):
            await VLLMBackend().infer({"prompt": "test"}, _ctx())

    async def test_infer_deadline_exceeded_raises(self) -> None:
        from nerva.backends.vllm import VLLMBackend

        async def _slow_generate(prompt: str, params: Any, req_id: str):  # type: ignore[no-untyped-def]
            await asyncio.sleep(10)
            yield MagicMock()

        vllm_mock = _make_vllm_mock()
        vllm_mock.AsyncLLMEngine.from_engine_args.return_value.generate = _slow_generate
        with patch.dict("sys.modules", {"vllm": vllm_mock}):
            backend = VLLMBackend()
            await backend.load_model(_config())
            with pytest.raises((RuntimeError, asyncio.TimeoutError, TimeoutError)):
                await backend.infer({"prompt": "test"}, _ctx(deadline_ms=50))

    async def test_infer_stream_yields_chunks(self) -> None:
        from nerva.backends.vllm import VLLMBackend
        with patch.dict("sys.modules", {"vllm": _make_vllm_mock(["A", "B", "C"])}):
            backend = VLLMBackend()
            await backend.load_model(_config())
            chunks = [
                chunk async for chunk in backend.infer_stream({"prompt": "Hi"}, _ctx())
            ]
            assert len(chunks) == 3
            assert all("text" in c for c in chunks)
            assert chunks[-1].get("finished") is True


class TestVLLMBackendRegistered:
    def test_registered_as_vllm(self) -> None:
        from nerva.backends.registry import list_backends
        import nerva.backends.vllm  # noqa: F401 — triggers @register_backend
        assert "vllm" in list_backends()
```

**Step 2: Run tests (expect failures)**

```bash
uv run pytest tests/test_vllm_backend.py -v
# Expected: ImportError — nerva.backends.vllm not yet created
```

**Step 3: Create `src/nerva/backends/vllm.py`**

```python
"""VLLMBackend — thin wrapper around vllm.AsyncLLMEngine.

Import of this module succeeds without vllm installed.
ImportError is deferred to load_model() to support optional dependency.

Inputs dict keys:
    prompt (str): Input prompt.
    max_tokens (int, optional): Default 256.
    temperature (float, optional): Default 1.0.
    top_p (float, optional): Default 1.0.

Unary output dict keys:
    text (str): Final generated text.

Streaming output dict keys (per chunk):
    text (str): Cumulative text so far.
    finished (bool): True on final chunk.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from nerva.backends.base import Backend, BatchMeta, InferContext, ModelConfig
from nerva.backends.registry import register_backend

logger = logging.getLogger(__name__)


@register_backend("vllm")
class VLLMBackend(Backend):
    """Backend wrapping vllm.AsyncLLMEngine for LLM text generation.

    Design decisions:
    - Bypasses Nerva DynamicBatcher; vLLM has continuous batching internally.
    - Deadline enforced via asyncio.wait_for().
    - SHM not used; vLLM manages its own GPU memory (PagedAttention).
    """

    def __init__(self) -> None:
        self._engine: Any | None = None
        self._config: ModelConfig | None = None

    async def load_model(self, config: ModelConfig) -> None:
        """Initialize vllm.AsyncLLMEngine.

        Args:
            config: backend_options must contain 'model_path' (str).
                    Optional: 'tensor_parallel_size' (int), 'max_model_len' (int).

        Raises:
            ImportError: vllm not installed.
            ValueError: model_path missing from backend_options.
            RuntimeError: model already loaded.
        """
        try:
            import vllm  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "vllm is required for VLLMBackend. "
                "Install it with: pip install 'nerva[vllm]'"
            ) from exc

        if self._engine is not None:
            name = self._config.model_name if self._config else "?"
            raise RuntimeError(f"Model already loaded: {name}. Call unload_model() first.")

        model_path: str | None = config.backend_options.get("model_path")
        if not model_path:
            raise ValueError("VLLMBackend requires 'model_path' in backend_options.")

        engine_args: dict[str, Any] = {
            "model": model_path,
            "tensor_parallel_size": config.backend_options.get("tensor_parallel_size", 1),
        }
        max_model_len: int | None = config.backend_options.get("max_model_len")
        if max_model_len is not None:
            engine_args["max_model_len"] = max_model_len

        logger.info("Loading VLLMBackend '%s' on %s", config.model_name, config.device)
        self._engine = vllm.AsyncLLMEngine.from_engine_args(vllm.AsyncEngineArgs(**engine_args))
        self._config = config
        logger.info("VLLMBackend '%s' loaded", config.model_name)

    async def unload_model(self) -> None:
        if self._engine is None:
            return
        name = self._config.model_name if self._config else "?"
        logger.info("Unloading VLLMBackend '%s'", name)
        self._engine = None
        self._config = None

    async def infer(
        self,
        inputs: dict[str, Any],
        context: InferContext,
        batch_meta: BatchMeta | None = None,
    ) -> dict[str, Any]:
        """Unary inference: consume all tokens, return final text."""
        engine = self._ensure_loaded()
        prompt: str = inputs.get("prompt", "")
        sampling_params = self._build_sampling_params(inputs)

        async def _consume() -> dict[str, Any]:
            final_text = ""
            async for output in engine.generate(prompt, sampling_params, context.request_id):
                if output.outputs:
                    final_text = output.outputs[0].text
            return {"text": final_text}

        try:
            return await asyncio.wait_for(_consume(), timeout=context.deadline_ms / 1000.0)
        except asyncio.TimeoutError as exc:
            raise RuntimeError("DEADLINE_EXCEEDED") from exc

    async def infer_stream(
        self,
        inputs: dict[str, Any],
        context: InferContext,
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming inference: yield each output chunk from vLLM."""
        engine = self._ensure_loaded()
        prompt: str = inputs.get("prompt", "")
        sampling_params = self._build_sampling_params(inputs)
        async for output in engine.generate(prompt, sampling_params, context.request_id):
            if output.outputs:
                yield {"text": output.outputs[0].text, "finished": output.finished}

    async def warmup(self) -> None:
        pass  # vLLM handles its own warmup internally.

    def health_check(self) -> bool:
        return self._engine is not None

    @property
    def is_loaded(self) -> bool:
        return self._engine is not None

    def _ensure_loaded(self) -> Any:
        if self._engine is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        return self._engine

    def _build_sampling_params(self, inputs: dict[str, Any]) -> Any:
        try:
            import vllm  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError("vllm not installed") from exc
        return vllm.SamplingParams(
            max_tokens=inputs.get("max_tokens", 256),
            temperature=inputs.get("temperature", 1.0),
            top_p=inputs.get("top_p", 1.0),
        )
```

**Step 4: Run tests (expect pass)**

```bash
uv run pytest tests/test_vllm_backend.py -v
uv run ruff check src/nerva/backends/vllm.py
uv run mypy src/nerva/backends/vllm.py
```

**Step 5: Commit**

```bash
git add src/nerva/backends/vllm.py tests/test_vllm_backend.py
git commit -m "feat(backends): implement VLLMBackend with conditional vllm import"
```

---

## Task 7: Update Exports + E2E Tests + Example

**Files:**
- Modify: `src/nerva/__init__.py`
- Create: `tests/test_phase5_e2e.py`
- Modify: `examples/02_multi_model_pipeline.py`

**Step 1: Update `src/nerva/__init__.py`**

Add imports and `__all__` entries:

```python
from nerva.observability.logging import configure_logging
from nerva.observability.metrics import NervaMetrics, get_metrics

# In __all__:
"NervaMetrics", "configure_logging", "get_metrics",
```

Note: do NOT auto-import `nerva.backends.vllm` — it's an optional backend, user must explicitly import it.

**Step 2: Create `tests/test_phase5_e2e.py`**

```python
"""Phase 5 E2E tests — full observability + vLLM pipeline (no GPU required)."""

from __future__ import annotations

import sys
import time
from typing import Any
from unittest.mock import MagicMock, patch

import msgpack
import pytest
from prometheus_client import CollectorRegistry

from nerva.observability.logging import configure_logging
from nerva.observability.metrics import NervaMetrics


def _make_vllm_mock() -> Any:
    async def _generate(prompt: str, params: Any, req_id: str):  # type: ignore[no-untyped-def]
        out = MagicMock()
        out.outputs = [MagicMock()]
        out.outputs[0].text = f"response to: {prompt}"
        out.finished = True
        yield out

    engine = MagicMock()
    engine.generate = _generate
    vllm_mod = MagicMock()
    vllm_mod.AsyncLLMEngine.from_engine_args = MagicMock(return_value=engine)
    vllm_mod.AsyncEngineArgs = MagicMock(return_value=MagicMock())
    vllm_mod.SamplingParams = MagicMock(return_value=MagicMock())
    return vllm_mod


class TestObservabilityPublicAPI:
    def test_configure_logging_exported(self) -> None:
        import nerva
        assert hasattr(nerva, "configure_logging")

    def test_get_metrics_exported(self) -> None:
        import nerva
        assert hasattr(nerva, "get_metrics")

    def test_nerva_metrics_exported(self) -> None:
        import nerva
        assert hasattr(nerva, "NervaMetrics")

    def test_configure_logging_no_crash(self) -> None:
        configure_logging(dev=True)

    def test_two_metrics_with_separate_registries(self) -> None:
        r1 = CollectorRegistry()
        r2 = CollectorRegistry()
        m1 = NervaMetrics(registry=r1)
        m2 = NervaMetrics(registry=r2)
        m1.request_total.labels(pipeline="p1", status="ok").inc()
        # m2 should be unaffected
        val = m2.request_total.labels(pipeline="p1", status="ok")._value.get()
        assert val == 0.0


class TestVLLMBackendE2E:
    async def test_full_infer_flow(self) -> None:
        from nerva.backends.base import InferContext, ModelConfig
        from nerva.backends.vllm import VLLMBackend

        with patch.dict("sys.modules", {"vllm": _make_vllm_mock()}):
            backend = VLLMBackend()
            await backend.load_model(ModelConfig(
                model_name="e2e-llm", model_class=object,
                device="cuda:0", backend_options={"model_path": "/models/test"},
            ))
            ctx = InferContext(request_id="e2e-001", deadline_ms=5000)
            result = await backend.infer({"prompt": "What is Nerva?"}, ctx)
            assert "text" in result
            assert len(result["text"]) > 0

    async def test_infer_stream_yields_final_chunk(self) -> None:
        from nerva.backends.base import InferContext, ModelConfig
        from nerva.backends.vllm import VLLMBackend

        with patch.dict("sys.modules", {"vllm": _make_vllm_mock()}):
            backend = VLLMBackend()
            await backend.load_model(ModelConfig(
                model_name="e2e-llm", model_class=object,
                device="cuda:0", backend_options={"model_path": "/models/test"},
            ))
            ctx = InferContext(request_id="e2e-stream-001", deadline_ms=5000)
            chunks = [c async for c in backend.infer_stream({"prompt": "Hello"}, ctx)]
            assert len(chunks) >= 1
            assert chunks[-1]["finished"] is True


class TestMetricsEndpointE2E:
    def test_metrics_endpoint_returns_200(self) -> None:
        from starlette.testclient import TestClient
        from nerva.server.app import build_app
        app = build_app(pipelines={}, model_info=[])
        resp = TestClient(app).get("/metrics")
        assert resp.status_code == 200

    def test_rpc_handler_metrics_recorded(self) -> None:
        from unittest.mock import AsyncMock
        from starlette.applications import Starlette
        from starlette.routing import Route
        from starlette.testclient import TestClient
        from nerva.server.protocol import Frame, FrameType, encode_frame
        from nerva.server.rpc import RpcHandler

        reg = CollectorRegistry()
        m = NervaMetrics(registry=reg)
        executor = AsyncMock()
        executor.execute.return_value = {"out": "ok"}
        handler = RpcHandler({"pipe": executor}, metrics=m)
        app = Starlette(routes=[Route("/rpc/{pipeline_name}", handler.handle, methods=["POST"])])

        open_body = msgpack.packb({"pipeline": "pipe"})
        data_body = msgpack.packb({"x": 1})
        body = (
            encode_frame(Frame(FrameType.OPEN, 1, 0, open_body))
            + encode_frame(Frame(FrameType.DATA, 1, 0, data_body))
            + encode_frame(Frame(FrameType.END, 1, 0, b""))
        )
        deadline = int(time.time() * 1000) + 30000
        TestClient(app).post(
            "/rpc/pipe", content=body,
            headers={"content-type": "application/x-nerva-rpc",
                     "x-nerva-deadline-ms": str(deadline),
                     "x-nerva-stream": "0"},
        )
        assert m.request_total.labels(pipeline="pipe", status="ok")._value.get() == 1.0
        assert m.request_in_flight.labels(pipeline="pipe")._value.get() == 0.0


@pytest.mark.gpu
class TestVLLMGPUIntegration:
    """Requires real GPU + vllm. Run with: pytest -m gpu"""
    async def test_placeholder(self) -> None:
        pytest.skip("GPU tests require real hardware.")
```

**Step 3: Update `examples/02_multi_model_pipeline.py`**

```python
"""Example: Tokenizer → LLM → Detokenizer pipeline.

Demonstrates a text generation pipeline using Nerva's DAG execution.
Uses a toy LLM stub — replace with VLLMBackend for production use.

Run (CPU, no GPU):
    uv run python examples/02_multi_model_pipeline.py

Production usage with vLLM:
    llm = model(name="llm", backend="vllm", device="cuda:0",
                backend_options={"model_path": "/path/to/model"})
"""

from __future__ import annotations

import asyncio
from typing import Any

from nerva import model, trace
from nerva.core.model import Model
from nerva.observability.logging import configure_logging

configure_logging(dev=True)


class TokenizerModel(Model):
    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        text: str = inputs["text"]
        return {"tokens": text.split(), "prompt": text}


class LLMModel(Model):
    """Toy LLM stub. Replace with VLLMBackend in production."""

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        prompt: str = inputs["prompt"]
        return {"raw_output": f"[LLM response to: {prompt[:40]}]"}


class DetokenizerModel(Model):
    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"text": inputs["raw_output"].strip()}


tokenizer_h = model(TokenizerModel, name="tokenizer", device="cpu")
llm_h = model(LLMModel, name="llm", device="cpu")
detokenizer_h = model(DetokenizerModel, name="detokenizer", device="cpu")


def pipeline(text_input: Any) -> Any:
    tok_out = tokenizer_h(text_input)
    llm_out = llm_h(tok_out)
    return detokenizer_h(llm_out)


if __name__ == "__main__":
    async def main() -> None:
        graph = trace(pipeline)
        print(f"Pipeline graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
        # To serve:
        # import nerva
        # nerva.serve({"text_gen": graph}, host="0.0.0.0", port=8080)

    asyncio.run(main())
```

**Step 4: Run full test suite**

```bash
uv run pytest tests/ -v --tb=short -m "not gpu"
uv run ruff check src/ tests/
uv run mypy
```

**Step 5: Commit**

```bash
git add src/nerva/__init__.py tests/test_phase5_e2e.py examples/02_multi_model_pipeline.py
git commit -m "feat(phase5): update exports, add e2e tests, update pipeline example"
```

---

## Task 8: Final Verification + Documentation Update

**Files:**
- Modify: `docs/plans/2026-02-25-mvp-roadmap.md`

**Step 1: Full verification**

```bash
# All tests (skip GPU)
uv run pytest tests/ -m "not gpu" -v --tb=short
# Expected: ~215+ tests passing

# Lint
uv run ruff check src/ tests/
# Expected: 0 errors

# Type check
uv run mypy
# Expected: 0 issues

# Phase 5 tests only
uv run pytest tests/test_observability.py tests/test_vllm_backend.py tests/test_phase5_e2e.py -v
```

**Step 2: Update roadmap**

In `docs/plans/2026-02-25-mvp-roadmap.md`, Phase 5 section:
- Change `**状态：** ⬜ 待设计` to `**状态：** ✅ 已完成 (2026-03-02)`
- Add verification results and output files list

**Step 3: Final commit**

```bash
git add docs/plans/2026-02-25-mvp-roadmap.md
git commit -m "docs: mark Phase 5 complete in MVP roadmap"
```

---

## Commit Sequence Summary

```
1. feat(observability): add NervaMetrics with isolated registry support
2. feat(observability): implement configure_logging with structlog; add tests
3. feat(batcher): wire optional metrics into DynamicBatcher
4. feat(server): add /metrics endpoint; instrument RpcHandler with metrics + structlog
5. feat(worker): add structlog to infer hot path; add worker_status metrics to manager
6. feat(backends): implement VLLMBackend with conditional vllm import
7. feat(phase5): update exports, add e2e tests, update pipeline example
8. docs: mark Phase 5 complete in MVP roadmap
```

---

## Risk Mitigations

| Risk | Mitigation |
|---|---|
| `prometheus_client` `_value.get()` is private API | Acceptable for tests; stable in >=0.21; alternative is `generate_latest()` text parsing |
| `@register_backend("vllm")` called multiple times in tests | Never del+reimport `nerva.backends.vllm` in tests; registration happens once on first import |
| `structlog.configure()` + `cache_logger_on_first_use=True` breaks test ordering | Use `False` in `configure_logging()` for test safety |
| mypy strict on vllm types | `self._engine: Any \| None`; `vllm` already in `ignore_missing_imports` overrides |
| `batcher.py` enqueue_time field breaks existing tests | Add `enqueue_time: float = field(default_factory=time.monotonic)` to `_PendingRequest`; default avoids breaking existing construction |
