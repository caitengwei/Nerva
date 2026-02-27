"""Phase 2 DAG Benchmark tests."""

from __future__ import annotations

import contextlib
import json
import platform
import subprocess
import time
from pathlib import Path
from statistics import mean, median, stdev
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

from nerva import model, trace
from nerva.backends.base import InferContext
from nerva.core.graph import Edge, Graph, Node
from nerva.core.primitives import parallel
from nerva.engine.executor import Executor
from nerva.engine.shm_pool import ShmPool, ShmSlot
from nerva.worker.manager import WorkerManager
from tests.helpers import (
    BenchClassifier,
    BenchFusionModel,
    BenchImageEncoder,
    BenchTextEncoder,
)

BENCH_DIR = Path("bench-results/phase2")


class _CountingShmPool(ShmPool):
    """ShmPool with allocation counter for benchmark observability."""

    def __init__(self) -> None:
        super().__init__()
        self.alloc_calls = 0

    def alloc(self, size: int) -> ShmSlot:
        self.alloc_calls += 1
        return super().alloc(size)


class _ShmAwareProxy:
    """Wrapper that always passes shm_pool to WorkerProxy.infer()."""

    def __init__(self, proxy: Any, shm_pool: ShmPool) -> None:
        self._proxy = proxy
        self._shm_pool = shm_pool

    async def infer(
        self,
        inputs: dict[str, Any],
        context: InferContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._proxy.infer(inputs, context, shm_pool=self._shm_pool)


def _env_info() -> dict[str, str]:
    """Collect environment metadata for benchmark results."""
    sha = "unknown"
    with contextlib.suppress(Exception):
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "commit_sha": sha,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _save_result(name: str, data: dict[str, Any]) -> None:
    """Save benchmark result as JSON."""
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    path = BENCH_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2))


class TestBenchModels:
    """Verify benchmark models produce correct output shapes."""

    async def test_image_encoder_output_shape(self) -> None:
        m = BenchImageEncoder()
        m._options = {"dim": 768, "delay_ms": 0}
        m.load()
        result = await m.infer({"image_bytes": b"\x00" * 4096})
        assert "features" in result
        assert len(result["features"]) == 768
        assert all(isinstance(v, float) for v in result["features"])

    async def test_text_encoder_output_shape(self) -> None:
        m = BenchTextEncoder()
        m._options = {"dim": 768, "delay_ms": 0}
        m.load()
        result = await m.infer({"text": "hello world"})
        assert "features" in result
        assert len(result["features"]) == 768

    async def test_fusion_model_output_shape(self) -> None:
        m = BenchFusionModel()
        m._options = {"dim": 512, "delay_ms": 0}
        m.load()
        result = await m.infer({
            "img_features": [1.0] * 768,
            "txt_features": [2.0] * 768,
        })
        assert "fused_features" in result
        assert len(result["fused_features"]) == 512

    async def test_classifier_output(self) -> None:
        m = BenchClassifier()
        m._options = {"delay_ms": 0}
        m.load()
        result = await m.infer({"fused_features": [1.0] * 512})
        assert "label" in result
        assert "score" in result
        assert isinstance(result["label"], str)
        assert isinstance(result["score"], float)


@pytest.mark.slow
class TestB1TraceOverhead:
    """B1: trace() overhead — 1000 iterations, pure CPU, no Workers."""

    def test_trace_overhead(self) -> None:
        img_enc = model("img_enc", BenchImageEncoder, backend="pytorch", device="cpu")
        txt_enc = model("txt_enc", BenchTextEncoder, backend="pytorch", device="cpu")
        fusion = model("fusion", BenchFusionModel, backend="pytorch", device="cpu")
        classifier = model("classifier", BenchClassifier, backend="pytorch", device="cpu")

        def pipeline(x: object) -> object:
            img_out, txt_out = parallel(
                lambda: img_enc(x),
                lambda: txt_enc(x),
            )
            fused = fusion({
                "img_features": img_out["features"],
                "txt_features": txt_out["features"],
            })
            return classifier({"fused_features": fused["fused_features"]})

        n_iters = 1000
        times_us: list[float] = []

        for _ in range(n_iters):
            t0 = time.perf_counter_ns()
            trace(pipeline)
            t1 = time.perf_counter_ns()
            times_us.append((t1 - t0) / 1000.0)

        avg_us = mean(times_us)
        std_us = stdev(times_us)

        result = {
            "benchmark": "B1_trace_overhead",
            "iterations": n_iters,
            "avg_us": round(avg_us, 2),
            "std_us": round(std_us, 2),
            "min_us": round(min(times_us), 2),
            "max_us": round(max(times_us), 2),
            "env": _env_info(),
        }
        _save_result("B1_trace_overhead", result)

        print(f"\n[B1] trace() overhead: avg={avg_us:.1f}us, std={std_us:.1f}us")

        # Sanity gate: < 100us per trace
        assert avg_us < 100, f"trace() too slow: {avg_us:.1f}us > 100us"


@pytest.mark.slow
class TestB2ExecutorOverhead:
    """B2: executor scheduling overhead — delay=0 models, total - infer time."""

    async def test_executor_overhead(self) -> None:
        total_infer_ns = 0

        def _make_timed_proxy(
            result_factory: Callable[[], dict[str, Any]],
        ) -> AsyncMock:
            async def _infer(inputs: Any, context: Any, **kwargs: Any) -> dict[str, Any]:
                nonlocal total_infer_ns
                t0 = time.perf_counter_ns()
                result = result_factory()
                total_infer_ns += time.perf_counter_ns() - t0
                return result

            proxy = AsyncMock()
            proxy.infer = AsyncMock(side_effect=_infer)
            return proxy

        # Build the 4-node multimodal DAG (no parallel node — flat graph).
        g = Graph()
        g.add_node(Node(id="img_enc", model_name="img_enc"))
        g.add_node(Node(id="txt_enc", model_name="txt_enc"))
        g.add_node(Node(id="fusion", model_name="fusion"))
        g.add_node(Node(id="classifier", model_name="classifier"))

        g.add_edge(Edge(
            src="img_enc", dst="fusion",
            src_field_path=("features",), dst_input_key="img_features",
        ))
        g.add_edge(Edge(
            src="txt_enc", dst="fusion",
            src_field_path=("features",), dst_input_key="txt_features",
        ))
        g.add_edge(Edge(
            src="fusion", dst="classifier",
            src_field_path=("fused_features",), dst_input_key="fused_features",
        ))

        n_iters = 100
        overhead_us_list: list[float] = []

        for _ in range(n_iters):
            total_infer_ns = 0

            proxies: dict[str, Any] = {
                "img_enc": _make_timed_proxy(lambda: {"features": [0.1] * 768}),
                "txt_enc": _make_timed_proxy(lambda: {"features": [0.1] * 768}),
                "fusion": _make_timed_proxy(lambda: {"fused_features": [0.3] * 512}),
                "classifier": _make_timed_proxy(lambda: {"label": "cat", "score": 0.95}),
            }
            ctx = InferContext(request_id="bench-b2", deadline_ms=30000)
            executor = Executor(g, proxies, ctx)

            t0 = time.perf_counter_ns()
            await executor.execute({"image_bytes": b"\x00", "text": "hello"})
            total_ns = time.perf_counter_ns() - t0

            overhead_ns = total_ns - total_infer_ns
            overhead_us_list.append(overhead_ns / 1000.0)

        avg_us = mean(overhead_us_list)
        std_us = stdev(overhead_us_list)

        result = {
            "benchmark": "B2_executor_overhead",
            "iterations": n_iters,
            "avg_us": round(avg_us, 2),
            "std_us": round(std_us, 2),
            "min_us": round(min(overhead_us_list), 2),
            "max_us": round(max(overhead_us_list), 2),
            "env": _env_info(),
        }
        _save_result("B2_executor_overhead", result)

        print(f"\n[B2] executor overhead: avg={avg_us:.1f}us, std={std_us:.1f}us")

        # Sanity gate: < 5ms (5000us)
        assert avg_us < 5000, f"executor overhead too high: {avg_us:.1f}us > 5000us"


@pytest.mark.slow
class TestB3ParallelSpeedup:
    """B3: parallel concurrency gain — delay=10ms models, parallel vs sequential."""

    async def test_parallel_speedup(self) -> None:
        delay_s = 0.01  # 10ms

        async def delayed_infer(inputs: Any, context: Any, **kwargs: Any) -> dict[str, Any]:
            import asyncio

            await asyncio.sleep(delay_s)
            return {"features": [0.1] * 768}

        # -- Sequential: img_enc -> txt_enc (linear chain) --
        g_seq = Graph()
        g_seq.add_node(Node(id="img_enc", model_name="img_enc"))
        g_seq.add_node(Node(id="txt_enc", model_name="txt_enc"))
        g_seq.add_edge(Edge(src="img_enc", dst="txt_enc"))

        # -- Parallel: img_enc || txt_enc --
        branch_img = Graph()
        branch_img.add_node(Node(id="img_enc", model_name="img_enc"))
        branch_txt = Graph()
        branch_txt.add_node(Node(id="txt_enc", model_name="txt_enc"))

        g_par = Graph()
        par_node = Node(
            id="par_1", model_name="parallel", node_type="parallel",
            branches=[branch_img, branch_txt],
        )
        g_par.add_node(par_node)

        n_iters = 50
        seq_times: list[float] = []
        par_times: list[float] = []

        mock_img_seq = AsyncMock()
        mock_img_seq.infer = AsyncMock(side_effect=delayed_infer)
        mock_txt_seq = AsyncMock()
        mock_txt_seq.infer = AsyncMock(side_effect=delayed_infer)
        proxies_seq: dict[str, Any] = {"img_enc": mock_img_seq, "txt_enc": mock_txt_seq}

        for _ in range(n_iters):
            ctx = InferContext(request_id="bench-b3-seq", deadline_ms=30000)
            executor = Executor(g_seq, proxies_seq, ctx)
            t0 = time.perf_counter()
            await executor.execute({"image_bytes": b"\x00"})
            seq_times.append(time.perf_counter() - t0)

        mock_img_par = AsyncMock()
        mock_img_par.infer = AsyncMock(side_effect=delayed_infer)
        mock_txt_par = AsyncMock()
        mock_txt_par.infer = AsyncMock(side_effect=delayed_infer)
        proxies_par: dict[str, Any] = {"img_enc": mock_img_par, "txt_enc": mock_txt_par}

        for _ in range(n_iters):
            ctx = InferContext(request_id="bench-b3-par", deadline_ms=30000)
            executor = Executor(g_par, proxies_par, ctx)
            t0 = time.perf_counter()
            await executor.execute({"image_bytes": b"\x00"})
            par_times.append(time.perf_counter() - t0)

        avg_seq = mean(seq_times)
        avg_par = mean(par_times)
        speedup = avg_seq / avg_par

        result = {
            "benchmark": "B3_parallel_speedup",
            "iterations": n_iters,
            "delay_ms": delay_s * 1000,
            "avg_sequential_ms": round(avg_seq * 1000, 2),
            "avg_parallel_ms": round(avg_par * 1000, 2),
            "speedup_ratio": round(speedup, 2),
            "env": _env_info(),
        }
        _save_result("B3_parallel_speedup", result)

        print(
            f"\n[B3] parallel speedup: seq={avg_seq * 1000:.1f}ms, "
            f"par={avg_par * 1000:.1f}ms, speedup={speedup:.2f}x"
        )

        # Sanity gate: > 1.5x speedup
        assert speedup > 1.5, f"parallel speedup too low: {speedup:.2f}x < 1.5x"


def _percentiles(data: list[float]) -> dict[str, float]:
    """Compute p50/p95/p99 from a list of values."""
    sorted_data = sorted(data)
    n = len(sorted_data)
    p50 = median(sorted_data)
    p95_idx = min(int(n * 0.95), n - 1)
    p99_idx = min(int(n * 0.99), n - 1)
    return {
        "p50": round(p50, 3),
        "p95": round(sorted_data[p95_idx], 3),
        "p99": round(sorted_data[p99_idx], 3),
    }


@pytest.mark.slow
class TestB4E2EPipeline:
    """B4: end-to-end pipeline — Small + Large payloads, 100 iterations each."""

    async def _run_pipeline(
        self,
        *,
        dim: int,
        delay_ms: float,
        image_bytes_size: int,
        text: str,
        n_iters: int,
    ) -> tuple[list[float], int]:
        """Run full 4-model DAG and return (latencies_ms, shm_alloc_calls)."""
        h_img = model("img_enc", BenchImageEncoder, backend="pytorch", device="cpu", dim=dim, delay_ms=delay_ms)
        h_txt = model("txt_enc", BenchTextEncoder, backend="pytorch", device="cpu", dim=dim, delay_ms=delay_ms)
        h_fusion = model("fusion", BenchFusionModel, backend="pytorch", device="cpu", dim=dim, delay_ms=delay_ms)
        h_cls = model("classifier", BenchClassifier, backend="pytorch", device="cpu", delay_ms=delay_ms)

        manager = WorkerManager()
        shm_pool = _CountingShmPool()
        try:
            p_img = await manager.start_worker(h_img)
            p_txt = await manager.start_worker(h_txt)
            p_fusion = await manager.start_worker(h_fusion)
            p_cls = await manager.start_worker(h_cls)

            # Build DAG: parallel(img, txt) -> fusion -> classifier
            branch_img = Graph()
            branch_img.add_node(Node(id="img_enc", model_name="img_enc"))
            branch_txt = Graph()
            branch_txt.add_node(Node(id="txt_enc", model_name="txt_enc"))

            g = Graph()
            par_node = Node(
                id="par_1", model_name="parallel", node_type="parallel",
                branches=[branch_img, branch_txt],
            )
            g.add_node(par_node)
            g.add_node(Node(id="fusion", model_name="fusion"))
            g.add_node(Node(id="classifier", model_name="classifier"))

            g.add_edge(Edge(src="par_1", dst="fusion", src_field_path=("0", "features"), dst_input_key="img_features"))
            g.add_edge(Edge(src="par_1", dst="fusion", src_field_path=("1", "features"), dst_input_key="txt_features"))
            g.add_edge(Edge(src="fusion", dst="classifier", src_field_path=("fused_features",), dst_input_key="fused_features"))

            proxies: dict[str, Any] = {
                "img_enc": _ShmAwareProxy(p_img, shm_pool),
                "txt_enc": _ShmAwareProxy(p_txt, shm_pool),
                "fusion": _ShmAwareProxy(p_fusion, shm_pool),
                "classifier": _ShmAwareProxy(p_cls, shm_pool),
            }

            pipeline_input: dict[str, Any] = {
                "image_bytes": b"\x00" * image_bytes_size,
                "text": text,
            }

            # Warmup
            ctx = InferContext(request_id="warmup", deadline_ms=30000)
            executor = Executor(g, proxies, ctx)
            await executor.execute(pipeline_input)

            latencies_ms: list[float] = []
            for i in range(n_iters):
                ctx = InferContext(request_id=f"b4-{i}", deadline_ms=30000)
                executor = Executor(g, proxies, ctx)
                t0 = time.perf_counter()
                result = await executor.execute(pipeline_input)
                latencies_ms.append((time.perf_counter() - t0) * 1000)

                # Verify correctness.
                assert "label" in result
                assert "score" in result

            return latencies_ms, shm_pool.alloc_calls
        finally:
            await manager.shutdown_all()
            shm_pool.close()

    async def test_e2e_small_payload(self) -> None:
        """Small payload: 4KB image, 100B text, dim=768."""
        latencies, shm_alloc_calls = await self._run_pipeline(
            dim=768,
            delay_ms=0,
            image_bytes_size=4096,
            text="a" * 100,
            n_iters=100,
        )

        pcts = _percentiles(latencies)
        result = {
            "benchmark": "B4_e2e_small",
            "iterations": 100,
            "payload": "small",
            "dim": 768,
            "image_bytes_size": 4096,
            "latency_ms": pcts,
            "avg_ms": round(mean(latencies), 3),
            "shm_alloc_calls": shm_alloc_calls,
            "env": _env_info(),
        }
        _save_result("B4_e2e_small", result)

        print(f"\n[B4-small] p50={pcts['p50']:.1f}ms, p95={pcts['p95']:.1f}ms, p99={pcts['p99']:.1f}ms")

    async def test_e2e_large_payload(self) -> None:
        """Large payload: 256KB image, 1KB text, dim=1280."""
        latencies, shm_alloc_calls = await self._run_pipeline(
            dim=1280,
            delay_ms=0,
            image_bytes_size=256 * 1024,
            text="a" * 1024,
            n_iters=100,
        )

        pcts = _percentiles(latencies)
        result = {
            "benchmark": "B4_e2e_large",
            "iterations": 100,
            "payload": "large",
            "dim": 1280,
            "image_bytes_size": 256 * 1024,
            "latency_ms": pcts,
            "avg_ms": round(mean(latencies), 3),
            "shm_alloc_calls": shm_alloc_calls,
            "env": _env_info(),
        }
        _save_result("B4_e2e_large", result)

        print(f"\n[B4-large] p50={pcts['p50']:.1f}ms, p95={pcts['p95']:.1f}ms, p99={pcts['p99']:.1f}ms")
        assert shm_alloc_calls > 0, "large payload did not trigger SHM allocations"
