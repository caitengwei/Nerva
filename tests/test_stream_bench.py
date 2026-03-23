"""Streaming performance benchmarks (SB1-SB5).

Run with: uv run pytest tests/test_stream_bench.py -v -m slow
Results saved to bench-results/stream/
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import platform
import subprocess
import time
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import httpx
import msgpack
import pytest

from nerva import model, trace
from nerva.backends.base import InferContext
from nerva.core.graph import Graph, Node
from nerva.engine.executor import Executor
from nerva.server.app import build_app
from nerva.server.protocol import Frame, FrameType, encode_frame
from nerva.server.serve import _build_pipelines
from nerva.worker.manager import WorkerManager
from tests.helpers import BenchStreamingModel

BENCH_DIR = Path("bench-results/stream")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _env_info() -> dict[str, str]:
    sha = "unknown"
    with contextlib.suppress(Exception):
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "commit_sha": sha,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _save_result(name: str, data: dict[str, Any]) -> None:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    path = BENCH_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2))
    print(f"\n[bench] saved → {path}")


def _percentiles(data: list[float]) -> dict[str, float]:
    sorted_data = sorted(data)
    n = len(sorted_data)
    p50 = median(sorted_data)
    p90_idx = min(int(n * 0.90), n - 1)
    p95_idx = min(int(n * 0.95), n - 1)
    p99_idx = min(int(n * 0.99), n - 1)
    return {
        "p50": round(p50, 3),
        "p90": round(sorted_data[p90_idx], 3),
        "p95": round(sorted_data[p95_idx], 3),
        "p99": round(sorted_data[p99_idx], 3),
    }


def _make_request_body(pipeline: str, inputs: dict[str, Any], request_id: int = 1) -> bytes:
    """Build OPEN + DATA + END frame bytes for RPC requests."""
    frames = b""
    frames += encode_frame(Frame(FrameType.OPEN, request_id, 0, msgpack.packb({"pipeline": pipeline})))
    frames += encode_frame(Frame(FrameType.DATA, request_id, 0, msgpack.packb(inputs, use_bin_type=True)))
    frames += encode_frame(Frame(FrameType.END, request_id, 0, b""))
    return frames


def _stream_headers(mode: str = "1", deadline_offset_ms: int = 60000) -> dict[str, str]:
    deadline = int(time.time() * 1000) + deadline_offset_ms
    return {
        "content-type": "application/x-nerva-rpc",
        "x-nerva-deadline-ms": str(deadline),
        "x-nerva-stream": mode,
    }


# ---------------------------------------------------------------------------
# SB1: Executor execute_stream scheduling overhead
# ---------------------------------------------------------------------------


class _MockStreamProxy:
    """Minimal InferableStreamProxy with zero-delay chunks."""

    def __init__(self, n_chunks: int = 5) -> None:
        self._n_chunks = n_chunks

    async def infer(
        self, inputs: dict[str, Any], context: InferContext, **kwargs: Any
    ) -> dict[str, Any]:
        return {}

    async def infer_stream(
        self, inputs: dict[str, Any], context: InferContext, **kwargs: Any
    ) -> Any:
        for i in range(self._n_chunks):
            yield {"tok": i}


@pytest.mark.slow
class TestSB1ExecutorStreamOverhead:
    """SB1: execute_stream() scheduling overhead — pure mock proxy, no Worker."""

    async def test_executor_stream_overhead(self) -> None:
        g = Graph()
        g.add_node(Node(id="model", model_name="model"))
        proxy = _MockStreamProxy(n_chunks=5)
        proxies: dict[str, Any] = {"model": proxy}

        n_iters = 100
        overhead_us_list: list[float] = []

        for i in range(n_iters):
            ctx = InferContext(request_id=f"sb1-{i}", deadline_ms=30000)
            executor = Executor(g, proxies, ctx)
            t0 = time.perf_counter_ns()
            async for _ in executor.execute_stream({}):
                pass
            elapsed_us = (time.perf_counter_ns() - t0) / 1000.0
            overhead_us_list.append(elapsed_us)

        avg_us = mean(overhead_us_list)
        std_us = stdev(overhead_us_list)

        result = {
            "benchmark": "SB1_executor_stream_overhead",
            "description": "execute_stream() total time with zero-delay mock proxy (5 chunks)",
            "iterations": n_iters,
            "avg_us": round(avg_us, 2),
            "std_us": round(std_us, 2),
            "min_us": round(min(overhead_us_list), 2),
            "max_us": round(max(overhead_us_list), 2),
            "env": _env_info(),
        }
        _save_result("SB1_executor_stream_overhead", result)
        print(f"\n[SB1] execute_stream overhead: avg={avg_us:.1f}us, std={std_us:.1f}us")


# ---------------------------------------------------------------------------
# SB2: WorkerProxy infer_stream IPC latency consistency
# ---------------------------------------------------------------------------

_SB2_COUNT = 100
_SB2_DELAY_MS = 30
_SB2_THEORETICAL_MS = _SB2_COUNT * _SB2_DELAY_MS  # 3000ms


@pytest.mark.slow
class TestSB2WorkerProxyStreamLatency:
    """SB2: WorkerProxy.infer_stream() latency consistency — real Worker, realistic delay.

    Measures how close actual stream latency is to theoretical (count * delay_ms).
    The gap reveals IPC + serialization overhead under realistic conditions.
    """

    async def test_worker_proxy_stream_latency(self) -> None:
        handle = model(
            "sb2-bench-stream",
            BenchStreamingModel,
            backend="pytorch",
            device="cpu",
            count=_SB2_COUNT,
            chunk_size=1024,
            delay_ms=_SB2_DELAY_MS,
        )
        manager = WorkerManager()
        try:
            proxy = await manager.start_worker(handle)

            n_iters = 5
            latencies_ms: list[float] = []

            # Warmup
            ctx = InferContext(request_id="sb2-warmup", deadline_ms=60000)
            async for _ in proxy.infer_stream({}, ctx):
                pass

            for i in range(n_iters):
                ctx = InferContext(request_id=f"sb2-{i}", deadline_ms=60000)
                t0 = time.perf_counter()
                async for _ in proxy.infer_stream({}, ctx):
                    pass
                elapsed_ms = (time.perf_counter() - t0) * 1000
                latencies_ms.append(elapsed_ms)

            pcts = _percentiles(latencies_ms)
            avg_ms = mean(latencies_ms)
            overhead_ms = round(avg_ms - _SB2_THEORETICAL_MS, 2)

            result = {
                "benchmark": "SB2_worker_proxy_stream_latency",
                "description": (
                    f"WorkerProxy.infer_stream() serial, count={_SB2_COUNT}, "
                    f"chunk_size=1024, delay_ms={_SB2_DELAY_MS}"
                ),
                "iterations": n_iters,
                "theoretical_ms": _SB2_THEORETICAL_MS,
                "avg_ms": round(avg_ms, 1),
                "overhead_ms": overhead_ms,
                "latency_ms": pcts,
                "env": _env_info(),
            }
            _save_result("SB2_worker_proxy_stream_latency", result)
            print(
                f"\n[SB2] IPC latency: avg={avg_ms:.0f}ms (theoretical={_SB2_THEORETICAL_MS}ms, "
                f"overhead={overhead_ms:+.1f}ms) p95={pcts['p95']:.0f}ms"
            )
        finally:
            await manager.shutdown_all()


# ---------------------------------------------------------------------------
# SB3: E2E single-stream latency (TTFT + full stream)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSB3E2EStreamLatency:
    """SB3: E2E execute_stream() TTFT and full-stream latency — real Worker, realistic delay.

    TTFT ≈ first delay_ms (time to receive first chunk through IPC + executor).
    Full stream ≈ count * delay_ms (theoretical). Gap shows E2E stack overhead.
    """

    async def test_e2e_stream_latency(self) -> None:
        handle = model(
            "sb3-bench-stream",
            BenchStreamingModel,
            backend="pytorch",
            device="cpu",
            count=100,
            chunk_size=1024,
            delay_ms=30,
        )
        graph = trace(lambda inp: handle(inp))
        manager = WorkerManager()
        try:
            executors, _ = await _build_pipelines({"sb3": graph}, manager)
            pipeline_executor = executors["sb3"]

            n_iters = 5
            ttft_ms_list: list[float] = []
            full_ms_list: list[float] = []

            # Warmup
            async for _ in pipeline_executor.execute_stream({}, deadline_ms=60000, request_id="sb3-warmup"):
                pass

            for i in range(n_iters):
                t_start = time.perf_counter()
                t_first: float | None = None
                async for _ in pipeline_executor.execute_stream(
                    {}, deadline_ms=60000, request_id=f"sb3-{i}"
                ):
                    if t_first is None:
                        t_first = time.perf_counter()
                t_end = time.perf_counter()

                ttft_ms_list.append(((t_first or t_end) - t_start) * 1000)
                full_ms_list.append((t_end - t_start) * 1000)

            ttft_pcts = _percentiles(ttft_ms_list)
            full_pcts = _percentiles(full_ms_list)

            result = {
                "benchmark": "SB3_e2e_stream_latency",
                "description": "E2E execute_stream() serial, count=100, chunk_size=1024, delay_ms=30",
                "iterations": n_iters,
                "theoretical_ttft_ms": 30,
                "theoretical_full_ms": 3000,
                "ttft_ms": {
                    "avg": round(mean(ttft_ms_list), 1),
                    **ttft_pcts,
                },
                "full_stream_ms": {
                    "avg": round(mean(full_ms_list), 1),
                    **full_pcts,
                },
                "env": _env_info(),
            }
            _save_result("SB3_e2e_stream_latency", result)
            print(
                f"\n[SB3] TTFT: avg={mean(ttft_ms_list):.1f}ms (theoretical=30ms) | "
                f"Full: avg={mean(full_ms_list):.0f}ms (theoretical=3000ms)"
            )
        finally:
            await manager.shutdown_all()


# ---------------------------------------------------------------------------
# SB4: HTTP concurrent streams — output streaming, auto bottleneck detection
# ---------------------------------------------------------------------------

_C_SEQUENCE = [1, 2, 4, 8, 16, 32, 64, 128]
_THEORETICAL_LATENCY_MS = 3000.0    # count=100, delay_ms=30
_BOTTLENECK_THRESHOLD_MS = 4500.0   # 1.5x theoretical
_ROUNDS_PER_C = 3


async def _one_stream_request(
    client: httpx.AsyncClient,
    body: bytes,
    headers: dict[str, str],
    url: str,
) -> float:
    """Issue one streaming POST and return total latency in ms."""
    t0 = time.perf_counter()
    await client.post(url, content=body, headers=headers)
    return (time.perf_counter() - t0) * 1000


@pytest.mark.slow
class TestSB4HttpConcurrentStreams:
    """SB4: x-nerva-stream=1, escalating concurrency until p90 > 4500ms."""

    async def test_http_concurrent_streams(self) -> None:
        handle = model(
            "sb4-bench-stream",
            BenchStreamingModel,
            backend="pytorch",
            device="cpu",
            count=100,
            chunk_size=1024,
            delay_ms=30,
        )
        graph = trace(lambda inp: handle(inp))
        manager = WorkerManager()
        try:
            executors, model_info = await _build_pipelines({"sb4": graph}, manager)
            app = build_app(pipelines=executors, model_info=model_info)
            transport = httpx.ASGITransport(app=app)

            body = _make_request_body("sb4", {"value": "x"})
            results: list[dict[str, Any]] = []
            bottleneck_c: int | None = None

            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver", timeout=120.0
            ) as client:
                # Warmup with C=1
                headers = _stream_headers("1", deadline_offset_ms=120000)
                await _one_stream_request(client, body, headers, "/rpc/sb4")

                for c in _C_SEQUENCE:
                    round_latencies: list[float] = []
                    round_qps_list: list[float] = []
                    headers = _stream_headers("1", deadline_offset_ms=120000)

                    for _ in range(_ROUNDS_PER_C):
                        tasks = [
                            _one_stream_request(client, body, headers, "/rpc/sb4")
                            for _ in range(c)
                        ]
                        t_round = time.perf_counter()
                        latencies = await asyncio.gather(*tasks)
                        wall_clock_s = time.perf_counter() - t_round
                        round_latencies.extend(latencies)
                        round_qps_list.append(c / (wall_clock_s + 1e-9))

                    pcts = _percentiles(round_latencies)
                    qps = round(mean(round_qps_list), 2)

                    entry = {
                        "c": c,
                        "p50_ms": pcts["p50"],
                        "p90_ms": pcts["p90"],
                        "p99_ms": pcts["p99"],
                        "qps": qps,
                    }
                    results.append(entry)
                    print(
                        f"\n[SB4] C={c:3d}: p50={pcts['p50']:.0f}ms "
                        f"p90={pcts['p90']:.0f}ms p99={pcts['p99']:.0f}ms qps={qps}"
                    )

                    if pcts["p90"] > _BOTTLENECK_THRESHOLD_MS:
                        bottleneck_c = c
                        print(
                            f"[SB4] Bottleneck detected at C={c} "
                            f"(p90={pcts['p90']:.0f}ms > {_BOTTLENECK_THRESHOLD_MS:.0f}ms)"
                        )
                        break

            result = {
                "benchmark": "SB4_http_concurrent_streams",
                "stream_mode": "1",
                "theoretical_latency_ms": _THEORETICAL_LATENCY_MS,
                "bottleneck_threshold_ms": _BOTTLENECK_THRESHOLD_MS,
                "bottleneck_c": bottleneck_c,
                "rounds_per_c": _ROUNDS_PER_C,
                "results": results,
                "env": _env_info(),
            }
            _save_result("SB4_http_concurrent_streams", result)
        finally:
            await manager.shutdown_all()


# ---------------------------------------------------------------------------
# SB5: HTTP full-duplex large body — x-nerva-stream=2
# ---------------------------------------------------------------------------

_SB5_INPUT_PAYLOAD_BYTES = 65536  # 64 KB


@pytest.mark.slow
class TestSB5HttpFullDuplexLargeBody:
    """SB5: x-nerva-stream=2, 64KB body, escalating concurrency until p90 > 4500ms."""

    async def test_http_fullduplex_large_body(self) -> None:
        handle = model(
            "sb5-bench-stream",
            BenchStreamingModel,
            backend="pytorch",
            device="cpu",
            count=100,
            chunk_size=1024,
            delay_ms=30,
        )
        graph = trace(lambda inp: handle(inp))
        manager = WorkerManager()
        try:
            executors, model_info = await _build_pipelines({"sb5": graph}, manager)
            app = build_app(pipelines=executors, model_info=model_info)
            transport = httpx.ASGITransport(app=app)

            # 64KB input payload embedded in DATA frame
            large_input = {"payload": b"x" * _SB5_INPUT_PAYLOAD_BYTES}
            body = _make_request_body("sb5", large_input)
            results: list[dict[str, Any]] = []
            bottleneck_c: int | None = None

            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver", timeout=120.0
            ) as client:
                # Warmup
                headers = _stream_headers("2", deadline_offset_ms=120000)
                await _one_stream_request(client, body, headers, "/rpc/sb5")

                for c in _C_SEQUENCE:
                    round_latencies: list[float] = []
                    round_qps_list: list[float] = []
                    headers = _stream_headers("2", deadline_offset_ms=120000)

                    for _ in range(_ROUNDS_PER_C):
                        tasks = [
                            _one_stream_request(client, body, headers, "/rpc/sb5")
                            for _ in range(c)
                        ]
                        t_round = time.perf_counter()
                        latencies = await asyncio.gather(*tasks)
                        wall_clock_s = time.perf_counter() - t_round
                        round_latencies.extend(latencies)
                        round_qps_list.append(c / (wall_clock_s + 1e-9))

                    pcts = _percentiles(round_latencies)
                    qps = round(mean(round_qps_list), 2)

                    entry = {
                        "c": c,
                        "p50_ms": pcts["p50"],
                        "p90_ms": pcts["p90"],
                        "p99_ms": pcts["p99"],
                        "qps": qps,
                    }
                    results.append(entry)
                    print(
                        f"\n[SB5] C={c:3d}: p50={pcts['p50']:.0f}ms "
                        f"p90={pcts['p90']:.0f}ms p99={pcts['p99']:.0f}ms qps={qps}"
                    )

                    if pcts["p90"] > _BOTTLENECK_THRESHOLD_MS:
                        bottleneck_c = c
                        print(
                            f"[SB5] Bottleneck at C={c} "
                            f"(p90={pcts['p90']:.0f}ms > {_BOTTLENECK_THRESHOLD_MS:.0f}ms)"
                        )
                        break

            result = {
                "benchmark": "SB5_http_fullduplex_large_body",
                "stream_mode": "2",
                "input_payload_bytes": _SB5_INPUT_PAYLOAD_BYTES,
                "theoretical_latency_ms": _THEORETICAL_LATENCY_MS,
                "bottleneck_threshold_ms": _BOTTLENECK_THRESHOLD_MS,
                "bottleneck_c": bottleneck_c,
                "rounds_per_c": _ROUNDS_PER_C,
                "results": results,
                "env": _env_info(),
            }
            _save_result("SB5_http_fullduplex_large_body", result)
        finally:
            await manager.shutdown_all()
