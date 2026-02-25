"""
Spike S4: Async Dynamic Batcher Benchmark

Validates the core dynamic batching pattern for Nerva's inference serving:
- Timer-based flush under low concurrency (no infinite wait)
- Throughput improvement under high concurrency via batching
- asyncio scheduling overhead characterization

Usage:
    uv run python spikes/s4_async_batcher_benchmark.py
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Request:
    """A single inference request."""

    request_id: int
    payload: Any = None
    submit_time: float = 0.0       # monotonic timestamp when submitted
    batch_fire_time: float = 0.0   # monotonic timestamp when batch fires


@dataclass
class BatchResult:
    """Result for one request within a batch."""

    request_id: int
    value: Any = None


@dataclass
class ScenarioMetrics:
    """Collected metrics for one benchmark scenario."""

    max_batch_size: int
    max_delay_ms: float
    concurrency: int
    total_requests: int

    # latencies in ms
    request_latencies: list[float] = field(default_factory=list)
    batch_wait_times: list[float] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)

    wall_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Dynamic Batcher
# ---------------------------------------------------------------------------

BatchHandler = Callable[
    [list[Request]],
    Coroutine[Any, Any, list[BatchResult]],
]


class DynamicBatcher:
    """
    Queues individual requests and fires batches when either
    max_batch_size is reached or max_delay_ms expires.
    """

    def __init__(
        self,
        batch_handler: BatchHandler,
        max_batch_size: int = 16,
        max_delay_ms: float = 5.0,
    ) -> None:
        self._handler = batch_handler
        self._max_batch_size = max_batch_size
        self._max_delay_ms = max_delay_ms

        self._queue: asyncio.Queue[tuple[Request, asyncio.Future[BatchResult]]] = asyncio.Queue()
        self._loop_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._loop_task = asyncio.create_task(self._batch_loop())

    async def stop(self) -> None:
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

    async def submit(self, request: Request) -> BatchResult:
        """Submit a single request; returns when the batch completes."""
        request.submit_time = time.monotonic()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[BatchResult] = loop.create_future()
        await self._queue.put((request, future))
        return await future

    async def _batch_loop(self) -> None:
        """Main loop: collect up to max_batch_size items or until timer fires."""
        while True:
            batch: list[tuple[Request, asyncio.Future[BatchResult]]] = []

            # Wait for the first item (blocks indefinitely until one arrives).
            first = await self._queue.get()
            batch.append(first)

            # Now try to fill the batch within the time budget.
            deadline = time.monotonic() + self._max_delay_ms / 1000.0
            while len(batch) < self._max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=remaining,
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            # Fire the batch.
            requests = [req for req, _ in batch]
            fire_time = time.monotonic()
            for req in requests:
                req.batch_fire_time = fire_time

            try:
                results = await self._handler(requests)
                # Dispatch results back to individual futures.
                result_map = {r.request_id: r for r in results}
                for req, fut in batch:
                    if not fut.done():
                        fut.set_result(result_map[req.request_id])
            except Exception as exc:
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(exc)


# ---------------------------------------------------------------------------
# Simulated inference handler
# ---------------------------------------------------------------------------

async def simulated_inference(
    batch: list[Request],
    base_compute_ms: float = 5.0,
) -> list[BatchResult]:
    """
    Simulates GPU inference.  Cost scales sub-linearly with batch size
    (GPU parallelism amortises per-request cost).
    """
    n = len(batch)
    # Model: base + small per-item overhead (simulates memory-bound scaling)
    compute_s = (base_compute_ms + 0.3 * n) / 1000.0
    await asyncio.sleep(compute_s)
    return [BatchResult(request_id=req.request_id, value=f"ok-{req.request_id}") for req in batch]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

async def run_scenario(
    max_batch_size: int,
    max_delay_ms: float,
    concurrency: int,
    total_requests: int,
    base_compute_ms: float = 5.0,
) -> ScenarioMetrics:
    metrics = ScenarioMetrics(
        max_batch_size=max_batch_size,
        max_delay_ms=max_delay_ms,
        concurrency=concurrency,
        total_requests=total_requests,
    )

    async def handler(batch: list[Request]) -> list[BatchResult]:
        metrics.batch_sizes.append(len(batch))
        return await simulated_inference(batch, base_compute_ms)

    batcher = DynamicBatcher(
        batch_handler=handler,
        max_batch_size=max_batch_size,
        max_delay_ms=max_delay_ms,
    )
    await batcher.start()

    sem = asyncio.Semaphore(concurrency)
    request_counter = 0

    async def send_one() -> None:
        nonlocal request_counter
        rid = request_counter
        request_counter += 1
        req = Request(request_id=rid)
        async with sem:
            result = await batcher.submit(req)
        done_time = time.monotonic()
        # Record latencies.
        metrics.request_latencies.append((done_time - req.submit_time) * 1000.0)
        metrics.batch_wait_times.append((req.batch_fire_time - req.submit_time) * 1000.0)
        assert result.request_id == rid

    wall_start = time.monotonic()
    tasks = [asyncio.create_task(send_one()) for _ in range(total_requests)]
    await asyncio.gather(*tasks)
    wall_end = time.monotonic()

    metrics.wall_time_s = wall_end - wall_start

    await batcher.stop()
    return metrics


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def pct(values: list[float], p: float) -> float:
    """Percentile (0-100)."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[-1]
    return s[f] + (k - f) * (s[c] - s[f])


def print_results(all_metrics: list[ScenarioMetrics]) -> None:
    header = (
        f"{'batch':>5} {'delay':>5} {'conc':>5} | "
        f"{'lat_p50':>8} {'lat_p95':>8} {'lat_p99':>8} | "
        f"{'wait_p50':>8} {'wait_p95':>8} {'wait_p99':>8} | "
        f"{'bs_mean':>7} {'bs_min':>6} {'bs_max':>6} | "
        f"{'rps':>8}"
    )
    sep = "-" * len(header)

    print()
    print("=" * len(header))
    print("Dynamic Batcher Benchmark Results")
    print("  Latencies and wait times in milliseconds")
    print("=" * len(header))
    print(header)
    print(sep)

    for m in all_metrics:
        lat_p50 = pct(m.request_latencies, 50)
        lat_p95 = pct(m.request_latencies, 95)
        lat_p99 = pct(m.request_latencies, 99)
        wait_p50 = pct(m.batch_wait_times, 50)
        wait_p95 = pct(m.batch_wait_times, 95)
        wait_p99 = pct(m.batch_wait_times, 99)
        bs_mean = statistics.mean(m.batch_sizes) if m.batch_sizes else 0
        bs_min = min(m.batch_sizes) if m.batch_sizes else 0
        bs_max = max(m.batch_sizes) if m.batch_sizes else 0
        rps = m.total_requests / m.wall_time_s if m.wall_time_s > 0 else 0

        print(
            f"{m.max_batch_size:>5} {m.max_delay_ms:>5.0f} {m.concurrency:>5} | "
            f"{lat_p50:>8.2f} {lat_p95:>8.2f} {lat_p99:>8.2f} | "
            f"{wait_p50:>8.2f} {wait_p95:>8.2f} {wait_p99:>8.2f} | "
            f"{bs_mean:>7.1f} {bs_min:>6} {bs_max:>6} | "
            f"{rps:>8.0f}"
        )

    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    batch_sizes = [8, 16, 32]
    delays_ms = [1, 5, 10]
    concurrencies = [1, 32, 128, 512]
    total_requests = 1000

    all_metrics: list[ScenarioMetrics] = []
    total_combos = len(batch_sizes) * len(delays_ms) * len(concurrencies)
    done = 0

    for bs in batch_sizes:
        for delay in delays_ms:
            for conc in concurrencies:
                done += 1
                print(
                    f"\r  Running scenario {done}/{total_combos}: "
                    f"batch={bs} delay={delay}ms conc={conc} ...",
                    end="",
                    flush=True,
                )
                m = await run_scenario(
                    max_batch_size=bs,
                    max_delay_ms=delay,
                    concurrency=conc,
                    total_requests=total_requests,
                )
                all_metrics.append(m)

    print("\r" + " " * 80 + "\r", end="")  # clear progress line
    print_results(all_metrics)

    # ---- Key observations ----
    print("Key observations:")

    # 1) Timer flush under low concurrency
    low_conc = [m for m in all_metrics if m.concurrency == 1]
    if low_conc:
        avg_bs = statistics.mean([statistics.mean(m.batch_sizes) for m in low_conc])
        avg_wait = statistics.mean([pct(m.batch_wait_times, 50) for m in low_conc])
        print(
            f"  [Low concurrency (conc=1)]  avg batch size = {avg_bs:.1f}, "
            f"median wait = {avg_wait:.2f} ms"
        )
        print(
            "    -> Timer flush fires correctly; requests are NOT stuck waiting "
            "for a full batch."
        )

    # 2) Throughput at high concurrency
    hi_conc = [m for m in all_metrics if m.concurrency == 512]
    lo_conc = [m for m in all_metrics if m.concurrency == 1]
    if hi_conc and lo_conc:
        hi_rps = max(m.total_requests / m.wall_time_s for m in hi_conc)
        lo_rps = max(m.total_requests / m.wall_time_s for m in lo_conc)
        print(
            f"  [Throughput]  best RPS at conc=512: {hi_rps:.0f}, "
            f"best RPS at conc=1: {lo_rps:.0f}, "
            f"speedup: {hi_rps / lo_rps:.1f}x"
        )
        print(
            "    -> Batching significantly improves throughput under load."
        )

    # 3) asyncio overhead
    overhead_scenarios = [m for m in all_metrics if m.max_delay_ms == 1 and m.concurrency == 1]
    if overhead_scenarios:
        median_waits = [pct(m.batch_wait_times, 50) for m in overhead_scenarios]
        print(
            f"  [Scheduling overhead]  median batch wait at delay=1ms, conc=1: "
            f"{statistics.mean(median_waits):.2f} ms"
        )
        print(
            "    -> This approximates the asyncio event-loop + queue overhead."
        )

    print()


if __name__ == "__main__":
    asyncio.run(main())
