from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class LoadgenResult:
    qps: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    error_rate: float
    max_in_flight: int
    total_requests: int
    error_count: int
    latencies_ms: list[float]


TargetFn = Callable[[dict[str, int], int], Awaitable[tuple[bool, str]]]


def _percentile(sorted_values: list[float], ratio: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(math.ceil(len(sorted_values) * ratio) - 1, len(sorted_values) - 1)
    return sorted_values[max(0, idx)]


async def run_closed_loop(
    target: TargetFn,
    *,
    concurrency: int,
    duration_s: float,
    deadline_ms: int,
) -> LoadgenResult:
    if concurrency <= 0:
        raise ValueError("concurrency must be > 0")
    if duration_s <= 0:
        raise ValueError("duration_s must be > 0")
    if deadline_ms <= 0:
        raise ValueError("deadline_ms must be > 0")

    semaphore = asyncio.Semaphore(concurrency)
    started_at = time.perf_counter()
    stop_at = started_at + duration_s

    latencies_ms: list[float] = []
    total_requests = 0
    error_count = 0
    in_flight = 0
    max_in_flight = 0
    payload_counter = 0

    async def one_request(seq: int) -> None:
        nonlocal total_requests
        nonlocal error_count
        nonlocal in_flight
        nonlocal max_in_flight

        async with semaphore:
            start_ns = time.perf_counter_ns()
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)

            ok = False
            try:
                ok, _err = await target({"seq": seq}, deadline_ms)
            except Exception:
                ok = False
            finally:
                latency = (time.perf_counter_ns() - start_ns) / 1_000_000.0
                latencies_ms.append(latency)
                total_requests += 1
                if not ok:
                    error_count += 1
                in_flight -= 1

    workers: set[asyncio.Task[None]] = set()
    try:
        while time.perf_counter() < stop_at:
            if len(workers) >= concurrency:
                done, pending = await asyncio.wait(workers, return_when=asyncio.FIRST_COMPLETED)
                workers = set(pending)
                # Drain finished tasks exceptions.
                for task in done:
                    task.result()

            payload_counter += 1
            workers.add(asyncio.create_task(one_request(payload_counter)))

        if workers:
            done, _ = await asyncio.wait(workers)
            for task in done:
                task.result()
    finally:
        for task in workers:
            if not task.done():
                task.cancel()

    elapsed_s = max(time.perf_counter() - started_at, 1e-9)
    sorted_lat = sorted(latencies_ms)
    p50 = _percentile(sorted_lat, 0.50)
    p95 = _percentile(sorted_lat, 0.95)
    p99 = _percentile(sorted_lat, 0.99)
    qps = total_requests / elapsed_s
    error_rate = (error_count / total_requests) if total_requests else 0.0

    return LoadgenResult(
        qps=qps,
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        error_rate=error_rate,
        max_in_flight=max_in_flight,
        total_requests=total_requests,
        error_count=error_count,
        latencies_ms=latencies_ms,
    )
