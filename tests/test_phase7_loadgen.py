from __future__ import annotations

import asyncio

from scripts.bench.loadgen import run_closed_loop


async def test_never_exceeds_concurrency_limit() -> None:
    async def target(payload: dict[str, int], deadline_ms: int) -> tuple[bool, str]:
        del payload, deadline_ms
        await asyncio.sleep(0.001)
        return True, ""

    result = await run_closed_loop(
        target,
        concurrency=1000,
        duration_s=0.5,
        deadline_ms=500,
    )
    assert result.max_in_flight <= 1000


async def test_stats_order_and_error_rate() -> None:
    counter = {"n": 0}

    async def target(payload: dict[str, int], deadline_ms: int) -> tuple[bool, str]:
        del payload, deadline_ms
        counter["n"] += 1
        current = counter["n"]
        await asyncio.sleep(0.0005)
        return (current % 4 != 0), "boom"

    result = await run_closed_loop(
        target,
        concurrency=16,
        duration_s=0.3,
        deadline_ms=500,
    )

    assert result.p50_ms <= result.p95_ms <= result.p99_ms
    assert 0.20 <= result.error_rate <= 0.30


async def test_qps_uses_wall_clock_elapsed_after_drain() -> None:
    async def target(payload: dict[str, int], deadline_ms: int) -> tuple[bool, str]:
        del payload, deadline_ms
        await asyncio.sleep(0.05)
        return True, ""

    result = await run_closed_loop(
        target,
        concurrency=1,
        duration_s=0.01,
        deadline_ms=500,
    )
    assert result.total_requests >= 1
    assert result.qps < 80.0
