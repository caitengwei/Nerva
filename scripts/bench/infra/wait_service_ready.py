from __future__ import annotations

import argparse
import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence

import httpx

StatusGetter = Callable[[str], Awaitable[int]]


async def _default_getter(url: str) -> int:
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, timeout=2.0)
        return resp.status_code


async def wait_service_ready(
    *,
    kind: str,
    url: str,
    timeout_seconds: float,
    interval_seconds: float = 1.0,
    getter: StatusGetter | None = None,
) -> bool:
    if kind not in {"vllm", "triton"}:
        raise ValueError("kind must be one of: vllm, triton")

    probe = getter or _default_getter
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        try:
            status = await probe(url)
            if status == 200:
                return True
        except Exception:
            pass
        await asyncio.sleep(interval_seconds)

    return False


def _cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait until service is ready")
    parser.add_argument("--kind", choices=["vllm", "triton"], required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    return parser.parse_args(argv)


async def _amain(argv: Sequence[str] | None = None) -> int:
    args = _cli(argv)
    ok = await wait_service_ready(
        kind=args.kind,
        url=args.url,
        timeout_seconds=args.timeout_seconds,
        interval_seconds=args.interval_seconds,
    )
    if ok:
        print(f"{args.kind} ready: {args.url}")
        return 0

    print(f"timeout waiting for {args.kind}: {args.url}")
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(_amain(argv))


if __name__ == "__main__":
    raise SystemExit(main())
