"""Tests for per-node multi-instance Worker support.

Covers:
- ModelHandle.instances field default and setting
- model() instances parameter and validation
- MultiInstanceProxy round-robin dispatch
- MultiInstanceProxy health_check any-healthy semantics
- WorkerManager spawning N workers for instances>1
- Uniform distribution of requests across instances
- Regression: instances=1 behavior unchanged
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nerva.backends.base import InferContext
from nerva.core.model import Model, ModelHandle, model
from nerva.worker.manager import WorkerManager
from nerva.worker.proxy import MultiInstanceProxy, WorkerProxy
from tests.helpers import EchoModel, PidModel

# ---------------------------------------------------------------------------
# Unit tests: ModelHandle.instances
# ---------------------------------------------------------------------------


class _DummyModel(Model):
    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {}


class TestModelHandleInstances:
    def test_default_is_one(self) -> None:
        handle = ModelHandle(
            name="test", model_class=_DummyModel, backend="pytorch", device="cpu"
        )
        assert handle.instances == 1

    def test_set_to_two(self) -> None:
        handle = ModelHandle(
            name="test",
            model_class=_DummyModel,
            backend="pytorch",
            device="cpu",
            instances=2,
        )
        assert handle.instances == 2

    def test_set_to_four(self) -> None:
        handle = ModelHandle(
            name="test",
            model_class=_DummyModel,
            backend="pytorch",
            device="cpu",
            instances=4,
        )
        assert handle.instances == 4


# ---------------------------------------------------------------------------
# Unit tests: model() function — instances parameter
# ---------------------------------------------------------------------------


class TestModelFunctionInstances:
    def test_default_is_one(self) -> None:
        handle = model("test_mi_default", _DummyModel)
        assert handle.instances == 1

    def test_set_to_two(self) -> None:
        handle = model("test_mi_two", _DummyModel, instances=2)
        assert handle.instances == 2

    def test_invalid_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="instances must be >= 1"):
            model("test_mi_zero", _DummyModel, instances=0)

    def test_invalid_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="instances must be >= 1"):
            model("test_mi_neg", _DummyModel, instances=-1)


# ---------------------------------------------------------------------------
# Unit tests: MultiInstanceProxy
# ---------------------------------------------------------------------------


def _make_mock_proxy(*, healthy: bool = True, result: dict[str, Any] | None = None) -> WorkerProxy:
    """Return a WorkerProxy mock with configurable behavior."""
    proxy = MagicMock(spec=WorkerProxy)
    proxy.infer = AsyncMock(return_value=result or {})
    proxy.health_check = AsyncMock(return_value=healthy)
    proxy.shutdown = AsyncMock()
    proxy.close = AsyncMock()
    return proxy  # type: ignore[return-value]


class TestMultiInstanceProxyRoundRobin:
    async def test_round_robin_two_proxies(self) -> None:
        """Requests cycle across proxies 0 → 1 → 0 → 1 ..."""
        p0 = _make_mock_proxy(result={"instance": 0})
        p1 = _make_mock_proxy(result={"instance": 1})
        mip = MultiInstanceProxy([p0, p1])
        ctx = InferContext(request_id="rr-001", deadline_ms=5000)

        results = []
        for _ in range(4):
            r = await mip.infer({}, ctx)
            results.append(r["instance"])

        # 4 calls with 2 proxies: 0,1,0,1
        assert results == [0, 1, 0, 1]

    async def test_round_robin_three_proxies(self) -> None:
        """Requests cycle evenly across 3 proxies."""
        proxies = [_make_mock_proxy(result={"i": i}) for i in range(3)]
        mip = MultiInstanceProxy(proxies)
        ctx = InferContext(request_id="rr-002", deadline_ms=5000)

        counts: Counter[int] = Counter()
        for _ in range(9):
            r = await mip.infer({}, ctx)
            counts[r["i"]] += 1

        # 9 calls / 3 proxies = 3 each
        assert counts == {0: 3, 1: 3, 2: 3}

    async def test_single_proxy_always_called(self) -> None:
        """With 1 proxy, all calls go to that proxy."""
        p0 = _make_mock_proxy(result={"instance": 0})
        mip = MultiInstanceProxy([p0])
        ctx = InferContext(request_id="rr-003", deadline_ms=5000)

        for _ in range(5):
            await mip.infer({}, ctx)

        assert p0.infer.call_count == 5  # type: ignore[attr-defined]

    def test_empty_proxies_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one proxy"):
            MultiInstanceProxy([])


class TestMultiInstanceProxyHealthCheck:
    async def test_all_healthy_returns_true(self) -> None:
        proxies = [_make_mock_proxy(healthy=True) for _ in range(3)]
        mip = MultiInstanceProxy(proxies)
        assert await mip.health_check() is True

    async def test_one_healthy_returns_true(self) -> None:
        """Any-healthy semantics: one healthy proxy is sufficient."""
        proxies = [
            _make_mock_proxy(healthy=False),
            _make_mock_proxy(healthy=True),
            _make_mock_proxy(healthy=False),
        ]
        mip = MultiInstanceProxy(proxies)
        assert await mip.health_check() is True

    async def test_all_unhealthy_returns_false(self) -> None:
        proxies = [_make_mock_proxy(healthy=False) for _ in range(3)]
        mip = MultiInstanceProxy(proxies)
        assert await mip.health_check() is False

    async def test_exception_treated_as_unhealthy(self) -> None:
        """If a health_check call raises, that instance is treated as unhealthy."""
        p0 = _make_mock_proxy(healthy=False)
        p0.health_check = AsyncMock(side_effect=RuntimeError("timeout"))  # type: ignore[attr-defined]
        p1 = _make_mock_proxy(healthy=True)
        mip = MultiInstanceProxy([p0, p1])
        assert await mip.health_check() is True

    async def test_all_raise_returns_false(self) -> None:
        p0 = _make_mock_proxy(healthy=False)
        p0.health_check = AsyncMock(side_effect=RuntimeError("timeout"))  # type: ignore[attr-defined]
        p1 = _make_mock_proxy(healthy=False)
        p1.health_check = AsyncMock(side_effect=RuntimeError("timeout"))  # type: ignore[attr-defined]
        mip = MultiInstanceProxy([p0, p1])
        assert await mip.health_check() is False


class TestMultiInstanceProxyInferStream:
    async def test_infer_stream_round_robins_to_first_proxy(self) -> None:
        """infer_stream() round-robins; first call goes to proxy 0."""
        chunks_p0 = [{"i": 0, "chunk": 0}, {"i": 0, "chunk": 1}]
        chunks_p1 = [{"i": 1, "chunk": 0}]

        async def _gen_p0(*_: Any, **__: Any) -> Any:
            for c in chunks_p0:
                yield c

        async def _gen_p1(*_: Any, **__: Any) -> Any:
            for c in chunks_p1:
                yield c

        p0 = _make_mock_proxy()
        p0.infer_stream = _gen_p0  # type: ignore[attr-defined]
        p1 = _make_mock_proxy()
        p1.infer_stream = _gen_p1  # type: ignore[attr-defined]

        mip = MultiInstanceProxy([p0, p1])
        ctx = InferContext(request_id="stream-rr-001", deadline_ms=5000)

        got: list[dict[str, Any]] = []
        async for chunk in mip.infer_stream({}, ctx):
            got.append(chunk)

        # Counter starts at 0 → first call routes to p0.
        assert got == chunks_p0

    async def test_infer_stream_second_call_round_robins_to_p1(self) -> None:
        """Two sequential infer_stream calls alternate between proxy 0 and 1."""
        async def _gen_p0(*_: Any, **__: Any) -> Any:
            yield {"instance": 0}

        async def _gen_p1(*_: Any, **__: Any) -> Any:
            yield {"instance": 1}

        p0 = _make_mock_proxy()
        p0.infer_stream = _gen_p0  # type: ignore[attr-defined]
        p1 = _make_mock_proxy()
        p1.infer_stream = _gen_p1  # type: ignore[attr-defined]

        mip = MultiInstanceProxy([p0, p1])
        ctx = InferContext(request_id="stream-rr-002", deadline_ms=5000)

        first: list[dict[str, Any]] = []
        async for chunk in mip.infer_stream({}, ctx):
            first.append(chunk)

        second: list[dict[str, Any]] = []
        async for chunk in mip.infer_stream({}, ctx):
            second.append(chunk)

        assert first == [{"instance": 0}]
        assert second == [{"instance": 1}]


class TestMultiInstanceProxyShutdownClose:
    async def test_shutdown_broadcasts_to_all(self) -> None:
        proxies = [_make_mock_proxy() for _ in range(3)]
        mip = MultiInstanceProxy(proxies)
        await mip.shutdown()
        for p in proxies:
            p.shutdown.assert_awaited_once()  # type: ignore[attr-defined]

    async def test_close_broadcasts_to_all(self) -> None:
        proxies = [_make_mock_proxy() for _ in range(3)]
        mip = MultiInstanceProxy(proxies)
        await mip.close()
        for p in proxies:
            p.close.assert_awaited_once()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Integration: WorkerManager spawns N workers for instances > 1
# ---------------------------------------------------------------------------


class TestWorkerManagerMultiInstance:
    async def test_instances_two_spawns_two_processes(self) -> None:
        """instances=2 starts 2 independent Worker processes."""
        mgr = WorkerManager()
        try:
            handle = model("echo_mi", EchoModel, async_infer=True, instances=2)
            proxy = await mgr.start_worker(handle)

            assert isinstance(proxy, MultiInstanceProxy)
            assert len(proxy._proxies) == 2

            # Both underlying processes should be alive.
            assert "echo_mi-0" in mgr._workers
            assert "echo_mi-1" in mgr._workers
            e0 = mgr._workers["echo_mi-0"]
            e1 = mgr._workers["echo_mi-1"]
            assert e0.process is not None and e0.process.is_alive()
            assert e1.process is not None and e1.process.is_alive()
            # They must be different processes.
            assert e0.process.pid != e1.process.pid
        finally:
            await mgr.shutdown_all()

    async def test_instance_group_registered(self) -> None:
        """WorkerManager._instance_groups records instance worker_ids."""
        mgr = WorkerManager()
        try:
            handle = model("echo_grp", EchoModel, async_infer=True, instances=2)
            await mgr.start_worker(handle)

            assert "echo_grp" in mgr._instance_groups
            ids = mgr._instance_groups["echo_grp"]
            assert ids == ["echo_grp-0", "echo_grp-1"]
        finally:
            await mgr.shutdown_all()

    async def test_multi_instance_infer_returns_correct_results(self) -> None:
        """MultiInstanceProxy infer() returns correct EchoModel output."""
        mgr = WorkerManager()
        try:
            handle = model("echo_infer_mi", EchoModel, async_infer=True, instances=2)
            proxy = await mgr.start_worker(handle)

            ctx = InferContext(request_id="mi-req-001", deadline_ms=30000)
            result = await proxy.infer({"value": 77}, ctx)
            assert result == {"echo": 77}
        finally:
            await mgr.shutdown_all()

    async def test_multi_instance_health_check_true(self) -> None:
        """Health check returns True when all instances are running."""
        mgr = WorkerManager()
        try:
            handle = model("echo_hc_mi", EchoModel, async_infer=True, instances=2)
            proxy = await mgr.start_worker(handle)

            assert await proxy.health_check() is True
        finally:
            await mgr.shutdown_all()

    async def test_shutdown_all_stops_all_instances(self) -> None:
        """shutdown_all() terminates all instance processes."""
        mgr = WorkerManager()
        handle = model("echo_shutdown_mi", EchoModel, async_infer=True, instances=2)
        await mgr.start_worker(handle)

        e0 = mgr._workers["echo_shutdown_mi-0"]
        e1 = mgr._workers["echo_shutdown_mi-1"]
        proc0 = e0.process
        proc1 = e1.process

        await mgr.shutdown_all()

        assert proc0 is not None and not proc0.is_alive()
        assert proc1 is not None and not proc1.is_alive()
        assert len(mgr._workers) == 0
        assert len(mgr._instance_groups) == 0

    async def test_requests_distributed_across_instances(self) -> None:
        """Requests are distributed across both Worker instances (verified by PID).

        Uses PidModel (returns os.getpid()) to confirm that at least two
        distinct worker PIDs appear in the responses — proving that both
        instances handled requests rather than all traffic going to one.
        """
        mgr = WorkerManager()
        try:
            handle = model("pid_dist", PidModel, async_infer=True, instances=2)
            proxy = await mgr.start_worker(handle)

            # Send 10 concurrent requests.
            n = 10
            ctx_list = [
                InferContext(request_id=f"dist-{i}", deadline_ms=30000) for i in range(n)
            ]
            results = await asyncio.gather(
                *(proxy.infer({"value": i}, ctx_list[i]) for i in range(n))
            )

            # Verify that both worker instances handled requests (distinct PIDs).
            pids = {r["pid"] for r in results}
            assert len(pids) == 2, (
                f"Expected requests distributed across 2 worker PIDs, got: {pids}"
            )
        finally:
            await mgr.shutdown_all()


# ---------------------------------------------------------------------------
# Regression: instances=1 behavior unchanged
# ---------------------------------------------------------------------------


class TestInstancesOneRegression:
    async def test_instances_one_returns_worker_proxy(self) -> None:
        """instances=1 (default) returns a plain WorkerProxy, not MultiInstanceProxy."""
        mgr = WorkerManager()
        try:
            handle = model("echo_single", EchoModel, async_infer=True)
            proxy = await mgr.start_worker(handle)

            assert isinstance(proxy, WorkerProxy)
            assert not isinstance(proxy, MultiInstanceProxy)
        finally:
            await mgr.shutdown_all()

    async def test_instances_one_infer_works(self) -> None:
        mgr = WorkerManager()
        try:
            handle = model("echo_single_infer", EchoModel, async_infer=True)
            proxy = await mgr.start_worker(handle)

            ctx = InferContext(request_id="single-001", deadline_ms=30000)
            result = await proxy.infer({"value": 55}, ctx)
            assert result == {"echo": 55}
        finally:
            await mgr.shutdown_all()

    async def test_instances_one_not_in_instance_groups(self) -> None:
        """instances=1 does not add an entry to _instance_groups."""
        mgr = WorkerManager()
        try:
            handle = model("echo_no_group", EchoModel, async_infer=True)
            await mgr.start_worker(handle)

            assert "echo_no_group" not in mgr._instance_groups
        finally:
            await mgr.shutdown_all()
