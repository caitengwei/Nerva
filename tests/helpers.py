"""Shared test model implementations."""

from __future__ import annotations

from typing import Any

from nerva.core.model import Model


class EchoModel(Model):
    """Returns {"echo": inputs.get("value")}."""

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"echo": inputs.get("value")}


class SlowModel(Model):
    """Sleeps for inputs.get("delay", 1.0) seconds."""

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        await asyncio.sleep(inputs.get("delay", 1.0))
        return {"done": True}


class CrashModel(Model):
    """Always raises RuntimeError on infer."""

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("CrashModel always fails")


class BigOutputModel(Model):
    """Returns payload with configurable size; default > inline threshold."""

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        size = int(inputs.get("size", 10000))
        return {"blob": "x" * size}


class UpperModel(Model):
    """Uppercases inputs["value"] and returns {"features": uppercased}."""

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        val = str(inputs.get("value", ""))
        return {"features": val.upper()}


class ConcatModel(Model):
    """Concatenates inputs["a"] and inputs["b"] into {"result": a+b}."""

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        a = str(inputs.get("a", ""))
        b = str(inputs.get("b", ""))
        return {"result": a + b}


class BenchImageEncoder(Model):
    """Benchmark model: simulates image encoding.

    Options: dim (int), delay_ms (float).
    Input: {"image_bytes": bytes}
    Output: {"features": [float] * dim}
    """

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        dim = int(self._options.get("dim", 768))
        delay_ms = float(self._options.get("delay_ms", 0))
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)
        return {"features": [0.1] * dim}


class BenchTextEncoder(Model):
    """Benchmark model: simulates text encoding.

    Options: dim (int), delay_ms (float).
    Input: {"text": str}
    Output: {"features": [float] * dim}
    """

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        dim = int(self._options.get("dim", 768))
        delay_ms = float(self._options.get("delay_ms", 0))
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)
        return {"features": [0.2] * dim}


class BenchFusionModel(Model):
    """Benchmark model: simulates multimodal fusion.

    Options: dim (int), delay_ms (float).
    Input: {"img_features": list, "txt_features": list}
    Output: {"fused_features": [float] * dim}
    """

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        dim = int(self._options.get("dim", 512))
        delay_ms = float(self._options.get("delay_ms", 0))
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)
        return {"fused_features": [0.3] * dim}


class PidModel(Model):
    """Returns {"pid": os.getpid()} — lets callers identify which Worker process ran the request."""

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import os

        return {"pid": os.getpid()}


class StreamingEchoModel(Model):
    """Yields {"chunk": i, "value": inputs["value"]} for i in range(count).

    inputs["count"] controls how many chunks to yield (default 3).
    """

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"echo": inputs.get("value")}

    async def infer_stream(self, inputs: dict[str, Any]):  # type: ignore[override]
        count = int(inputs.get("count", 3))
        for i in range(count):
            yield {"chunk": i, "value": inputs.get("value")}


class StreamingCrashModel(Model):
    """Yields one chunk then raises RuntimeError."""

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {}

    async def infer_stream(self, inputs: dict[str, Any]):  # type: ignore[override]
        yield {"chunk": 0}
        raise RuntimeError("StreamingCrashModel always fails mid-stream")


class SlowStreamingModel(Model):
    """Yields first chunk immediately, then sleeps 500ms before yielding second chunk.

    Useful for testing deadline enforcement during streaming iteration.
    """

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {}

    async def infer_stream(self, inputs: dict[str, Any]):  # type: ignore[override]
        import asyncio

        yield {"chunk": 0}
        await asyncio.sleep(0.5)  # 500ms — triggers deadline if deadline_ms < 500
        yield {"chunk": 1}


class BenchClassifier(Model):
    """Benchmark model: simulates classification head.

    Options: delay_ms (float).
    Input: {"fused_features": list}
    Output: {"label": str, "score": float}
    """

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import asyncio

        delay_ms = float(self._options.get("delay_ms", 0))
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)
        return {"label": "cat", "score": 0.95}


class BenchStreamingModel(Model):
    """Streaming bench model: yields `count` chunks of `chunk_size` bytes each.

    Options:
        count (int, default 100): number of chunks to yield.
        chunk_size (int, default 1024): bytes per chunk.
        delay_ms (float, default 30): sleep between chunks (simulates inference).
    Output per chunk: {"chunk": i, "payload": bytes}
    """

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        chunk_size = int(self._options.get("chunk_size", 1024))
        return {"chunk": 0, "payload": b"x" * chunk_size}

    async def infer_stream(self, inputs: dict[str, Any]):  # type: ignore[no-untyped-def]
        import asyncio

        count = int(self._options.get("count", 100))
        chunk_size = int(self._options.get("chunk_size", 1024))
        delay_ms = float(self._options.get("delay_ms", 30))
        payload = b"x" * chunk_size
        for i in range(count):
            if delay_ms > 0:
                await asyncio.sleep(delay_ms / 1000.0)
            yield {"chunk": i, "payload": payload}
