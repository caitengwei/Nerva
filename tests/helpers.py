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
