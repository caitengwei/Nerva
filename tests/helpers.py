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
