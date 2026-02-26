"""Tests for nerva.backends.pytorch — PyTorch backend."""

from typing import Any

import pytest

from nerva import Model
from nerva.backends.base import InferContext, ModelConfig
from nerva.backends.pytorch import PyTorchBackend


class EchoModel(Model):
    """Test model that echoes inputs."""

    def load(self) -> None:
        self.loaded = True

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"echo": inputs}


class StreamModel(Model):
    """Test model that streams tokens."""

    def load(self) -> None:
        pass

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"result": "full"}

    async def infer_stream(self, inputs: dict[str, Any]):  # type: ignore[override]
        for i in range(3):
            yield {"token": i}


def _make_context(request_id: str = "test-req") -> InferContext:
    return InferContext(request_id=request_id, deadline_ms=9999999)


def _make_config(model_class: type[Model], name: str = "test") -> ModelConfig:
    return ModelConfig(
        model_name=name,
        model_class=model_class,
        device="cpu",
    )


class TestPyTorchBackendLifecycle:
    async def test_load_and_health(self) -> None:
        backend = PyTorchBackend()
        assert not backend.is_loaded
        assert not backend.health_check()

        await backend.load_model(_make_config(EchoModel))
        assert backend.is_loaded
        assert backend.health_check()

    async def test_unload(self) -> None:
        backend = PyTorchBackend()
        await backend.load_model(_make_config(EchoModel))
        await backend.unload_model()
        assert not backend.is_loaded

    async def test_unload_when_not_loaded(self) -> None:
        backend = PyTorchBackend()
        await backend.unload_model()  # Should not raise

    async def test_double_load_raises(self) -> None:
        backend = PyTorchBackend()
        await backend.load_model(_make_config(EchoModel))
        with pytest.raises(RuntimeError, match="already loaded"):
            await backend.load_model(_make_config(EchoModel))


class TestPyTorchBackendInfer:
    async def test_infer(self) -> None:
        backend = PyTorchBackend()
        await backend.load_model(_make_config(EchoModel))

        result = await backend.infer({"value": 42}, _make_context())
        assert result == {"echo": {"value": 42}}

    async def test_infer_without_load_raises(self) -> None:
        backend = PyTorchBackend()
        with pytest.raises(RuntimeError, match="No model loaded"):
            await backend.infer({}, _make_context())

    async def test_infer_cancelled_raises(self) -> None:
        backend = PyTorchBackend()
        await backend.load_model(_make_config(EchoModel))

        ctx = InferContext(request_id="test", deadline_ms=0, cancelled=True)
        with pytest.raises(RuntimeError, match="already cancelled"):
            await backend.infer({}, ctx)


class TestPyTorchBackendStream:
    async def test_infer_stream(self) -> None:
        backend = PyTorchBackend()
        await backend.load_model(_make_config(StreamModel))

        chunks = []
        async for chunk in backend.infer_stream({"prompt": "hi"}, _make_context()):
            chunks.append(chunk)

        assert chunks == [{"token": 0}, {"token": 1}, {"token": 2}]

    async def test_infer_stream_cancelled_raises(self) -> None:
        backend = PyTorchBackend()
        await backend.load_model(_make_config(StreamModel))

        ctx = InferContext(request_id="test", deadline_ms=0, cancelled=True)
        with pytest.raises(RuntimeError, match="already cancelled"):
            async for _ in backend.infer_stream({}, ctx):
                pass
