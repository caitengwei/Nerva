"""Tests for nerva.core.model — Model base class and model() declaration."""

from typing import Any

import pytest

from nerva import Model, ModelHandle, model


class DummyModel(Model):
    """Minimal Model for testing."""

    def load(self) -> None:
        self.loaded = True

    async def infer(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"echo": inputs.get("value")}


class BadNotAModel:
    """Not a Model subclass."""


class TestModelDeclaration:
    def test_model_returns_handle(self) -> None:
        handle = model("test", DummyModel, backend="pytorch", device="cpu")
        assert isinstance(handle, ModelHandle)
        assert handle.name == "test"
        assert handle.model_class is DummyModel
        assert handle.backend == "pytorch"
        assert handle.device == "cpu"

    def test_model_with_options(self) -> None:
        handle = model("test", DummyModel, backend="pytorch", device="cuda:0", max_batch=32)
        assert handle.options == {"max_batch": 32}

    def test_model_rejects_non_model_class(self) -> None:
        with pytest.raises(TypeError, match=r"subclass of nerva\.Model"):
            model("bad", BadNotAModel, backend="pytorch", device="cpu")  # type: ignore[arg-type]

    def test_handle_is_frozen(self) -> None:
        handle = model("test", DummyModel)
        with pytest.raises(AttributeError):
            handle.name = "changed"  # type: ignore[misc]

    def test_handle_call_outside_pipeline_raises(self) -> None:
        handle = model("test", DummyModel)
        with pytest.raises(RuntimeError, match="outside of a pipeline context"):
            handle({"value": 1})


class TestModelLifecycle:
    def test_load_is_called(self) -> None:
        instance = DummyModel()
        instance.load()
        assert instance.loaded is True

    async def test_infer(self) -> None:
        instance = DummyModel()
        instance.load()
        result = await instance.infer({"value": 42})
        assert result == {"echo": 42}

    async def test_infer_stream_not_implemented(self) -> None:
        instance = DummyModel()
        instance.load()
        with pytest.raises(NotImplementedError, match="does not support streaming"):
            async for _ in instance.infer_stream({}):
                pass

    def test_unload_is_noop_by_default(self) -> None:
        instance = DummyModel()
        instance.load()
        instance.unload()  # Should not raise
