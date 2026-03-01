"""Tests for nerva.core.model — Model base class and model() declaration."""

from typing import Any

import pytest

from nerva import Model, ModelHandle, model
from nerva.core.model import get_model_handle, list_model_handles


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


class TestModelRegistry:
    def test_model_registers_handle(self) -> None:
        handle = model("test_reg", DummyModel)
        assert get_model_handle("test_reg") is handle

    def test_get_missing_raises(self) -> None:
        with pytest.raises(KeyError, match="test_missing"):
            get_model_handle("test_missing")

    def test_list_handles(self) -> None:
        model("a", DummyModel)
        model("b", DummyModel)
        handles = list_model_handles()
        assert set(handles.keys()) == {"a", "b"}

    def test_duplicate_name_overwrites(self) -> None:
        model("dup", DummyModel, device="cpu")
        handle2 = model("dup", DummyModel, device="cuda:0")
        assert get_model_handle("dup") is handle2


def test_model_with_batch_config() -> None:
    """model() accepts batch_config and stores it on ModelHandle."""
    from nerva.engine.batcher import BatchConfig

    cfg = BatchConfig(max_batch_size=8, max_delay_ms=5.0)
    handle = model("m", DummyModel, batch_config=cfg)
    assert handle.batch_config is cfg
    assert handle.batch_config.max_batch_size == 8


def test_model_without_batch_config() -> None:
    """batch_config defaults to None — existing behavior unaffected."""
    handle = model("m", DummyModel)
    assert handle.batch_config is None
