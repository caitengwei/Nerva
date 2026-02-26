"""Tests for nerva.backends.registry — backend registration."""

from typing import Any

import pytest

from nerva.backends.base import Backend, InferContext, ModelConfig
from nerva.backends.registry import (
    clear_registry,
    get_backend,
    list_backends,
    register_backend,
)


class TestBackendRegistry:
    def setup_method(self) -> None:
        """Save and clear registry for test isolation."""
        # Import to ensure built-in backends are registered
        import nerva.backends.pytorch  # noqa: F401

        self._saved: dict[str, type[Backend]] = {}
        from nerva.backends.registry import _REGISTRY

        self._saved = dict(_REGISTRY)

    def teardown_method(self) -> None:
        """Restore registry after each test."""
        from nerva.backends.registry import _REGISTRY

        _REGISTRY.clear()
        _REGISTRY.update(self._saved)

    def test_pytorch_registered_by_default(self) -> None:
        assert "pytorch" in list_backends()

    def test_get_backend_returns_class(self) -> None:
        from nerva.backends.pytorch import PyTorchBackend

        cls = get_backend("pytorch")
        assert cls is PyTorchBackend

    def test_get_backend_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="not found"):
            get_backend("nonexistent")

    def test_register_custom_backend(self) -> None:
        @register_backend("custom_test")
        class CustomBackend(Backend):
            async def load_model(self, config: ModelConfig) -> None:
                pass

            async def unload_model(self) -> None:
                pass

            async def infer(
                self, inputs: dict[str, Any], context: InferContext, batch_meta: Any = None
            ) -> dict[str, Any]:
                return {}

            async def infer_stream(self, inputs: dict[str, Any], context: InferContext):  # type: ignore[override]
                yield {}

        assert "custom_test" in list_backends()
        assert get_backend("custom_test") is CustomBackend

    def test_register_duplicate_raises(self) -> None:
        @register_backend("dup_test")
        class First(Backend):
            async def load_model(self, config: ModelConfig) -> None:
                pass

            async def unload_model(self) -> None:
                pass

            async def infer(
                self, inputs: dict[str, Any], context: InferContext, batch_meta: Any = None
            ) -> dict[str, Any]:
                return {}

            async def infer_stream(self, inputs: dict[str, Any], context: InferContext):  # type: ignore[override]
                yield {}

        with pytest.raises(ValueError, match="already registered"):

            @register_backend("dup_test")
            class Second(Backend):
                async def load_model(self, config: ModelConfig) -> None:
                    pass

                async def unload_model(self) -> None:
                    pass

                async def infer(
                    self, inputs: dict[str, Any], context: InferContext, batch_meta: Any = None
                ) -> dict[str, Any]:
                    return {}

                async def infer_stream(self, inputs: dict[str, Any], context: InferContext):  # type: ignore[override]
                    yield {}

    def test_clear_registry(self) -> None:
        clear_registry()
        assert list_backends() == []
