"""PyTorch Backend implementation.

Wraps user-defined Model subclasses for in-process inference.
Handles model lifecycle, device placement, and inference_mode optimization.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from nerva.backends.base import Backend, BatchMeta, InferContext, ModelConfig
from nerva.backends.registry import register_backend

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from nerva.core.model import Model

logger = logging.getLogger(__name__)


@register_backend("pytorch")
class PyTorchBackend(Backend):
    """Backend for user-defined PyTorch models.

    Instantiates the user's Model subclass, calls load(),
    and delegates infer() calls to it.
    """

    def __init__(self) -> None:
        self._model: Model | None = None
        self._config: ModelConfig | None = None

    async def load_model(self, config: ModelConfig) -> None:
        """Instantiate user's Model subclass and call load()."""
        if self._model is not None:
            raise RuntimeError(
                f"Model already loaded: {self._config.model_name if self._config else '?'}. "
                "Call unload_model() first."
            )
        self._config = config
        logger.info(
            "Loading model '%s' on device '%s'",
            config.model_name,
            config.device,
        )
        instance = config.model_class()
        instance._options = config.backend_options
        instance.load()
        self._model = instance
        logger.info("Model '%s' loaded successfully", config.model_name)

    async def unload_model(self) -> None:
        """Call Model.unload() and release reference."""
        if self._model is None:
            return
        name = self._config.model_name if self._config else "?"
        logger.info("Unloading model '%s'", name)
        self._model.unload()
        self._model = None
        self._config = None

    async def infer(
        self,
        inputs: dict[str, Any],
        context: InferContext,
        batch_meta: BatchMeta | None = None,
    ) -> dict[str, Any]:
        """Delegate inference to the user's Model."""
        model = self._ensure_loaded()
        if context.cancelled:
            raise RuntimeError(f"Request {context.request_id} already cancelled")
        return await model.infer(inputs)

    async def infer_stream(
        self,
        inputs: dict[str, Any],
        context: InferContext,
    ) -> AsyncIterator[dict[str, Any]]:
        """Delegate streaming inference to the user's Model."""
        model = self._ensure_loaded()
        if context.cancelled:
            raise RuntimeError(f"Request {context.request_id} already cancelled")
        async for chunk in model.infer_stream(inputs):
            yield chunk

    async def warmup(self) -> None:
        """No-op warmup for PyTorch backend."""

    def health_check(self) -> bool:
        return self._model is not None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> Model:
        if self._model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        return self._model
