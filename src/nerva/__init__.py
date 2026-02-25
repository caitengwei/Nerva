"""Nerva — High-performance model inference serving framework."""

__version__ = "0.1.0"

# Ensure built-in backends are registered on import.
import nerva.backends.pytorch as _pytorch_backend  # noqa: F401
from nerva.backends.base import Backend, BatchMeta, InferContext, ModelConfig
from nerva.backends.registry import get_backend, list_backends, register_backend
from nerva.core.model import Model, ModelHandle, model
from nerva.worker.manager import WorkerManager
from nerva.worker.proxy import WorkerProxy

__all__ = [
    "Backend",
    "BatchMeta",
    "InferContext",
    "Model",
    "ModelConfig",
    "ModelHandle",
    "WorkerManager",
    "WorkerProxy",
    "get_backend",
    "list_backends",
    "model",
    "register_backend",
]
