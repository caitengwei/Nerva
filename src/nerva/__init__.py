"""Nerva — High-performance model inference serving framework."""

__version__ = "0.1.0"

# Ensure built-in backends are registered on import.
import nerva.backends.pytorch as _pytorch_backend  # noqa: F401
from nerva.backends.base import Backend, BatchMeta, InferContext, ModelConfig
from nerva.backends.registry import get_backend, list_backends, register_backend
from nerva.core.graph import Edge, Graph, Node
from nerva.core.model import Model, ModelHandle, get_model_handle, list_model_handles, model
from nerva.core.primitives import cond, parallel
from nerva.core.proxy import Proxy, trace
from nerva.engine.batcher import BatchConfig, DynamicBatcher
from nerva.engine.executor import Executor
from nerva.server.serve import serve
from nerva.worker.manager import WorkerManager
from nerva.worker.proxy import WorkerProxy

__all__ = [
    "Backend",
    "BatchConfig",
    "BatchMeta",
    "DynamicBatcher",
    "Edge",
    "Executor",
    "Graph",
    "InferContext",
    "Model",
    "ModelConfig",
    "ModelHandle",
    "Node",
    "Proxy",
    "WorkerManager",
    "WorkerProxy",
    "cond",
    "get_backend",
    "get_model_handle",
    "list_backends",
    "list_model_handles",
    "model",
    "parallel",
    "register_backend",
    "serve",
    "trace",
]
