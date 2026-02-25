"""Backend registration mechanism."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nerva.backends.base import Backend

_REGISTRY: dict[str, type[Backend]] = {}


def register_backend(name: str) -> Any:
    """Decorator to register a Backend implementation.

    Usage:
        @register_backend("pytorch")
        class PyTorchBackend(Backend):
            ...
    """

    def decorator(cls: type[Backend]) -> type[Backend]:
        if name in _REGISTRY:
            raise ValueError(f"Backend '{name}' is already registered")
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_backend(name: str) -> type[Backend]:
    """Retrieve a registered backend class by name.

    Raises:
        KeyError: If the backend name is not registered.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none)"
        raise KeyError(f"Backend '{name}' not found. Available: {available}")
    return _REGISTRY[name]


def list_backends() -> list[str]:
    """Return sorted list of registered backend names."""
    return sorted(_REGISTRY.keys())


def clear_registry() -> None:
    """Clear all registered backends. For testing only."""
    _REGISTRY.clear()
