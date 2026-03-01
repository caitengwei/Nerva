"""Shared pytest fixtures for Nerva test suite."""

from __future__ import annotations

import pytest

from nerva.core.model import _model_registry


@pytest.fixture(autouse=True)
def _clean_model_registry() -> None:  # type: ignore[misc]
    """Ensure model registry is clean before and after each test."""
    _model_registry.clear()
    yield  # type: ignore[misc]
    _model_registry.clear()
