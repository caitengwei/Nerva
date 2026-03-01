"""Shared pytest fixtures for Nerva test suite."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from nerva.core.model import _model_registry

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def _clean_model_registry() -> Generator[None, None, None]:
    """Ensure model registry is clean before and after each test."""
    _model_registry.clear()
    yield
    _model_registry.clear()
