"""Shared pytest fixtures for Nerva test suite."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from prometheus_client import CollectorRegistry

from nerva.core.model import _model_registry
from nerva.observability.metrics import NervaMetrics

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def _clean_model_registry() -> Generator[None, None, None]:
    """Ensure model registry is clean before and after each test."""
    _model_registry.clear()
    yield
    _model_registry.clear()


@pytest.fixture()
def metrics() -> NervaMetrics:
    """Isolated NervaMetrics for tests — avoids 'Duplicated timeseries' errors."""
    return NervaMetrics(registry=CollectorRegistry())
