"""Smoke test to verify project setup."""

import nerva


def test_version() -> None:
    assert nerva.__version__ == "0.1.0"
