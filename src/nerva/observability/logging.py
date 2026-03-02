"""Structured logging configuration for Nerva using structlog.

Usage:
    from nerva.observability.logging import configure_logging
    configure_logging(dev=True)   # human-friendly (local)
    configure_logging(dev=False)  # JSON (production)

Per-request context binding (async-safe via contextvars):
    import structlog.contextvars
    structlog.contextvars.bind_contextvars(request_id="req-abc", pipeline="chat")
    structlog.contextvars.clear_contextvars()
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(
    *,
    dev: bool = False,
    level: int = logging.INFO,
) -> None:
    """Configure structlog for Nerva.

    Args:
        dev: If True, use ConsoleRenderer (human-friendly).
             If False, use JSONRenderer (production).
        level: stdlib logging level. Defaults to INFO.
    """
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
        force=True,
    )

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if dev:
        processors: list[structlog.types.Processor] = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(),
        ]
    else:
        processors = [
            *shared_processors,
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,  # False for test safety
    )
