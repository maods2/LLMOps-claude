"""
Structured logging configuration using structlog.

Provides a pre-configured logger factory with:
  - JSON output in production
  - Human-readable output in development
  - Automatic context fields (service, environment, timestamp)
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import structlog


def configure_logging(
    service_name: str = "llm-stack",
    log_level: str = "INFO",
    json_logs: bool = True,
    environment: str = "production",
) -> None:
    """Configure structlog for the application.

    Call this once at application startup.

    Args:
        service_name: Service identifier included in every log entry.
        log_level:    Python logging level string.
        json_logs:    If True, emit JSON; otherwise human-readable.
        environment:  E.g. 'development', 'production', 'test'.
    """
    log_level_int = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level_int,
    )

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_logs:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)  # type: ignore[assignment]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Inject global context
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(service=service_name, env=environment)


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """Return a configured structlog logger.

    Args:
        name: Optional logger name (defaults to calling module).
    Returns:
        A bound structlog logger.
    """
    return structlog.get_logger(name)


def bind_context(**kwargs) -> None:
    """Bind key-value pairs to the current context (all future log calls)."""
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind_context(*keys: str) -> None:
    """Remove specific keys from the logging context."""
    structlog.contextvars.unbind_contextvars(*keys)
