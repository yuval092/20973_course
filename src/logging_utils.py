"""Structured logging helpers for RoboChess runtime events."""

from __future__ import annotations

import logging
from typing import Any


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def format_fields(**fields: Any) -> str:
    """Format structured key/value logging fields in a stable order."""
    parts = []
    for key in sorted(fields):
        value = fields[key]
        if value is None:
            continue
        parts.append(f"{key}={_format_value(value)}")
    return " ".join(parts)


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    """Emit a structured log event."""
    suffix = format_fields(**fields)
    message = event if not suffix else f"{event} | {suffix}"
    logger.log(level, message)
