"""
Structured logging helpers for RoboChess runtime events.

This module provides tools for emitting structured log events with consistent
formatting, making it easier to parse and analyze runtime logs.
"""

import logging


class EventLogger:
    """Helper class for structured event logging."""

    @staticmethod
    def format_value(value):
        """
        Format a single value for logging.
        
        Args:
            value: The value to format.
            
        Returns:
            A string representation of the value.
        """
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    @classmethod
    def format_fields(cls, **fields):
        """
        Format structured key/value logging fields in a stable order.
        
        Args:
            **fields: Arbitrary keyword arguments to format.
            
        Returns:
            A formatted string of key=value pairs.
        """
        parts = []
        for key in sorted(fields):
            value = fields[key]
            if value is None:
                continue
            parts.append(f"{key}={cls.format_value(value)}")
        return " ".join(parts)

    @classmethod
    def log_event(cls, logger, level, event, **fields):
        """
        Emit a structured log event.
        
        Args:
            logger: The logger instance to use.
            level: The logging level (e.g., logging.INFO).
            event: The name of the event.
            **fields: Additional structured data for the event.
        """
        suffix = cls.format_fields(**fields)
        message = event if not suffix else f"{event} | {suffix}"
        logger.log(level, message)


def log_event(logger, level, event, **fields):
    """
    Legacy wrapper for log_event to maintain compatibility.
    
    Args:
        logger: The logger instance to use.
        level: The logging level.
        event: The name of the event.
        **fields: Additional structured data.
    """
    EventLogger.log_event(logger, level, event, **fields)
