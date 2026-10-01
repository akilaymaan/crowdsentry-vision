"""Structured logging.

Log lines carry machine-readable ``key=value`` context after the message, so per-camera
processing can be followed by eye in the console *and* grepped or parsed without a log
pipeline::

    18:42:07 INFO     realtime      window scored  camera=CAM-02-CONCOURSE level=HIGH
                                                   score=64.2 density=1.412 people=17

Use :func:`get_logger` for a plain logger and :meth:`StructuredLogger.bind` to attach
context that every subsequent line should carry (the camera name, typically).
"""

from __future__ import annotations

import logging
import sys
from typing import Any

_CONFIGURED = False

# Values that are noise in a console line.
_SKIP = (None, "")


def _render(context: dict[str, Any], fields: dict[str, Any]) -> str:
    parts = []
    for key, value in {**context, **fields}.items():
        if value in _SKIP:
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:.3f}".rstrip("0").rstrip("."))
        else:
            parts.append(f"{key}={value}")
    return "  ".join(parts)


class StructuredLogger:
    """A thin logging wrapper that appends ``key=value`` pairs to each message."""

    def __init__(self, logger: logging.Logger, context: dict[str, Any] | None = None) -> None:
        self._logger = logger
        self._context = context or {}

    def bind(self, **context: Any) -> StructuredLogger:
        """Return a new logger carrying additional persistent context."""
        return StructuredLogger(self._logger, {**self._context, **context})

    # Every parameter here is positional-only (note the `/`). Without it a field named
    # `level`, `message` or `exc_info` -- and `level` is the single most natural field
    # name in this codebase -- collides with the parameter and raises TypeError instead
    # of being logged.
    def _log(self, level: int, message: str, exc_info: bool, /, **fields: Any) -> None:
        if not self._logger.isEnabledFor(level):
            return
        rendered = _render(self._context, fields)
        self._logger.log(
            level, "%s%s", message.ljust(22), f"  {rendered}" if rendered else "",
            exc_info=exc_info,
        )

    def debug(self, message: str, /, **fields: Any) -> None:
        self._log(logging.DEBUG, message, False, **fields)

    def info(self, message: str, /, **fields: Any) -> None:
        self._log(logging.INFO, message, False, **fields)

    def warning(self, message: str, /, **fields: Any) -> None:
        self._log(logging.WARNING, message, False, **fields)

    def error(self, message: str, /, **fields: Any) -> None:
        self._log(logging.ERROR, message, False, **fields)

    def exception(self, message: str, /, **fields: Any) -> None:
        self._log(logging.ERROR, message, True, **fields)


def get_logger(name: str, **context: Any) -> StructuredLogger:
    return StructuredLogger(logging.getLogger(name), context)


def configure_logging(level: str = "INFO", quiet_libraries: bool = True) -> None:
    """Set up console logging. Idempotent, so calling it twice does not double output."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)-22s %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    if quiet_libraries:
        # These are chatty at INFO and drown out the per-camera lines that matter.
        for noisy in ("ultralytics", "matplotlib", "PIL", "pymongo", "pymongo.command"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


__all__ = ["StructuredLogger", "configure_logging", "get_logger"]
