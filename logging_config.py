from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logs.

    Produces a single-line JSON object per record with a stable set of
    top-level keys and merges any LoggerAdapter extra context.
    """

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        # Base envelope
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Merge in any structured context that was attached via LoggerAdapter
        # or passed in the "extra" dict. Avoid duplicating standard attributes.
        reserved = set(payload.keys()) | {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "levelno",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
        }

        for key, value in record.__dict__.items():
            if key in reserved or key.startswith("_"):
                continue
            payload[key] = self._normalize(value)

        # Include exception info, if present.
        if record.exc_info:
            payload["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
            payload["exc_message"] = str(record.exc_info[1]) if record.exc_info[1] else None

        return json.dumps(payload, ensure_ascii=False)

    def _normalize(self, value: Any) -> Any:
        """Best-effort conversion of complex objects into JSON-serializable data."""
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, Mapping):
            return {k: self._normalize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._normalize(v) for v in value]
        try:
            json.dumps(value)
            return value
        except TypeError:
            return repr(value)


def setup_logging() -> None:
    """Configure application-wide structured logging.

    Idempotent: safe to call multiple times (e.g. from Streamlit entrypoints).
    """
    if getattr(setup_logging, "_configured", False):  # type: ignore[attr-defined]
        return

    level_name = os.getenv("VIDEOCLONER_LOG_LEVEL", "INFO").upper()
    try:
        level = getattr(logging, level_name)
    except AttributeError:
        level = logging.INFO

    root = logging.getLogger()
    root.setLevel(level)

    # Clear any default handlers that Streamlit or other frameworks may attach,
    # so we have a predictable log format.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)

    # Make sure we do not propagate library debug noise by default.
    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("gspread").setLevel(logging.WARNING)

    setattr(setup_logging, "_configured", True)  # type: ignore[attr-defined]


class _MergingLoggerAdapter(logging.LoggerAdapter):
    """LoggerAdapter that merges per-call extra with the adapter's base context.

    The stdlib LoggerAdapter.process() overwrites kwargs["extra"] with
    self.extra, silently dropping any per-call extra fields. This subclass
    merges both dicts so per-call keys (like ``reason``) are preserved
    alongside adapter-level keys (like ``job_key``).
    """

    def process(
        self, msg: str, kwargs: Any
    ) -> tuple[str, Any]:
        call_extra = kwargs.get("extra", {}) or {}
        kwargs["extra"] = {**self.extra, **call_extra}
        return msg, kwargs


def get_logger(name: str, **context: Any) -> logging.LoggerAdapter:
    """Return a LoggerAdapter bound to the given name and context.

    Example:
        logger = get_logger(__name__, job_key=ctx.job_key)
        logger.info("Starting job")
    """
    setup_logging()
    base = logging.getLogger(name)
    return _MergingLoggerAdapter(base, extra=context or {})

