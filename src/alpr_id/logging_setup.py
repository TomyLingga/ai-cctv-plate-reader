"""Structured logging: one JSON object per line, to the console and to a rotating file."""
from __future__ import annotations

import logging
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import structlog

# Greedy up to the last '@' before the path: a password may itself contain an unencoded '@'.
_URL_CREDENTIALS = re.compile(r"//([^:/@\s]+):[^/\s]*@")
# Some cameras take the password as a parameter in the path instead.
_SECRET_PARAM = re.compile(r"(?i)\b(pass(?:word|wd)?|pwd)=[^&\s/'\"]*")


def mask_url(text: str) -> str:
    """Hide the password of every URL inside `text`.

    Covers scheme://user:password@host and password=... parameters. Used for everything that
    leaves the process: log lines, API responses, error messages of the stream library.
    """
    return _SECRET_PARAM.sub(r"\1=***", _URL_CREDENTIALS.sub(r"//\1:***@", text))


def _mask_event(_logger: object, _method: str, event: dict) -> dict:
    """Last line of defence: no log line leaves the process with a camera password in it."""
    for key, value in event.items():
        if isinstance(value, str) and "://" in value:
            event[key] = mask_url(value)
    return event


def configure_logging(log_dir: str | Path | None = "data/logs", level: str = "INFO") -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_dir is not None:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(path / "alpr.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"))
    logging.basicConfig(level=level, format="%(message)s", handlers=handlers, force=True)
    # uvicorn's access log would add one line per request; clients poll every second.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            _mask_event,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
