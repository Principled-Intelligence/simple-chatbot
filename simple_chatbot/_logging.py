import logging
import os
import sys
from typing import Literal

from loguru import logger

_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "openai",
    "chromadb",
    "chromadb.telemetry",
    "LiteLLM",
    "LiteLLM Proxy",
    "LiteLLM Router",
    "litellm",
)

LogFormat = Literal["pretty", "json"]


class _InterceptHandler(logging.Handler):
    """Route stdlib logging records into loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno  # type: ignore[assignment]

        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def _format_extra(extra: dict) -> str:
    if not extra:
        return ""

    parts = []
    for key in sorted(extra):
        value = str(extra[key]).replace("{", "{{").replace("}", "}}")
        parts.append(f"{key}={value}")
    return " | " + " ".join(parts)


def _pretty_format(record: dict) -> str:
    extra = _format_extra(record["extra"])
    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
        f"{extra} — <level>{{message}}</level>\n{{exception}}"
    )


def _silence_litellm() -> None:
    """Stop litellm from print()-ing request/response bodies and embedding vectors.

    litellm prints via its own `print_verbose` helper (bypasses stdlib logging
    entirely) when `set_verbose=True` or when debug info isn't suppressed.
    It also uses a stdlib logger named `LiteLLM` (with a capital L) that
    needs to be silenced separately from the `litellm` module logger.
    """
    os.environ.setdefault("LITELLM_LOG", "WARNING")

    try:
        import litellm

        litellm.set_verbose = False
        litellm.suppress_debug_info = True
        if hasattr(litellm, "telemetry"):
            litellm.telemetry = False
    except Exception:
        pass


def setup_logging(level: str = "INFO", log_format: LogFormat = "pretty") -> None:
    logger.remove()
    if log_format == "json":
        logger.add(
            sys.stderr,
            level=level,
            serialize=True,
            enqueue=True,
            backtrace=False,
            diagnose=False,
        )
    else:
        logger.add(
            sys.stderr,
            level=level,
            format=_pretty_format,
            colorize=True,
            enqueue=True,
            backtrace=False,
            diagnose=False,
        )

    _silence_litellm()

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

    # Keep chatty third-party loggers at WARNING so they don't spam at INFO.
    # Their records still flow through loguru via the root InterceptHandler,
    # so levels stay consistent across the whole app.
    for name in _NOISY_LOGGERS:
        log = logging.getLogger(name)
        log.handlers = [_InterceptHandler()]
        log.setLevel(logging.WARNING)
        log.propagate = False

    # uvicorn / fastapi logging we want to keep at the user-chosen level.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        log = logging.getLogger(name)
        log.handlers = [_InterceptHandler()]
        log.propagate = False
