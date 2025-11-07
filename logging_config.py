import logging
import logging.handlers
import os
import sys


LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
APP_LOG_FILE = os.path.join(LOGS_DIR, "app.log")
EXC_LOG_FILE = os.path.join(LOGS_DIR, "exceptions.log")


def _ensure_logs_dir_exists() -> None:
    os.makedirs(LOGS_DIR, exist_ok=True)
    # Touch files so they exist even before first log write
    for path in (APP_LOG_FILE, EXC_LOG_FILE):
        try:
            if not os.path.exists(path):
                with open(path, "a", encoding="utf-8"):
                    pass
        except OSError:
            # If touching fails, continue; handlers will try to create later
            pass


def configure_logging() -> logging.Logger:
    """Configure root logger with two file handlers: general and exceptions.

    Returns the app logger for convenience.
    """
    _ensure_logs_dir_exists()

    logger = logging.getLogger("agentic_workflow")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if configure_logging is called multiple times
    if logger.handlers:
        return logger

    # General rotating file handler
    app_handler = logging.handlers.RotatingFileHandler(
        APP_LOG_FILE, maxBytes=2 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    app_handler.setLevel(logging.INFO)
    app_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    app_handler.setFormatter(app_formatter)

    # Exceptions-only file handler
    exc_handler = logging.handlers.RotatingFileHandler(
        EXC_LOG_FILE, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    exc_handler.setLevel(logging.ERROR)
    exc_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    exc_handler.setFormatter(exc_formatter)

    logger.addHandler(app_handler)
    logger.addHandler(exc_handler)

    # Also echo to console during development
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(app_formatter)
    logger.addHandler(console_handler)

    # Hook uncaught exceptions to the exception log
    def _handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = _handle_exception

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a child logger after ensuring configuration."""
    base = configure_logging()
    return base.getChild(name) if name else base


