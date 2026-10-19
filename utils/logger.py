# utils/logger.py
"""
Shared application logger.

Logs to stdout (UTF-8, so the emoji status markers render on Windows too) and
to LOG_FILE. LOG_LEVEL and LOG_FILE come from the environment / .env.
"""
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def setup_logger(name: str = "ai_analyst", log_file: str = None, level: str = "INFO") -> logging.Logger:
    """
    Setup logger with console and file handlers

    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level

    Returns:
        Configured logger
    """

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()

    # Formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)

    # Fix for Windows emoji support
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    logger.addHandler(console_handler)

    # File handler with UTF-8 encoding
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


# Create default logger
logger = setup_logger(
    name="ai_analyst",
    log_file=os.getenv("LOG_FILE", "./logs/app.log"),
    level=os.getenv("LOG_LEVEL", "INFO"),
)


def log_section(title: str):
    """Log a section header"""
    logger.info("=" * 70)
    logger.info(f" {title}")
    logger.info("=" * 70)


def log_success(message: str):
    """Log success message"""
    logger.info(f"✅ {message}")


def log_error(message: str):
    """Log error message"""
    logger.error(f"❌ {message}")


def log_warning(message: str):
    """Log warning message"""
    logger.warning(f"⚠️  {message}")