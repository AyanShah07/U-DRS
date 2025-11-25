"""
Logging utilities for U-DRS system.
Provides structured logging with file and console output.
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(log_file: Path = None, level: str = "INFO"):
    """
    Configure logger with file and console handlers.
    
    Args:
        log_file: Path to log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Remove default handler
    logger.remove()
    
    # Console handler with color
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True
    )
    
    # File handler
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=level,
            rotation="10 MB",
            retention="1 week",
            compression="zip"
        )
    
    return logger


def get_logger(name: str = "U-DRS"):
    """Get logger instance with specified name."""
    return logger.bind(name=name)


# Default logger instance
default_logger = get_logger()
