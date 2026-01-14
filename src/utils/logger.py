"""Utility functions and logging setup."""
from loguru import logger
import sys
from pathlib import Path
from config import LOG_FILE, LOG_LEVEL

def setup_logging():
    """Configure loguru logger."""
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=LOG_LEVEL,
        colorize=True
    )
    
    # Add file handler
    logger.add(
        LOG_FILE,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=LOG_LEVEL,
        rotation="10 MB",
        retention="1 month",
        compression="zip"
    )
    
    return logger

def get_logger(name: str):
    """Get a logger instance."""
    return logger.bind(name=name)
