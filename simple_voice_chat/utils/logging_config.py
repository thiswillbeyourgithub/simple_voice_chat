import sys
from pathlib import Path
from typing import Optional

from loguru import logger

def setup_logging(
    console_log_level: str,
    log_file_path: Optional[Path],
    verbose_mode: bool,
):
    """
    Configures Loguru logging for the application.

    Args:
        console_log_level: The logging level for the console (e.g., "INFO", "DEBUG").
        log_file_path: The full path to the log file. If None, file logging is disabled.
        verbose_mode: If True, enables more detailed console logging and backtrace/diagnose.
    """
    logger.remove()  # Remove default handler

    # Define console format based on level (with color)
    log_format_debug_console = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    log_format_info_console = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    log_format_console = log_format_debug_console if console_log_level == "DEBUG" else log_format_info_console

    # Add console logger
    logger.add(
        sys.stderr,
        level=console_log_level,
        format=log_format_console,
        colorize=True,
        enqueue=True,
        backtrace=verbose_mode,  # Enable backtrace/diagnose based on verbose_mode
        diagnose=verbose_mode,
    )

    if log_file_path:
        try:
            # Define file format (without color)
            log_format_file = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"

            # Add file logger
            logger.add(
                log_file_path,
                level="DEBUG",  # Always log DEBUG level to file
                format=log_format_file,
                rotation="10 MB",
                retention=5,
                encoding="utf-8",
                enqueue=True,
                backtrace=True,  # Always include backtrace in file logs
                diagnose=True,  # Always include diagnose info in file logs
            )
            # Log the path *after* adding the file sink
            logger.debug(f"Logging to file: {log_file_path}")
        except Exception as e:
            # If file logging setup fails, console logger is already active.
            logger.error(f"Failed to set up file logging to {log_file_path}: {e}. File logging disabled.")
    else:
        logger.info("File logging is disabled as no log file path was provided (or directory setup failed).")
