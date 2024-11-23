import os
import logging
import colorlog
from datetime import datetime


def get_logger(name="main", log_dir="logs", log_level=logging.INFO):
    """
    Creates and returns a logger with colored console logs and file logging.

    Args:
        name (str): The name of the logger.
        log_dir (str): The directory where log files will be stored.
        log_level (int): The logging level (e.g., logging.DEBUG, logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """

    # Ensure the log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Create a log file name with a timestamp
    log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Create a file handler for logging to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)

    # Create a console handler for colored logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # Define log colors for different levels
    log_colors = {
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    }

    # Set formatter for file handler (no colors)
    file_formatter = logging.Formatter('[%(levelname)s] %(asctime)s: %(name)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Set formatter for console handler (with colors)
    console_formatter = colorlog.ColoredFormatter(
        '[%(levelname)s] %(asctime)s: %(name)s - %(message)s',
        log_colors=log_colors,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Prevent duplicate logging
    logger.propagate = False

    return logger
