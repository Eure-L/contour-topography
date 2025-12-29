import logging
from typing import Optional

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages based on level."""

    COLORS = {
        'DEBUG': '\033[90m',    # Light grey
        'INFO': '\033[94m',     # Blue
        'WARNING': '\033[93m', # Yellow
        'ERROR': '\033[91m',   # Red
        'CRITICAL': '\033[91m' # Red
    }

    RESET = '\033[0m'

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.msg = f"{self.COLORS[levelname]}{record.msg}{self.RESET}"
            record.levelname = f"{self.COLORS[levelname]}{record.levelname}{self.RESET}"

            formatted_message = super().format(record)

            formatted_message = formatted_message.replace(self.RESET, '')
            formatted_message = formatted_message.replace(self.COLORS[levelname], '')

            return f"{self.COLORS[levelname]}{formatted_message}{self.RESET}"
        return super().format(record)

def set_logger(level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure and return a logger with the specified settings.

    :param name: Name of the logger
    :param level: Logging level (default: logging.INFO)
    :param log_file: Optional file path to save logs to
    :return: Configured logger instance
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = ColoredFormatter('%(asctime)s | %(levelname)s | %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger