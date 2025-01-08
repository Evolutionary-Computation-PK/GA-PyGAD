import logging
import os
import sys


def get_logger():
    logger_name = str(os.path.basename(sys.argv[0]).replace(".py", ""))
    return logging.getLogger(logger_name)


def setup_logging(log_file: str = None):
    """
    Configures the logging settings.

    Args:
        :param log_file: The path to the log file.
    """
    logger_name = os.path.basename(sys.argv[0]).replace(".py", "")

    if log_file is None:
        log_file = f"{logger_name}.log"

    log_dir = 'logs'
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger(str(logger_name))
    logger.setLevel(logging.DEBUG)

    while logger.handlers:
        logger.handlers.pop()

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger
