import os
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def remove_optuna_files():
    # Path to the logs directory (we will check for 'logs' directory first, if it doesn't exist, we'll check the current directory)
    logs_directory = 'logs'
    db_pattern = '*.db'  # Pattern for matching multiple SQLite database files

    # If 'logs' directory doesn't exist, look for log files in the current directory
    if os.path.exists(logs_directory):
        log_files = glob.glob(os.path.join(logs_directory, '*.log'))  # Search for *.log files in the 'logs' directory
    else:
        log_files = glob.glob('*.log')  # Search for *.log files in the current directory

    if log_files:
        for log_file in log_files:
            try:
                os.remove(log_file)
                logger.info(f"Removed log file: {log_file}")
            except OSError as e:
                logger.error(f"Error while removing log file {log_file}: {e}")
    else:
        logger.warning("No log files found to delete.")

    # Removing SQLite database files matching the pattern (e.g., 'optuna_*.db')
    db_files = glob.glob(db_pattern)  # Find all files matching the pattern 'optuna_*.db'

    if db_files:
        for db_file in db_files:
            try:
                os.remove(db_file)
                logger.info(f"Removed database file: {db_file}")
            except OSError as e:
                logger.error(f"Error while removing database file {db_file}: {e}")
    else:
        logger.warning(f"No database files matching the pattern '{db_pattern}' were found.")


# Example call to the function
remove_optuna_files()
