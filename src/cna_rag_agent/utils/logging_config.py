# src/cna_rag_agent/utils/logging_config.py
import logging
import sys
from pathlib import Path # Added for type hinting if needed, not strictly necessary here

# Attempt to import config values. This creates a slight circular dependency risk
# if this module is imported before config can resolve its own paths.
# A common pattern is to pass config values to this setup function or have config call it.
# For simplicity now, direct import, assuming config.py is loaded early.
try:
    from ..config import LOG_LEVEL, LOG_FILE_PATH
except ImportError:
    # Fallback if this module is somehow loaded in a context where relative import fails
    # or config isn't set up (e.g., during a unit test of this module itself).
    print("WARNING: Could not import LOG_LEVEL, LOG_FILE_PATH from config. Using defaults for logging.")
    LOG_LEVEL = "INFO"
    LOG_FILE_PATH = Path(".") / "temp_app.log" # Temporary fallback

_logger_initialized = False

def setup_logging(log_level_str: str = LOG_LEVEL, log_file: Path = LOG_FILE_PATH) -> logging.Logger:
    """
    Configures and returns a logger for the application.
    Avoids adding multiple handlers if called multiple times.
    """
    global _logger_initialized
    logger = logging.getLogger("cna_rag_agent") # Get a specific logger for our app

    if _logger_initialized and logger.hasHandlers():
         # Logger already configured by a previous call in this session
        return logger

    numeric_level = getattr(logging, log_level_str.upper(), None)
    if not isinstance(numeric_level, int):
        # Fallback to INFO if an invalid level string is provided
        print(f"WARNING: Invalid log level '{log_level_str}'. Defaulting to INFO.")
        numeric_level = logging.INFO

    logger.setLevel(numeric_level)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level) # Set level for handler too

    # File Handler
    # Ensure the directory for the log file exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode='a') # Append mode
    file_handler.setLevel(numeric_level) # Set level for handler too

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - [%(module)s.%(funcName)s:%(lineno)d] - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.propagate = False # Prevent root logger from also handling these messages if it's configured
    _logger_initialized = True

    # Test message to confirm setup
    # logger.info(f"Logging initialized. Level: {log_level_str}. File: {log_file}")
    return logger

# Initialize and expose the logger for other modules to import
logger = setup_logging()