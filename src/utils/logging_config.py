# src/utils/logging_config.py
import logging

def setup_logging(level=logging.INFO):
    """Configure logging with a specific level."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )