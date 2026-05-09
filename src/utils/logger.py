import logging
import os

def setup_logger(name: str, log_file: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s"))
        logger.addHandler(handler)
    return logger
