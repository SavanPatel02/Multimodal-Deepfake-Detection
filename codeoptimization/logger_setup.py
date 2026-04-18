"""
Shared logging utility for all codeoptimization scripts.

Every run saves a timestamped log file to codeoptimization/logs/.
Logs are written to both console and file simultaneously.

Usage:
    from logger_setup import setup_logger
    log = setup_logger("train_pair_M1")  # → logs/train_pair_M1_20260401_143022.log
    log.info("Starting training...")
"""

import os
import logging
from datetime import datetime

LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def setup_logger(name: str) -> logging.Logger:
    """
    Create a logger that writes to both console and
    codeoptimization/logs/<name>_<timestamp>.log

    Args:
        name: Identifier for this run (e.g. "train_pair_M1", "evaluate", "predict_video")

    Returns:
        Configured logging.Logger instance.
    """
    os.makedirs(LOGS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = os.path.join(LOGS_DIR, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if setup_logger is called more than once
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S"
    )

    # Console handler — INFO and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # File handler — DEBUG and above (captures everything)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.info(f"Logging to: {log_file}")
    return logger
