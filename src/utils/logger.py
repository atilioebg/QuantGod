import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from src.config import settings

def setup_logger(name: str = "SAIMP", log_level: int = logging.INFO) -> logging.Logger:
    """
    Configures a logger with console and file handlers.
    
    Args:
        name: Name of the logger.
        log_level: Logging level (default columns: INFO).
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Prevent duplicate handlers
    if logger.hasHandlers():
        return logger

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(module)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (Rotating)
    log_dir = settings.BASE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        filename=log_dir / "app.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def log_memory_usage(logger=None):
    """
    Logs current memory usage (RAM and VRAM).
    """
    if logger is None:
        logger = logging.getLogger("SAIMP")
        
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        ram_gb = mem_info.rss / (1024 ** 3)
        logger.info(f"[MEMORY] RAM: {ram_gb:.2f} GB")
    except ImportError:
        pass
        
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            vram_cached_gb = torch.cuda.memory_reserved() / (1024 ** 3)
            logger.info(f"[MEMORY] VRAM: {vram_gb:.2f} GB (Allocated) | {vram_cached_gb:.2f} GB (Reserved)")
    except ImportError:
        pass

logger = setup_logger()
UKNOWN_LOG_LEVEL = logging.INFO # fallback