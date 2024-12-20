import logging
import os
from pathlib import Path

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    # Create logger
    logger = logging.getLogger(name)
    
    # Only add handlers if none exist
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create logs directory if it doesn't exist
        log_dir = Path(os.getcwd()) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create handlers
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(log_dir / 'processing.log')
        
        # Create formatters and add it to handlers
        log_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(log_format)
        file_handler.setFormatter(log_format)
        
        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger

def setup_logger(name: str) -> logging.Logger:
    """Setup and return a logger instance."""
    return get_logger(name) 