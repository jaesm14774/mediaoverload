"""Logging with daily file rotation and execution time tracking."""
import logging
import os
import time
from datetime import datetime
from functools import wraps
import asyncio

def setup_logger(name):
    """Create logger with daily rotating file handler and console output."""
    if not os.path.exists('logs'):
        os.makedirs('logs')

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    class DailyRotatingFileHandler(logging.FileHandler):
        """Automatically switches to new file when date changes."""
        def __init__(self, name):
            super().__init__(self._get_log_filename(), encoding='utf-8')
            self.name = name
            self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self.setFormatter(self.formatter)

        def _get_log_filename(self):
            return f"logs/{datetime.now().strftime('%Y-%m-%d')}.log"

        def emit(self, record):
            current_filename = self._get_log_filename()
            if self.baseFilename != current_filename:
                if self.stream:
                    self.stream.close()
                    self.stream = None
                self.baseFilename = current_filename
                self.stream = self._open()
            super().emit(record)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(DailyRotatingFileHandler(name))
        logger.addHandler(console_handler)

    return logger

def log_execution_time(logger):
    """Decorator to log function execution time (works with sync/async)."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            logger.info(f"開始執行 {func.__name__}")
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"{func.__name__} 執行完成，耗時: {execution_time:.2f} 秒")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} 執行失敗，耗時: {execution_time:.2f} 秒，錯誤: {str(e)}")
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            logger.info(f"開始執行 {func.__name__}")
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"{func.__name__} 執行完成，耗時: {execution_time:.2f} 秒")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} 執行失敗，耗時: {execution_time:.2f} 秒，錯誤: {str(e)}")
                raise

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
