import logging
import os
import time
from datetime import datetime
from functools import wraps
import asyncio

def setup_logger(name):
    # 確保logs目錄存在
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # 設定logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    class DailyRotatingFileHandler(logging.FileHandler):
        def __init__(self, name):
            super().__init__(self._get_log_filename(), encoding='utf-8')  # 修正初始化順序
            self.name = name
            self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self.setFormatter(self.formatter)
        
        def _get_log_filename(self):
            return f"logs/{datetime.now().strftime('%Y-%m-%d')}.log"
        
        def emit(self, record):
            # 檢查是否需要切換到新的日誌文件
            current_filename = self._get_log_filename()
            if self.baseFilename != current_filename:
                if self.stream:
                    self.stream.close()
                    self.stream = None
                self.baseFilename = current_filename
                self.stream = self._open()
            super().emit(record)
    
    # 控制台處理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 設定格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # 添加處理器
    if not logger.handlers:
        logger.addHandler(DailyRotatingFileHandler(name))
        logger.addHandler(console_handler)
    
    return logger

def log_execution_time(logger):
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            logger.info(f"開始執行 {func.__name__}")
            try:
                result = await func(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time
                logger.info(f"{func.__name__} 執行完成，耗時: {execution_time:.2f} 秒")
                return result
            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time
                logger.error(f"{func.__name__} 執行失敗，耗時: {execution_time:.2f} 秒，錯誤: {str(e)}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            logger.info(f"開始執行 {func.__name__}")
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time
                logger.info(f"{func.__name__} 執行完成，耗時: {execution_time:.2f} 秒")
                return result
            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time
                logger.error(f"{func.__name__} 執行失敗，耗時: {execution_time:.2f} 秒，錯誤: {str(e)}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator 