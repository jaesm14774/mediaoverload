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
    
    # 設定日誌檔案名稱（使用當前日期）
    log_filename = f"logs/{datetime.now().strftime('%Y-%m-%d')}.log"
    
    # 檔案處理器
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 控制台處理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 設定格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加處理器
    if not logger.handlers:
        logger.addHandler(file_handler)
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