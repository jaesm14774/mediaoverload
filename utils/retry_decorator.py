import time
import functools
from typing import Callable, Any, Optional, Union, Tuple, Type


def retry(max_attempts: int = 3,
          delay: float = 2.0,
          backoff_factor: float = 1.5,
          exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
          on_retry: Optional[Callable[[int, Exception], None]] = None,
          on_failure: Optional[Callable[[Exception], None]] = None):
    """
    重試裝飾器，提供靈活的重試機制
    
    Args:
        max_attempts: 最大重試次數（包括第一次嘗試）
        delay: 初始重試延遲時間（秒）
        backoff_factor: 退避係數，每次重試延遲時間會乘以此係數
        exceptions: 需要重試的異常類型，可以是單個異常或異常元組
        on_retry: 重試時的回調函數，接收 (attempt, exception) 參數
        on_failure: 所有重試失敗後的回調函數，接收 exception 參數
    
    Returns:
        裝飾後的函數
    
    Example:
        @retry(max_attempts=3, delay=1.0, backoff_factor=2.0)
        def unstable_api_call():
            # 可能失敗的 API 調用
            pass
            
        @retry(
            max_attempts=5,
            delay=0.5,
            exceptions=(ConnectionError, TimeoutError),
            on_retry=lambda attempt, e: print(f"重試第 {attempt} 次: {e}")
        )
        def network_request():
            # 網絡請求
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    # 執行原函數
                    result = func(*args, **kwargs)
                    
                    # 如果是第一次以外的嘗試成功，記錄成功信息
                    if attempt > 1:
                        print(f"函數 {func.__name__} 在第 {attempt} 次嘗試後成功")
                    
                    return result
                    
                except exceptions as e:
                    last_exception = e
                    
                    # 如果不是最後一次嘗試
                    if attempt < max_attempts:
                        # 執行重試回調
                        if on_retry:
                            on_retry(attempt, e)
                        else:
                            print(f"函數 {func.__name__} 第 {attempt} 次嘗試失敗: {str(e)}")
                            print(f"等待 {current_delay:.1f} 秒後重試...")
                        
                        # 延遲
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        # 最後一次嘗試失敗
                        print(f"函數 {func.__name__} 所有 {max_attempts} 次嘗試都失敗")
                        
                        # 執行失敗回調
                        if on_failure:
                            on_failure(e)
                        
                        # 重新拋出最後一次的異常
                        raise e
            
            # 理論上不會執行到這裡
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


# 預設的重試裝飾器配置
def api_retry(max_attempts: int = 3):
    """API 調用專用的重試裝飾器"""
    return retry(
        max_attempts=max_attempts,
        delay=2.0,
        backoff_factor=1.5,
        exceptions=Exception
    )


def network_retry(max_attempts: int = 5):
    """網絡請求專用的重試裝飾器"""
    return retry(
        max_attempts=max_attempts,
        delay=1.0,
        backoff_factor=2.0,
        exceptions=(ConnectionError, TimeoutError, OSError)
    )


def vision_api_retry(max_attempts: int = 10):
    """視覺 API 專用的重試裝飾器，包含詳細的日誌"""
    def custom_on_retry(attempt: int, exception: Exception):
        print(f"視覺 API 第 {attempt} 次嘗試失敗: {str(exception)}")
        delay_time = 2.0 * (1.5 ** (attempt - 1))
        print(f"等待 {delay_time:.1f} 秒後重試...")
    
    return retry(
        max_attempts=max_attempts,
        delay=2.0,
        backoff_factor=1.5,
        exceptions=Exception,
        on_retry=custom_on_retry
    )
