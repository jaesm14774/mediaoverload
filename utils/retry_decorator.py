"""Retry decorator with exponential backoff."""
import time
import functools
from typing import Callable, Any, Optional, Union, Tuple, Type


def retry(max_attempts: int = 5,
          delay: float = 10.0,
          backoff_factor: float = 1.5,
          exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
          on_retry: Optional[Callable[[int, Exception], None]] = None,
          on_failure: Optional[Callable[[Exception], None]] = None):
    """Retry decorator with exponential backoff.

    Args:
        max_attempts: Maximum retry attempts (including first try)
        delay: Initial delay in seconds
        backoff_factor: Delay multiplier after each retry
        exceptions: Exception types to retry on
        on_retry: Callback on each retry (attempt, exception)
        on_failure: Callback after all retries fail (exception)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 1:
                        print(f"函數 {func.__name__} 在第 {attempt} 次嘗試後成功")
                    return result

                except exceptions as e:
                    last_exception = e

                    if attempt < max_attempts:
                        if on_retry:
                            on_retry(attempt, e)
                        else:
                            print(f"函數 {func.__name__} 第 {attempt} 次嘗試失敗: {str(e)}")
                            print(f"等待 {current_delay:.1f} 秒後重試...")

                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        print(f"函數 {func.__name__} 所有 {max_attempts} 次嘗試都失敗")
                        if on_failure:
                            on_failure(e)
                        raise e

            if last_exception:
                raise last_exception

        return wrapper
    return decorator


def api_retry(max_attempts: int = 3):
    """Pre-configured retry for API calls."""
    return retry(
        max_attempts=max_attempts,
        delay=2.0,
        backoff_factor=1.5,
        exceptions=Exception
    )


def network_retry(max_attempts: int = 5):
    """Pre-configured retry for network requests."""
    return retry(
        max_attempts=max_attempts,
        delay=1.0,
        backoff_factor=2.0,
        exceptions=(ConnectionError, TimeoutError, OSError)
    )


def vision_api_retry(max_attempts: int = 10):
    """Pre-configured retry for vision API calls with detailed logging."""
    def custom_on_retry(attempt: int, exception: Exception):
        # 檢查是否為 Google API 超時錯誤
        exception_type = type(exception).__name__
        exception_msg = str(exception)
        
        if 'DeadlineExceeded' in exception_type or 'Deadline' in exception_msg or '504' in exception_msg:
            print(f"⚠️  視覺 API 請求超時（第 {attempt} 次嘗試）: {exception_msg}")
            print(f"   這可能是因為圖片處理需要較長時間，將增加等待時間後重試...")
            # 對於超時錯誤，使用更長的延遲時間
            delay_time = 5.0 * (1.5 ** (attempt - 1))  # 從 5 秒開始，而不是 2 秒
        else:
            print(f"視覺 API 第 {attempt} 次嘗試失敗: {exception_msg}")
            delay_time = 2.0 * (1.5 ** (attempt - 1))
        
        print(f"等待 {delay_time:.1f} 秒後重試...")

    return retry(
        max_attempts=max_attempts,
        delay=10.0,
        backoff_factor=3,
        exceptions=Exception,  # 捕獲所有異常，包括 DeadlineExceeded
        on_retry=custom_on_retry
    )
