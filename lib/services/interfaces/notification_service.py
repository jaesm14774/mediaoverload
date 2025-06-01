"""通知服務介面"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class INotificationService(ABC):
    """通知服務介面
    
    統一處理成功和失敗的通知
    """
    
    @abstractmethod
    def notify_success(self,
                      character: str,
                      execution_time: float,
                      image_count: int,
                      additional_info: Optional[Dict[str, Any]] = None) -> None:
        """發送成功通知
        
        Args:
            character: 角色名稱
            execution_time: 執行時間（秒）
            image_count: 成功上傳的圖片數量
            additional_info: 額外資訊
        """
        pass
    
    @abstractmethod
    def notify_error(self,
                    character: str,
                    error_message: str,
                    additional_info: Optional[Dict[str, Any]] = None) -> None:
        """發送錯誤通知
        
        Args:
            character: 角色名稱
            error_message: 錯誤訊息
            additional_info: 額外資訊
        """
        pass
    
    @abstractmethod
    def configure_webhook(self, webhook_type: str, webhook_url: str) -> None:
        """配置 webhook
        
        Args:
            webhook_type: webhook 類型（如 'success', 'error'）
            webhook_url: webhook URL
        """
        pass 