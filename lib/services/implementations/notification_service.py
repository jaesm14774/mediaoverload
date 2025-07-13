"""通知服務實現"""
from typing import Dict, Any, Optional
from datetime import datetime
from lib.services.interfaces.notification_service import INotificationService
from lib.discord import DiscordNotify
from utils.logger import setup_logger


class NotificationService(INotificationService):
    """通知服務實現"""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.webhooks = {}
        self.discord_notify = DiscordNotify()
    
    def notify_success(self,
                      character: str,
                      execution_time: float,
                      media_count: int,
                      additional_info: Optional[Dict[str, Any]] = None) -> None:
        """發送成功通知"""
        if 'success' not in self.webhooks:
            self.logger.warning("成功通知的 webhook 未配置")
            return
        
        # 計算執行時間
        hours = int(execution_time // 3600)
        minutes = int((execution_time % 3600) // 60)
        seconds = int(execution_time % 60)
        
        # 構建通知訊息
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        success_message = (
            f"✨ 任務完成通知 ✨\n"
            f"🎭 角色: {character}\n"
            f"⏰ 完成時間: {current_time}\n"
            f"⌛ 總執行時間: {hours}小時 {minutes}分鐘 {seconds}秒\n"
            f"📸 成功上傳圖片數量: {media_count}張"
        )
        
        # 添加額外資訊
        if additional_info:
            for key, value in additional_info.items():
                success_message += f"\n{key}: {value}"
        
        # 發送通知
        self.discord_notify.webhook_url = self.webhooks['success']
        self.discord_notify.notify(success_message)
        self.logger.info("成功通知已發送")
    
    def notify_error(self,
                    character: str,
                    error_message: str,
                    additional_info: Optional[Dict[str, Any]] = None) -> None:
        """發送錯誤通知"""
        if 'error' not in self.webhooks:
            self.logger.warning("錯誤通知的 webhook 未配置")
            return
        
        # 構建錯誤訊息
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        error_notification = (
            f"❌ 錯誤通知 ❌\n"
            f"🎭 角色: {character}\n"
            f"⏰ 錯誤時間: {current_time}\n"
            f"💥 錯誤訊息: {error_message}\n"
        )
        
        # 添加額外資訊
        if additional_info:
            for key, value in additional_info.items():
                error_notification += f"\n{key}: {value}"
        
        # 發送通知
        self.discord_notify.webhook_url = self.webhooks['error']
        self.discord_notify.notify(error_notification)
        self.logger.info("錯誤通知已發送")
    
    def configure_webhook(self, webhook_type: str, webhook_url: str) -> None:
        """配置 webhook"""
        self.webhooks[webhook_type] = webhook_url
        self.logger.info(f"已配置 {webhook_type} webhook") 