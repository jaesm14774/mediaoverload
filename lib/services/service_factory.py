"""服務工廠 - 負責創建和組裝服務"""
import os
from typing import Optional
from dotenv import load_dotenv
from lib.database import db_pool
from lib.repositories.character_repository import CharacterRepository
from lib.repositories.news_repository import NewsRepository
from lib.services.implementations import (
    PromptService,
    ContentGenerationService,
    ReviewService,
    PublishingService,
    NotificationService,
    OrchestrationService
)


class ServiceFactory:
    """服務工廠 - 負責創建和組裝所有服務"""
    
    def __init__(self):
        # 載入環境變數
        load_dotenv('media_overload.env')
        load_dotenv('configs/social_media/discord/Discord.env')
        
        # 初始化資料庫連接
        self._init_database()
        
        # 服務實例
        self._prompt_service = None
        self._content_service = None
        self._review_service = None
        self._publishing_service = None
        self._notification_service = None
        self._orchestration_service = None
        
        # Repository 實例
        self._character_repository = None
        self._news_repository = None
    
    def _init_database(self):
        """初始化資料庫連接"""
        db_pool.initialize(
            'mysql',
            host=os.environ['mysql_host'],
            port=int(os.environ['mysql_port']),
            user=os.environ['mysql_user'],
            password=os.environ['mysql_password'],
            db_name=os.environ['mysql_db_name']
        )
        self.mysql_conn = db_pool.get_connection('mysql')
    
    def get_character_repository(self) -> CharacterRepository:
        """獲取角色資料庫存取層"""
        if self._character_repository is None:
            self._character_repository = CharacterRepository(self.mysql_conn)
        return self._character_repository
    
    def get_news_repository(self) -> NewsRepository:
        """獲取新聞資料庫存取層"""
        if self._news_repository is None:
            self._news_repository = NewsRepository(self.mysql_conn)
        return self._news_repository
    
    def get_prompt_service(self) -> PromptService:
        """獲取提示詞生成服務"""
        if self._prompt_service is None:
            self._prompt_service = PromptService(
                news_repository=self.get_news_repository(),
                character_repository=self.get_character_repository()
            )
        return self._prompt_service
    
    def get_content_service(self) -> ContentGenerationService:
        """獲取內容生成服務"""
        if self._content_service is None:
            self._content_service = ContentGenerationService(
                character_repository=self.get_character_repository()
            )
        return self._content_service
    
    def get_review_service(self) -> ReviewService:
        """獲取審核服務"""
        if self._review_service is None:
            self._review_service = ReviewService()
            # 配置 Discord
            self._review_service.configure_discord(
                token=os.environ.get('discord_review_bot_token', ''),
                channel_id=int(os.environ.get('discord_review_channel_id', 0))
            )
        return self._review_service
    
    def get_publishing_service(self) -> PublishingService:
        """獲取發布服務"""
        if self._publishing_service is None:
            self._publishing_service = PublishingService()
        return self._publishing_service
    
    def get_notification_service(self) -> NotificationService:
        """獲取通知服務"""
        if self._notification_service is None:
            self._notification_service = NotificationService()
            # 配置 webhooks
            self._notification_service.configure_webhook(
                'success',
                os.environ.get('程式執行狀態', '')
            )
            self._notification_service.configure_webhook(
                'error',
                os.environ.get('程式bug權杖', '')
            )
        return self._notification_service
    
    def get_orchestration_service(self) -> OrchestrationService:
        """獲取協調服務（新的 ContentProcessor）"""
        if self._orchestration_service is None:
            self._orchestration_service = OrchestrationService(
                character_repository=self.get_character_repository()
            )
            
            # 配置所有依賴的服務
            self._orchestration_service.configure_services(
                prompt_service=self.get_prompt_service(),
                content_service=self.get_content_service(),
                review_service=self.get_review_service(),
                publishing_service=self.get_publishing_service(),
                notification_service=self.get_notification_service()
            )
        
        return self._orchestration_service
    
    def cleanup(self):
        """清理資源"""
        db_pool.close_all() 