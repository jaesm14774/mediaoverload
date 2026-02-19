"""發布服務實現"""
import os
import re
from typing import List, Dict, Any, Optional
from lib.services.interfaces.publishing_service import IPublishingService
from lib.social_media import MediaPost, SocialMediaManager, InstagramPlatform, TwitterPlatform, FacebookPlatform
from utils.image import ImageProcessor
from utils.logger import setup_logger


class PublishingService(IPublishingService):
    """發布服務實現
    
    處理媒體檔案格式轉換並發布到社群媒體平台。
    """
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.social_media_manager = SocialMediaManager()
    
    def process_media(self, media_paths: List[str], output_dir: str) -> List[str]:
        """處理媒體檔案（格式轉換等）
        
        Args:
            media_paths: 媒體檔案路徑列表
            output_dir: 輸出目錄
            
        Returns:
            處理後的媒體檔案路徑列表
        """
        from utils.image import ImageConverter
        converter = ImageConverter(quality=95)
        
        self.logger.info("開始處理媒體檔案")
        processed_paths = []
        
        for media_path in media_paths:
            if not os.path.exists(media_path):
                self.logger.warning(f"媒體檔案不存在，跳過: {media_path}")
                continue
            
            if media_path.lower().endswith(('.mp4', '.avi', '.mov', '.gif', '.webm')):
                processed_paths.append(media_path)
                continue
            
            if media_path.lower().endswith('.jpg'):
                processed_paths.append(media_path)
                continue
            
            converted_path = converter.convert_to_jpg(media_path, output_dir=None)
            if converted_path and os.path.exists(converted_path):
                processed_paths.append(converted_path)
                try:
                    os.remove(media_path)
                except Exception as e:
                    self.logger.warning(f"無法刪除原始檔案 {media_path}: {e}")
            else:
                self.logger.warning(f"圖片轉換失敗，使用原始檔案: {media_path}")
                processed_paths.append(media_path)
        
        self.logger.info(f"媒體處理完成，共處理 {len(processed_paths)} 個檔案")
        return processed_paths
    
    def publish_to_social_media(self,
                              post: MediaPost,
                              platforms: Optional[List[str]] = None) -> Dict[str, bool]:
        """發布到社群媒體
        
        Args:
            post: 媒體貼文物件
            platforms: 要發布的平台列表（None 則發布到所有平台）
            
        Returns:
            各平台發布結果的字典
        """
        self.logger.info("開始社群媒體上傳")
        
        if platforms:
            results = {}
            for platform in platforms:
                result = self.social_media_manager.upload_to_platform(platform, post)
                results[platform] = result
                if result:
                    self.logger.info(f'{platform} 上傳成功')
                else:
                    self.logger.warning(f'{platform} 上傳失敗')
            return results
        else:
            return self.social_media_manager.upload_to_all(post)
    
    def register_platform(self, 
                         platform_name: str,
                         platform_config: Dict[str, Any]) -> None:
        """註冊社群媒體平台
        
        Args:
            platform_name: 平台名稱
            platform_config: 平台配置字典
        """
        platform_name_lower = platform_name.lower()
        
        if platform_name_lower == 'instagram':
            platform = InstagramPlatform(
                config_folder_path=platform_config['config_folder_path'],
                prefix=platform_config.get('prefix', '')
            )
            self.social_media_manager.register_platform(platform_name, platform)
            self.logger.info(f"已註冊平台: {platform_name}")
        elif platform_name_lower == 'twitter':
            platform = TwitterPlatform(
                config_folder_path=platform_config['config_folder_path'],
                prefix=platform_config.get('prefix', '')
            )
            self.social_media_manager.register_platform(platform_name, platform)
            self.logger.info(f"已註冊平台: {platform_name}")
        elif platform_name_lower == 'facebook':
            platform = FacebookPlatform(
                config_folder_path=platform_config['config_folder_path'],
                prefix=platform_config.get('prefix', '')
            )
            self.social_media_manager.register_platform(platform_name, platform)
            self.logger.info(f"已註冊平台: {platform_name}")
        else:
            raise ValueError(f"不支援的平台: {platform_name}") 