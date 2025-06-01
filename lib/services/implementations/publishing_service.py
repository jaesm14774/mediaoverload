"""發布服務實現"""
import re
from typing import List, Dict, Any, Optional
from lib.services.interfaces.publishing_service import IPublishingService
from lib.social_media import MediaPost, SocialMediaManager, InstagramPlatform
from utils.image import ImageProcessor
from utils.logger import setup_logger


class PublishingService(IPublishingService):
    """發布服務實現"""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.social_media_manager = SocialMediaManager()
    
    def process_images(self, image_paths: List[str], output_dir: str) -> List[str]:
        """處理圖片（格式轉換等）"""
        self.logger.info("開始圖片處理")
        
        # 使用 ImageProcessor 處理圖片
        image_processor = ImageProcessor(output_dir)
        image_processor.main_process()
        
        # 將圖片路徑轉換為 jpg 格式
        processed_paths = []
        for image_path in image_paths:
            processed_path = re.sub(string=image_path, pattern=r'\.png|\.jpeg', repl='.jpg')
            processed_paths.append(processed_path)
        
        self.logger.info(f"圖片處理完成，共處理 {len(processed_paths)} 張圖片")
        return processed_paths
    
    def publish_to_social_media(self,
                              post: MediaPost,
                              platforms: Optional[List[str]] = None) -> Dict[str, bool]:
        """發布到社群媒體"""
        self.logger.info("開始社群媒體上傳")
        
        if platforms:
            results = {}
            for platform in platforms:
                # try:
                #     result = self.social_media_manager.upload_to_platform(platform, post)
                #     results[platform] = result
                #     if result:
                #         self.logger.info(f'{platform} 上傳成功')
                #     else:
                #         self.logger.warning(f'{platform} 上傳失敗')
                # except Exception as e:
                #     self.logger.error(f'{platform} 上傳時發生錯誤: {str(e)}')
                #     results[platform] = False
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
        """註冊社群媒體平台"""
        # 根據平台類型創建對應的平台實例
        if platform_name.lower() == 'instagram':
            platform = InstagramPlatform(
                config_folder_path=platform_config['config_folder_path'],
                prefix=platform_config.get('prefix', '')
            )
            self.social_media_manager.register_platform(platform_name, platform)
            self.logger.info(f"已註冊平台: {platform_name}")
        else:
            raise ValueError(f"不支援的平台: {platform_name}") 