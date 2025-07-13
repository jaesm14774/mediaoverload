"""發布服務介面"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from lib.social_media import MediaPost


class IPublishingService(ABC):
    """發布服務介面
    
    負責圖片的後續處理和社群媒體的上傳
    """    
    @abstractmethod
    def process_media(self, media_paths: List[str], output_dir: str) -> List[str]:
        """處理圖片（格式轉換等）
        
        Args:
            media_paths: 原始圖片路徑列表
            output_dir: 輸出目錄
            
        Returns:
            處理後的圖片路徑列表
        """
        pass
    
    @abstractmethod
    def publish_to_social_media(self,
                              post: MediaPost,
                              platforms: Optional[List[str]] = None) -> Dict[str, bool]:
        """發布到社群媒體
        
        Args:
            post: 要發布的內容
            platforms: 要發布的平台列表（None 表示所有平台）
            
        Returns:
            發布結果字典，key 為平台名稱，value 為是否成功
        """
        pass
    
    @abstractmethod
    def register_platform(self, 
                         platform_name: str,
                         platform_config: Dict[str, Any]) -> None:
        """註冊社群媒體平台
        
        Args:
            platform_name: 平台名稱
            platform_config: 平台配置
        """
        pass 