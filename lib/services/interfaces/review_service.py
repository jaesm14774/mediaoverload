"""審核服務介面"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional


class IReviewService(ABC):
    """審核服務介面
    
    處理與 Discord 的互動，進行內容審核
    """
    
    @abstractmethod
    async def review_content(self,
                           text: str,
                           media_paths: List[str],
                           timeout: int = 3600) -> Tuple[bool, str, Optional[str], Optional[List[int]]]:
        """審核內容
        
        Args:
            text: 要審核的文字內容
            media_paths: 要審核的圖片路徑列表
            timeout: 審核超時時間（秒）
            
        Returns:
            Tuple 包含：
            - result: 審核結果（通過/拒絕）
            - user: 審核用戶
            - edited_content: 編輯後的內容（如果有）
            - selected_indices: 選中的圖片索引列表
        """
        pass
    
    @abstractmethod
    def configure_discord(self, token: str, channel_id: int) -> None:
        """配置 Discord 設定
        
        Args:
            token: Discord bot token
            channel_id: Discord 頻道 ID
        """
        pass 