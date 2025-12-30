"""新聞資料服務介面"""
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class INewsDataService(ABC):
    """新聞資料服務介面"""
    
    @abstractmethod
    def get_random_news(self, 
                       date_filter: str,
                       exclude_categories: list[str] = None) -> Optional[Dict[str, Any]]:
        """獲取隨機新聞
        
        Args:
            date_filter: 日期篩選條件
            exclude_categories: 要排除的類別列表
            
        Returns:
            包含 title, keyword 的字典，如果沒有則返回 None
        """
        pass

