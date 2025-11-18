from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

@dataclass
class GenerationConfig:
    """基礎生成配置類"""
    def __init__(self, **kwargs):
        self._attributes = kwargs  # 儲存所有屬性
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __getattr__(self, name):
        return None
    
    def get_all_attributes(self):
        """獲取所有設置的屬性"""
        return self._attributes

class ContentStrategy(ABC):
    """內容生成策略的抽象基類"""
    
    def load_config(self, config: GenerationConfig):
        self.config = config
    
    @abstractmethod
    def generate_description(self):
        """生成內容的抽象方法"""
        pass
    
    @abstractmethod
    def generate_media(self):
        """生成媒體的抽象方法（圖片或視頻）"""
        pass
    
    @abstractmethod
    def analyze_media_text_match(self, similarity_threshold):
        """分析媒體文本匹配的抽象方法"""
        pass
    
    @abstractmethod
    def generate_article_content(self):
        """生成發文的抽象方法"""
        pass
    
    def needs_user_review(self) -> bool:
        """檢查是否需要使用者審核
        
        預設返回 False，子類可以覆寫此方法來指示需要使用者審核
        例如：Text2Image2Video 在圖片生成後需要使用者選擇圖片
        
        Returns:
            bool: 如果需要使用者審核返回 True，否則返回 False
        """
        return False
    
    def get_review_items(self, max_items: int = 10) -> List[Dict[str, Any]]:
        """獲取需要審核的項目
        
        預設返回 filter_results，子類可以覆寫此方法來提供自定義的審核項目
        注意：Discord API 一次最多上傳 10 張圖片
        
        Args:
            max_items: 最大項目數量（預設 10，符合 Discord 限制）
            
        Returns:
            List[Dict[str, Any]]: 需要審核的項目列表，每個項目包含 media_path 等資訊
        """
        if hasattr(self, 'filter_results'):
            return self.filter_results[:max_items]
        return []
    
    def continue_after_review(self, selected_indices: List[int]) -> bool:
        """在使用者審核後繼續執行後續階段
        
        預設返回 False（不需要後續操作），子類可以覆寫此方法來執行後續階段
        例如：Text2Image2Video 在使用者選擇圖片後需要生成影片
        
        Args:
            selected_indices: 使用者選擇的項目索引列表（相對於 get_review_items 返回的列表）
            
        Returns:
            bool: 如果成功繼續執行返回 True，否則返回 False
        """
        return False
    
    def should_generate_article_now(self) -> bool:
        """判斷是否應該現在生成文章內容
        
        預設返回 True，子類可以覆寫此方法來延遲文章生成
        例如：Text2Image2Video 應該在影片生成後才生成文章內容
        
        Returns:
            bool: 如果應該現在生成文章內容返回 True，否則返回 False
        """
        return True