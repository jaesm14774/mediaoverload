"""提示詞生成服務介面"""
from abc import ABC, abstractmethod
from typing import Optional


class IPromptService(ABC):
    """提示詞生成服務介面
    
    負責根據角色和策略生成提示詞
    """
    
    @abstractmethod
    def generate_prompt(self, 
                       character: str, 
                       method: str = 'default',
                       temperature: float = 1.0,
                       extra_info: Optional[str] = None) -> str:
        """生成提示詞
        
        Args:
            character: 角色名稱
            method: 生成方法 ('default', 'news' 等)
            temperature: LLM 溫度參數
            extra_info: 額外資訊
            
        Returns:
            生成的提示詞
        """
        pass
    
    @abstractmethod
    def generate_by_arbitrary(self, character: str, temperature: float = 1.0) -> str:
        """使用默認方法生成提示詞"""
        pass
    
    @abstractmethod
    def generate_by_news(self, character: str, temperature: float = 1.0) -> str:
        """使用新聞資訊生成提示詞"""
        pass 