"""內容生成服務介面"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from lib.media_auto.strategies.base_strategy import GenerationConfig


class IContentGenerationService(ABC):
    """內容生成服務介面
    
    封裝 StrategyFactory 和具體的生成策略，負責實際的內容生成
    """
    
    @abstractmethod
    def generate_content(self, config: GenerationConfig) -> Dict[str, Any]:
        """生成內容
        
        Args:
            config: 生成配置
            
        Returns:
            包含生成結果的字典，包括：
            - descriptions: 生成的描述
            - images: 生成的圖片路徑
            - filter_results: 篩選後的結果
            - article_content: 生成的文章內容
        """
        pass
    
    @abstractmethod
    def generate_descriptions(self, config: GenerationConfig) -> List[str]:
        """生成描述文字"""
        pass
    
    @abstractmethod
    def generate_images(self, config: GenerationConfig, descriptions: List[str]) -> List[str]:
        """根據描述生成圖片"""
        pass
    
    @abstractmethod
    def analyze_image_text_match(self, 
                               images: List[str], 
                               descriptions: List[str],
                               similarity_threshold: float = 0.9) -> List[Dict[str, Any]]:
        """分析圖文匹配度"""
        pass
    
    @abstractmethod
    def generate_article(self, config: GenerationConfig, filter_results: List[Dict[str, Any]]) -> str:
        """生成文章內容"""
        pass 