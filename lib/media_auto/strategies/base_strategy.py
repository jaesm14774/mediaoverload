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
    def generate_article_content(self):
        """生成發文的抽象方法"""
        pass
