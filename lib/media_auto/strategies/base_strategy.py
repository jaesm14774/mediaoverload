from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

@dataclass
class GenerationConfig:
    """基礎生成配置類"""
    output_dir: str
    character: str
    prompt: str
    generation_type: str  # 'text2img', 'img2img', 'text2video' etc.
    workflow_path: Optional[str] = None
    default_hashtags: List[str] = field(default_factory=list)
    additional_params: Dict[str, Any] = None

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
