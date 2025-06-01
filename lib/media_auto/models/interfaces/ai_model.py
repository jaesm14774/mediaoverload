from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """模型配置類"""
    model_name: str
    temperature: float = 0.3
    max_tokens: Optional[int] = None

class AIModelInterface(ABC):
    """AI 模型接口"""
    @abstractmethod
    def chat_completion(self, 
                       messages: List[dict],
                       images: Optional[List[str]] = None,
                       **kwargs) -> str:
        """執行聊天完成"""
        pass 