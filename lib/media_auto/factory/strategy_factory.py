from typing import Dict, Type
from lib.media_auto.strategies.base_strategy import ContentStrategy
from lib.media_auto.strategies.image_strategies import (
    Text2ImageStrategy,
    Image2ImageStrategy,
    Text2VideoStrategy
)

class StrategyFactory:
    """策略工廠類"""
    
    _strategies: Dict[str, Type[ContentStrategy]] = {
        'text2img': Text2ImageStrategy,
        'img2img': Image2ImageStrategy,
        'text2video': Text2VideoStrategy
    }
    
    @classmethod
    def get_strategy(cls, strategy_type: str) -> ContentStrategy:
        """獲取對應的策略實例"""
        strategy_class = cls._strategies.get(strategy_type)
        if not strategy_class:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        return strategy_class()
    
    @classmethod
    def register_strategy(cls, strategy_type: str, strategy_class: Type[ContentStrategy]):
        """註冊新的策略"""
        cls._strategies[strategy_type] = strategy_class