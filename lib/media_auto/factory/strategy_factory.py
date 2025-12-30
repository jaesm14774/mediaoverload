from typing import Dict, Type
from lib.media_auto.strategies.base_strategy import ContentStrategy
from lib.media_auto.strategies import (
    Text2ImageStrategy,
    Image2ImageStrategy,
    Text2Image2ImageStrategy,
    Text2VideoStrategy,
    Text2Image2VideoStrategy,
    Text2LongVideoStrategy,
    StickerPackStrategy
)

class StrategyFactory:
    """策略工廠類"""
    
    _strategies: Dict[str, Type[ContentStrategy]] = {
        # 支援舊命名
        'text2img': Text2ImageStrategy,
        'img2img': Image2ImageStrategy,
        # 新命名
        'text2image': Text2ImageStrategy,
        'image2image': Image2ImageStrategy,
        # 文生圖 -> 圖生圖策略
        'text2image2image': Text2Image2ImageStrategy,
        't2i2i': Text2Image2ImageStrategy,
        # 影片策略
        'text2video': Text2VideoStrategy,
        't2v': Text2VideoStrategy,
        # 文生圖 -> 圖生影片策略
        'text2image2video': Text2Image2VideoStrategy,
        't2i2v': Text2Image2VideoStrategy,
        # 文生長片策略（尾幀驅動）
        'text2longvideo': Text2LongVideoStrategy,
        't2lv': Text2LongVideoStrategy,
        # 貼圖包生成策略
        'sticker_pack': StickerPackStrategy,
        'stickerpack': StickerPackStrategy
    }
    
    @classmethod
    def get_strategy(cls, strategy_type: str, character_data_service=None, vision_manager=None) -> ContentStrategy:
        """獲取對應的策略實例"""
        strategy_class = cls._strategies.get(strategy_type)
        if not strategy_class:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        return strategy_class(character_data_service=character_data_service, vision_manager=vision_manager)
    
    @classmethod
    def register_strategy(cls, strategy_type: str, strategy_class: Type[ContentStrategy]):
        """註冊新的策略"""
        cls._strategies[strategy_type] = strategy_class