"""Quick Draw 使用案例模組

此模組提供各種圖片生成使用案例，基於 mediaoverload 框架。
使用簡化的內容生成服務，適合快速範例和測試。
"""

from .base_use_case import BaseUseCase
from .single_character import SingleCharacterUseCase
from .character_interaction import CharacterInteractionUseCase
from .news_based import NewsBasedUseCase
from .buddhist_style import BuddhistStyleUseCase
from .black_humor import BlackHumorUseCase
from .cinematic import CinematicUseCase

__all__ = [
    'BaseUseCase',
    'SingleCharacterUseCase',
    'CharacterInteractionUseCase',
    'NewsBasedUseCase',
    'BuddhistStyleUseCase',
    'BlackHumorUseCase',
    'CinematicUseCase',
]

