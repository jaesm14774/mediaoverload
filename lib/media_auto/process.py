from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from abc import ABC

from lib.social_media import *

@dataclass
class CharacterConfig:
    """角色基礎配置"""
    character: str
    output_dir: str
    workflow_path: str
    similarity_threshold: float = 0.9
    generation_type: str = 'text2img'
    default_hashtags: list[str] = field(default_factory=list)
    additional_params: Optional[Dict[str, Any]] = field(default_factory=dict)
    group_name: str = ''

class BaseCharacter(ABC):
    """角色基礎類別"""
    config: CharacterConfig
    group_name: str = ''

    def __init__(self):
        self.config = self.get_default_config()

    def get_default_config(self) -> CharacterConfig:
        """返回角色的默認配置"""
        return CharacterConfig(
            character=self.character.lower(),
            output_dir=self.output_dir,
            workflow_path=self.workflow_path,
            similarity_threshold=self.similarity_threshold,
            generation_type=self.generation_type,
            default_hashtags=self.default_hashtags,
            additional_params=self.additional_params,
            group_name=self.group_name,
        )
    
    def get_generation_config(self, prompt: str) -> Dict[str, Any]:
        """具體實現生成配置"""
        config = {k: v for k, v in self.config.__dict__.items()}
        config.update({'prompt': prompt})
        return config

class KirbyProcess(BaseCharacter, SocialMediaMixin):
    character = 'kirby'
    output_dir = f'/app/output_image'
    workflow_path = '/app/configs/workflow/nova-anime-xl.json'
    similarity_threshold = 0.9
    generation_type = 'text2img'
    default_hashtags = ['nintendo']
    additional_params = {
        'images_per_description': 3,
        'is_negative': False
    }

    def __init__(self):
        BaseCharacter.__init__(self)
        SocialMediaMixin.__init__(self)
        # Register social media platforms
        self.register_social_media({
            'instagram': (InstagramPlatform, f'/app/configs/social_media/ig', self.character),
        })


class WobbuffetProcess(BaseCharacter, SocialMediaMixin):
    character = 'wobbuffet'
    output_dir = f'/app/output_image'
    workflow_path = '/app/configs/workflow/nova-anime-xl.json'
    similarity_threshold = 0.7
    generation_type = 'text2img'
    default_hashtags = ['pokemon']
    group_name = 'Pokemon'
    additional_params = {
        'images_per_description': 3,
        'is_negative': False
    }
    
    def __init__(self):
        BaseCharacter.__init__(self)
        SocialMediaMixin.__init__(self)
        # Register social media platforms
        self.register_social_media({
            'instagram': (InstagramPlatform, f'/app/configs/social_media/ig', self.character),
        })

class WaddledeeProcess(BaseCharacter, SocialMediaMixin):
    character = 'waddledee'
    output_dir = f'/app/output_image'
    workflow_path = '/app/configs/workflow/nova-anime-xl.json'
    similarity_threshold = 0.7
    generation_type = 'text2img'
    default_hashtags = ['kirby']
    group_name = 'Kirby'
    additional_params = {
        'images_per_description': 3,
        'is_negative': False
    }
    
    def __init__(self):
        BaseCharacter.__init__(self)
        SocialMediaMixin.__init__(self)
        # Register social media platforms
        self.register_social_media({
            'instagram': (InstagramPlatform, f'/app/configs/social_media/ig', self.character),
        })