"""新的角色基類 - 支援配置外部化"""
from abc import ABC
from typing import Dict, Any, Optional
from lib.media_auto.character_config import CharacterConfig, BaseCharacter
from lib.config_loader import ConfigLoader
from lib.social_media import SocialMediaMixin


class ConfigurableCharacter(ABC):
    """可配置的角色基類"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化角色
        
        Args:
            config_path: 配置檔案路徑，如果提供則從檔案載入配置
        """
        if config_path:
            # 從配置檔案載入
            config_dict = ConfigLoader.load_character_config(config_path)
            self.config = ConfigLoader.create_character_config(config_dict)
            self._social_media_config = ConfigLoader.get_social_media_config(config_dict)
            
            # 設定屬性以保持向後相容
            self.character = self.config.character
            self.output_dir = self.config.output_dir
            self.workflow_path = self.config.workflow_path
            self.similarity_threshold = self.config.similarity_threshold
            self.generation_type = self.config.generation_type
            self.default_hashtags = self.config.default_hashtags
            self.group_name = self.config.group_name
            self.generate_prompt_method = self.config.generate_prompt_method
            self.image_system_prompt = self.config.image_system_prompt
            self.additional_params = self.config.additional_params
        else:
            # 使用子類定義的屬性
            self.config = self.get_default_config()
            self._social_media_config = {}
    
    def get_default_config(self) -> CharacterConfig:
        """返回角色的默認配置"""
        return CharacterConfig(
            character=getattr(self, 'character', '').lower(),
            output_dir=getattr(self, 'output_dir', '/app/output_image'),
            workflow_path=getattr(self, 'workflow_path', ''),
            similarity_threshold=getattr(self, 'similarity_threshold', 0.9),
            generation_type=getattr(self, 'generation_type', 'text2img'),
            default_hashtags=getattr(self, 'default_hashtags', []),
            additional_params=getattr(self, 'additional_params', {}),
            group_name=getattr(self, 'group_name', ''),
            generate_prompt_method=getattr(self, 'generate_prompt_method', 'default'),
            image_system_prompt=getattr(self, 'image_system_prompt', 'default')
        )
    
    def get_generation_config(self, prompt: str) -> Dict[str, Any]:
        """具體實現生成配置"""
        config = {k: v for k, v in self.config.__dict__.items()}
        config.update({'prompt': prompt})
        return config


class ConfigurableCharacterWithSocialMedia(ConfigurableCharacter, SocialMediaMixin):
    """支援社群媒體功能的可配置角色"""
    
    def __init__(self, config_path: Optional[str] = None):
        ConfigurableCharacter.__init__(self, config_path)
        SocialMediaMixin.__init__(self)
        
        # 如果有社群媒體配置，自動註冊
        if self._social_media_config:
            self._register_platforms_from_config()
    
    def _register_platforms_from_config(self):
        """從配置註冊社群媒體平台"""
        from lib.social_media import InstagramPlatform
        
        platform_mapping = {
            'instagram': InstagramPlatform
        }
        
        platforms_to_register = {}
        for platform_name, platform_config in self._social_media_config.items():
            if platform_name in platform_mapping:
                platform_class = platform_mapping[platform_name]
                platforms_to_register[platform_name] = (
                    platform_class,
                    platform_config['config_folder_path'],
                    platform_config.get('prefix', self.character)
                )
        if platforms_to_register:
            self.register_social_media(platforms_to_register) 