from abc import ABC
from typing import Dict, Any, Optional
from lib.media_auto.character_config import CharacterConfig
from lib.config_loader import ConfigLoader
from lib.social_media import SocialMediaMixin


class ConfigurableCharacter(ABC):
    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            config_dict = ConfigLoader.load_character_config(config_path)
            self.config = ConfigLoader.create_character_config(config_dict)
            self._social_media_config = ConfigLoader.get_social_media_config(config_dict)

            # Mirror attributes for backward compatibility
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
            self.config = self.get_default_config()
            self._social_media_config = {}

    def get_default_config(self) -> CharacterConfig:
        return CharacterConfig(
            character=getattr(self, 'character', '').lower(),
            output_dir=getattr(self, 'output_dir', '/app/output_media'),
            workflow_path=getattr(self, 'workflow_path', ''),
            similarity_threshold=getattr(self, 'similarity_threshold', 0.9),
            generation_type=getattr(self, 'generation_type', 'text2img'),
            default_hashtags=getattr(self, 'default_hashtags', []),
            additional_params=getattr(self, 'additional_params', {}),
            group_name=getattr(self, 'group_name', ''),
            generate_prompt_method=getattr(self, 'generate_prompt_method', 'arbitrary'),
            image_system_prompt=getattr(self, 'image_system_prompt', 'stable_diffusion_prompt'),
            style=getattr(self, 'style', '')
        )

    def get_generation_config(self, prompt: str) -> Dict[str, Any]:
        config = {k: v for k, v in self.config.__dict__.items()}
        config.update({'prompt': prompt})
        return config


class ConfigurableCharacterWithSocialMedia(ConfigurableCharacter, SocialMediaMixin):
    def __init__(self, config_path: Optional[str] = None):
        ConfigurableCharacter.__init__(self, config_path)
        SocialMediaMixin.__init__(self)

        if self._social_media_config:
            self._register_platforms_from_config()

    def _register_platforms_from_config(self):
        from lib.social_media import InstagramPlatform, TwitterPlatform, FacebookPlatform

        platform_mapping = {
            'instagram': InstagramPlatform,
            'twitter': TwitterPlatform,
            'facebook': FacebookPlatform
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
