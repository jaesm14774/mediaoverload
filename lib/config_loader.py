import os
import yaml
import numpy as np
from typing import Dict, Any
from lib.media_auto.character_config import CharacterConfig


class ConfigLoader:
    @staticmethod
    def load_character_config(config_path: str) -> Dict[str, Any]:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @staticmethod
    def process_weighted_choice(weights: Dict[str, float]) -> str:
        choices = list(weights.keys())
        probabilities = list(weights.values())

        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]

        return str(np.random.choice(choices, size=1, p=probabilities)[0])

    @staticmethod
    def create_character_config(config_dict: Dict[str, Any]) -> CharacterConfig:
        character_info = config_dict.get('character', {})
        generation_info = config_dict.get('generation', {})
        social_media_info = config_dict.get('social_media', {})

        prompt_method = 'arbitrary'
        if 'prompt_method_weights' in generation_info:
            prompt_method = ConfigLoader.process_weighted_choice(
                generation_info['prompt_method_weights']
            )

        image_system_prompt = 'stable_diffusion_prompt'
        if 'image_system_prompt_weights' in generation_info:
            image_system_prompt = ConfigLoader.process_weighted_choice(
                generation_info['image_system_prompt_weights']
            )

        style = generation_info.get('style', '')
        if 'style_weights' in generation_info:
            style = ConfigLoader.process_weighted_choice(
                generation_info['style_weights']
            )

        generation_type = generation_info.get('generation_type', 'text2img')
        if 'generation_type_weights' in generation_info:
            generation_type = ConfigLoader.process_weighted_choice(
                generation_info['generation_type_weights']
            )

        workflow_path = generation_info.get('workflow_path', '')
        if 'workflows' in generation_info and generation_type in generation_info['workflows']:
            workflow_path = generation_info['workflows'][generation_type]

        return CharacterConfig(
            character=character_info.get('name', '').lower(),
            output_dir=generation_info.get('output_dir', '/app/output_media'),
            workflow_path=workflow_path,
            similarity_threshold=generation_info.get('similarity_threshold', 0.9),
            generation_type=generation_type,
            default_hashtags=social_media_info.get('default_hashtags', []),
            additional_params=config_dict.get('additional_params', {}),
            group_name=character_info.get('group_name', ''),
            generate_prompt_method=prompt_method,
            image_system_prompt=image_system_prompt,
            style=style
        )

    @staticmethod
    def get_social_media_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        social_media_info = config_dict.get('social_media', {})
        platforms = social_media_info.get('platforms', {})

        platform_configs = {}
        for platform_name, platform_config in platforms.items():
            if platform_config.get('enabled', False):
                platform_configs[platform_name] = {
                    'config_folder_path': platform_config.get('config_folder_path', ''),
                    'prefix': platform_config.get('prefix', '')
                }

        return platform_configs
