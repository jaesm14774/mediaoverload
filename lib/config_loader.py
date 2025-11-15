"""配置載入器"""
import os
import yaml
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
from lib.media_auto.character_config import CharacterConfig


class ConfigLoader:
    """配置載入器，負責從 YAML 檔案載入角色配置"""
    
    @staticmethod
    def load_character_config(config_path: str) -> Dict[str, Any]:
        """載入角色配置檔案
        
        Args:
            config_path: 配置檔案路徑
            
        Returns:
            配置字典
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置檔案不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    @staticmethod
    def process_weighted_choice(weights: Dict[str, float]) -> str:
        """根據權重隨機選擇一個選項
        
        Args:
            weights: 選項及其權重的字典
            
        Returns:
            選中的選項
        """
        choices = list(weights.keys())
        probabilities = list(weights.values())
        
        # 確保權重總和為 1
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
        
        return str(np.random.choice(choices, size=1, p=probabilities)[0])
    
    @staticmethod
    def create_character_config(config_dict: Dict[str, Any]) -> CharacterConfig:
        """根據配置字典創建 CharacterConfig 實例
        
        Args:
            config_dict: 配置字典
            
        Returns:
            CharacterConfig 實例
        """
        character_info = config_dict.get('character', {})
        generation_info = config_dict.get('generation', {})
        social_media_info = config_dict.get('social_media', {})
        
        # 處理提示詞生成方法
        prompt_method = 'arbitrary'
        if 'prompt_method_weights' in generation_info:
            prompt_method = ConfigLoader.process_weighted_choice(
                generation_info['prompt_method_weights']
            )
        
        # 處理圖片系統提示
        image_system_prompt = 'stable_diffusion_prompt'
        if 'image_system_prompt_weights' in generation_info:
            image_system_prompt = ConfigLoader.process_weighted_choice(
                generation_info['image_system_prompt_weights']
            )

        # 處理風格選擇
        style = generation_info.get('style', '')
        if 'style_weights' in generation_info:
            style = ConfigLoader.process_weighted_choice(
                generation_info['style_weights']
            )

        # 處理生成類型選擇
        generation_type = generation_info.get('generation_type', 'text2img')
        if 'generation_type_weights' in generation_info:
            generation_type = ConfigLoader.process_weighted_choice(
                generation_info['generation_type_weights']
            )
        
        # 根據生成類型選擇工作流
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
        """獲取社群媒體配置
        
        Args:
            config_dict: 配置字典
            
        Returns:
            社群媒體配置字典
        """
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