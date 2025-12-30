"""配置建構器

簡化 GenerationConfig 的建立過程
"""

import sys
from pathlib import Path

# 確保可以導入 mediaoverload 模組
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from lib.media_auto.strategies.base_strategy import GenerationConfig
from typing import Dict, List, Any, Optional


class ConfigBuilder:
    """配置建構器

    提供流暢的 API 來建立 GenerationConfig
    """

    def __init__(self):
        self._config = {}

    def with_character(self, character: str) -> 'ConfigBuilder':
        """設定主角色

        Args:
            character: 角色名稱

        Returns:
            self
        """
        self._config['character'] = character
        return self

    def with_workflow(self, workflow_path: str) -> 'ConfigBuilder':
        """設定工作流路徑

        Args:
            workflow_path: 工作流 JSON 檔案路徑

        Returns:
            self
        """
        self._config['workflow_path'] = workflow_path
        return self

    def with_output_dir(self, output_dir: str) -> 'ConfigBuilder':
        """設定輸出目錄

        Args:
            output_dir: 輸出目錄路徑

        Returns:
            self
        """
        self._config['output_dir'] = output_dir
        return self

    def with_prompt(self, prompt: str) -> 'ConfigBuilder':
        """設定提示詞

        Args:
            prompt: 生成提示詞

        Returns:
            self
        """
        self._config['prompt'] = prompt
        return self

    def with_style(self, style: str) -> 'ConfigBuilder':
        """設定風格

        Args:
            style: 風格描述

        Returns:
            self
        """
        self._config['style'] = style
        return self

    def with_image_system_prompt(self, prompt_key: str) -> 'ConfigBuilder':
        """設定圖片系統提示詞

        Args:
            prompt_key: 提示詞鍵值，可選值:
                - 'stable_diffusion_prompt'
                - 'two_character_interaction_generate_system_prompt'
                - 'buddhist_combined_image_system_prompt'
                - 'black_humor_system_prompt'
                - 'unbelievable_world_system_prompt'

        Returns:
            self
        """
        self._config['image_system_prompt'] = prompt_key
        return self

    def with_generation_type(self, gen_type: str) -> 'ConfigBuilder':
        """設定生成類型

        Args:
            gen_type: 生成類型，可選值:
                - 'text2img' 或 'text2image'
                - 'text2video' 或 't2v'
                - 'img2img' 或 'image2image'

        Returns:
            self
        """
        self._config['generation_type'] = gen_type
        return self

    def with_group_name(self, group_name: str) -> 'ConfigBuilder':
        """設定角色群組名稱

        Args:
            group_name: 角色群組名稱

        Returns:
            self
        """
        self._config['group_name'] = group_name
        return self

    def with_images_per_description(self, count: int) -> 'ConfigBuilder':
        """設定每個描述生成的圖片數量

        Args:
            count: 圖片數量

        Returns:
            self
        """
        if 'additional_params' not in self._config:
            self._config['additional_params'] = {}
        self._config['additional_params']['images_per_description'] = count
        return self

    def with_videos_per_description(self, count: int) -> 'ConfigBuilder':
        """設定每個描述生成的視頻數量

        Args:
            count: 視頻數量

        Returns:
            self
        """
        if 'additional_params' not in self._config:
            self._config['additional_params'] = {}
        self._config['additional_params']['videos_per_description'] = count
        return self

    def with_image_size(self, width: int, height: int) -> 'ConfigBuilder':
        """設定圖片尺寸

        Args:
            width: 寬度
            height: 高度

        Returns:
            self
        """
        if 'additional_params' not in self._config:
            self._config['additional_params'] = {}

        custom_updates = [
            {
                "node_type": "PrimitiveInt",
                "node_index": 0,
                "inputs": {"value": width}
            },
            {
                "node_type": "PrimitiveInt",
                "node_index": 1,
                "inputs": {"value": height}
            }
        ]
        self._config['additional_params']['custom_node_updates'] = custom_updates
        return self

    def with_negative_prompt(self, enabled: bool = True) -> 'ConfigBuilder':
        """設定是否使用負面提示詞

        Args:
            enabled: 是否啟用

        Returns:
            self
        """
        if 'additional_params' not in self._config:
            self._config['additional_params'] = {}
        self._config['additional_params']['is_negative'] = enabled
        return self

    def with_similarity_threshold(self, threshold: float) -> 'ConfigBuilder':
        """設定圖文相似度閾值

        Args:
            threshold: 相似度閾值 (0.0 ~ 1.0)

        Returns:
            self
        """
        self._config['similarity_threshold'] = threshold
        return self

    def with_default_hashtags(self, hashtags: List[str]) -> 'ConfigBuilder':
        """設定預設 hashtags

        Args:
            hashtags: hashtag 列表

        Returns:
            self
        """
        self._config['default_hashtags'] = hashtags
        return self

    def with_keywords(self, keywords: List[str]) -> 'ConfigBuilder':
        """設定關鍵字列表

        Args:
            keywords: 關鍵字列表，用於描述生成或搜尋

        Returns:
            self
        """
        self._config['keywords'] = keywords
        return self

    def with_video_workflow(self, workflow_path: str) -> 'ConfigBuilder':
        """設定影片工作流路徑（便捷方法）

        Args:
            workflow_path: 影片工作流 JSON 檔案路徑

        Returns:
            self
        """
        self._config['workflow_path'] = workflow_path
        self._config['generation_type'] = 'text2video'
        return self

    def with_secondary_character(self, character: str) -> 'ConfigBuilder':
        """設定次要角色

        Args:
            character: 次要角色名稱

        Returns:
            self
        """
        self._config['secondary_character'] = character
        return self

    def with_extract_description(self, enabled: bool = True) -> 'ConfigBuilder':
        """設定是否從圖片提取描述（用於 image2image）

        Args:
            enabled: 是否啟用

        Returns:
            self
        """
        self._config['extract_description'] = enabled
        return self

    def with_input_image(self, image_path: str) -> 'ConfigBuilder':
        """設定輸入圖片路徑（用於 image2image）

        Args:
            image_path: 輸入圖片路徑

        Returns:
            self
        """
        self._config['input_image_path'] = image_path
        self._config['generation_type'] = 'image2image'
        return self

    def with_denoise(self, denoise: float) -> 'ConfigBuilder':
        """設定 denoise 權重（用於 image2image）

        Args:
            denoise: denoise 權重 (0.0 ~ 1.0)

        Returns:
            self
        """
        if 'additional_params' not in self._config:
            self._config['additional_params'] = {}
        if 'image' not in self._config['additional_params']:
            self._config['additional_params']['image'] = {}
        self._config['additional_params']['image']['denoise'] = denoise
        return self

    def with_additional_params(self, **params) -> 'ConfigBuilder':
        """設定額外參數

        Args:
            **params: 額外參數

        Returns:
            self
        """
        if 'additional_params' not in self._config:
            self._config['additional_params'] = {}
        self._config['additional_params'].update(params)
        return self

    def build(self) -> GenerationConfig:
        """建立 GenerationConfig

        Returns:
            GenerationConfig 實例
        """
        return GenerationConfig(**self._config)

    @classmethod
    def quick_image_config(cls,
                          character: str,
                          workflow_path: str,
                          output_dir: str,
                          prompt: str,
                          style: str = 'minimalist style',
                          images_per_description: int = 4) -> GenerationConfig:
        """快速建立圖片生成配置

        Args:
            character: 角色名稱
            workflow_path: 工作流路徑
            output_dir: 輸出目錄
            prompt: 提示詞
            style: 風格
            images_per_description: 每個描述生成的圖片數量

        Returns:
            GenerationConfig 實例
        """
        return cls() \
            .with_character(character) \
            .with_workflow(workflow_path) \
            .with_output_dir(output_dir) \
            .with_prompt(prompt) \
            .with_style(style) \
            .with_generation_type('text2img') \
            .with_images_per_description(images_per_description) \
            .build()

    @classmethod
    def quick_video_config(cls,
                          character: str,
                          workflow_path: str,
                          output_dir: str,
                          prompt: str,
                          videos_per_description: int = 2) -> GenerationConfig:
        """快速建立視頻生成配置

        Args:
            character: 角色名稱
            workflow_path: 工作流路徑
            output_dir: 輸出目錄
            prompt: 提示詞
            videos_per_description: 每個描述生成的視頻數量

        Returns:
            GenerationConfig 實例
        """
        return cls() \
            .with_character(character) \
            .with_workflow(workflow_path) \
            .with_output_dir(output_dir) \
            .with_prompt(prompt) \
            .with_generation_type('text2video') \
            .with_videos_per_description(videos_per_description) \
            .build()

