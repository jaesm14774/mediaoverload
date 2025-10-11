"""單角色圖片生成使用案例

基於單一角色和主題生成圖片
"""

from .base_use_case import BaseUseCase
from lib.media_auto.strategies.base_strategy import GenerationConfig
from typing import Optional


class SingleCharacterUseCase(BaseUseCase):
    """單角色圖片生成

    使用案例：為指定角色生成單一主題的圖片

    範例：
        >>> use_case = SingleCharacterUseCase()
        >>> result = use_case.execute(
        ...     character='Kirby',
        ...     topic='peaceful sleeping',
        ...     style='minimalist style, simple white background'
        ... )
    """

    def build_config(self,
                    character: Optional[str] = None,
                    workflow_name: str = 'nova-anime-xl',
                    topic: str = '',
                    style: str = 'minimalist style, simple white background',
                    images_per_description: int = 4,
                    group_name: Optional[str] = None) -> GenerationConfig:
        """建立單角色生成配置

        Args:
            character: 角色名稱，如果為 None 則隨機選擇
            workflow_name: 工作流名稱
            topic: 主題描述
            style: 風格描述
            images_per_description: 每個描述生成的圖片數量
            group_name: 角色群組名稱，用於隨機選擇角色

        Returns:
            GenerationConfig 實例
        """
        # 如果沒有指定角色，隨機選擇一個
        if character is None:
            character = self.get_random_character(group_name)
            print(f"隨機選擇角色: {character}")

        # 建立提示詞
        prompt = f"""
Main character: {character}
topic: {topic}
style: {style}
        """.strip()

        # 建立配置
        config = GenerationConfig(
            character=character,
            workflow_path=self.load_workflow(workflow_name),
            output_dir=self.output_folder,
            prompt=prompt,
            style=style,
            generation_type='text2img',
            image_system_prompt='stable_diffusion_prompt',
            group_name=group_name or '',
            additional_params={
                'images_per_description': images_per_description,
                'is_negative': False
            }
        )

        return config

    @classmethod
    def quick_execute(cls,
                     character: str,
                     topic: str = 'peaceful moment',
                     **kwargs):
        """快速執行單角色生成

        Args:
            character: 角色名稱
            topic: 主題
            **kwargs: 其他參數

        Returns:
            生成結果
        """
        use_case = cls()
        return use_case.execute(character=character, topic=topic, **kwargs)

