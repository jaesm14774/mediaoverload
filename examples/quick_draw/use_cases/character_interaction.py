"""雙角色互動圖片生成使用案例

生成兩個角色的互動場景
"""

from .base_use_case import BaseUseCase
from lib.media_auto.strategies.base_strategy import GenerationConfig
from typing import Optional
import random


class CharacterInteractionUseCase(BaseUseCase):
    """雙角色互動圖片生成

    使用案例：生成兩個角色互動的場景

    範例：
        >>> use_case = CharacterInteractionUseCase()
        >>> result = use_case.execute(
        ...     main_character='Kirby',
        ...     secondary_character='Waddle Dee',
        ...     topic='friendship',
        ...     style='warm and cozy'
        ... )
    """

    def build_config(self,
                    main_character: Optional[str] = None,
                    secondary_character: Optional[str] = None,
                    workflow_name: str = 'nova-anime-xl',
                    topic: str = 'friendship and companionship',
                    style: str = 'minimalist style, simple white background',
                    images_per_description: int = 4,
                    group_name: Optional[str] = None) -> GenerationConfig:
        """建立雙角色互動生成配置

        Args:
            main_character: 主角色名稱，如果為 None 則隨機選擇
            secondary_character: 次要角色名稱，如果為 None 則隨機選擇
            workflow_name: 工作流名稱
            topic: 互動主題
            style: 風格描述
            images_per_description: 每個描述生成的圖片數量
            group_name: 角色群組名稱

        Returns:
            GenerationConfig 實例
        """
        # 隨機選擇角色
        if main_character is None:
            main_character = self.get_random_character(group_name)
            print(f"隨機選擇主角色: {main_character}")

        if secondary_character is None:
            # 確保次要角色與主角色不同
            characters = self.get_characters_by_group(group_name) if group_name else self.character_df['role_name_en'].tolist()
            available_characters = [c for c in characters if c != main_character]
            if available_characters:
                secondary_character = random.choice(available_characters)
                print(f"隨機選擇次要角色: {secondary_character}")
            else:
                # 如果找不到不同的角色，使用預設角色
                secondary_character = 'Pikachu' if main_character != 'Pikachu' else 'Mario'

        # 建立提示詞 - 使用雙角色系統提示詞
        prompt = f"""
Main Role: {main_character}
Secondary Role: {secondary_character}
topic: {topic}
style: {style}
        """.strip()

        # 建立配置
        config = GenerationConfig(
            character=main_character,
            secondary_character=secondary_character,  # 加入次要角色
            workflow_path=self.load_workflow(workflow_name),
            output_dir=self.output_folder,
            prompt=prompt,
            style=style,
            generation_type='text2img',
            image_system_prompt='two_character_interaction_generate_system_prompt',
            group_name=group_name or '',
            additional_params={
                'images_per_description': images_per_description,
                'is_negative': False
            }
        )

        return config

    def execute_batch(self,
                     main_character: str,
                     group_name: Optional[str] = None,
                     batch_size: int = 10,
                     **kwargs):
        """批次生成多組互動圖片

        Args:
            main_character: 固定的主角色
            group_name: 角色群組
            batch_size: 批次數量
            **kwargs: 其他參數

        Returns:
            所有批次的結果列表
        """
        results = []

        for i in range(batch_size):
            print(f"\n=== 批次 {i+1}/{batch_size} ===")

            # 每次隨機選擇不同的次要角色
            result = self.execute(
                main_character=main_character,
                secondary_character=None,  # 自動隨機選擇
                group_name=group_name,
                **kwargs
            )
            results.append(result)

        return results

    @classmethod
    def quick_execute(cls,
                     main_character: str,
                     secondary_character: str,
                     topic: str = 'friendship',
                     **kwargs):
        """快速執行雙角色互動生成

        Args:
            main_character: 主角色
            secondary_character: 次要角色
            topic: 互動主題
            **kwargs: 其他參數

        Returns:
            生成結果
        """
        use_case = cls()
        return use_case.execute(
            main_character=main_character,
            secondary_character=secondary_character,
            topic=topic,
            **kwargs
        )

