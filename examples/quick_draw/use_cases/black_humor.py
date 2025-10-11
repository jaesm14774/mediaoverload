"""黑色幽默圖片生成使用案例

生成具有諷刺意味和黑色幽默的圖片
"""

from .base_use_case import BaseUseCase
from lib.media_auto.strategies.base_strategy import GenerationConfig
from typing import Optional
import random


class BlackHumorUseCase(BaseUseCase):
    """黑色幽默圖片生成

    使用案例：生成具有反差和諷刺意味的黑色幽默圖片

    範例:
        >>> use_case = BlackHumorUseCase()
        >>> result = use_case.execute(
        ...     main_character='Kirby',
        ...     secondary_character='Waddle Dee'
        ... )
    """

    def build_config(self,
                    main_character: Optional[str] = None,
                    secondary_character: Optional[str] = None,
                    workflow_name: str = 'nova-anime-xl',
                    style: str = 'minimalist style, pure background',
                    images_per_description: int = 4,
                    group_name: Optional[str] = None) -> GenerationConfig:
        """建立黑色幽默生成配置

        Args:
            main_character: 主角色名稱，如果為 None 則隨機選擇
            secondary_character: 次要角色名稱，如果為 None 則隨機選擇
            workflow_name: 工作流名稱
            style: 風格描述
            images_per_description: 每個描述生成的圖片數量
            group_name: 角色群組名稱

        Returns:
            GenerationConfig 實例
        """
        # 隨機選擇主角色
        if main_character is None:
            main_character = self.get_random_character(group_name)
            print(f"隨機選擇主角色: {main_character}")

        # 隨機選擇次要角色
        if secondary_character is None:
            characters = self.get_characters_by_group(group_name) if group_name else self.character_df['role_name_en'].tolist()
            available_characters = [c for c in characters if c != main_character]
            if available_characters:
                secondary_character = random.choice(available_characters)
            else:
                secondary_character = 'Pikachu' if main_character != 'Pikachu' else 'Mario'
            print(f"隨機選擇次要角色: {secondary_character}")

        # 建立提示詞 - 使用黑色幽默系統提示詞
        prompt = f"""
Main Role: {main_character}
Secondary Role: {secondary_character}
style: {style}
        """.strip()

        # 建立配置
        config = GenerationConfig(
            character=main_character,
            workflow_path=self.load_workflow(workflow_name),
            output_dir=self.output_folder,
            prompt=prompt,
            style=style,
            generation_type='text2img',
            image_system_prompt='black_humor_system_prompt',
            group_name=group_name or '',
            additional_params={
                'images_per_description': images_per_description,
                'is_negative': False
            }
        )

        return config

    def execute_batch(self,
                     main_character: Optional[str] = None,
                     group_name: Optional[str] = None,
                     batch_size: int = 20,
                     **kwargs):
        """批次生成多組黑色幽默圖片

        Args:
            main_character: 固定的主角色，如果為 None 則每次隨機選擇
            group_name: 角色群組
            batch_size: 批次數量
            **kwargs: 其他參數

        Returns:
            所有批次的結果列表
        """
        results = []

        for i in range(batch_size):
            print(f"\n=== 批次 {i+1}/{batch_size} ===")

            # 執行生成（角色會在 build_config 中自動隨機選擇）
            result = self.execute(
                main_character=main_character,
                secondary_character=None,  # 自動隨機選擇
                group_name=group_name,
                **kwargs
            )
            results.append(result)

        return results

    def execute_with_scenarios(self,
                              main_character: str,
                              scenarios: Optional[list] = None,
                              batch_size: int = 10,
                              **kwargs):
        """使用預定義場景批次生成

        Args:
            main_character: 主角色
            scenarios: 場景描述列表
            batch_size: 批次數量
            **kwargs: 其他參數

        Returns:
            所有批次的結果列表
        """
        # 預設黑色幽默場景
        if scenarios is None:
            scenarios = [
                'minimalist style, pure white background, ironic situation',
                'minimalist style, pure background, unexpected danger',
                'minimalist style, naive protagonist in perilous environment',
                'minimalist style, comedic tragedy, dark irony',
                'minimalist style, blissfully unaware of impending doom',
            ]

        results = []

        for i in range(batch_size):
            print(f"\n=== 批次 {i+1}/{batch_size} ===")

            # 隨機選擇場景
            style = random.choice(scenarios)
            print(f"場景: {style}")

            result = self.execute(
                main_character=main_character,
                style=style,
                **kwargs
            )
            results.append(result)

        return results

    @classmethod
    def quick_execute(cls,
                     main_character: str,
                     secondary_character: Optional[str] = None,
                     **kwargs):
        """快速執行黑色幽默生成

        Args:
            main_character: 主角色
            secondary_character: 次要角色
            **kwargs: 其他參數

        Returns:
            生成結果
        """
        use_case = cls()
        return use_case.execute(
            main_character=main_character,
            secondary_character=secondary_character,
            **kwargs
        )

