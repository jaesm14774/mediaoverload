"""電影級別圖片生成使用案例

生成電影感、高質感的寬螢幕比例圖片
"""

from .base_use_case import BaseUseCase
from lib.media_auto.strategies.base_strategy import GenerationConfig
from typing import Optional, Tuple
import random
import numpy as np


class CinematicUseCase(BaseUseCase):
    """電影級別圖片生成

    使用案例：生成具有電影感的高質量圖片

    範例:
        >>> use_case = CinematicUseCase()
        >>> result = use_case.execute(
        ...     main_character='Kirby',
        ...     aspect_ratio='cinematic'  # 16:9
        ... )
    """

    # 預定義的長寬比
    ASPECT_RATIOS = {
        'cinematic': (1280, 720),    # 16:9
        'wide': (1920, 1080),         # 16:9 Full HD
        'ultrawide': (2560, 1080),    # 21:9
        'cinema_scope': (2048, 858),  # 2.39:1
        'standard': (1024, 768),      # 4:3
    }

    def build_config(self,
                    main_character: Optional[str] = None,
                    secondary_character: Optional[str] = None,
                    workflow_name: str = 'nova-anime-xl',
                    aspect_ratio: str = 'cinematic',
                    custom_size: Optional[Tuple[int, int]] = None,
                    use_news: bool = True,
                    news_count: int = 30,
                    images_per_description: int = 4,
                    group_name: Optional[str] = None) -> GenerationConfig:
        """建立電影級別生成配置

        Args:
            main_character: 主角色名稱，如果為 None 則隨機選擇
            secondary_character: 次要角色名稱，如果為 None 則隨機選擇
            workflow_name: 工作流名稱
            aspect_ratio: 長寬比預設值 ('cinematic', 'wide', 'ultrawide', 'cinema_scope', 'standard')
            custom_size: 自定義尺寸 (width, height)，會覆蓋 aspect_ratio
            use_news: 是否結合新聞資訊
            news_count: 使用的新聞數量
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

        # 決定圖片尺寸
        if custom_size:
            width, height = custom_size
        elif aspect_ratio in self.ASPECT_RATIOS:
            width, height = self.ASPECT_RATIOS[aspect_ratio]
        else:
            print(f"警告：未知的長寬比 '{aspect_ratio}'，使用預設 'cinematic'")
            width, height = self.ASPECT_RATIOS['cinematic']

        print(f"圖片尺寸: {width}x{height}")

        # 獲取新聞資訊
        news_extra = ""
        if use_news:
            news_df = self.get_latest_news()
            if len(news_df) > 0:
                chosen_idx = np.random.choice(range(min(news_count, len(news_df))))
                keyword = news_df.loc[chosen_idx, 'keyword']
                title = news_df.loc[chosen_idx, 'title']
                news_extra = f"{title} ; {keyword}"

        # 建立提示詞
        prompt = f"""
Main Role: {main_character}
Secondary Role: {secondary_character}
additional info: {news_extra}
        """.strip()

        # 建立自定義節點更新（設定圖片尺寸）
        custom_updates = [
            {
                "node_type": "PrimitiveInt",
                "node_index": 0,  # 第一個 PrimitiveInt 節點 (width)
                "inputs": {"value": width}
            },
            {
                "node_type": "PrimitiveInt",
                "node_index": 1,  # 第二個 PrimitiveInt 節點 (height)
                "inputs": {"value": height}
            }
        ]

        # 建立配置
        config = GenerationConfig(
            character=main_character,
            workflow_path=self.load_workflow(workflow_name),
            output_dir=self.output_folder,
            prompt=prompt,
            generation_type='text2img',
            image_system_prompt='cinematic_stable_diffusion_prompt',  # 使用電影級別提示詞
            group_name=group_name or '',
            additional_params={
                'images_per_description': images_per_description,
                'is_negative': False,
                'custom_node_updates': custom_updates
            }
        )

        return config

    def execute_batch_with_news(self,
                                main_character: str,
                                secondary_character: Optional[str] = None,
                                batch_size: int = 10,
                                aspect_ratio: str = 'cinematic',
                                **kwargs):
        """批次生成帶新聞資訊的電影級別圖片

        每次使用不同的新聞資訊

        Args:
            main_character: 主角色
            secondary_character: 次要角色
            batch_size: 批次數量
            aspect_ratio: 長寬比
            **kwargs: 其他參數

        Returns:
            所有批次的結果列表
        """
        results = []

        for i in range(batch_size):
            print(f"\n=== 批次 {i+1}/{batch_size} ===")

            result = self.execute(
                main_character=main_character,
                secondary_character=secondary_character,
                aspect_ratio=aspect_ratio,
                use_news=True,
                **kwargs
            )
            results.append(result)

        return results

    def execute_multi_aspect_ratios(self,
                                   main_character: str,
                                   aspect_ratios: Optional[list] = None,
                                   **kwargs):
        """使用多種長寬比生成圖片

        Args:
            main_character: 主角色
            aspect_ratios: 長寬比列表
            **kwargs: 其他參數

        Returns:
            所有長寬比的結果字典
        """
        if aspect_ratios is None:
            aspect_ratios = ['cinematic', 'wide', 'cinema_scope']

        results = {}

        for ratio in aspect_ratios:
            print(f"\n=== 長寬比: {ratio} ===")

            result = self.execute(
                main_character=main_character,
                aspect_ratio=ratio,
                **kwargs
            )
            results[ratio] = result

        return results

    @classmethod
    def quick_execute(cls,
                     main_character: str,
                     aspect_ratio: str = 'cinematic',
                     **kwargs):
        """快速執行電影級別生成

        Args:
            main_character: 主角色
            aspect_ratio: 長寬比
            **kwargs: 其他參數

        Returns:
            生成結果
        """
        use_case = cls()
        return use_case.execute(
            main_character=main_character,
            aspect_ratio=aspect_ratio,
            **kwargs
        )

