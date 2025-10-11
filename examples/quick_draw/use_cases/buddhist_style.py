"""佛性/靈性風格圖片生成使用案例

融合佛教、道教、基督教等宗教元素的圖片生成
"""

from .base_use_case import BaseUseCase
from lib.media_auto.strategies.base_strategy import GenerationConfig
from typing import Optional
import random
import numpy as np


class BuddhistStyleUseCase(BaseUseCase):
    """佛性/靈性風格圖片生成

    使用案例：生成融合宗教/靈性元素的圖片

    範例:
        >>> use_case = BuddhistStyleUseCase()
        >>> result = use_case.execute(
        ...     character='Kirby',
        ...     spiritual_theme='meditation and enlightenment'
        ... )
    """

    def build_config(self,
                    character: Optional[str] = None,
                    workflow_name: str = 'nova-anime-xl',
                    spiritual_theme: str = 'meditation and enlightenment',
                    extra_info: str = '',
                    use_news: bool = True,
                    news_count: int = 10,
                    images_per_description: int = 4,
                    group_name: Optional[str] = None) -> GenerationConfig:
        """建立佛性風格生成配置

        Args:
            character: 角色名稱，如果為 None 則隨機選擇
            workflow_name: 工作流名稱
            spiritual_theme: 靈性主題
            extra_info: 額外資訊
            use_news: 是否結合新聞關鍵字
            news_count: 使用的新聞數量
            images_per_description: 每個描述生成的圖片數量
            group_name: 角色群組名稱

        Returns:
            GenerationConfig 實例
        """
        # 隨機選擇角色
        if character is None:
            character = self.get_random_character(group_name)
            print(f"隨機選擇角色: {character}")

        # 如果使用新聞，獲取新聞資訊
        news_extra = ""
        if use_news:
            news_df = self.get_latest_news()
            if len(news_df) > 0:
                # 隨機選擇一條新聞
                chosen_idx = np.random.choice(range(min(news_count, len(news_df))))
                keyword = news_df.loc[chosen_idx, 'keyword']
                title = news_df.loc[chosen_idx, 'title']
                news_extra = f"{title} ; {keyword}"

        # 合併額外資訊
        combined_extra = f"{extra_info} {news_extra}".strip()

        # 建立提示詞
        prompt = f"""
Main Role: {character}
topic: {spiritual_theme}
extra: {combined_extra}
        """.strip()

        # 建立配置
        config = GenerationConfig(
            character=character,
            workflow_path=self.load_workflow(workflow_name),
            output_dir=self.output_folder,
            prompt=prompt,
            generation_type='text2img',
            image_system_prompt='buddhist_combined_image_system_prompt',
            group_name=group_name or '',
            additional_params={
                'images_per_description': images_per_description,
                'is_negative': False
            }
        )

        return config

    def execute_themed_batch(self,
                            character: str,
                            themes: Optional[list] = None,
                            batch_size: int = 10,
                            **kwargs):
        """批次生成不同靈性主題的圖片

        Args:
            character: 角色名稱
            themes: 靈性主題列表，如果為 None 則使用預設主題
            batch_size: 批次數量
            **kwargs: 其他參數

        Returns:
            所有批次的結果列表
        """
        # 預設靈性主題
        if themes is None:
            themes = [
                'meditation and enlightenment',
                'compassion and kindness',
                'inner peace and harmony',
                'spiritual awakening',
                'divine grace and blessing',
                'wisdom and understanding',
                'transcendence and liberation',
                'sacred journey',
                'celestial beauty',
                'eternal light'
            ]

        results = []

        for i in range(batch_size):
            print(f"\n=== 批次 {i+1}/{batch_size} ===")

            # 隨機選擇主題
            theme = random.choice(themes)
            print(f"靈性主題: {theme}")

            # 執行生成
            result = self.execute(
                character=character,
                spiritual_theme=theme,
                **kwargs
            )
            results.append(result)

        return results

    @classmethod
    def quick_execute(cls,
                     character: str,
                     spiritual_theme: str = 'meditation',
                     **kwargs):
        """快速執行佛性風格生成

        Args:
            character: 角色名稱
            spiritual_theme: 靈性主題
            **kwargs: 其他參數

        Returns:
            生成結果
        """
        use_case = cls()
        return use_case.execute(
            character=character,
            spiritual_theme=spiritual_theme,
            **kwargs
        )

