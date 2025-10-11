"""基於新聞的圖片生成使用案例

根據每日新聞關鍵字生成相關圖片
"""

from .base_use_case import BaseUseCase
from lib.media_auto.strategies.base_strategy import GenerationConfig
from typing import Optional, List
import numpy as np


class NewsBasedUseCase(BaseUseCase):
    """基於新聞的圖片生成

    使用案例：根據最新新聞的標題和關鍵字生成相關圖片

    範例:
        >>> use_case = NewsBasedUseCase()
        >>> result = use_case.execute(
        ...     character='Kirby',
        ...     news_count=10
        ... )
    """

    def build_config(self,
                    character: str,
                    workflow_name: str = 'nova-anime-xl',
                    news_count: int = 10,
                    optional_characters: Optional[List[str]] = None,
                    images_per_description: int = 4,
                    **kwargs) -> GenerationConfig:
        """建立基於新聞的生成配置

        Args:
            character: 主角色名稱
            workflow_name: 工作流名稱
            news_count: 要使用的新聞數量
            optional_characters: 可選的次要角色列表
            images_per_description: 每個描述生成的圖片數量
            **kwargs: 其他參數

        Returns:
            GenerationConfig 實例
        """
        # 獲取最新新聞
        news_df = self.get_latest_news()

        if len(news_df) == 0:
            raise ValueError("沒有找到符合條件的新聞")

        # 隨機選擇新聞
        if news_count > len(news_df):
            news_count = len(news_df)

        chosen_indices = np.random.choice(range(len(news_df)), size=news_count, replace=False)

        # 設定預設的可選角色
        if optional_characters is None:
            optional_characters = ['Waddle Dee', 'Magolor', 'Totoro', 'snoopy', 'Woodstock']

        # 建立提示詞 - 整合新聞資訊
        news_prompts = []
        for idx in chosen_indices:
            keyword = news_df.loc[idx, 'keyword']
            title = news_df.loc[idx, 'title']

            prompt = f"""
main_character: {character}
optional_character: {', '.join(optional_characters)}
extra_info: {title} ; {keyword}
            """.strip()

            news_prompts.append(prompt)

        # 使用第一個新聞作為主要提示詞
        main_prompt = news_prompts[0] if news_prompts else ""

        # 建立配置
        config = GenerationConfig(
            character=character,
            workflow_path=self.load_workflow(workflow_name),
            output_dir=self.output_folder,
            prompt=main_prompt,
            generation_type='text2img',
            image_system_prompt='stable_diffusion_prompt',
            additional_params={
                'images_per_description': images_per_description,
                'is_negative': False
            },
            # 儲存所有新聞提示詞供後續使用
            _news_prompts=news_prompts,
            _news_indices=chosen_indices.tolist()
        )

        return config

    def execute(self, **kwargs):
        """執行基於新聞的圖片生成

        這個方法會為每個新聞生成一組圖片

        Returns:
            生成結果
        """
        # 建立配置
        config = self.build_config(**kwargs)

        # 獲取所有新聞提示詞
        news_prompts = getattr(config, '_news_prompts', [])

        all_results = []

        # 為每個新聞生成圖片
        for i, prompt in enumerate(news_prompts):
            print(f"\n=== 處理新聞 {i+1}/{len(news_prompts)} ===")
            print(f"提示詞: {prompt[:100]}...")

            # 更新配置的提示詞
            config.prompt = prompt

            # 執行生成
            result = self.content_service.generate_content(config)
            all_results.append(result)

        return {
            'all_results': all_results,
            'total_news': len(news_prompts),
            'summary': self._summarize_results(all_results)
        }

    def _summarize_results(self, results: List) -> dict:
        """總結生成結果

        Args:
            results: 結果列表

        Returns:
            總結字典
        """
        total_descriptions = sum(len(r.get('descriptions', [])) for r in results)
        total_media = sum(len(r.get('media_files', [])) for r in results)
        total_filtered = sum(len(r.get('filter_results', [])) for r in results)

        return {
            'total_descriptions': total_descriptions,
            'total_media_files': total_media,
            'total_filtered': total_filtered,
            'success_rate': total_filtered / total_media if total_media > 0 else 0
        }

    @classmethod
    def quick_execute(cls,
                     character: str,
                     news_count: int = 5,
                     **kwargs):
        """快速執行新聞生成

        Args:
            character: 角色名稱
            news_count: 新聞數量
            **kwargs: 其他參數

        Returns:
            生成結果
        """
        use_case = cls()
        return use_case.execute(character=character, news_count=news_count, **kwargs)

