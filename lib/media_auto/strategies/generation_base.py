from typing import List, Dict, Any
import random
import time
import re
import json
import numpy as np
import os

from lib.media_auto.strategies.base_strategy import ContentStrategy
from lib.media_auto.models.vision.vision_manager import VisionManagerBuilder
from lib.media_auto.models.vision.model_switcher import ModelSwitcher
from lib.comfyui.websockets_api import ComfyUICommunicator
from lib.comfyui.node_manager import NodeManager


class BaseGenerationStrategy(ContentStrategy):
    """生成策略的基礎類別，封裝共同的邏輯和屬性"""

    def __init__(self, character_repository=None, vision_manager=None):
        """初始化基礎策略

        Args:
            character_repository: 角色資料庫
            vision_manager: 外部傳入的視覺管理器（可選）
        """
        # 初始化內部視覺管理器
        self._initialize_vision_managers()

        # 如果有外部傳入的 VisionManager，優先使用
        if vision_manager is not None:
            self.current_vision_manager = vision_manager
            self.external_vision_manager = True
        else:
            self.external_vision_manager = False

        # 共同的元件
        self.node_manager = NodeManager()
        self.character_repository = character_repository

        # 共同的狀態屬性
        self.descriptions: List[str] = []
        self.filter_results: List[Dict[str, Any]] = []
        self.article_content: str = ""
        self.communicator = None

    def _initialize_vision_managers(self):
        # 建立 Gemini 管理器
        self.gemini_vision_manager = VisionManagerBuilder() \
            .with_vision_model('gemini', model_name='gemini-flash-lite-latest') \
            .with_text_model('gemini', model_name='gemini-flash-lite-latest') \
            .build()

        # 建立 OpenRouter 管理器
        self.openrouter_vision_manager = VisionManagerBuilder() \
            .with_vision_model('openrouter') \
            .with_text_model('openrouter') \
            .with_random_models(True) \
            .build()

        self.ollama_vision_manager = VisionManagerBuilder() \
            .with_vision_model('ollama', model_name='llava:13b') \
            .with_text_model('ollama', model_name='llama3.2:latest') \
            .build()
        self.ollama_switcher = ModelSwitcher(self.ollama_vision_manager)

        # 預設使用 OpenRouter
        self.current_vision_manager = self.openrouter_vision_manager

    def set_vision_provider(self, provider: str = 'openrouter'):
        """設置視覺模型提供者

        Args:
            provider: 'ollama', 'gemini', 或 'openrouter'
        """
        if self.external_vision_manager:
            print(f"警告：正在使用外部傳入的 VisionManager，無法切換提供者到 {provider}")
            return

        if provider == 'ollama':
            self.current_vision_manager = self.ollama_vision_manager
        elif provider == 'gemini':
            self.current_vision_manager = self.gemini_vision_manager
        elif provider == 'openrouter':
            self.current_vision_manager = self.openrouter_vision_manager
        else:
            raise ValueError(f"不支援的視覺模型提供者: {provider}")
        print(f"已切換至 {provider} 視覺模型提供者")

    def _load_workflow(self, path: str) -> Dict[str, Any]:
        """載入工作流配置"""
        with open(path, "r", encoding='utf-8') as f:
            return json.loads(f.read())

    def _get_strategy_config(self, strategy_type: str, stage: str = None) -> Dict[str, Any]:
        """獲取策略專用配置，支援 general 參數覆蓋
        
        Args:
            strategy_type: 策略類型 (text2img, text2image2video, 等)
            stage: 階段名稱 (first_stage, second_stage, video, 等)，可選
        
        Returns:
            合併後的配置字典（策略專用參數覆蓋 general 參數）
        """
        additional_params = getattr(self.config, 'additional_params', {})
        general_params = additional_params.get('general', {})
        strategies = additional_params.get('strategies', {})
        strategy_config = strategies.get(strategy_type, {})
        
        if stage:
            stage_config = strategy_config.get(stage, {})
            # 合併：general -> strategy -> stage（後者覆蓋前者）
            return {**general_params, **strategy_config, **stage_config}
        else:
            # 合併：general -> strategy（strategy 覆蓋 general）
            return {**general_params, **strategy_config}

    def _process_weighted_choice(self, weights: Dict[str, float], exclude: list = None) -> str:
        """根據權重隨機選擇選項（自動正規化）
        
        Args:
            weights: 權重字典，例如 {'option1': 0.5, 'option2': 0.3}
            exclude: 要排除的選項列表（例如：排除雙角色互動）
        
        Returns:
            選中的選項字串，如果沒有可用選項則返回 None
        """
        if not weights:
            return None
        
        # 過濾掉要排除的選項
        filtered_weights = weights.copy()
        if exclude:
            for key in exclude:
                filtered_weights.pop(key, None)
        
        if not filtered_weights:
            return None
        
        choices = list(filtered_weights.keys())
        probabilities = list(filtered_weights.values())
        
        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]
        else:
            # 如果所有權重都是 0，均勻分配
            probabilities = [1.0 / len(choices)] * len(choices)
        
        return str(np.random.choice(choices, size=1, p=probabilities)[0])

    def _get_random_secondary_character(self, main_character: str, character_repository) -> str:
        """獲取隨機的 Secondary Role"""
        try:
            # 如果沒有character_repository，嘗試延遲導入
            if character_repository is None:
                try:
                    from lib.services.service_factory import ServiceFactory
                    service_factory = ServiceFactory()
                    character_repository = service_factory.get_character_repository()
                except ImportError:
                    print("無法導入ServiceFactory，使用預設角色")
                    return self._get_default_secondary_character(main_character)

            # 從角色配置中獲取 group_name 和 workflow
            group_name = getattr(self.config, 'group_name', '')
            workflow_path = getattr(self.config, 'workflow_path', '')

            # 從 workflow_path 中提取 workflow 名稱（去掉路徑和副檔名）
            workflow_name = os.path.splitext(os.path.basename(workflow_path))[0] if workflow_path else ''

            print(f"嘗試從群組 '{group_name}' 和工作流 '{workflow_name}' 中獲取角色")

            if group_name and workflow_name:
                # 從資料庫中獲取同群組的角色
                characters = character_repository.get_characters_by_group(group_name, workflow_name)

                # 過濾掉主角色
                available_characters = [char for char in characters if char.lower() != main_character.lower()]

                if available_characters:
                    selected_character = random.choice(available_characters)
                    print(f"從資料庫獲取到 Secondary Role: {selected_character}")
                    return selected_character
                else:
                    print(f"群組 '{group_name}' 中沒有其他可用角色")
            else:
                print(f"角色配置中缺少 group_name 或 workflow_path")

            # 如果無法從資料庫獲取，使用預設角色
            return self._get_default_secondary_character(main_character)

        except Exception as e:
            print(f"獲取 Secondary Role 時發生錯誤: {e}")
            return self._get_default_secondary_character(main_character)

    def _get_default_secondary_character(self, main_character: str) -> str:
        """獲取預設的 Secondary Role"""
        default_characters = ["wobbuffet", "Pikachu", "Mario", "fantastic"]
        available_defaults = [char for char in default_characters if char.lower() != main_character.lower()]
        if available_defaults:
            selected_default = random.choice(available_defaults)
            print(f"使用預設 Secondary Role: {selected_default}")
            return selected_default
        return None

    def _generate_two_character_interaction_description(self, prompt: str, style: str = '') -> str:
        """生成雙角色互動描述

        這個方法會從資料庫中獲取一個Secondary Role，並使用雙角色互動系統提示詞
        包含用戶原始prompt
        """
        try:
            # 優先使用 config 中指定的 secondary_character
            secondary_character = getattr(self.config, 'secondary_character', None)

            if not secondary_character:
                # 如果 config 中沒有指定，才從資料庫隨機獲取
                secondary_character = self._get_random_secondary_character(
                    self.config.character,
                    self.character_repository
                )
            else:
                print(f'使用指定的 Secondary Role: {secondary_character}')

            if secondary_character:
                print(f'雙角色互動：Main Role: {self.config.character}, Secondary Role: {secondary_character}')

                # 傳遞原始prompt給雙角色互動生成
                descriptions = self.current_vision_manager.generate_two_character_interaction_prompt(
                    main_character=self.config.character,
                    secondary_character=secondary_character,
                    prompt=prompt,
                    style=style  # 直接傳遞 style，不強制預設值
                )
                return descriptions
            else:
                print('無法獲取 Secondary Role，使用預設方法')
                return self.current_vision_manager.generate_image_prompts(prompt, 'stable_diffusion_prompt')

        except Exception as e:
            print(f'雙角色互動生成時發生錯誤: {e}，使用預設方法')
            return self.current_vision_manager.generate_image_prompts(prompt, 'stable_diffusion_prompt')

    def _upload_image_to_comfyui(self, image_path: str) -> str:
        """上傳圖片到 ComfyUI 伺服器

        Args:
            image_path: 本地圖片路徑

        Returns:
            上傳後的圖片文件名
        """
        try:
            image_filename = self.communicator.upload_image(image_path)
            print(f"✅ 圖片已上傳: {image_filename}")
            return image_filename
        except Exception as e:
            # 如果上傳失敗，嘗試直接使用文件名（假設圖片已經在 ComfyUI 的 input 目錄）
            print(f"⚠️ 圖片上傳失敗: {e}")
            print(f"   嘗試直接使用文件名: {os.path.basename(image_path)}")
            return os.path.basename(image_path)

    def prevent_hashtag_count_too_more(self, hashtag_text):
        """防止 hashtag 數量過多"""
        hashtag_candidate_list=[part.lower() for part in re.split(pattern='\n|#', string=hashtag_text) if part != '']

        deduplicate_list = []
        for part in hashtag_candidate_list:
            if part not in deduplicate_list:
                deduplicate_list.append(part)

        if len(deduplicate_list) > 30:
            hashtag_text = deduplicate_list[0] + '\n#' + '#'.join(deduplicate_list[1:2] + np.random.choice(deduplicate_list[2:], size=27, replace=False).tolist())

        return hashtag_text.lower().strip()

    def generate_article_content(self):
        """生成文章內容 - 通用實現"""
        start_time = time.time()

        if not self.filter_results:
            character = getattr(self.config, 'character', '')
            strategy_name = self.__class__.__name__.replace('Strategy', '').lower()
            self.article_content = f"#{character} #AI #{strategy_name}"
            return self

        # 整合角色名稱、描述和預設標籤
        # 限制最多使用3張圖片的描述來生成文章內容
        limited_results = self.filter_results[:3]
        content_parts = [
            getattr(self.config, 'character', ''),
            *list(set([row['description'] for row in limited_results])),
            getattr(self.config, 'prompt', '')
        ]

        article_content = self.current_vision_manager.generate_seo_hashtags('\n\n'.join(content_parts))

        # 加入預設標籤
        if hasattr(self.config, 'default_hashtags') and self.config.default_hashtags:
            article_content += ' #' + ' #'.join([tag.lstrip('#') for tag in self.config.default_hashtags])

        if '</think>' in article_content:
            article_content = article_content.split('</think>')[-1].strip()

        self.article_content = article_content.replace('"', '').replace('*', '').lower()

        print(f'產生文章內容花費: {time.time() - start_time}')
        return self
