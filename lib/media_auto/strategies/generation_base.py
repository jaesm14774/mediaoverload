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

        # 預設使用 Gemini（更穩定且便宜）
        self.current_vision_manager = self.gemini_vision_manager

    def set_vision_provider(self, provider: str = 'gemini'):
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
        # 確保 additional_params 是字典類型
        if not isinstance(additional_params, dict):
            print(f"⚠️ additional_params 不是字典類型: {type(additional_params)}, 使用空字典")
            additional_params = {}
        
        general_params = additional_params.get('general', {}) or {}
        strategies = additional_params.get('strategies', {}) or {}
        strategy_config = strategies.get(strategy_type, {}) or {}
        
        if stage:
            stage_config = strategy_config.get(stage, {}) or {}
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

    def upscale_images(self, image_paths: List[str], output_dir: str, workflow_path: str = None) -> List[str]:
        """使用 upscale workflow 放大圖片
        
        Args:
            image_paths: 要放大的圖片路徑列表
            output_dir: 輸出路徑
            workflow_path: upscale workflow 路徑（可選，預設從配置讀取）
            
        Returns:
            放大後的圖片路徑列表
        """
        import glob
        
        # 從配置中獲取 upscale 設置（支援策略特定配置覆蓋）
        strategy_type = self.__class__.__name__.replace('Strategy', '').lower()
        # 將類名映射到配置中的策略類型名稱
        if strategy_type == 'text2image2video':
            strategy_type = 'text2image2video'
        elif strategy_type == 'text2image2image':
            strategy_type = 'text2image2image'
        elif strategy_type == 'text2image':
            strategy_type = 'text2img'
        elif strategy_type == 'image2image':
            strategy_type = 'image2image'
        
        # 獲取策略配置（策略特定配置會覆蓋 general 配置）
        strategy_config = self._get_strategy_config(strategy_type)
        
        # 調試輸出：顯示讀取到的配置
        print(f"🔍 策略類型: {strategy_type}")
        print(f"🔍 enable_upscale 配置值: {strategy_config.get('enable_upscale', '未找到')}")
        print(f"🔍 upscale_workflow_path 配置值: {strategy_config.get('upscale_workflow_path', '未找到')}")
        
        # 檢查是否啟用 upscale
        enable_upscale = strategy_config.get('enable_upscale', False)
        if not enable_upscale:
            print("⚠️ Upscale 功能未啟用，跳過放大流程")
            return image_paths
        
        # 獲取 workflow 路徑
        if not workflow_path:
            workflow_path = strategy_config.get('upscale_workflow_path', 'configs/workflow/Tile Upscaler SDXL.json')
        
        if not os.path.exists(workflow_path):
            print(f"⚠️ Upscale workflow 不存在: {workflow_path}，跳過放大流程")
            return image_paths
        
        print(f"\n{'=' * 60}")
        print(f"開始放大 {len(image_paths)} 張圖片")
        print(f"{'=' * 60}")
        
        # 載入 workflow
        upscale_workflow = self._load_workflow(workflow_path)
        
        # 確保 WebSocket 連接存在
        if not self.communicator or not self.communicator.ws or not self.communicator.ws.connected:
            self.communicator = ComfyUICommunicator()
            self.communicator.connect_websocket()
            print("已建立 WebSocket 連接")
        
        upscaled_paths = []
        upscale_output_dir = os.path.join(output_dir, 'upscaled')
        os.makedirs(upscale_output_dir, exist_ok=True)
        
        try:
            for idx, image_path in enumerate(image_paths):
                print(f"\n[{idx+1}/{len(image_paths)}] 放大圖片: {os.path.basename(image_path)}")
                
                # 上傳圖片到 ComfyUI
                image_filename = self._upload_image_to_comfyui(image_path)
                
                # 準備更新：更新 LoadImage 節點（節點 225）載入圖片
                # 必須使用 "type": "direct_update" 格式，才能讓 process_workflow 正確識別並應用更新
                updates = [
                    {
                        "type": "direct_update",
                        "node_id": "225",
                        "inputs": {"image": image_filename}
                    }
                ]
                
                # 處理 workflow
                is_last_image = (idx == len(image_paths) - 1)
                success, saved_files = self.communicator.process_workflow(
                    workflow=upscale_workflow,
                    updates=updates,
                    output_path=upscale_output_dir,
                    file_name=f"upscaled_{idx}",
                    auto_close=False
                )
                
                if success and saved_files:
                    # 找到最新生成的圖片（通常是放大後的圖片）
                    upscaled_image = saved_files[-1] if saved_files else None
                    if upscaled_image and os.path.exists(upscaled_image):
                        upscaled_paths.append(upscaled_image)
                        print(f"✅ 圖片已放大: {os.path.basename(upscaled_image)}")
                    else:
                        print(f"⚠️ 無法找到放大後的圖片，使用原始圖片")
                        upscaled_paths.append(image_path)
                else:
                    print(f"⚠️ 放大失敗，使用原始圖片")
                    upscaled_paths.append(image_path)
        
        finally:
            # 不關閉 WebSocket，讓後續流程繼續使用
            pass
        
        print(f"\n✅ 完成放大流程，共處理 {len(upscaled_paths)} 張圖片")
        return upscaled_paths

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
