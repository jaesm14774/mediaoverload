from typing import List, Dict, Any
import random
import time
import glob
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

class Text2ImageStrategy(BaseGenerationStrategy):
    """
    文生圖策略
    實驗顯示 ollama 在判斷 文圖匹配時，不論是llama3.2-vision 或是 llava 都沒有達到期望
    改成google 大幅 增加準確性
    """

    def __init__(self, character_repository=None, vision_manager=None):
        super().__init__(character_repository, vision_manager)

        # Text2Image 專用的額外管理器
        if not self.external_vision_manager:
            self.ollama_vision_manager = VisionManagerBuilder() \
                .with_vision_model('ollama', model_name='llava:13b') \
                .with_text_model('ollama', model_name='llama3.2:latest') \
                .build()
            self.ollama_switcher = ModelSwitcher(self.ollama_vision_manager)

    def generate_description(self):
        """描述生成"""
        start_time = time.time()

        # 獲取策略專用配置
        image_config = self._get_strategy_config('text2img')
        
        # 獲取 style 和 image_system_prompt
        style = image_config.get('style') or getattr(self.config, 'style', '')
        
        image_system_prompt_weights = image_config.get('image_system_prompt_weights')
        
        # 如果有權重配置，使用權重選擇
        if image_system_prompt_weights:
            image_system_prompt = self._process_weighted_choice(image_system_prompt_weights)
        else:
            image_system_prompt = image_config.get('image_system_prompt') or getattr(self.config, 'image_system_prompt', 'stable_diffusion_prompt')

        prompt = self.config.prompt
        # 只有當 style 不為空時才加上 style: 前綴
        if style and style.strip():
            prompt = f"""{prompt}\nstyle: {style}""".strip()

        # 如果有指定角色，且 prompt 中不包含角色信息，則加入角色
        if self.config.character:
            character_lower = self.config.character.lower()
            prompt_lower = prompt.lower()
            # 檢查 prompt 中是否已包含角色名稱或 "main character" 等關鍵字
            if character_lower not in prompt_lower and "main character" not in prompt_lower:
                prompt = f"Main character: {self.config.character}\n{prompt}"
                print(f"已自動加入主角色到 prompt: {self.config.character}")

        # 檢查是否使用雙角色互動系統提示詞
        if image_system_prompt == 'two_character_interaction_generate_system_prompt':
            # 使用雙角色互動生成邏輯
            descriptions = self._generate_two_character_interaction_description(prompt, style)
        else:
            # 使用原有的生成邏輯
            descriptions = self.current_vision_manager.generate_image_prompts(prompt, image_system_prompt)
        
        print('All generated descriptions : ', descriptions)
        if self.config.character:
            character = self.config.character.lower()
            self.descriptions = [descriptions] if descriptions.replace(' ', '').lower().find(character) != -1 or descriptions.lower().find(character) != -1 else []

        print(f'Image descriptions : {self.descriptions}\n')
        print(f'生成描述花費 : {time.time() - start_time}')
        return self
        
    def generate_media(self):
        start_time = time.time()
        workflow = self._load_workflow(self.config.workflow_path)
        self.communicator = ComfyUICommunicator()
        
        try:
            # 建立 WebSocket 連接（僅一次）
            self.communicator.connect_websocket()
            print("已建立 WebSocket 連接，開始批次生成圖片")
            
            # 獲取策略專用配置
            image_config = self._get_strategy_config('text2img')
            
            images_per_description = image_config.get('images_per_description', 4)
            
            total_images = len(self.descriptions) * images_per_description
            current_image = 0
            
            for idx, description in enumerate(self.descriptions):
                seed_start = random.randint(1, 999999999999)
                for i in range(images_per_description):
                    current_image += 1
                    print(f'\n[{current_image}/{total_images}] 為描述 {idx+1}/{len(self.descriptions)}，生成第 {i+1}/{images_per_description} 張圖片')
                    
                    # 獲取自定義節點更新配置
                    custom_updates = image_config.get('custom_node_updates', [])
                    
                    # 合併配置參數
                    merged_params = {**self.config.additional_params, **image_config}
                    
                    # 使用統一的更新方法（自動處理衝突檢查）
                    updates = self.node_manager.generate_updates(
                        workflow=workflow,
                        updates_config=custom_updates,
                        description=description,
                        seed=seed_start + i,
                        **merged_params
                    )
                    
                    # 處理工作流，但不自動關閉 WebSocket（auto_close=False）
                    # 只有在最後一張圖片時才關閉（通過 finally 處理）
                    is_last_image = (idx == len(self.descriptions) - 1 and i == images_per_description - 1)
                    self.communicator.process_workflow(
                        workflow=workflow,
                        updates=updates,
                        output_path=f"{self.config.output_dir}",
                        file_name=f"{self.config.character}_d{idx}_{i}",
                        auto_close=False  # 不自動關閉，保持連接以供下次使用
                    )
                    
        finally:
            # 確保在所有圖片生成完成後關閉 WebSocket
            if self.communicator and self.communicator.ws and self.communicator.ws.connected:
                print("\n所有圖片生成完成，關閉 WebSocket 連接")
                self.communicator.ws.close()
                
        print(f'\n✅ 生成圖片總耗時: {time.time() - start_time:.2f} 秒')
        return self

    def analyze_media_text_match(self, similarity_threshold) -> Dict[str, Any]:
        """分析生成的圖片 - 隨機選擇分析模型"""
        media_paths = glob.glob(f'{self.config.output_dir}/*')

        # 隨機選擇分析管理器：Gemini 或 OpenRouter
        available_managers = [self.gemini_vision_manager, self.openrouter_vision_manager]
        selected_manager = np.random.choice(available_managers, p=[0.5, 0.5])

        # 輸出所選擇的模型類型
        if selected_manager == self.gemini_vision_manager:
            print("使用 Gemini 進行圖像相似度分析")
        else:
            print("使用 OpenRouter 隨機模型進行圖像相似度分析")

        self.filter_results = selected_manager.analyze_media_text_match(
            media_paths=media_paths,
            descriptions=self.descriptions,
            main_character=self.config.character,
            similarity_threshold=similarity_threshold,
            temperature=0.3
        )
        return self


class Image2ImageStrategy(BaseGenerationStrategy):
    """圖生圖策略 - 直接使用現有圖片進行 image to image 生成"""

    def __init__(self, character_repository=None, vision_manager=None):
        super().__init__(character_repository, vision_manager)
    
    def generate_description(self):
        """生成描述 - 對於 image2image，描述可以從圖片中提取或直接使用配置中的描述"""
        start_time = time.time()
        
        # 如果配置中有圖片路徑，可以提取描述
        if hasattr(self.config, 'input_image_path') and self.config.input_image_path:
            # 可選：從圖片中提取描述
            if hasattr(self.config, 'extract_description') and self.config.extract_description:
                description = self.current_vision_manager.extract_image_content(
                    self.config.input_image_path
                )
                self.descriptions = [description]
            else:
                # 使用配置中的描述
                self.descriptions = [self.config.prompt] if hasattr(self.config, 'prompt') else ['']
        else:
            # 使用配置中的描述
            self.descriptions = [self.config.prompt] if hasattr(self.config, 'prompt') else ['']
        
        print(f'Image2Image 描述: {self.descriptions}')
        print(f'生成描述花費: {time.time() - start_time}')
        return self
    
    def generate_media(self):
        """生成圖片 - 使用 image to image 工作流"""
        start_time = time.time()
        
        # 檢查是否有輸入圖片路徑
        input_image_path = getattr(self.config, 'input_image_path', None)
        if not input_image_path:
            raise ValueError("Image2ImageStrategy 需要 input_image_path 參數")
        
        if not os.path.exists(input_image_path):
            raise FileNotFoundError(f"找不到輸入圖片: {input_image_path}")
        
        # 獲取策略專用配置
        image_config = self._get_strategy_config('image2image')
        
        # 載入 image to image 工作流
        workflow_path = image_config.get('workflow_path') or getattr(self.config, 'workflow_path', 'configs/workflow/example/image_to_image.json')
        workflow = self._load_workflow(workflow_path)
        self.communicator = ComfyUICommunicator()
        
        try:
            # 建立 WebSocket 連接
            self.communicator.connect_websocket()
            print("已建立 WebSocket 連接，開始 image to image 生成")
            
            # 上傳圖片到 ComfyUI（如果需要）
            image_filename = self._upload_image_to_comfyui(input_image_path)
            
            # 獲取參數
            images_per_input = image_config.get('images_per_input', 1)
            
            # 獲取 denoise 參數（權重 0.5-0.7）
            denoise = image_config.get('denoise', 0.6)
            # 確保 denoise 在合理範圍內
            denoise = max(0.5, min(0.7, denoise))
            
            total_images = images_per_input
            current_image = 0
            
            for i in range(images_per_input):
                current_image += 1
                print(f'\n[{current_image}/{total_images}] 生成第 {i+1}/{images_per_input} 張圖片')
                
                # 獲取自定義節點更新配置
                custom_updates = image_config.get('custom_node_updates', []).copy()  # 複製列表避免修改原始配置
                
                # 添加 LoadImage 節點更新（載入輸入圖片）
                custom_updates.append({
                    "node_type": "LoadImage",
                    "node_index": 0,
                    "inputs": {"image": image_filename}
                })
                
                # 添加 KSampler 節點更新（設置 denoise）
                custom_updates.append({
                    "node_type": "KSampler",
                    "node_index": 0,
                    "inputs": {"denoise": denoise}
                })
                
                # 合併配置參數
                merged_params = {**self.config.additional_params, **image_config}
                
                # 使用統一的更新方法
                updates = self.node_manager.generate_updates(
                    workflow=workflow,
                    updates_config=custom_updates,
                    description=self.descriptions[0] if self.descriptions else '',
                    seed=random.randint(1, 999999999999) + i,
                    **merged_params
                )
                
                # 處理工作流
                is_last_image = (i == images_per_input - 1)
                self.communicator.process_workflow(
                    workflow=workflow,
                    updates=updates,
                    output_path=f"{self.config.output_dir}",
                    file_name=f"{self.config.character}_i2i_{i}" if hasattr(self.config, 'character') else f"i2i_{i}",
                    auto_close=False
                )
                
        finally:
            # 確保在所有圖片生成完成後關閉 WebSocket
            if self.communicator and self.communicator.ws and self.communicator.ws.connected:
                print("\n所有圖片生成完成，關閉 WebSocket 連接")
                self.communicator.ws.close()
        
        print(f'\n✅ Image to Image 生成總耗時: {time.time() - start_time:.2f} 秒')
        return self

    def analyze_media_text_match(self, similarity_threshold):
        """分析生成的圖片與描述的匹配度 - 隨機選擇分析模型"""
        media_paths = glob.glob(f'{self.config.output_dir}/*')

        # 過濾出圖片文件
        image_paths = [p for p in media_paths if any(p.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp'])]

        if not image_paths:
            self.filter_results = []
            return self

        # 隨機選擇分析管理器：Gemini 或 OpenRouter
        if not self.external_vision_manager:
            available_managers = [self.gemini_vision_manager, self.openrouter_vision_manager]
            selected_manager = np.random.choice(available_managers, p=[0.8, 0.2])

            # 輸出所選擇的模型類型
            if selected_manager == self.gemini_vision_manager:
                print("使用 Gemini 進行圖像相似度分析")
            else:
                print("使用 OpenRouter 隨機模型進行圖像相似度分析")
        else:
            selected_manager = self.current_vision_manager
            print("使用外部傳入的 VisionManager 進行圖像相似度分析")

        # 使用選定的視覺管理器分析匹配度
        self.filter_results = selected_manager.analyze_media_text_match(
            media_paths=image_paths,
            descriptions=self.descriptions,
            main_character=getattr(self.config, 'character', ''),
            similarity_threshold=similarity_threshold,
            temperature=0.3
        )

        return self


class Text2Image2ImageStrategy(BaseGenerationStrategy):
    """文生圖 -> 圖生圖策略 - 先進行 text to image，篩選後再用 image to image 二次生成"""

    def __init__(self, character_repository=None, vision_manager=None):
        super().__init__(character_repository, vision_manager)
        self.first_stage_images: List[str] = []  # 第一階段生成的圖片路徑
    
    def generate_description(self):
        """生成描述 - 使用與 Text2ImageStrategy 相同的邏輯"""
        start_time = time.time()

        # 獲取策略專用配置
        image_config = self._get_strategy_config('text2image2image')
        
        # 獲取 style 和 image_system_prompt
        style = image_config.get('style') or getattr(self.config, 'style', '')
        
        image_system_prompt_weights = image_config.get('image_system_prompt_weights')
        
        # 如果有權重配置，使用權重選擇
        if image_system_prompt_weights:
            image_system_prompt = self._process_weighted_choice(image_system_prompt_weights)
        else:
            image_system_prompt = image_config.get('image_system_prompt') or getattr(self.config, 'image_system_prompt', 'stable_diffusion_prompt')

        prompt = self.config.prompt
        # 只有當 style 不為空時才加上 style: 前綴
        if style and style.strip():
            prompt = f"""{prompt}\nstyle: {style}""".strip()

        # 如果有指定角色，且 prompt 中不包含角色信息，則加入角色
        if self.config.character:
            character_lower = self.config.character.lower()
            prompt_lower = prompt.lower()
            if character_lower not in prompt_lower and "main character" not in prompt_lower:
                prompt = f"Main character: {self.config.character}\n{prompt}"
                print(f"已自動加入主角色到 prompt: {self.config.character}")

        # 檢查是否使用雙角色互動系統提示詞
        if image_system_prompt == 'two_character_interaction_generate_system_prompt':
            descriptions = self._generate_two_character_interaction_description(prompt, style)
        else:
            descriptions = self.current_vision_manager.generate_image_prompts(prompt, image_system_prompt)
        
        print('All generated descriptions : ', descriptions)
        if self.config.character:
            character = self.config.character.lower()
            self.descriptions = [descriptions] if descriptions.replace(' ', '').lower().find(character) != -1 or descriptions.lower().find(character) != -1 else []
        else:
            self.descriptions = [descriptions] if descriptions else []

        print(f'Image descriptions : {self.descriptions}\n')
        print(f'生成描述花費 : {time.time() - start_time}')
        return self
    
    def generate_media(self):
        """生成圖片 - 分兩階段：text2image -> image2image"""
        start_time = time.time()
        
        # 第一階段：Text to Image
        print("=" * 60)
        print("第一階段：Text to Image 生成")
        print("=" * 60)
        
        # 獲取策略專用配置
        first_stage_config = self._get_strategy_config('text2image2image', 'first_stage')
        
        first_stage_workflow = self._load_workflow(self.config.workflow_path)
        self.communicator = ComfyUICommunicator()
        
        try:
            self.communicator.connect_websocket()
            print("已建立 WebSocket 連接，開始第一階段生成")
            
            # 第一階段參數
            images_per_description = first_stage_config.get('images_per_description', 4)
            
            total_first_stage = len(self.descriptions) * images_per_description
            current_image = 0
            
            for idx, description in enumerate(self.descriptions):
                seed_start = random.randint(1, 999999999999)
                for i in range(images_per_description):
                    current_image += 1
                    print(f'\n[第一階段 {current_image}/{total_first_stage}] 為描述 {idx+1}/{len(self.descriptions)}，生成第 {i+1}/{images_per_description} 張圖片')
                    
                    custom_updates = first_stage_config.get('custom_node_updates', [])
                    
                    merged_params = {**self.config.additional_params, **first_stage_config}
                    
                    updates = self.node_manager.generate_updates(
                        workflow=first_stage_workflow,
                        updates_config=custom_updates,
                        description=description,
                        seed=seed_start + i,
                        **merged_params
                    )
                    
                    is_last_image = (idx == len(self.descriptions) - 1 and i == images_per_description - 1)
                    self.communicator.process_workflow(
                        workflow=first_stage_workflow,
                        updates=updates,
                        output_path=f"{self.config.output_dir}/first_stage",
                        file_name=f"{self.config.character}_d{idx}_{i}",
                        auto_close=False
                    )
            
            # 等待第一階段完成
            print("\n第一階段生成完成，開始篩選圖片...")
            
            # 篩選第一階段生成的圖片
            first_stage_output_dir = f"{self.config.output_dir}/first_stage"
            first_stage_images = glob.glob(f'{first_stage_output_dir}/*')
            image_paths = [p for p in first_stage_images if any(p.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp'])]
            
            if not image_paths:
                print("警告：第一階段沒有生成任何圖片")
                return self
            
            # 分析圖文匹配度並篩選
            similarity_threshold = getattr(self.config, 'similarity_threshold', 0.9)
            print(f"使用相似度閾值: {similarity_threshold} 進行篩選")
            
            filter_results = self.current_vision_manager.analyze_media_text_match(
                media_paths=image_paths,
                descriptions=self.descriptions,
                main_character=self.config.character,
                similarity_threshold=similarity_threshold,
                temperature=0.3
            )
            
            if not filter_results:
                print("警告：沒有圖片通過篩選")
                return self
            
            print(f"篩選後共有 {len(filter_results)} 張圖片符合描述")
            self.first_stage_images = [result['media_path'] for result in filter_results]
            
            # 第二階段：Image to Image
            print("\n" + "=" * 60)
            print("第二階段：Image to Image 二次生成")
            print("=" * 60)
            
            # 獲取第二階段配置
            second_stage_config = self._get_strategy_config('text2image2image', 'second_stage')
            
            # 載入 image to image 工作流
            i2i_workflow_path = second_stage_config.get('i2i_workflow_path') or 'configs/workflow/example/image_to_image.json'
            i2i_workflow = self._load_workflow(i2i_workflow_path)
            
            # 第二階段參數
            images_per_input = second_stage_config.get('images_per_input', 1)
            
            # 獲取 denoise 參數（權重 0.5-0.7）
            denoise = second_stage_config.get('denoise', 0.6)
            denoise = max(0.5, min(0.7, denoise))
            print(f"使用 denoise 權重: {denoise}")
            
            total_second_stage = len(self.first_stage_images) * images_per_input
            current_i2i = 0
            
            for img_idx, input_image_path in enumerate(self.first_stage_images):
                image_filename = self._upload_image_to_comfyui(input_image_path)
                desc_index = img_idx % len(self.descriptions)
                
                for i in range(images_per_input):
                    current_i2i += 1
                    print(f'\n[第二階段 {current_i2i}/{total_second_stage}] 使用圖片 {img_idx+1}/{len(self.first_stage_images)}，生成第 {i+1}/{images_per_input} 張圖片')
                    
                    custom_updates = second_stage_config.get('custom_node_updates', []).copy()  # 複製列表避免修改原始配置
                    
                    # 添加 LoadImage 節點更新
                    custom_updates.append({
                        "node_type": "LoadImage",
                        "node_index": 0,
                        "inputs": {"image": image_filename}
                    })
                    
                    # 添加 KSampler 節點更新（設置 denoise）
                    custom_updates.append({
                        "node_type": "KSampler",
                        "node_index": 0,
                        "inputs": {"denoise": denoise}
                    })
                    
                    merged_params = {**self.config.additional_params, **second_stage_config}
                    
                    updates = self.node_manager.generate_updates(
                        workflow=i2i_workflow,
                        updates_config=custom_updates,
                        description=self.descriptions[desc_index],
                        seed=random.randint(1, 999999999999) + i,
                        **merged_params
                    )
                    
                    is_last_i2i = (img_idx == len(self.first_stage_images) - 1 and i == images_per_input - 1)
                    self.communicator.process_workflow(
                        workflow=i2i_workflow,
                        updates=updates,
                        output_path=f"{self.config.output_dir}/second_stage",
                        file_name=f"{self.config.character}_i2i_{img_idx}_{i}",
                        auto_close=False
                    )
            
        finally:
            if self.communicator and self.communicator.ws and self.communicator.ws.connected:
                print("\n所有圖片生成完成，關閉 WebSocket 連接")
                self.communicator.ws.close()
        
        print(f'\n✅ Text2Image2Image 生成總耗時: {time.time() - start_time:.2f} 秒')
        return self

    def analyze_media_text_match(self, similarity_threshold):
        """分析第二階段生成的圖片與描述的匹配度"""
        second_stage_output_dir = f"{self.config.output_dir}/second_stage"
        media_paths = glob.glob(f'{second_stage_output_dir}/*')

        image_paths = [p for p in media_paths if any(p.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp'])]

        if not image_paths:
            self.filter_results = []
            return self

        self.filter_results = self.current_vision_manager.analyze_media_text_match(
            media_paths=image_paths,
            descriptions=self.descriptions,
            main_character=self.config.character,
            similarity_threshold=similarity_threshold,
            temperature=0.3
        )

        return self


class Text2VideoStrategy(BaseGenerationStrategy):
    """文生影片策略"""

    def __init__(self, character_repository=None, vision_manager=None):
        super().__init__(character_repository, vision_manager)

        # Text2Video 專用的額外管理器
        if not self.external_vision_manager:
            self.ollama_vision_manager = VisionManagerBuilder() \
                .with_vision_model('ollama', model_name='qwen2.5vl:7b') \
                .with_text_model('ollama', model_name='gemma3:4b') \
                .build()

    def set_vision_provider(self, provider: str = 'openrouter'):
        """設置視覺模型提供者

        Args:
            provider: 'ollama' 或 'openrouter'
        """
        if self.external_vision_manager:
            print(f"警告：正在使用外部傳入的 VisionManager，無法切換提供者到 {provider}")
            return

        if provider == 'ollama':
            self.current_vision_manager = self.ollama_vision_manager
        elif provider == 'openrouter':
            self.current_vision_manager = self.openrouter_vision_manager
        else:
            raise ValueError(f"不支援的視覺模型提供者: {provider}")
        print(f"已切換至 {provider} 視覺模型提供者")
    
    def generate_description(self):
        """生成視頻描述"""
        start_time = time.time()
        
        # 獲取策略專用配置
        video_config = self._get_strategy_config('text2video')
        
        # 獲取基本配置
        style = video_config.get('style') or getattr(self.config, 'style', '')
        
        image_system_prompt = video_config.get('image_system_prompt') or getattr(self.config, 'image_system_prompt', 'unbelievable_world_system_prompt')
        
        prompt = self.config.prompt
        if style:
            prompt = f"""{prompt}\nstyle:{style}""".strip()

        # 如果有指定角色，且 prompt 中不包含角色信息，則加入角色
        if self.config.character:
            character_lower = self.config.character.lower()
            prompt_lower = prompt.lower()
            # 檢查 prompt 中是否已包含角色名稱或 "main character" 等關鍵字
            if character_lower not in prompt_lower and "main character" not in prompt_lower:
                prompt = f"Main character: {self.config.character}\n{prompt}"
                print(f"已自動加入主角色到 prompt: {self.config.character}")
        
        # 第一階段：使用角色名稱生成基礎描述
        character_description = self.current_vision_manager.generate_image_prompts(
            user_input=self.config.character,
            system_prompt_key=image_system_prompt
        )
        
        # 第二階段：使用視頻系統提示詞生成詳細的視頻描述
        video_description = self.current_vision_manager.generate_video_prompts(
            user_input=character_description,
            system_prompt_key='video_description_system_prompt'
        )
        
        print(f'Character description: {character_description}')
        print(f'Video description: {video_description}')
        
        # 檢查角色名稱是否存在於描述中
        if self.config.character:
            character = self.config.character.lower()
            if video_description.replace(' ', '').lower().find(character) != -1 or video_description.lower().find(character) != -1:
                self.descriptions = [video_description]
            else:
                self.descriptions = []
        else:
            self.descriptions = [video_description]
        
        print(f'Final video descriptions: {self.descriptions}')
        print(f'生成描述花費: {time.time() - start_time}')
        return self
    
    def generate_media(self):
        """生成視頻或圖片"""
        start_time = time.time()
        
        if not self.descriptions:
            print("沒有描述可供生成視頻")
            return self
        
        # 獲取策略專用配置
        video_config = self._get_strategy_config('text2video')
        
        # 載入工作流
        workflow = self._load_workflow(self.config.workflow_path)
        self.communicator = ComfyUICommunicator()
        
        try:
            # 建立 WebSocket 連接（僅一次）
            self.communicator.connect_websocket()
            print("已建立 WebSocket 連接，開始批次生成視頻")
            
            # 為每個描述生成視頻
            videos_per_description = video_config.get('videos_per_description', 2)
            
            total_videos = len(self.descriptions) * videos_per_description
            current_video = 0
            
            for idx, description in enumerate(self.descriptions):
                seed_start = random.randint(1, 999999999999)
                
                for i in range(videos_per_description):
                    current_video += 1
                    print(f'\n[{current_video}/{total_videos}] 為描述 {idx+1}/{len(self.descriptions)}，生成第 {i+1}/{videos_per_description} 個視頻')
                    
                    # 獲取自定義節點更新配置
                    default_video_updates = [
                        # 更新寬度和高度
                        {
                            "node_type": "PrimitiveInt",
                            "inputs": {"value": 512}
                        },
                        # 更新影片長度
                        {
                            "node_type": "EmptyHunyuanLatentVideo",
                            "inputs": {"length": 97}
                        }
                    ]
                    
                    custom_updates = video_config.get('custom_node_updates', default_video_updates)
                    
                    # 合併配置參數
                    merged_params = {**self.config.additional_params, **video_config}
                    
                    # 使用統一的更新方法（自動處理衝突檢查）
                    updates = self.node_manager.generate_updates(
                        workflow=workflow,
                        updates_config=custom_updates,
                        description=description,
                        seed=seed_start + i,
                        **merged_params
                    )
                    
                    # 處理工作流，但不自動關閉 WebSocket（auto_close=False）
                    # 只有在最後一個視頻時才關閉（通過 finally 處理）
                    self.communicator.process_workflow(
                        workflow=workflow,
                        updates=updates,
                        output_path=f"{self.config.output_dir}",
                        file_name=f"{self.config.character}_video_d{idx}_{i}",
                        auto_close=False  # 不自動關閉，保持連接以供下次使用
                    )
                    
        finally:
            # 確保在所有視頻生成完成後關閉 WebSocket
            if self.communicator and self.communicator.ws and self.communicator.ws.connected:
                print("\n所有視頻生成完成，關閉 WebSocket 連接")
                self.communicator.ws.close()
        
        print(f'\n✅ 生成視頻總耗時: {time.time() - start_time:.2f} 秒')
        return self

    def analyze_media_text_match(self, similarity_threshold) -> Dict[str, Any]:
        """分析生成的視頻（暫時不做文本匹配分析）"""
        # 對於視頻，我們暫時不做文本匹配分析
        # 可以在未來擴展此功能
        video_paths = glob.glob(f'{self.config.output_dir}/*')

        # 簡單地返回所有生成的視頻
        self.filter_results = []
        for video_path in video_paths:
            if any(video_path.endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.gif']):
                self.filter_results.append({
                    'media_path': video_path,  # 保持與現有接口一致
                    'description': self.descriptions[0] if self.descriptions else '',
                    'similarity': 1.0  # 暫時設為1.0
                })

        return self


class Text2Image2VideoStrategy(BaseGenerationStrategy):
    """文生圖 -> 圖生影片策略
    
    流程：
    1. text 2 image (使用 nova-anime-xl，可選)
    2. 透過 Discord 讓使用者選擇圖片（不做 AI 篩選）
    3. 透過 image 產生影片描述
    4. image + 影片描述 產生 音頻描述
    5. 使用 wan2.2_gguf_i2v workflow 產生含音頻的影片
    6. 使用影片生成文章內容
    7. 上傳到社群媒體
    """

    def __init__(self, character_repository=None, vision_manager=None):
        super().__init__(character_repository, vision_manager)
        self.first_stage_images: List[str] = []  # 第一階段生成的圖片路徑
        self.video_descriptions: Dict[str, str] = {}  # 圖片路徑 -> 影片描述
        self.audio_descriptions: Dict[str, str] = {}  # 圖片路徑 -> 音頻描述
        self._videos_generated = False  # 標記影片是否已生成
        self._videos_reviewed = False  # 標記影片是否已審核
    
    def needs_user_review(self) -> bool:
        """檢查是否需要使用者審核 - 在圖片生成後或影片生成後需要"""
        # 如果第一階段圖片已生成但影片未生成，則需要使用者審核圖片
        if len(self.first_stage_images) > 0 and not self._videos_generated:
            return True
        # 如果影片已生成但尚未審核，則需要使用者審核影片
        if self._videos_generated and not self._videos_reviewed:
            return True
        return False
    
    def get_review_items(self, max_items: int = 10) -> List[Dict[str, Any]]:
        """獲取需要審核的項目（限制最多 10 張，符合 Discord API 限制）"""
        # 如果影片已生成但尚未審核，返回影片供審核
        if self._videos_generated and not self._videos_reviewed:
            video_output_dir = f"{self.config.output_dir}/videos"
            if os.path.exists(video_output_dir):
                media_paths = glob.glob(f'{video_output_dir}/*')
                video_paths = [p for p in media_paths if any(p.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.gif', '.webm'])]
                
                if video_paths:
                    review_items = []
                    for video_path in video_paths[:max_items]:
                        # 嘗試從文件名匹配對應的圖片和描述
                        match = re.search(rf'{re.escape(self.config.character)}_i2v_(\d+)_\d+\.', video_path, re.IGNORECASE)
                        if match:
                            img_idx = int(match.group(1))
                            if img_idx < len(self.first_stage_images):
                                img_path = self.first_stage_images[img_idx]
                                video_desc = self.video_descriptions.get(img_path, '')
                                review_items.append({
                                    'media_path': video_path,
                                    'description': video_desc,
                                    'similarity': 1.0
                                })
                        else:
                            review_items.append({
                                'media_path': video_path,
                                'description': self.video_descriptions.get(self.first_stage_images[0], '') if self.first_stage_images else '',
                                'similarity': 1.0
                            })
                    
                    if len(video_paths) > max_items:
                        print(f"警告：共有 {len(video_paths)} 個影片，但 Discord 限制一次最多 {max_items} 個，只發送前 {max_items} 個供審核")
                    
                    return review_items
        
        # 如果第一階段圖片已生成但影片未生成，返回圖片供審核
        if not self.first_stage_images:
            return []
        
        # 如果圖片超過 10 張，只返回前 10 張
        review_images = self.first_stage_images[:max_items]
        
        review_items = []
        for img_path in review_images:
            review_items.append({
                'media_path': img_path,
                'description': self.descriptions[0] if self.descriptions else '',
                'similarity': 1.0
            })
        
        if len(self.first_stage_images) > max_items:
            print(f"警告：共有 {len(self.first_stage_images)} 張圖片，但 Discord 限制一次最多 {max_items} 張，只發送前 {max_items} 張供審核")
        
        return review_items
    
    def continue_after_review(self, selected_indices: List[int]) -> bool:
        """在使用者審核後繼續執行後續階段 - 生成影片"""
        if not selected_indices:
            print("警告：沒有選擇任何圖片")
            return False
        
        # 將 selected_indices 映射回原始 first_stage_images 的索引
        # selected_indices 是相對於 get_review_items 返回的列表（最多 10 張）
        # 獲取審核項目（與發送給 Discord 的一致）
        review_items = self.get_review_items(max_items=10)
        selected_image_paths = [review_items[i]['media_path'] for i in selected_indices if i < len(review_items)]
        
        if not selected_image_paths:
            print("警告：無法獲取選擇的圖片路徑")
            return False
        
        # 找到這些圖片在原始 first_stage_images 中的索引
        original_indices = []
        for img_path in selected_image_paths:
            try:
                idx = self.first_stage_images.index(img_path)
                original_indices.append(idx)
            except ValueError:
                print(f"警告：無法找到圖片路徑對應的索引: {img_path}")
        
        if not original_indices:
            print("警告：無法找到選擇的圖片對應的索引")
            return False
        
        # 生成影片
        print(f"開始使用 {len(original_indices)} 張使用者選擇的圖片生成影片")
        self.generate_videos_from_selected_images(original_indices)
        self._videos_generated = True
        self._videos_reviewed = False  # 影片生成後需要審核
        return True
    
    def should_generate_article_now(self) -> bool:
        """判斷是否應該現在生成文章內容 - 應該在影片生成後才生成"""
        return self._videos_generated
    
    def generate_description(self):
        """生成描述 - 使用策略專用配置（text2image2video 的 text2image 階段）"""
        start_time = time.time()

        # 獲取策略專用配置
        first_stage_config = self._get_strategy_config('text2image2video', 'first_stage')
        
        # 獲取 style（優先使用策略專用配置）
        style = first_stage_config.get('style') or getattr(self.config, 'style', '')
        
        # 獲取 image_system_prompt（支援加權選擇或多個選項）
        # 優先檢查是否有 image_system_prompt_weights（加權選擇）
        image_system_prompt_weights = first_stage_config.get('image_system_prompt_weights')
        
        if image_system_prompt_weights:
            # 使用加權選擇，但排除雙角色互動
            image_system_prompt = self._process_weighted_choice(
                image_system_prompt_weights,
                exclude=['two_character_interaction_generate_system_prompt']
            )
        else:
            # 使用單一的 image_system_prompt 配置
            image_system_prompt = first_stage_config.get('image_system_prompt') or getattr(self.config, 'image_system_prompt', 'stable_diffusion_prompt')
        
        prompt = self.config.prompt
        # 只有當 style 不為空時才加上 style: 前綴
        if style and style.strip():
            prompt = f"""{prompt}\nstyle: {style}""".strip()

        # 如果有指定角色，且 prompt 中不包含角色信息，則加入角色
        if self.config.character:
            character_lower = self.config.character.lower()
            prompt_lower = prompt.lower()
            if character_lower not in prompt_lower and "main character" not in prompt_lower:
                prompt = f"Main character: {self.config.character}\n{prompt}"
                print(f"已自動加入主角色到 prompt: {self.config.character}")

        # 對於 text2image2video，強制不使用雙角色互動系統提示詞
        # 確保使用乾淨簡單的背景
        if image_system_prompt == 'two_character_interaction_generate_system_prompt':
            print("警告：text2image2video 不支援雙角色互動，改用 stable_diffusion_prompt")
            image_system_prompt = 'stable_diffusion_prompt'
        
        descriptions = self.current_vision_manager.generate_image_prompts(prompt, image_system_prompt)
        
        print('All generated descriptions : ', descriptions)
        if self.config.character:
            character = self.config.character.lower()
            self.descriptions = [descriptions] if descriptions.replace(' ', '').lower().find(character) != -1 or descriptions.lower().find(character) != -1 else []
        else:
            self.descriptions = [descriptions] if descriptions else []

        print(f'Image descriptions : {self.descriptions}\n')
        print(f'生成描述花費 : {time.time() - start_time}')
        return self
    
    def generate_media(self):
        """生成媒體 - 分多階段：text2image -> 篩選 -> 生成影片描述 -> 生成音頻描述 -> i2v"""
        start_time = time.time()
        
        # 第一階段：Text to Image
        print("=" * 60)
        print("第一階段：Text to Image 生成")
        print("=" * 60)
        
        # 獲取策略專用配置
        first_stage_config = self._get_strategy_config('text2image2video', 'first_stage')
        
        # 獲取 workflow 路徑
        t2i_workflow_path = first_stage_config.get('t2i_workflow_path') or self.config.workflow_path
        
        first_stage_workflow = self._load_workflow(t2i_workflow_path)
        self.communicator = ComfyUICommunicator()
        
        try:
            self.communicator.connect_websocket()
            print("已建立 WebSocket 連接，開始第一階段生成")
            
            # 第一階段參數
            images_per_description = first_stage_config.get('images_per_description', 4)
            
            total_first_stage = len(self.descriptions) * images_per_description
            current_image = 0
            
            for idx, description in enumerate(self.descriptions):
                seed_start = random.randint(1, 999999999999)
                for i in range(images_per_description):
                    current_image += 1
                    print(f'\n[第一階段 {current_image}/{total_first_stage}] 為描述 {idx+1}/{len(self.descriptions)}，生成第 {i+1}/{images_per_description} 張圖片')
                    
                    custom_updates = first_stage_config.get('custom_node_updates', [])
                    
                    merged_params = {**self.config.additional_params, **first_stage_config}
                    
                    updates = self.node_manager.generate_updates(
                        workflow=first_stage_workflow,
                        updates_config=custom_updates,
                        description=description,
                        seed=seed_start + i,
                        **merged_params
                    )
                    
                    is_last_image = (idx == len(self.descriptions) - 1 and i == images_per_description - 1)
                    self.communicator.process_workflow(
                        workflow=first_stage_workflow,
                        updates=updates,
                        output_path=f"{self.config.output_dir}/first_stage",
                        file_name=f"{self.config.character}_d{idx}_{i}",
                        auto_close=False
                    )
            
            # 等待第一階段完成
            print("\n第一階段生成完成，收集所有圖片供使用者審核...")
            
            # 收集第一階段生成的所有圖片（不做 AI 篩選）
            first_stage_output_dir = f"{self.config.output_dir}/first_stage"
            first_stage_images = glob.glob(f'{first_stage_output_dir}/*')
            image_paths = [p for p in first_stage_images if any(p.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp'])]
            
            if not image_paths:
                print("警告：第一階段沒有生成任何圖片")
                return self
            
            # 保存所有圖片路徑（不做 AI 篩選，等待使用者選擇）
            self.first_stage_images = sorted(image_paths)  # 排序以確保順序一致
            print(f"第一階段共生成 {len(self.first_stage_images)} 張圖片，將透過 Discord 供使用者審核選擇")
            
            # 在第一階段圖片生成後，立即生成發文內文（使用圖 + 描述）
            print("\n" + "=" * 60)
            print("生成發文內文（基於第一階段圖片和描述）")
            print("=" * 60)
            
            # 為每張圖片提取內容並生成描述（最多3張）
            image_descriptions = []
            limited_images = self.first_stage_images[:3]  # 限制最多3張圖片
            print(f"將使用前 {len(limited_images)} 張圖片來生成文章內容（共 {len(self.first_stage_images)} 張）")
            for img_path in limited_images:
                image_content = self.current_vision_manager.extract_image_content(img_path)
                image_descriptions.append({
                    'media_path': img_path,
                    'description': image_content
                })
            
            # 暫時設置 filter_results 用於生成文章內容
            self.filter_results = image_descriptions
            
            # 生成文章內容
            self.generate_article_content()
            print(f"發文內文已生成: {self.article_content[:100]}..." if len(self.article_content) > 100 else f"發文內文已生成: {self.article_content}")
            
            # 注意：後續的影片生成將在使用者選擇圖片後進行
            # 請調用 generate_videos_from_selected_images 方法繼續流程
            
        finally:
            if self.communicator and self.communicator.ws and self.communicator.ws.connected:
                print("\n所有媒體生成完成，關閉 WebSocket 連接")
                self.communicator.ws.close()
        
        print(f'\n✅ Text2Image2Video 生成總耗時: {time.time() - start_time:.2f} 秒')
        return self

    def generate_videos_from_selected_images(self, selected_image_indices: List[int]):
        """根據使用者選擇的圖片生成影片
        
        Args:
            selected_image_indices: 使用者選擇的圖片索引列表（對應 self.first_stage_images）
        """
        if not selected_image_indices:
            print("警告：沒有選擇任何圖片")
            return self
        
        # 根據選擇的索引獲取圖片
        selected_images = [self.first_stage_images[i] for i in selected_image_indices if i < len(self.first_stage_images)]
        
        if not selected_images:
            print("警告：選擇的圖片索引無效")
            return self
        
        print(f"使用者選擇了 {len(selected_images)} 張圖片，開始生成影片...")
        
        # 更新 first_stage_images 為選擇的圖片
        self.first_stage_images = selected_images
        
        start_time = time.time()
        
        # 第二階段：為每張圖片生成影片描述
        print("\n" + "=" * 60)
        print("第二階段：生成影片描述")
        print("=" * 60)
        
        for img_path in self.first_stage_images:
            print(f"為圖片生成影片描述: {img_path}")
            # 從圖片中提取內容描述
            image_content = self.current_vision_manager.extract_image_content(img_path)
            # 生成影片描述
            video_description = self.current_vision_manager.generate_video_prompts(
                user_input=image_content,
                system_prompt_key='video_description_system_prompt'
            )
            self.video_descriptions[img_path] = video_description
            print(f"影片描述: {video_description}")
        
        # 第三階段：為每張圖片生成音頻描述
        print("\n" + "=" * 60)
        print("第三階段：生成音頻描述")
        print("=" * 60)
        
        for img_path in self.first_stage_images:
            print(f"為圖片生成音頻描述: {img_path}")
            video_desc = self.video_descriptions.get(img_path, '')
            audio_description = self.current_vision_manager.generate_audio_description(
                image_path=img_path,
                video_description=video_desc
            )
            self.audio_descriptions[img_path] = audio_description
            print(f"音頻描述: {audio_description}")
        
        # 第四階段：使用 wan2.2 workflow 生成含音頻的影片
        print("\n" + "=" * 60)
        print("第四階段：Image to Video 生成（含音頻）")
        print("=" * 60)
        
        # 確保 WebSocket 連接存在
        if not self.communicator or not self.communicator.ws or not self.communicator.ws.connected:
            self.communicator = ComfyUICommunicator()
            self.communicator.connect_websocket()
            print("已重新建立 WebSocket 連接")
        
        try:
            # 獲取策略專用配置
            video_config = self._get_strategy_config('text2image2video', 'video')
            
            # 載入 i2v workflow
            i2v_workflow_path = video_config.get('i2v_workflow_path') or 'configs/workflow/wan2.2_gguf_i2v_audio.json'
            i2v_workflow = self._load_workflow(i2v_workflow_path)
            
            videos_per_image = video_config.get('videos_per_image', 1)
            
            # 檢查是否使用 noise_seed
            use_noise_seed = video_config.get('use_noise_seed', False)
            
            total_videos = len(self.first_stage_images) * videos_per_image
            current_video = 0
            
            for img_idx, input_image_path in enumerate(self.first_stage_images):
                image_filename = self._upload_image_to_comfyui(input_image_path)
                video_desc = self.video_descriptions.get(input_image_path, '')
                audio_desc = self.audio_descriptions.get(input_image_path, '')
                
                # 為每張圖片生成影片時，使用不同的 seed 起始值
                seed_start = random.randint(1, 999999999999)
                
                for i in range(videos_per_image):
                    current_video += 1
                    print(f'\n[第四階段 {current_video}/{total_videos}] 使用圖片 {img_idx+1}/{len(self.first_stage_images)}，生成第 {i+1}/{videos_per_image} 個影片')
                    
                    custom_updates = video_config.get('custom_node_updates', []).copy()  # 複製列表避免修改原始配置
                    
                    # 添加 LoadImage 節點更新
                    custom_updates.append({
                        "node_type": "LoadImage",
                        "node_index": 0,
                        "inputs": {"image": image_filename}
                    })
                    
                    # 添加 Positive prompt 更新（影片描述）
                    # 使用 node_id 直接更新，避免過濾條件問題
                    # 根據 workflow JSON，節點 70 是 "Positive prompt" (PrimitiveString)
                    custom_updates.append({
                        "node_id": "70",
                        "inputs": {"value": video_desc}
                    })
                    
                    # 添加 Audio prompt 更新
                    # 根據 workflow JSON，節點 94 是 "Audio prompt" (PrimitiveString)
                    custom_updates.append({
                        "node_id": "94",
                        "inputs": {"value": audio_desc}
                    })
                    
                    merged_params = {**self.config.additional_params, **video_config}
                    # 從 merged_params 中移除 use_noise_seed，因為它已經被明確傳遞
                    merged_params.pop('use_noise_seed', None)
                    
                    # 確保每個影片使用不同的 seed（基於圖片索引和影片索引）
                    video_seed = seed_start + (img_idx * videos_per_image) + i
                    
                    updates = self.node_manager.generate_updates(
                        workflow=i2v_workflow,
                        updates_config=custom_updates,
                        description=video_desc,
                        seed=video_seed,
                        use_noise_seed=use_noise_seed,  # 傳遞 use_noise_seed 參數
                        **merged_params
                    )
                    
                    is_last_video = (img_idx == len(self.first_stage_images) - 1 and i == videos_per_image - 1)
                    self.communicator.process_workflow(
                        workflow=i2v_workflow,
                        updates=updates,
                        output_path=f"{self.config.output_dir}/videos",
                        file_name=f"{self.config.character}_i2v_{img_idx}_{i}",
                        auto_close=False
                    )
        finally:
            if self.communicator and self.communicator.ws and self.communicator.ws.connected:
                print("\n所有影片生成完成，關閉 WebSocket 連接")
                self.communicator.ws.close()
        
        # 更新 filter_results 為生成的影片結果，以便後續生成文章內容
        video_output_dir = f"{self.config.output_dir}/videos"
        if os.path.exists(video_output_dir):
            media_paths = glob.glob(f'{video_output_dir}/*')
            video_paths = [p for p in media_paths if any(p.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.gif', '.webm'])]
            
            if video_paths:
                self.filter_results = []
                for video_path in video_paths:
                    # 嘗試從文件名匹配對應的圖片和描述
                    match = re.search(rf'{re.escape(self.config.character)}_i2v_(\d+)_\d+\.', video_path, re.IGNORECASE)
                    if match:
                        img_idx = int(match.group(1))
                        if img_idx < len(self.first_stage_images):
                            img_path = self.first_stage_images[img_idx]
                            video_desc = self.video_descriptions.get(img_path, '')
                            self.filter_results.append({
                                'media_path': video_path,
                                'description': video_desc,
                                'similarity': 1.0
                            })
                    else:
                        # 如果無法匹配，使用第一個描述
                        self.filter_results.append({
                            'media_path': video_path,
                            'description': self.video_descriptions.get(self.first_stage_images[0], '') if self.first_stage_images else '',
                            'similarity': 1.0
                        })
                print(f"已更新 filter_results，包含 {len(self.filter_results)} 個影片")
        
        print(f'\n✅ 影片生成總耗時: {time.time() - start_time:.2f} 秒')
        return self
    
    def analyze_media_text_match(self, similarity_threshold):
        """分析生成的媒體
        
        對於 Text2Image2Video：
        - 如果影片已生成，返回所有影片（不做 AI 判讀）
        - 如果影片未生成，返回所有第一階段的圖片供 Discord 審核
        """
        # 檢查是否已有生成的影片
        video_output_dir = f"{self.config.output_dir}/videos"
        if os.path.exists(video_output_dir):
            media_paths = glob.glob(f'{video_output_dir}/*')
            video_paths = [p for p in media_paths if any(p.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.gif', '.webm'])]

            if video_paths:
                # 影片已生成，返回所有影片（不做 AI 判讀）
                self.filter_results = []
                for video_path in video_paths:
                    # 嘗試從文件名匹配對應的圖片和描述
                    match = re.search(rf'{re.escape(self.config.character)}_i2v_(\d+)_\d+\.', video_path, re.IGNORECASE)
                    if match:
                        img_idx = int(match.group(1))
                        if img_idx < len(self.first_stage_images):
                            img_path = self.first_stage_images[img_idx]
                            video_desc = self.video_descriptions.get(img_path, '')
                            self.filter_results.append({
                                'media_path': video_path,
                                'description': video_desc,
                                'similarity': 1.0  # 不做 AI 判讀，設為 1.0
                            })
                    else:
                        # 如果無法匹配，使用第一個描述
                        self.filter_results.append({
                            'media_path': video_path,
                            'description': self.video_descriptions.get(self.first_stage_images[0], '') if self.first_stage_images else '',
                            'similarity': 1.0
                        })
                return self
        
        # 影片未生成，返回所有第一階段的圖片供 Discord 審核（不做 AI 篩選）
        if not self.first_stage_images:
            self.filter_results = []
            return self
        
        # 返回所有圖片，不做 AI 篩選
        self.filter_results = []
        for img_path in self.first_stage_images:
            self.filter_results.append({
                'media_path': img_path,
                'description': self.descriptions[0] if self.descriptions else '',
                'similarity': 1.0  # 不做 AI 判讀，設為 1.0
            })
        
        print(f"返回 {len(self.filter_results)} 張圖片供 Discord 審核（不做 AI 篩選）")
        return self
        