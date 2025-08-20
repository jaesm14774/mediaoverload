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

class Text2ImageStrategy(ContentStrategy):
    """
    文生圖策略
    實驗顯示 ollama 在判斷 文圖匹配時，不論是llama3.2-vision 或是 llava 都沒有達到期望
    改成google 大幅 增加準確性
    """
    
    def __init__(self, character_repository=None, vision_manager=None):
        # 保留原有的邏輯作為後備
        self.ollama_vision_manager = VisionManagerBuilder() \
            .with_vision_model('ollama', model_name='llava:13b') \
            .with_text_model('ollama', model_name='llama3.2:latest') \
            .build()
        self.gemini_vision_manager = VisionManagerBuilder() \
            .with_vision_model('gemini', model_name='gemini-2.5-flash-lite-preview-06-17') \
            .with_text_model('gemini', model_name='gemini-2.5-flash-lite-preview-06-17') \
            .build()
        # 添加 OpenRouter 視覺管理器 - 使用隨機 free 模型
        self.openrouter_vision_manager = VisionManagerBuilder() \
            .with_vision_model('openrouter') \
            .with_text_model('openrouter') \
            .with_random_models(True) \
            .build()
        self.ollama_switcher = ModelSwitcher(self.ollama_vision_manager)
        # 如果有外部傳入的 VisionManager，優先使用
        if vision_manager is not None:
            self.current_vision_manager = vision_manager
            self.external_vision_manager = True
        else:
            self.current_vision_manager = self.openrouter_vision_manager
            self.external_vision_manager = False
        
        self.node_manager = NodeManager()
        self.descriptions: List[str] = []
        self.character_repository = character_repository

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

    def generate_description(self):
        """描述生成"""        
        start_time = time.time()
        
        style = getattr(self.config, 'style', '')
        prompt = self.config.prompt
        if style:
            prompt = f"""{prompt}\nstyle:{style}""".strip()

        # 檢查是否使用雙角色互動系統提示詞
        if self.config.image_system_prompt == 'two_character_interaction_generate_system_prompt':
            # 使用雙角色互動生成邏輯
            descriptions = self._generate_two_character_interaction_description(prompt, style)
        else:
            # 使用原有的生成邏輯
            descriptions = self.current_vision_manager.generate_image_prompts(prompt, self.config.image_system_prompt)
        
        print('All generated descriptions : ', descriptions)
        if self.config.character:
            character = self.config.character.lower()
            self.descriptions = [descriptions] if descriptions.replace(' ', '').lower().find(character) != -1 or descriptions.lower().find(character) != -1 else []

        print(f'Image descriptions : {self.descriptions}\n')
        print(f'生成描述花費 : {time.time() - start_time}')
        return self

    def _generate_two_character_interaction_description(self, prompt: str, style: str = '') -> str:
        """生成雙角色互動描述
        
        這個方法會從資料庫中獲取一個Secondary Role，並使用雙角色互動系統提示詞
        包含用戶原始prompt
        """
        try:
            # 獲取 Secondary Role
            secondary_character = self._get_random_secondary_character(
                self.config.character, 
                self.character_repository
            )
            
            if secondary_character:
                print(f'雙角色互動：Main Role: {self.config.character}, Secondary Role: {secondary_character}')
                
                # 修復：傳遞原始prompt給雙角色互動生成
                descriptions = self.current_vision_manager.generate_two_character_interaction_prompt(
                    main_character=self.config.character,
                    secondary_character=secondary_character,
                    prompt=prompt,  # 添加原始prompt
                    style=style if style else 'minimalist'
                )
                return descriptions
            else:
                print('無法獲取 Secondary Role，使用預設方法')
                # 回退到原有方法，保留原始prompt
                return self.current_vision_manager.generate_image_prompts(prompt, 'stable_diffusion_prompt')
                
        except Exception as e:
            print(f'雙角色互動生成時發生錯誤: {e}，使用預設方法')
            # 回退到原有方法，保留原始prompt
            return self.current_vision_manager.generate_image_prompts(prompt, 'stable_diffusion_prompt')
    
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
        
    def generate_media(self):
        start_time = time.time()
        workflow = self._load_workflow(self.config.workflow_path)
        self.communicator = ComfyUICommunicator()
        self.communicator.connect_websocket()
        
        # 為每個描述生成圖片
        # 優先使用圖片專用參數，然後是通用參數
        image_params = self.config.additional_params.get('image', {})
        images_per_description = image_params.get('images_per_description', 
                                                self.config.additional_params.get('images_per_description', 4))
        
        for idx, description in enumerate(self.descriptions):
            seed_start = random.randint(1, 999999999999)
            for i in range(images_per_description):
                print(f'為第{idx}描述，生成第{i}張圖片\n')
                
                # 檢查是否有自定義的節點更新配置
                # 優先使用圖片專用配置，然後是通用配置
                custom_updates = (image_params.get('custom_node_updates') or 
                                self.config.additional_params.get('custom_node_updates', []))
                
                # 合併圖片專用參數和通用參數
                merged_params = {**self.config.additional_params, **image_params}
                
                # 使用統一的更新方法（自動處理衝突檢查）
                updates = self.node_manager.generate_updates(
                    workflow=workflow,
                    updates_config=custom_updates,
                    description=description,
                    seed=seed_start + i,
                    **merged_params
                )
                try:
                    self.communicator.process_workflow(
                        workflow=workflow,
                        updates=updates,
                        output_path=f"{self.config.output_dir}",
                        file_name=f"{self.config.character}_d{idx}_{i}"
                    )
                finally:
                    if self.communicator and self.communicator.ws:
                        self.communicator.ws.close()
        print(f'生成圖片花費 : {time.time() - start_time}')
        return self

    def analyze_media_text_match(self, similarity_threshold) -> Dict[str, Any]:
        """分析生成的圖片 - 隨機選擇分析模型"""
        media_paths = glob.glob(f'{self.config.output_dir}/*')
        
        # 隨機選擇分析管理器：Gemini 或 OpenRouter
        available_managers = [self.gemini_vision_manager, self.openrouter_vision_manager]
        selected_manager = np.random.choice(available_managers, p=[0.8, 0.2])
        
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

    def prevent_hashtag_count_too_more(self, hashtag_text):
        hashtag_candidate_list=[part.lower() for part in re.split(pattern='\n|#', string=hashtag_text) if part != '']
    
        deduplicate_list = []
        for part in hashtag_candidate_list:
            if part not in deduplicate_list:
                deduplicate_list.append(part)
        
        if len(deduplicate_list) > 30:
            hashtag_text = deduplicate_list[0] + '\n#' + '#'.join(deduplicate_list[1:2] + np.random.choice(deduplicate_list[2:], size=27, replace=False).tolist())
    
        return hashtag_text.lower().strip()

    def generate_article_content(self):
        start_time = time.time()
        # 需要時可以動態切換模型
        self.ollama_switcher.switch_text_model('ollama', model_name='gemma3:12b')

        # 整合角色名稱、描述和預設標籤
        content_parts = [
            self.config.character,
            *list(set([row['description'] for row in self.filter_results])),
            self.config.prompt
        ]
        article_content = self.current_vision_manager.generate_seo_hashtags('\n\n'.join(content_parts))

        # 加入預設標籤
        if self.config.default_hashtags:
            article_content+=' #'+ ' #'.join([tag.lstrip('#') for tag in self.config.default_hashtags])

        if '</think>' in article_content:  # deepseek r1 will have <think>...</think> format
            article_content = article_content.split('</think>')[-1].strip()        
        
        article_content = self.prevent_hashtag_count_too_more(article_content)
        self.article_content = article_content.replace('"', '').replace('*', '').lower()

        print(f'產生文章內容花費 : {time.time() - start_time}')
        return self


class Image2ImageStrategy(ContentStrategy):
    """圖生圖策略"""
    
    def generate_description(self):
        # 實現圖生圖的邏輯
        pass
    
    def generate_media(self):
        # 實現圖生圖的媒體生成邏輯
        pass
    
    def analyze_media_text_match(self, similarity_threshold):
        # 實現圖生圖的媒體文本匹配邏輯
        pass
        
    def generate_article_content(self):
        pass


class Text2VideoStrategy(ContentStrategy):
    """文生影片策略"""
    
    def __init__(self, character_repository=None, vision_manager=None):
        # 如果有外部傳入的 VisionManager，優先使用
        if vision_manager is not None:
            self.current_vision_manager = vision_manager
            self.external_vision_manager = True
        else:
            # 保留原有的邏輯作為後備
            self.ollama_vision_manager = VisionManagerBuilder() \
                .with_vision_model('ollama', model_name='qwen2.5vl:7b') \
                .with_text_model('ollama', model_name='gemma3:4b') \
                .build()
            # 添加 OpenRouter 視覺管理器用於視頻生成 - 使用隨機 free 模型
            self.openrouter_vision_manager = VisionManagerBuilder() \
                .with_vision_model('openrouter') \
                .with_text_model('openrouter') \
                .with_random_models(True) \
                .build()
            # 預設使用 openrouter，可以透過 set_vision_provider 方法切換
            self.current_vision_manager = self.openrouter_vision_manager
            self.external_vision_manager = False
        
        self.node_manager = NodeManager()
        self.descriptions: List[str] = []
        self.character_repository = character_repository
        self.communicator = None

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
        
    def _load_workflow(self, path: str) -> Dict[str, Any]:
        """載入工作流配置"""
        with open(path, "r", encoding='utf-8') as f:
            return json.loads(f.read())
    
    def generate_description(self):
        """生成視頻描述"""
        start_time = time.time()
        
        # 獲取基本配置
        style = getattr(self.config, 'style', '')
        prompt = self.config.prompt
        if style:
            prompt = f"""{prompt}\nstyle:{style}""".strip()
        
        # 第一階段：使用角色名稱生成基礎描述
        character_description = self.current_vision_manager.generate_image_prompts(
            user_input=self.config.character,
            system_prompt_key=self.config.image_system_prompt or 'unbelievable_world_system_prompt'
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
        
        # 載入工作流
        workflow = self._load_workflow(self.config.workflow_path)
        self.communicator = ComfyUICommunicator()
        self.communicator.connect_websocket()
        
        try:
            # 為每個描述生成視頻
            # 優先使用視頻專用參數，然後是通用參數
            video_params = self.config.additional_params.get('video', {})
            videos_per_description = video_params.get('videos_per_description', 
                                                    self.config.additional_params.get('videos_per_description', 2))
            
            for idx, description in enumerate(self.descriptions):
                seed_start = random.randint(1, 999999999999)
                
                for i in range(videos_per_description):
                    print(f'為第{idx}描述，生成第{i}個視頻\n')
                    
                    # 準備自定義的節點更新配置
                    # 優先使用視頻專用配置，然後是通用配置
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
                    
                    custom_updates = (video_params.get('custom_node_updates') or 
                                    self.config.additional_params.get('custom_node_updates', default_video_updates))
                    
                    # 合併視頻專用參數和通用參數
                    merged_params = {**self.config.additional_params, **video_params}
                    
                    # 使用統一的更新方法（自動處理衝突檢查）
                    updates = self.node_manager.generate_updates(
                        workflow=workflow,
                        updates_config=custom_updates,
                        description=description,
                        seed=seed_start + i,
                        **merged_params
                    )
                    
                    # 處理工作流
                    self.communicator.process_workflow(
                        workflow=workflow,
                        updates=updates,
                        output_path=f"{self.config.output_dir}",
                        file_name=f"{self.config.character}_video_d{idx}_{i}"
                    )
                    
        finally:
            if self.communicator and self.communicator.ws:
                self.communicator.ws.close()
        
        print(f'生成視頻花費: {time.time() - start_time}')
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
    
    def generate_article_content(self):
        """生成文章內容"""
        start_time = time.time()
        
        # 使用現有的邏輯生成文章內容
        if not self.filter_results:
            self.article_content = f"#{self.config.character} #AI #video"
            return self
        
        # 整合角色名稱、描述和預設標籤
        content_parts = [
            self.config.character,
            *list(set([row['description'] for row in self.filter_results])),
            self.config.prompt
        ]
        
        article_content = self.current_vision_manager.generate_seo_hashtags('\n\n'.join(content_parts))
        
        # 加入預設標籤
        if hasattr(self.config, 'default_hashtags') and self.config.default_hashtags:
            article_content += ' #' + ' #'.join([tag.lstrip('#') for tag in self.config.default_hashtags])
        
        if '</think>' in article_content:  # deepseek r1 will have <think>...</think> format
            article_content = article_content.split('</think>')[-1].strip()
        
        self.article_content = article_content.replace('"', '').replace('*', '').lower()
        
        print(f'產生文章內容花費: {time.time() - start_time}')
        return self
        