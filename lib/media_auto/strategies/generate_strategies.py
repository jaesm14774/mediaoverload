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
    
    def __init__(self, character_repository=None):
        self.ollama_vision_manager = VisionManagerBuilder() \
            .with_vision_model('ollama', model_name='llava:13b') \
            .with_text_model('ollama', model_name='llama3.2:latest') \
            .build()
        self.gemini_vision_manager = VisionManagerBuilder() \
            .with_vision_model('gemini', model_name='gemini-2.5-flash-lite-preview-06-17') \
            .with_text_model('gemini', model_name='gemini-2.5-flash-lite-preview-06-17') \
            .build()
        self.node_manager = NodeManager()
        self.descriptions: List[str] = []
        self.ollama_switcher = ModelSwitcher(self.ollama_vision_manager)
        self.character_repository = character_repository

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
            descriptions = self.ollama_vision_manager.generate_image_prompts(prompt, self.config.image_system_prompt)
        
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
                descriptions = self.ollama_vision_manager.generate_two_character_interaction_prompt(
                    main_character=self.config.character,
                    secondary_character=secondary_character,
                    prompt=prompt,  # 添加原始prompt
                    style=style if style else 'minimalist'
                )
                return descriptions
            else:
                print('無法獲取 Secondary Role，使用預設方法')
                # 回退到原有方法，保留原始prompt
                return self.ollama_vision_manager.generate_image_prompts(prompt, 'stable_diffusion_prompt')
                
        except Exception as e:
            print(f'雙角色互動生成時發生錯誤: {e}，使用預設方法')
            # 回退到原有方法，保留原始prompt
            return self.ollama_vision_manager.generate_image_prompts(prompt, 'stable_diffusion_prompt')
    
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

    def generate_image(self):
        start_time = time.time()
        workflow = self._load_workflow(self.config.workflow_path)
        self.communicator = ComfyUICommunicator()
        self.communicator.connect_websocket()
        
        # 為每個描述生成圖片
        images_per_description = self.config.additional_params.get('images_per_description', 4)
        for idx, description in enumerate(self.descriptions):
            seed_start = random.randint(1, 999999999999)
            for i in range(images_per_description):
                print(f'為第{idx}描述，生成第{i}張圖片\n')
                
                # 檢查是否有自定義的節點更新配置
                custom_updates = self.config.additional_params.get('custom_node_updates', [])
                if custom_updates:
                    # 使用通用的節點更新方法
                    updates = self.node_manager.generate_generic_updates(
                        workflow=workflow,
                        updates_config=custom_updates
                    )
                    
                    # 添加文字和種子更新（如果沒有在 custom_updates 中指定）
                    has_text_update = any(u.get('node_type') in ['PrimitiveString', 'CLIPTextEncode'] for u in custom_updates)
                    if not has_text_update:
                        text_updates = self.node_manager.generate_text_updates(
                            workflow=workflow,
                            description=description,
                            **self.config.additional_params
                        )
                        updates.extend(text_updates)
                    
                    sampler_updates = self.node_manager.generate_sampler_updates(
                        workflow=workflow,
                        seed=seed_start + i
                    )
                    updates.extend(sampler_updates)
                else:
                    # 使用原本的方法
                    updates = self.node_manager.generate_updates(
                        workflow=workflow,
                        description=description,
                        seed=seed_start + i,
                        **self.config.additional_params
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

    def analyze_image_text_match(self, similarity_threshold) -> Dict[str, Any]:
        """分析生成的圖片"""
        image_paths = glob.glob(f'{self.config.output_dir}/*')
        self.filter_results = self.gemini_vision_manager.analyze_image_text_match(
            image_paths=image_paths,
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
        article_content = self.ollama_vision_manager.generate_seo_hashtags('\n\n'.join(content_parts))

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
        
    def generate_article_content(self):
        pass


class Text2VideoStrategy(ContentStrategy):
    """文生影片策略"""
    
    def generate_description(self):
        # 實現文生影片的邏輯
        pass

    def generate_article_content(self):
        pass