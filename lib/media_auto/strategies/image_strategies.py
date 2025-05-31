from typing import List, Dict, Any
import random
import time
import glob
import re
import json
import numpy as np

from lib.media_auto.strategies.base_strategy import ContentStrategy
from lib.content_generation.image_content_generator import VisionManagerBuilder, ModelSwitcher
from lib.comfyui.websockets_api import ComfyUICommunicator
from lib.comfyui.node_manager import NodeManager

class Text2ImageStrategy(ContentStrategy):
    """
    文生圖策略
    實驗顯示 ollama 在判斷 文圖匹配時，不論是llama3.2-vision 或是 llava 都沒有達到期望
    改成google 大幅 增加準確性
    """
    
    def __init__(self):
        self.ollama_vision_manager = VisionManagerBuilder() \
            .with_vision_model('ollama', model_name='llava:13b') \
            .with_text_model('ollama', model_name='llama3.2') \
            .build()
        self.gemini_vision_manager = VisionManagerBuilder() \
            .with_vision_model('gemini', model_name='gemini-2.0-flash-lite') \
            .with_text_model('gemini', model_name='gemini-2.0-flash-lite') \
            .build()
        self.node_manager = NodeManager()
        self.descriptions: List[str] = []
        self.ollama_switcher = ModelSwitcher(self.ollama_vision_manager)

    def _load_workflow(self, path: str) -> Dict[str, Any]:
        """載入工作流配置"""
        with open(path, "r", encoding='utf-8') as f:
            return json.loads(f.read())

    def generate_description(self):
        """描述生成"""        
        start_time = time.time()
        descriptions = [
            desc for desc in self.ollama_vision_manager.generate_image_prompts(self.config.prompt)
            if not desc.startswith('Here are')
        ]
        print('All generated descriptions : ', descriptions)
        if self.config.character:
            character = self.config.character.lower()
            self.descriptions = [
                desc for desc in descriptions
                if desc.replace(' ', '').lower().find(character) != -1 or desc.lower().find(character) != -1
            ]
        print(f'Image descriptions : {self.descriptions}\n')
        print(f'生成描述花費 : {time.time() - start_time}')
        return self

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
                    has_seed_update = True
                    if not has_text_update:
                        text_updates = self.node_manager.generate_text_updates(
                            workflow=workflow,
                            description=description,
                            **self.config.additional_params
                        )
                        updates.extend(text_updates)
                    
                    if has_seed_update:
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
                self.communicator.process_workflow(
                    workflow=workflow,
                    updates=updates,
                    output_path=f"{self.config.output_dir}",
                    file_name=f"{self.config.character}_d{idx}_{i}"
                )
        
        self.communicator.ws.close()
        print(f'生成圖片花費 : {time.time() - start_time}')
        return self

    def analyze_image_text_match(self, similarity_threshold) -> Dict[str, Any]:
        """分析生成的圖片"""
        start_time = time.time()
        total_results = []
        image_paths = glob.glob(f'{self.config.output_dir}/*')
        
        for image_path in image_paths:
            print(f'進行文圖匹配程度判斷中 : {image_path}\n')
            desc_index = int(re.search(f'{self.config.character}_d(\d+)_\d+\.', image_path).group(1))
            similarity = self.gemini_vision_manager.analyze_image_text_similarity(
                text=self.descriptions[desc_index],
                image_path=image_path,
                main_character=self.config.character,
                temperature=0.3
            )
            
            total_results.append({
                'image_path': image_path,
                'description': self.descriptions[desc_index],
                'similarity': similarity
            })
            time.sleep(8) #google free tier rate limit

        self.total_results = total_results

        # 過濾結果
        self.filter_results = []
        for row in total_results:
            try:
                similarity = float(row['similarity'].strip())
                if similarity >= similarity_threshold:
                    self.filter_results.append(row)
            except:
                continue
        print(f'分析圖文匹配程度 花費 : {time.time() - start_time}')
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
        # # 重整ig article post content
        # messages = [
        #     {'role': 'system', 'content': self.ollama_vision_manager.prompts['guide_seo_article_system_prompt']},
        #     {'role': 'user', 'content': article_content}
        # ]
        # article_content = self.ollama_vision_manager.text_model.chat_completion(messages=messages)
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