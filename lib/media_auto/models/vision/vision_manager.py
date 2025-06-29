from typing import List, Optional, Dict, Any
import re
import time
from lib.media_auto.models.interfaces.ai_model import AIModelInterface, ModelConfig
from lib.media_auto.models.vision.model_registry import ModelRegistry
from configs.prompt.image_system_guide import *

class VisionContentManager:
    """處理圖片內容分析與生成的類別"""
    def __init__(self, 
                 vision_model: AIModelInterface,
                 text_model: AIModelInterface,
                 prompts_config: dict):
        self.vision_model = vision_model
        self.text_model = text_model
        self.prompts = prompts_config
    
    def extract_image_content(self, image_path: str, **kwargs) -> str:
        """分析已有圖片並提取內容描述"""
        messages = [{
            'role': 'user',
            'content': self.prompts['describe_image_prompt']
        }]
        return self.vision_model.chat_completion(
            messages=messages,
            images=[image_path],
            **kwargs
        )
    
    def analyze_image_text_similarity(self, 
                                    text: str, 
                                    image_path: str, 
                                    main_character: str = '',
                                    **kwargs) -> str:
        """分析已有圖片並提取內容描述"""
        messages = [
            {
                'role': 'system',
                'content': self.prompts['text_image_similarity_prompt']
            },
            {
                'role': 'user',
                'content': f'main_character: {main_character} and image description: {text}'
            }
        ]
        return self.vision_model.chat_completion(
            messages=messages,
            images=[image_path],
            **kwargs
        )
    
    def generate_image_prompts(self, user_input: str, system_prompt_key: str = 'stable_diffusion_prompt', **kwargs) -> List[str]:
        """根據用戶輸入生成圖片描述提示詞"""
        
        actual_key_to_use = system_prompt_key
        if system_prompt_key not in self.prompts:
            print(f"Warning: Prompt key '{system_prompt_key}' not found in prompts configuration. Falling back to default 'stable_diffusion_prompt'.")
            actual_key_to_use = 'stable_diffusion_prompt'
            # if actual_key_to_use not in self.prompts:
            #     raise ValueError(f"Default prompt key '{actual_key_to_use}' is also missing from prompts configuration.")

        print(f"Using image system prompt key: {actual_key_to_use}")
        messages = [
            {'role': 'system', 'content': self.prompts[actual_key_to_use]},
            {'role': 'user', 'content': user_input}
        ]
        result = self.vision_model.chat_completion(messages=messages, **kwargs)
        if '</think>' in result:  # deepseek r1 will have <think>...</think> format
            result = result.split('</think>')[-1].strip()
        return result
    
    def generate_seo_hashtags(self, description: str, **kwargs) -> str:
        """生成 SEO 優化的 hashtags"""
        messages = [
            {'role': 'system', 'content': self.prompts['seo_hashtag_prompt']},
            {'role': 'user', 'content': description}
        ]
        return self.text_model.chat_completion(messages=messages, **kwargs)

    def generate_input_prompt(self, character, extra='', prompt_type='arbitrary_input_system_prompt') -> str:
        """生成任意輸入的轉換結果"""
        messages = [
            {'role': 'system', 'content': self.prompts[prompt_type]},
            {'role': 'assistant', 'content': "Check main character name must be in the output"},
            {'role': 'user', 'content': f"""main character: {character} {extra}"""}
        ]
        result = self.text_model.chat_completion(messages=messages)    
        if '</think>' in result:  # deepseek r1 will have <think>...</think> format
            result = result.split('</think>')[-1].strip()
        
        messages = [
            {'role': 'system', 'content': "As an expert editor, distill the text's essence. You must preserve the main character's name, along with the original style and emotion. Keep the output under 30 words."},
            {'role': 'user', 'content': f"""{result}"""}
        ]
        result = self.text_model.chat_completion(messages=messages)   

        if '</think>' in result:  # deepseek r1 will have <think>...</think> format
            result = result.split('</think>')[-1].strip()         
        
        return result

    def generate_two_character_interaction_prompt(self, main_character, secondary_character, prompt='', style='minimalist', **kwargs) -> str:
        """生成雙角色互動的提示詞"""
        # 構建輸入格式，包含所有必要字段
        user_input = f"""Main Role: {main_character}
Secondary Role: {secondary_character}
Style: {style}"""
        
        # 如果有原始prompt，將其納入輸入
        if prompt and prompt.strip():
            user_input += f"""
Original Context: {prompt.strip()}"""
        
        messages = [
            {'role': 'system', 'content': self.prompts['two_character_interaction_generate_system_prompt']},
            {'role': 'user', 'content': user_input}
        ]
        
        result = self.text_model.chat_completion(messages=messages, **kwargs)
        if '</think>' in result:  # deepseek r1 will have <think>...</think> format
            result = result.split('</think>')[-1].strip()
        
        return result

    def analyze_image_text_match(self, 
                               image_paths: List[str],
                               descriptions: List[str],
                               main_character: str,
                               similarity_threshold: float = 0.9,
                               **kwargs) -> List[Dict[str, Any]]:
        """分析圖文匹配度並過濾結果
        
        Args:
            image_paths: 圖片路徑列表
            descriptions: 描述文字列表
            main_character: 主要角色名稱
            similarity_threshold: 相似度閾值
            **kwargs: 其他參數
            
        Returns:
            過濾後的結果列表，每個結果包含：
            - image_path: 圖片路徑
            - description: 對應的描述
            - similarity: 相似度分數
        """
        start_time = time.time()
        total_results = []
        
        for image_path in image_paths:
            print(f'進行文圖匹配程度判斷中 : {image_path}\n')
            desc_index = int(re.search(f'{main_character}_d(\d+)_\d+\.', image_path).group(1))
            similarity = self.analyze_image_text_similarity(
                text=descriptions[desc_index],
                image_path=image_path,
                main_character=main_character,
                **kwargs
            )
            
            total_results.append({
                'image_path': image_path,
                'description': descriptions[desc_index],
                'similarity': similarity
            })
            time.sleep(3)  # google free tier rate limit

        # 過濾結果
        filter_results = []
        for row in total_results:
            try:
                similarity = float(row['similarity'].strip())
                if similarity >= similarity_threshold:
                    filter_results.append(row)
            except:
                continue
                
        print(f'分析圖文匹配程度 花費 : {time.time() - start_time}')
        return filter_results

class VisionManagerBuilder:
    """Vision Manager 建構器"""
    def __init__(self):
        self.vision_model_type = 'ollama'
        self.text_model_type = 'ollama'
        self.vision_config = {'model_name': 'llava:13b', 'temperature': 0.3}
        self.text_config = {'model_name': 'llama3.2', 'temperature': 0.3}
        self.prompts_config = {
            'seo_hashtag_prompt': seo_hashtag_prompt,
            'stable_diffusion_prompt': stable_diffusion_prompt,
            'describe_image_prompt': describe_image_prompt,
            'text_image_similarity_prompt': text_image_similarity_prompt,
            'arbitrary_input_system_prompt': arbitrary_input_system_prompt,
            'guide_seo_article_system_prompt': guide_seo_article_system_prompt,
            'unbelievable_world_system_prompt': unbelievable_world_system_prompt,
            'buddhist_combined_image_system_prompt': buddhist_combined_image_system_prompt,
            'fill_missing_details_system_prompt': fill_missing_details_system_prompt,
            'two_character_interaction_generate_system_prompt': two_character_interaction_generate_system_prompt
        }
    
    def with_vision_model(self, model_type: str, **config):
        """設置視覺模型"""
        self.vision_model_type = model_type
        if config:
            self.vision_config.update(config)
        return self
    
    def with_text_model(self, model_type: str, **config):
        """設置文本模型"""
        self.text_model_type = model_type
        if config:
            self.text_config.update(config)
        return self
    
    def with_prompts(self, prompts: dict):
        """設置提示詞配置"""
        if not isinstance(prompts, dict):
            raise ValueError("prompts must be a dictionary")
        
        # 確保必要的提示詞存在
        required_prompts = [
            'seo_hashtag_prompt', 'stable_diffusion_prompt', 'describe_image_prompt', 
            'text_image_similarity_prompt', 'arbitrary_input_system_prompt', 'guide_seo_article_system_prompt', 
            'unbelievable_world_system_prompt', 'buddhist_combined_image_system_prompt', 'two_character_interaction_generate_system_prompt'
        ]
        for prompt in required_prompts:
            if prompt not in prompts and prompt not in self.prompts_config:
                raise ValueError(f"Missing required prompt: {prompt}")
        
        self.prompts_config.update(prompts)
        return self
    
    def build(self) -> 'VisionContentManager':
        """建構 VisionContentManager 實例"""
        vision_model_class = ModelRegistry.get_model(self.vision_model_type)
        text_model_class = ModelRegistry.get_model(self.text_model_type)
        
        vision_model = vision_model_class(ModelConfig(**self.vision_config))
        text_model = text_model_class(ModelConfig(**self.text_config))
        
        # 確保必要的提示詞存在
        if 'stable_diffusion_prompt' not in self.prompts_config:
            raise ValueError("Missing required prompt: stable_diffusion_prompt")
        
        return VisionContentManager(
            vision_model=vision_model,
            text_model=text_model,
            prompts_config=self.prompts_config
        ) 