from abc import ABC, abstractmethod
from typing import List, Optional, Type, Dict
from dataclasses import dataclass
import ollama
import os
import re
from dotenv import load_dotenv
from configs.prompt.image_system_guide import *

@dataclass
class ModelConfig:
    """模型配置類"""
    model_name: str
    temperature: float = 0.3
    max_tokens: Optional[int] = None
    
class AIModelInterface(ABC):
    """AI 模型接口"""
    @abstractmethod
    def chat_completion(self, 
                       messages: List[dict],
                       images: Optional[List[str]] = None,
                       **kwargs) -> str:
        """執行聊天完成"""
        pass

class OllamaModel(AIModelInterface):
    """Ollama 模型實現"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = ollama.Client(
            host='http://host.docker.internal:11434'
        )

        
    def chat_completion(self, 
                       messages: List[dict],
                       images: Optional[List[str]] = None,
                       **kwargs) -> str:
        if images:
            messages[-1]['images'] = images
            
        response = self.client.chat(
            model=self.config.model_name,
            messages=messages,
            options={
                "temperature": self.config.temperature
            }
        )
        return response.message.content

class GeminiModel(AIModelInterface):
    """Gemini 模型實現"""
    def __init__(self, config: ModelConfig):
        self.config = config
        import google.generativeai as genai
        self.genai = genai
        # 這裡需要設置你的 API 金鑰
        load_dotenv(f'media_overload.env')
        genai.configure(api_key=os.environ['gemini_api_token'])

    def chat_completion(self, 
                       messages: List[dict],
                       images: Optional[List[str]] = None,
                       **kwargs) -> str:
        # 將 role-based messages 轉換為純文本
        combined_prompt = self._combine_messages(messages)
        
        model = self.genai.GenerativeModel(self.config.model_name)
        if images:
            # 處理圖片輸入
            image_parts = [self._load_image(img_path) for img_path in images]
            response = model.generate_content([combined_prompt, *image_parts])
        else:
            response = model.generate_content(combined_prompt)
            
        return response.text
    
    def _combine_messages(self, messages: List[dict]) -> str:
        """將角色型消息合併為單一提示詞"""
        combined = ""
        for msg in messages:
            content = msg.get('content', '')
            if msg.get('role') == 'system':
                combined += f"Instructions: {content}\n"
            else:
                combined += f"{content}\n"
        return combined.strip()
    
    def _load_image(self, image_path: str):
        """載入圖片"""
        from PIL import Image
        return Image.open(image_path)

class ModelRegistry:
    """模型註冊表，用於管理不同類型的模型實例"""
    _models: Dict[str, Type['AIModelInterface']] = {}
    
    @classmethod
    def register(cls, name: str, model_class: Type['AIModelInterface']):
        """註冊新的模型類別"""
        cls._models[name] = model_class
        
    @classmethod
    def get_model(cls, name: str) -> Type['AIModelInterface']:
        """獲取指定名稱的模型類別"""
        if name not in cls._models:
            raise ValueError(f"Model {name} not registered")
        return cls._models[name]
    
    @classmethod
    def available_models(cls) -> List[str]:
        """獲取所有已註冊的模型名稱"""
        return list(cls._models.keys())


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
    
    def generate_image_prompts(self, user_input: str, **kwargs) -> List[str]:
        """根據用戶輸入生成圖片描述提示詞"""
        messages = [
            {'role': 'system', 'content': self.prompts['image_system_prompt']},
            {'role': 'user', 'content': user_input}
        ]
        response = self.vision_model.chat_completion(messages=messages, **kwargs)
        return [re.sub(r'^\d+\.* ', '', part) for part in response.split('\n') if part]
    
    def generate_seo_hashtags(self, description: str, **kwargs) -> str:
        """生成 SEO 優化的 hashtags"""
        messages = [
            {'role': 'system', 'content': self.prompts['hashtag_system_prompt']},
            {'role': 'user', 'content': description}
        ]
        return self.text_model.chat_completion(messages=messages, **kwargs)

    def generate_arbitrary_input(self, character, extra='', prompt_type='default') -> str:
        """生成 SEO 優化的 hashtags"""
        if prompt_type == 'default':
            prompt_type = 'arbitrary_input_system_prompt'
        messages = [
            {'role': 'system', 'content': self.prompts[prompt_type]},
            {'role': 'assistant', 'content': 'Translation any input into precisely English and only one response without explanation'},
            {'role': 'user', 'content': f"""main character: {character}\n{extra}"""}
        ]
        result=self.text_model.chat_completion(messages=messages)    
        if '</think>' in result: #deepseek r1 will have <think>...</think> format
            result =result.split('</think>')[-1].strip()
        
        return result
        

class VisionManagerBuilder:
    """Vision Manager 建構器"""
    def __init__(self):
        self.vision_model_type = 'ollama'
        self.text_model_type = 'ollama'
        self.vision_config = {'model_name': 'llava:13b', 'temperature': 0.3}
        self.text_config = {'model_name': 'llama3.2', 'temperature': 0.3}
        self.prompts_config = {
            'hashtag_system_prompt': seo_hashtag_prompt,
            'image_system_prompt': stable_diffusion_prompt,
            'describe_image_prompt': describe_image_prompt,
            'text_image_similarity_prompt': text_image_similarity_prompt,
            'arbitrary_input_system_prompt': arbitrary_input_system_prompt,
            'guide_seo_article_system_prompt': guide_seo_article_system_prompt,
            'unbelievable_world_system_prompt': unbelievable_world_system_prompt
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
        self.prompts_config.update(prompts)
        return self
    
    def build(self) -> 'VisionContentManager':
        """建構 VisionContentManager 實例"""
        vision_model_class = ModelRegistry.get_model(self.vision_model_type)
        text_model_class = ModelRegistry.get_model(self.text_model_type)
        
        vision_model = vision_model_class(ModelConfig(**self.vision_config))
        text_model = text_model_class(ModelConfig(**self.text_config))
        
        return VisionContentManager(
            vision_model=vision_model,
            text_model=text_model,
            prompts_config=self.prompts_config
        )


class ModelSwitcher:
    """模型切換器，用於動態更換模型"""
    def __init__(self, vision_manager: 'VisionContentManager'):
        self.manager = vision_manager
    
    def switch_vision_model(self, model_type: str, **config):
        """切換視覺模型"""
        model_class = ModelRegistry.get_model(model_type)
        self.manager.vision_model = model_class(ModelConfig(**config))
    
    def switch_text_model(self, model_type: str, **config):
        """切換文本模型"""
        model_class = ModelRegistry.get_model(model_type)
        self.manager.text_model = model_class(ModelConfig(**config))


# 註冊模型
ModelRegistry.register('ollama', OllamaModel)
ModelRegistry.register('gemini', GeminiModel)