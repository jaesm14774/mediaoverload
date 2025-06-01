from typing import Dict, Type, List, Optional
from lib.media_auto.models.interfaces.ai_model import AIModelInterface, ModelConfig
import ollama
from dotenv import load_dotenv
import os

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

# 註冊預設模型
ModelRegistry.register('ollama', OllamaModel) 
ModelRegistry.register('gemini', GeminiModel)