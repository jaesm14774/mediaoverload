from typing import Dict, Type, List, Optional
from lib.media_auto.models.interfaces.ai_model import AIModelInterface, ModelConfig
import ollama
from dotenv import load_dotenv
import os
import requests
import json
import random

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


class OpenRouterModel(AIModelInterface):
    """OpenRouter 模型實現 - 支援透過 API 呼叫任何模型"""
    
    # OpenRouter 免費模型列表
    FREE_TEXT_MODELS = [
        "tngtech/deepseek-r1t2-chimera:free",
        "qwen/qwen3-235b-a22b:free"
    ]
    
    FREE_VISION_MODELS = [
        "qwen/qwen2.5-vl-72b-instruct:free",
        "google/gemma-3-27b-it:free"
    ]
    
    def __init__(self, config: ModelConfig):
        self.config = config
        load_dotenv(f'media_overload.env')
        self.api_key = os.environ.get('open_router_token')
        if not self.api_key:
            raise ValueError("open_router_token not found in environment variables")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    @classmethod
    def get_random_free_text_model(self) -> str:
        """隨機選擇一個免費文本模型"""
        return random.choice(self.FREE_TEXT_MODELS)
    
    @classmethod
    def get_random_free_vision_model(self) -> str:
        """隨機選擇一個免費視覺模型"""
        return random.choice(self.FREE_VISION_MODELS)
    
    def chat_completion(self, 
                       messages: List[dict],
                       images: Optional[List[str]] = None,
                       **kwargs) -> str:
        """使用 OpenRouter API 進行聊天完成"""
        # 處理圖片輸入 - 將圖片轉換為 base64
        processed_messages = self._process_messages_with_images(messages, images)
        # 準備 API 請求數據
        data = {
            "model": self.config.model_name,
            "messages": processed_messages,
            "temperature": self.config.temperature
        }
        
        # 添加額外的參數
        for key, value in kwargs.items():
            if key not in ['images']:  # 排除已處理的參數
                data[key] = value
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                data=json.dumps(data),
                timeout=120
            )
            response.raise_for_status()
            
            response_data = response.json()
            return response_data['choices'][0]['message']['content']
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"OpenRouter API 請求失敗: {e}")
        except KeyError as e:
            raise Exception(f"OpenRouter API 回應格式錯誤: {e}")
    
    def _process_messages_with_images(self, messages: List[dict], images: Optional[List[str]] = None) -> List[dict]:
        """處理包含圖片的訊息"""
        processed_messages = []
        
        for message in messages:
            if message.get('role') != 'user':
                processed_messages.append(message)
                continue

            processed_message = message.copy()
            
            # 如果是有圖片，添加圖片內容到user message 最前面
            if images:
                content_parts = []
                # 添加圖片內容
                for image_path in images:
                    try:
                        image_base64 = self._encode_image_to_base64(image_path)
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": image_base64
                            }
                        })
                    except Exception as e:
                        print(f"無法處理圖片 {image_path}: {e}")
                        continue
                
                # 添加文字內容
                if isinstance(message.get('content'), str):
                    content_parts.append({
                        "type": "text",
                        "text": message['content']
                    })
                
                processed_message['content'] = content_parts
            processed_messages.append(processed_message)
        
        return processed_messages
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        """將圖片檔案轉換為 Base64 編碼的字串"""
        import base64
        try:
            # 1. 以「二進位讀取」模式 (rb) 開啟本地圖檔
            with open(image_path, "rb") as image_file:
                # 2. 讀取檔案的全部二進位內容
                binary_data = image_file.read()
                # 3. 進行 Base64 編碼，得到 bytes 型態的結果
                base64_encoded_data = base64.b64encode(binary_data)
                # 4. 將 bytes 解碼成 utf-8 字串
                base64_string = base64_encoded_data.decode('utf-8')
                # 5. 組合成 API 需要的 "Data URL" 格式
                return f"data:image/jpeg;base64,{base64_string}"
        except FileNotFoundError:
            print(f"錯誤：找不到圖片檔案 '{image_path}'")
            return None

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
ModelRegistry.register('openrouter', OpenRouterModel)