from typing import Dict, Type, List, Optional
from lib.media_auto.models.interfaces.ai_model import AIModelInterface, ModelConfig
import ollama
from dotenv import load_dotenv
import os
import requests
import json
import random
import time
import logging
from google import genai
from google.genai import types

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
        # 這裡需要設置你的 API 金鑰
        load_dotenv(f'media_overload.env')
        self.client = genai.Client(
            api_key=os.environ['gemini_api_token'],
            http_options=types.HttpOptions(timeout=300000) # timeout is in milliseconds
        )

    def chat_completion(self, 
                       messages: List[dict],
                       images: Optional[List[str]] = None,
                       **kwargs) -> str:
        # 將 role-based messages 轉換為純文本
        combined_prompt = self._combine_messages(messages)
        
        if images:
            # 處理圖片輸入
            image_parts = [self._load_image(img_path) for img_path in images]
            response = self.client.models.generate_content(
                model=self.config.model_name,
                contents=[combined_prompt, *image_parts]
            )
        else:
            response = self.client.models.generate_content(
                model=self.config.model_name,
                contents=combined_prompt
            )
            
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
        "qwen/qwen3-235b-a22b:free",
        "minimax/minimax-m2:free",
        "deepseek/deepseek-chat-v3.1:free",
        "z-ai/glm-4.5-air:free",
        "deepseek/deepseek-r1-0528:free",
        "moonshotai/kimi-k2:free",
        "allenai/olmo-3.1-32b-think:free",
        "xiaomi/mimo-v2-flash:free",
        "mistralai/devstral-2512:free"
    ]
    
    FREE_VISION_MODELS = [
        "nvidia/nemotron-nano-12b-v2-vl:free",
        "qwen/qwen2.5-vl-32b-instruct:free",
        "google/gemma-3-27b-it:free",
        "google/gemini-2.0-flash-exp:free"
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
        self.logger = logging.getLogger(__name__)
    
    @classmethod
    def get_random_free_text_model(self) -> str:
        """隨機選擇一個免費文本模型"""
        selected_model = random.choice(self.FREE_TEXT_MODELS)
        return selected_model
    
    @classmethod
    def get_random_free_vision_model(self) -> str:
        """隨機選擇一個免費視覺模型"""
        selected_model = random.choice(self.FREE_VISION_MODELS)
        return selected_model
    
    def chat_completion(self,
                       messages: List[dict],
                       images: Optional[List[str]] = None,
                       max_retries: int = 10,
                       initial_retry_delay: float = 5,
                       **kwargs) -> str:
        """使用 OpenRouter API 進行聊天完成（含增強的重試機制）

        Args:
            messages: 訊息列表
            images: 圖片列表（可選）
            max_retries: 最大重試次數（預設 10 次）
            initial_retry_delay: 初始重試間隔秒數（預設 5 秒）
            **kwargs: 其他參數
        """
        # 處理圖片輸入 - 將圖片轉換為 base64
        processed_messages = self._process_messages_with_images(messages, images)
        
        # 判斷當前模型是否為免費模型，並準備備用模型列表
        current_model = self.config.model_name
        
        # 根據是否有圖片選擇對應的免費模型列表
        if images:
            available_models = self.FREE_VISION_MODELS.copy()
            is_free_model = current_model in self.FREE_VISION_MODELS
        else:
            available_models = self.FREE_TEXT_MODELS.copy()
            is_free_model = current_model in self.FREE_TEXT_MODELS
        
        # 記錄已嘗試過的模型（只有當當前模型在對應的免費模型列表中時才記錄）
        tried_models = [current_model] if is_free_model else []
        
        # 準備 API 請求數據
        data = {
            "model": current_model,
            "messages": processed_messages,
            "temperature": self.config.temperature
        }

        # 添加額外的參數
        for key, value in kwargs.items():
            if key not in ['images', 'max_retries', 'initial_retry_delay']:  # 排除已處理的參數
                data[key] = value

        retry_delay = initial_retry_delay
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=data,  # 使用 json 參數而不是 data=json.dumps()
                    timeout=(10, 30)  # (連接超時, 讀取超時) 秒 - 縮短讀取超時以加快重試
                )
                response.raise_for_status()

                response_data = response.json()
                if attempt > 0:
                    model_info = f"（使用模型: {data['model']}）" if is_free_model else ""
                    self.logger.info(f"OpenRouter API 請求在第 {attempt + 1} 次嘗試後成功{model_info}")
                return response_data['choices'][0]['message']['content']

            except requests.exceptions.HTTPError as e:
                last_exception = e
                status_code = e.response.status_code if e.response else None
                
                # 判斷是否應該重試
                should_retry = self._should_retry(status_code, attempt, max_retries)
                
                if should_retry:
                    # 如果是免費模型且還有其他模型可選，切換模型
                    if is_free_model and attempt < max_retries - 1:
                        new_model = self._switch_to_another_free_model(
                            available_models, tried_models, current_model
                        )
                        if new_model:
                            data['model'] = new_model
                            tried_models.append(new_model)
                            self.logger.info(
                                f"切換模型: {current_model} -> {new_model} "
                                f"（嘗試 {attempt + 1}/{max_retries}）"
                            )
                            current_model = new_model
                    
                    # 根據狀態碼調整退避策略
                    delay_multiplier = self._get_delay_multiplier(status_code)
                    current_delay = retry_delay * (delay_multiplier ** attempt)
                    
                    # 添加隨機抖動（jitter），避免雷群效應
                    jitter = random.uniform(0.8, 1.2)
                    actual_delay = current_delay * jitter
                    
                    model_info = f"（當前模型: {data['model']}）" if is_free_model else ""
                    self.logger.warning(
                        f"OpenRouter API 請求失敗 (嘗試 {attempt + 1}/{max_retries}): "
                        f"HTTP {status_code} - {str(e)}。{model_info}"
                        f"等待 {actual_delay:.1f} 秒後重試..."
                    )
                    time.sleep(actual_delay)
                else:
                    # 不應該重試的錯誤，直接拋出
                    self.logger.error(f"OpenRouter API 請求失敗: HTTP {status_code} - {str(e)}")
                    raise ValueError(f"OpenRouter API 請求失敗（HTTP {status_code}）: {e}")
                    
            except (requests.exceptions.ConnectionError, 
                    requests.exceptions.Timeout,
                    requests.exceptions.RequestException) as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # 如果是免費模型且還有其他模型可選，切換模型
                    if is_free_model:
                        new_model = self._switch_to_another_free_model(
                            available_models, tried_models, current_model
                        )
                        if new_model:
                            data['model'] = new_model
                            tried_models.append(new_model)
                            self.logger.info(
                                f"切換模型: {current_model} -> {new_model} "
                                f"（嘗試 {attempt + 1}/{max_retries}）"
                            )
                            current_model = new_model
                    
                    # 網絡錯誤使用指數退避
                    current_delay = retry_delay * (1.5 ** attempt)
                    jitter = random.uniform(0.8, 1.2)
                    actual_delay = current_delay * jitter
                    
                    model_info = f"（當前模型: {data['model']}）" if is_free_model else ""
                    self.logger.warning(
                        f"OpenRouter API 網絡錯誤 (嘗試 {attempt + 1}/{max_retries}): {str(e)}。{model_info}"
                        f"等待 {actual_delay:.1f} 秒後重試..."
                    )
                    time.sleep(actual_delay)
                else:
                    self.logger.error(f"OpenRouter API 網絡錯誤（已重試 {max_retries} 次）: {e}")
                    raise ValueError(f"OpenRouter API 網絡錯誤（已重試 {max_retries} 次）: {e}")
                    
            except json.JSONDecodeError as e:
                last_exception = e
                if attempt < max_retries - 1:
                    current_delay = retry_delay * (1.2 ** attempt)
                    self.logger.warning(
                        f"OpenRouter API JSON 解析錯誤 (嘗試 {attempt + 1}/{max_retries}): {str(e)}。"
                        f"等待 {current_delay:.1f} 秒後重試..."
                    )
                    time.sleep(current_delay)
                else:
                    self.logger.error(f"OpenRouter API JSON 解析錯誤（已重試 {max_retries} 次）: {e}")
                    raise RuntimeError(f"OpenRouter API JSON 解析錯誤（已重試 {max_retries} 次）: {e}")
                    
            except KeyError as e:
                # KeyError 通常是回應格式問題，不應該重試
                self.logger.error(f"OpenRouter API 回應格式錯誤: {e}")
                raise ValueError(f"OpenRouter API 回應格式錯誤: {e}")
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # 如果是免費模型且還有其他模型可選，切換模型
                    if is_free_model:
                        new_model = self._switch_to_another_free_model(
                            available_models, tried_models, current_model
                        )
                        if new_model:
                            data['model'] = new_model
                            tried_models.append(new_model)
                            self.logger.info(
                                f"切換模型: {current_model} -> {new_model} "
                                f"（嘗試 {attempt + 1}/{max_retries}）"
                            )
                            current_model = new_model
                    
                    current_delay = retry_delay * (1.3 ** attempt)
                    jitter = random.uniform(0.9, 1.1)
                    actual_delay = current_delay * jitter
                    
                    model_info = f"（當前模型: {data['model']}）" if is_free_model else ""
                    self.logger.warning(
                        f"OpenRouter API 未知錯誤 (嘗試 {attempt + 1}/{max_retries}): {str(e)}。{model_info}"
                        f"等待 {actual_delay:.1f} 秒後重試..."
                    )
                    time.sleep(actual_delay)
                else:
                    self.logger.error(f"OpenRouter API 請求失敗（已重試 {max_retries} 次）: {e}")
                    raise ValueError(f"OpenRouter API 請求失敗（已重試 {max_retries} 次）: {e}")
        
        # 如果所有重試都失敗
        if last_exception:
            self.logger.error(f"OpenRouter API 請求最終失敗（已重試 {max_retries} 次）: {last_exception}")
            raise ValueError(f"OpenRouter API 請求失敗（已重試 {max_retries} 次）: {last_exception}")
    
    def _switch_to_another_free_model(self, 
                                       available_models: List[str], 
                                       tried_models: List[str], 
                                       current_model: str) -> Optional[str]:
        """切換到另一個免費模型
        
        Args:
            available_models: 可用的免費模型列表
            tried_models: 已嘗試過的模型列表
            current_model: 當前使用的模型
            
        Returns:
            新的模型名稱，如果沒有可用模型則返回 None
        """
        # 找出尚未嘗試過的模型
        untried_models = [m for m in available_models if m not in tried_models]
        
        if not untried_models:
            # 如果所有模型都嘗試過了，記錄警告但繼續使用當前模型
            self.logger.warning(
                f"所有免費模型都已嘗試過，繼續使用當前模型: {current_model}"
            )
            return None
        
        # 隨機選擇一個未嘗試過的模型
        new_model = random.choice(untried_models)
        return new_model
    
    def _should_retry(self, status_code: Optional[int], attempt: int, max_retries: int) -> bool:
        """判斷是否應該重試
        
        Args:
            status_code: HTTP 狀態碼
            attempt: 當前嘗試次數
            max_retries: 最大重試次數
            
        Returns:
            是否應該重試
        """
        if attempt >= max_retries - 1:
            return False
        
        if status_code is None:
            return True
        
        # 可重試的狀態碼
        retryable_status_codes = {
            404,  # 暫時的服務不可用
            408,  # 請求超時
            429,  # 速率限制
            500,  # 服務器錯誤
            502,  # 網關錯誤
            503,  # 服務不可用
            504,  # 網關超時
        }
        
        # 不可重試的狀態碼
        non_retryable_status_codes = {
            400,  # 錯誤請求
            401,  # 未授權
            403,  # 禁止訪問
        }
        
        if status_code in retryable_status_codes:
            return True
        elif status_code in non_retryable_status_codes:
            return False
        else:
            # 其他狀態碼，保守地重試
            return True
    
    def _get_delay_multiplier(self, status_code: Optional[int]) -> float:
        """根據 HTTP 狀態碼獲取延遲倍數
        
        Args:
            status_code: HTTP 狀態碼
            
        Returns:
            延遲倍數
        """
        if status_code is None:
            return 1.5
        
        # 對於速率限制（429），使用更長的退避時間
        if status_code == 429:
            return 2.0
        # 對於服務器錯誤（5xx），使用中等退避時間
        elif status_code and 500 <= status_code < 600:
            return 1.8
        # 對於404等暫時錯誤，使用較短的退避時間
        elif status_code == 404:
            return 1.3
        else:
            return 1.5
    
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