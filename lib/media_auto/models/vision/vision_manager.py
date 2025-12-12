from typing import List, Optional, Dict, Any
import re
import os
import time
from lib.media_auto.models.interfaces.ai_model import AIModelInterface, ModelConfig
from lib.media_auto.models.vision.model_registry import ModelRegistry
from configs.prompt.image_system_guide import *
from configs.prompt.video_system_guide import *
from utils.retry_decorator import vision_api_retry
from utils.logger import setup_logger

class VisionContentManager:
    """處理圖片內容分析與生成的類別"""
    def __init__(self, 
                 vision_model: AIModelInterface,
                 text_model: AIModelInterface,
                 prompts_config: dict):
        self.vision_model = vision_model
        self.text_model = text_model
        self.prompts = prompts_config
    
    @vision_api_retry(max_attempts=5)
    def extract_image_content(self, image_path: str, **kwargs) -> str:
        """分析已有圖片並提取內容描述"""
        print(f"提取圖片內容 {image_path}...")
        messages = [{
            'role': 'user',
            'content': self.prompts['describe_image_prompt']
        }]
        result = self.vision_model.chat_completion(
            messages=messages,
            images=[image_path],
            **kwargs
        )
        print(f"圖片 {image_path} 內容提取成功")
        return result
    
    @vision_api_retry(max_attempts=3)
    def analyze_image_text_similarity(self, 
                                    text: str, 
                                    image_path: str, 
                                    main_character: str = '',
                                    **kwargs) -> str:
        """分析圖片與文本描述的相似度
        
        返回 LLM 的原始響應字符串，需要後續解析為數值。
        響應格式可能多樣，例如："0.85", "相似度: 0.85", "0.85/1.0", "85%" 等。
        """
        print(f"分析圖片 {image_path}...")
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
        
        result = self.vision_model.chat_completion(
            messages=messages,
            images=[image_path],
            **kwargs
        )
        print(f"圖片 {image_path} 分析成功")
        return result
    
    @vision_api_retry(max_attempts=5)
    def generate_image_prompts(self, user_input: str, system_prompt_key: str = 'stable_diffusion_prompt', **kwargs) -> str:
        """根據用戶輸入生成圖片描述提示詞"""
        
        # 驗證 user_input 不為空
        if not user_input or not str(user_input).strip():
            raise ValueError(f"user_input (關鍵詞) 不能為空！當前值: {repr(user_input)}")
        
        user_input = str(user_input).strip()
        
        actual_key_to_use = system_prompt_key
        if system_prompt_key not in self.prompts:
            print(f"Warning: Prompt key '{system_prompt_key}' not found in prompts configuration. Falling back to default 'stable_diffusion_prompt'.")
            actual_key_to_use = 'stable_diffusion_prompt'

        print(f"Using image system prompt key: {actual_key_to_use}")
        print(f"📝 傳遞給 LLM 的 user_input (關鍵詞): {user_input}")
        
        messages = [
            {'role': 'system', 'content': self.prompts[actual_key_to_use]},
            {'role': 'user', 'content': user_input}
        ]
        result = self.text_model.chat_completion(messages=messages, **kwargs)
        if not result or not result.strip():
            raise ValueError("API 返回空結果")
        if '</think>' in result:  # deepseek r1 will have <think>...</think> format
            result = result.split('</think>')[-1].strip()
        return result

    def generate_video_prompts(self, user_input: str, system_prompt_key: str = 'video_description_system_prompt', **kwargs) -> str:
        """根據用戶輸入生成視頻描述提示詞"""
        
        actual_key_to_use = system_prompt_key
        if system_prompt_key not in self.prompts:
            print(f"Warning: Prompt key '{system_prompt_key}' not found in prompts configuration. Falling back to default 'video_description_system_prompt'.")
            actual_key_to_use = 'video_description_system_prompt'

        print(f"Using video system prompt key: {actual_key_to_use}")
        messages = [
            {'role': 'system', 'content': self.prompts[actual_key_to_use]},
            {'role': 'user', 'content': user_input}
        ]
        result = self.text_model.chat_completion(messages=messages, **kwargs)
        if '</think>' in result:  # deepseek r1 will have <think>...</think> format
            result = result.split('</think>')[-1].strip()
        return result
    
    @vision_api_retry(max_attempts=3)
    def generate_audio_description(self, image_path: str, video_description: str = '', **kwargs) -> str:
        """根據圖片和影片描述生成音頻描述
        
        Args:
            image_path: 圖片路徑
            video_description: 影片描述（可選）
            **kwargs: 其他參數
            
        Returns:
            音頻描述（1-3個關鍵字，用逗號分隔）
        """
        print(f"生成音頻描述，圖片: {image_path}, 影片描述: {video_description}")
        
        # 構建輸入內容
        user_content = ""
        if video_description:
            user_content = f"Video description: {video_description}\n\nAnalyze the image and generate audio keywords."
        else:
            user_content = "Analyze the image and generate audio keywords."
        
        messages = [
            {'role': 'system', 'content': self.prompts.get('audio_description_prompt', 'Generate 1-3 English sound keywords based on the image.')},
            {'role': 'user', 'content': user_content}
        ]
        
        result = self.vision_model.chat_completion(
            messages=messages,
            images=[image_path],
            **kwargs
        )
        
        # 清理結果，只保留關鍵字
        result = result.strip()
        # 移除可能的 "Sound:" 等前綴
        if ':' in result:
            result = result.split(':', 1)[-1].strip()
        
        print(f"音頻描述生成成功: {result}")
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
            {'role': 'user', 'content': f"""Central Figure: {character}, central figure's name must be in the final response! {extra}"""}
        ]
        result = self.text_model.chat_completion(messages=messages)    
        if '</think>' in result:  # deepseek r1 will have <think>...</think> format
            result = result.split('</think>')[-1].strip()
        
        messages = [
            {'role': 'system', 'content': f"As an expert editor, distill the text's essence. You must preserve the principal character's name, along with the original style and emotion. Keep the output under 30 words."},
            {'role': 'user', 'content': f"""Central Figure: {character}, central figure's name must be in the final response! {result}"""}
        ]
        result = self.text_model.chat_completion(messages=messages)   

        if '</think>' in result:  # deepseek r1 will have <think>...</think> format
            result = result.split('</think>')[-1].strip()         
        
        return result

    def generate_two_character_interaction_prompt(self, main_character, secondary_character, prompt='', style='', **kwargs) -> str:
        """生成雙角色互動的提示詞"""
        # 構建輸入格式，包含所有必要字段
        user_input = f"""
        Main Role: {main_character}
        Secondary Role: {secondary_character}
        """

        # 只有當 style 不為空時才加上 Style 欄位
        if style and style.strip():
            user_input += f"""
        Style: {style}
        """
        
        # 如果有原始prompt，將其納入輸入
        if prompt and prompt.strip():
            user_input += f"""
            Original Context: {prompt.strip()}
            """
                    
        messages = [
            {'role': 'system', 'content': self.prompts['two_character_interaction_generate_system_prompt']},
            {'role': 'user', 'content': user_input}
        ]
        
        result = self.text_model.chat_completion(messages=messages, **kwargs)
        if '</think>' in result:  # deepseek r1 will have <think>...</think> format
            result = result.split('</think>')[-1].strip()
        
        return result

    def analyze_media_text_match(self, 
                               media_paths: List[str],
                               descriptions: List[str],
                               main_character: str,
                               similarity_threshold: float = 0.9,
                               **kwargs) -> List[Dict[str, Any]]:
        """分析圖文匹配度並過濾結果
        
        Args:
            media_paths: 圖片路徑列表
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
        
        logger = setup_logger('mediaoverload')
        
        for media_path in media_paths:
            print(f'進行文圖匹配程度判斷中 : {media_path}\n')
            logger.debug(f'分析文件: {media_path}')

            # 嘗試匹配第一階段的文件名格式: {anything}_d{idx}_{i}
            # 使用更靈活的模式，不依賴角色名稱的精確匹配
            match = re.search(r'_d(\d+)_\d+\.', media_path, re.IGNORECASE)
            
            # 如果第一階段匹配失敗，嘗試匹配第二階段的文件名格式: {anything}_i2i_{idx}_{i}
            if not match:
                match = re.search(r'_i2i_(\d+)_\d+\.', media_path, re.IGNORECASE)
            
            # 如果都匹配失敗，嘗試匹配影片格式: {anything}_i2v_{idx}_{i}
            if not match:
                match = re.search(r'_i2v_(\d+)_\d+\.', media_path, re.IGNORECASE)
            
            # 如果都匹配失敗，嘗試匹配影片格式: {anything}_video_d{idx}_{i}
            if not match:
                match = re.search(r'_video_d(\d+)_\d+\.', media_path, re.IGNORECASE)
            
            # 如果都匹配失敗，嘗試匹配 sticker 格式: {anything}_sticker_{idx}_{i}
            if not match:
                match = re.search(r'_sticker_(\d+)_\d+\.', media_path, re.IGNORECASE)
            
            # 如果都匹配失敗，跳過這個文件
            if not match:
                logger.warning(f'⚠️  警告：無法從文件名解析描述索引: {media_path}')
                print(f'⚠️  警告：無法從文件名解析描述索引: {media_path}')
                continue

            desc_index = int(match.group(1))

            # 確保索引在有效範圍內
            # 如果索引超出範圍，使用第一個描述（適用於單一描述對應多張圖片的情況）
            if desc_index >= len(descriptions):
                logger.debug(f'描述索引 {desc_index} 超出範圍（共有 {len(descriptions)} 個描述），使用第一個描述')
                desc_index = 0

            similarity_raw = self.analyze_image_text_similarity(
                text=descriptions[desc_index],
                image_path=media_path,
                main_character=main_character,
                **kwargs
            )
            
            total_results.append({
                'media_path': media_path,
                'description': descriptions[desc_index],
                'similarity': similarity_raw  # 原始字符串響應
            })
            time.sleep(3)  # google free tier rate limit
        
        logger = setup_logger('mediaoverload')
        logger.info(f'開始解析 {len(total_results)} 個相似度分析結果')
        
        # 過濾結果
        filter_results = []
        for row in total_results:
            try:
                similarity_str = str(row['similarity']).strip()
                logger.debug(f'原始相似度響應: {similarity_str[:100]}')
                
                # 從字符串中提取數字（處理 LLM 可能返回的各種格式）
                # 例如："0.85", "相似度: 0.85", "0.85分", "0.85/1.0", "0.85/1", "85%" 等
                # 優先嘗試提取 0-1 之間的小數
                similarity_value = None
                
                # 策略1: 直接匹配 0-1 之間的小數（包括 0.0, 1.0, 0, 1）
                match = re.search(r'\b(0(?:\.\d+)?|1(?:\.0+)?|0\.\d+|1\.0)\b', similarity_str)
                if match:
                    similarity_value = float(match.group())
                else:
                    # 策略2: 匹配百分比格式（如 "85%" -> 0.85）
                    percent_match = re.search(r'(\d+(?:\.\d+)?)\s*%', similarity_str, re.IGNORECASE)
                    if percent_match:
                        similarity_value = float(percent_match.group(1)) / 100.0
                    else:
                        # 策略3: 匹配分數格式（如 "0.85/1.0" 或 "85/100"）
                        fraction_match = re.search(r'(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)', similarity_str)
                        if fraction_match:
                            numerator = float(fraction_match.group(1))
                            denominator = float(fraction_match.group(2))
                            if denominator > 0:
                                similarity_value = numerator / denominator
                        else:
                            # 策略4: 提取任何數字並判斷是否在合理範圍內
                            number_match = re.search(r'(\d+(?:\.\d+)?)', similarity_str)
                            if number_match:
                                num = float(number_match.group(1))
                                # 如果數字在 0-100 範圍內，可能是百分比
                                if 0 <= num <= 100:
                                    similarity_value = num / 100.0
                                # 如果數字在 0-1 範圍內，直接使用
                                elif 0 <= num <= 1:
                                    similarity_value = num
                
                if similarity_value is not None:
                    # 確保分數在 0-1 範圍內
                    similarity_value = max(0.0, min(1.0, similarity_value))
                    
                    # 更新 row 中的 similarity 為數字
                    row['similarity'] = similarity_value
                    
                    logger.info(f'圖片 {os.path.basename(row["media_path"])} 相似度: {similarity_value:.3f} (閾值: {similarity_threshold:.3f})')
                    
                    if similarity_value >= similarity_threshold:
                        filter_results.append(row)
                        logger.info(f'  ✅ 通過篩選')
                    else:
                        logger.info(f'  ❌ 未通過篩選（低於閾值 {similarity_threshold:.3f}）')
                else:
                    logger.warning(f'⚠️  無法從響應中提取相似度分數: {similarity_str[:100]}...')
                    # 如果無法解析，記錄原始響應以便調試
                    row['similarity_raw'] = similarity_str
                    row['similarity'] = None
                    continue
            except Exception as e:
                logger.error(f'⚠️  解析相似度分數時發生錯誤: {e}, 響應: {str(row.get("similarity", ""))[:100]}...')
                import traceback
                logger.error(traceback.format_exc())
                continue
                
        logger.info(f'分析圖文匹配程度完成，共分析 {len(total_results)} 張圖片，篩選出 {len(filter_results)} 張（閾值: {similarity_threshold:.3f}）')
        logger.info(f'分析圖文匹配程度 花費 : {time.time() - start_time:.2f} 秒')
        
        # 如果沒有通過篩選的結果，記錄所有相似度分數以便調試
        if len(filter_results) == 0 and len(total_results) > 0:
            logger.warning('⚠️  沒有任何圖片通過相似度篩選，以下是所有相似度分數：')
            for row in total_results:
                similarity_val = row.get('similarity')
                if isinstance(similarity_val, (int, float)):
                    logger.warning(f'  {os.path.basename(row["media_path"])}: {similarity_val:.3f}')
                else:
                    logger.warning(f'  {os.path.basename(row["media_path"])}: 無法解析 ({str(similarity_val)[:50]})')
        
        return filter_results

class VisionManagerBuilder:
    """Vision Manager 建構器"""
    def __init__(self):
        self.vision_model_type = 'ollama'
        self.text_model_type = 'ollama'
        self.vision_config = {'model_name': 'llava:13b', 'temperature': 0.3}
        self.text_config = {'model_name': 'llama3.2', 'temperature': 0.3}
        self.use_random_models = False  # 新增：是否使用隨機模型選擇
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
            'two_character_interaction_generate_system_prompt': two_character_interaction_generate_system_prompt,
            'black_humor_system_prompt': black_humor_system_prompt,
            'video_description_system_prompt': video_description_system_prompt,
            'sticker_prompt_system_prompt': sticker_prompt_system_prompt,
            'warm_scene_description_system_prompt': warm_scene_description_system_prompt,
            'cinematic_stable_diffusion_prompt': cinematic_stable_diffusion_prompt,
            'conceptual_logo_design_prompt': conceptual_logo_design_prompt,
            'audio_description_prompt': audio_description_prompt,
            'sticker_motion_system_prompt': sticker_motion_system_prompt

        }
    
    def with_vision_model(self, model_type: str, **config):
        """設置視覺模型"""
        self.vision_model_type = model_type
        # 如果切換了模型類型，重置配置為該類型的預設值
        if model_type == 'gemini':
            self.vision_config = {'model_name': 'gemini-flash-lite-latest', 'temperature': 0.3}
        elif model_type == 'ollama':
            self.vision_config = {'model_name': 'llava:13b', 'temperature': 0.3}
        elif model_type == 'openrouter':
            self.vision_config = {'temperature': 0.3}  # model_name 將由隨機選擇或明確指定
        else:
            # 未知類型，重置為基本配置
            self.vision_config = {'temperature': 0.3}
        
        # 更新提供的配置
        if config:
            self.vision_config.update(config)
        return self
    
    def with_text_model(self, model_type: str, **config):
        """設置文本模型"""
        self.text_model_type = model_type
        # 如果切換了模型類型，重置配置為該類型的預設值
        if model_type == 'gemini':
            self.text_config = {'model_name': 'gemini-flash-lite-latest', 'temperature': 0.3}
        elif model_type == 'ollama':
            self.text_config = {'model_name': 'llama3.2:latest', 'temperature': 0.3}
        elif model_type == 'openrouter':
            self.text_config = {'temperature': 0.3}  # model_name 將由隨機選擇或明確指定
        else:
            # 未知類型，重置為基本配置
            self.text_config = {'temperature': 0.3}
        
        # 更新提供的配置
        if config:
            self.text_config.update(config)
        return self
    
    def with_random_models(self, enabled: bool = True):
        """啟用隨機模型選擇 (僅適用於 OpenRouter)"""
        self.use_random_models = enabled
        return self
    
    def build(self) -> 'VisionContentManager':
        """建構 VisionContentManager 實例"""
        vision_model_class = ModelRegistry.get_model(self.vision_model_type)
        text_model_class = ModelRegistry.get_model(self.text_model_type)
        
        # 如果啟用隨機模型選擇且使用 OpenRouter，則隨機選擇模型
        vision_config = self.vision_config.copy()
        text_config = self.text_config.copy()
        
        logger = setup_logger('mediaoverload')
        
        if self.use_random_models and self.vision_model_type == 'openrouter':
            from lib.media_auto.models.vision.model_registry import OpenRouterModel
            vision_config['model_name'] = OpenRouterModel.get_random_free_vision_model()
            logger.info(f"隨機選擇的 Vision 模型: {vision_config['model_name']}")
            
        if self.use_random_models and self.text_model_type == 'openrouter':
            from lib.media_auto.models.vision.model_registry import OpenRouterModel
            text_config['model_name'] = OpenRouterModel.get_random_free_text_model()
            logger.info(f"隨機選擇的 Text 模型: {text_config['model_name']}")
        
        vision_model = vision_model_class(ModelConfig(**vision_config))
        text_model = text_model_class(ModelConfig(**text_config))
        
        return VisionContentManager(
            vision_model=vision_model,
            text_model=text_model,
            prompts_config=self.prompts_config
        ) 