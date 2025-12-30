"""提示詞生成服務實現"""
import os
import random
import re
from typing import Optional
import datetime
from lib.services.interfaces.prompt_service import IPromptService
from lib.services.interfaces.news_data_service import INewsDataService
from lib.services.interfaces.character_data_service import ICharacterDataService
from lib.media_auto.models.vision.vision_manager import VisionManagerBuilder
from utils.logger import setup_logger


class PromptService(IPromptService):
    """提示詞生成服務實現
    
    支援多種生成方式：任意生成、新聞生成、雙角色互動生成。
    """
    
    def __init__(self, news_data_service: INewsDataService = None, character_data_service: ICharacterDataService = None, vision_manager=None):
        """初始化提示詞服務
        
        Args:
            news_data_service: 新聞資料服務（可選）
            character_data_service: 角色資料服務（可選）
            vision_manager: 視覺模型管理器（可選）
        """
        self.news_data_service = news_data_service
        self.character_data_service = character_data_service
        self.logger = setup_logger(__name__)
        self.vision_manager = vision_manager
    
    def _get_vision_manager(self, temperature: float = 1.0):
        """獲取或創建 VisionManager
        
        Args:
            temperature: 生成溫度
            
        Returns:
            VisionManager 實例
        """
        if self.vision_manager is None:
            self.logger.warning("沒有從外部傳入 VisionManager，使用默認配置創建")
            self.vision_manager = VisionManagerBuilder() \
                .with_vision_model('ollama', model_name='llama3.2-vision') \
                .with_text_model('ollama', model_name='deepseek-r1:8b', temperature=temperature) \
                .build()
        return self.vision_manager
    
    def generate_prompt(self, 
                       character: str, 
                       method: str = 'arbitrary',
                       temperature: float = 1.0,
                       extra_info: Optional[str] = None,
                       character_config: Optional[object] = None) -> str:
        """生成提示詞
        
        Args:
            character: 角色名稱
            method: 生成方法（arbitrary/news/two_character_interaction）
            temperature: 生成溫度
            extra_info: 額外資訊（可選）
            character_config: 角色配置（可選）
            
        Returns:
            生成的提示詞字串
        """
        self.logger.info(f'開始為角色生成提示詞: {character}, 方法: {method}')
        
        if method == 'arbitrary':
            prompt = self.generate_by_arbitrary(character, temperature)
        elif method == 'news':
            prompt = self.generate_by_news(character, temperature)
        elif method == 'two_character_interaction':
            prompt = self.generate_by_two_character_interaction(character, temperature, extra_info, character_config)
        else:
            raise ValueError(f"未知的生成方法: {method}")
        
        prompt = self._process_prompt_adjustments(prompt)
        self.logger.info(f'生成的提示詞: {prompt}')
        
        return prompt.lower()
    
    def generate_by_arbitrary(self, character: str, temperature: float = 1.0) -> str:
        """使用默認方法生成提示詞
        
        Args:
            character: 角色名稱
            temperature: 生成溫度
            
        Returns:
            生成的提示詞字串
        """
        vision_manager = self._get_vision_manager(temperature)
        
        # 生成第一層提示詞
        prompt = vision_manager.generate_input_prompt(
            character=character,
            prompt_type='arbitrary_input_system_prompt'
        )
        prompt = prompt.replace("'", '').replace('"', '')
        
        return prompt
    
    def generate_by_news(self, character: str, temperature: float = 1.0) -> str:
        """使用新聞資訊生成提示詞
        
        Args:
            character: 角色名稱
            temperature: 生成溫度
            
        Returns:
            生成的提示詞字串
        """
        if self.news_data_service is None:
            raise ValueError("新聞資料服務未設定")
        
        now_date = datetime.datetime.now().strftime('%Y-%m-%d')
        news_info = self.news_data_service.get_random_news(now_date)
        
        if news_info is None:
            self.logger.warning("無法獲取新聞資訊，切換到默認方法")
            return self.generate_by_arbitrary(character, temperature)
        
        vision_manager = self._get_vision_manager(temperature)
        
        info = f"""additional reference information : news_title: {news_info['title']} ; news_keyword: {news_info['keyword']}""".strip()
        prompt = vision_manager.generate_input_prompt(
            character=character,
            extra=info,
            prompt_type='fill_missing_details_system_prompt'
        )
        prompt = prompt.replace("'", '').replace('"', '')
        
        return prompt
    
    def generate_by_two_character_interaction(self, character: str, temperature: float = 1.0, extra_info: Optional[str] = None, character_config: Optional[object] = None) -> str:
        """使用雙角色互動生成提示詞
        
        Args:
            character: 主角色名稱
            temperature: 生成溫度
            extra_info: 額外資訊（可選）
            character_config: 角色配置（可選）
            
        Returns:
            生成的提示詞字串
        """
        if self.character_data_service is None:
            raise ValueError("角色資料服務未設定")
        
        vision_manager = self._get_vision_manager(temperature)
        
        secondary_character = extra_info if extra_info else self._get_random_secondary_character_with_config(character, character_config)
        
        if not secondary_character:
            raise ValueError("無法獲取 Secondary Role")
        
        self.logger.info(f'雙角色互動：Main Role: {character}, Secondary Role: {secondary_character}')
        prompt = vision_manager.generate_two_character_interaction_prompt(
            main_character=character,
            secondary_character=secondary_character
        )
        prompt = prompt.replace("'", '').replace('"', '')
        
        return prompt
    
    def _get_random_secondary_character_with_config(
        self, 
        main_character: str, 
        character_config: Optional[object],
        same_group_probability: float = 0.6
    ) -> Optional[str]:
        """使用角色配置獲取隨機的 Secondary Role
        
        Args:
            main_character: 主角色名稱
            character_config: 角色配置
            same_group_probability: 選擇同 group 角色的機率（預設 0.6 = 60%）
        """
        try:
            if character_config:
                group_name = getattr(character_config, 'group_name', '')
                workflow_path = getattr(character_config, 'workflow_path', '')
                workflow_name = os.path.splitext(os.path.basename(workflow_path))[0] if workflow_path else ''
                
                if group_name:
                    use_same_group = random.random() < same_group_probability
                    
                    if use_same_group:
                        characters = self.character_data_service.get_characters_by_group(group_name, workflow_name)
                        self.logger.info(f"選擇同 group ({group_name}) 角色")
                    else:
                        characters = self.character_data_service.get_characters_outside_group(group_name)
                        self.logger.info(f"選擇其他 group 角色 (排除 {group_name})")
                    
                    available_characters = [char for char in characters if char.lower() != main_character.lower()]
                    
                    if available_characters:
                        selected_character = random.choice(available_characters)
                        return selected_character
                    
                    # 如果該選擇沒有可用角色，嘗試另一個選擇
                    if use_same_group:
                        fallback_characters = self.character_data_service.get_characters_outside_group(group_name)
                        self.logger.info("同 group 無可用角色，fallback 到其他 group")
                    else:
                        fallback_characters = self.character_data_service.get_characters_by_group(group_name, workflow_name)
                        self.logger.info("其他 group 無可用角色，fallback 到同 group")
                    
                    available_fallback = [char for char in fallback_characters if char.lower() != main_character.lower()]
                    if available_fallback:
                        return random.choice(available_fallback)
            
            # 如果無法從資料庫獲取，使用預設角色
            default_characters = ["waddledee", "wobbuffet", "pikachu", "mario", "sonic"]
            available_defaults = [char for char in default_characters if char.lower() != main_character.lower()]
            if available_defaults:
                selected_default = random.choice(available_defaults)
                self.logger.info(f"使用預設 Secondary Role: {selected_default}")
                return selected_default
                
        except Exception as e:
            self.logger.error(f"獲取 Secondary Role 時發生錯誤: {e}")
        
        return None
    
    def _process_prompt_adjustments(self, prompt: str) -> str:
        """處理提示詞的特殊調整
        
        例如：將 "waddledee" 調整為 "waddle dee"
        
        Args:
            prompt: 原始提示詞
            
        Returns:
            調整後的提示詞
        """
        if re.sub(string=prompt.lower(), pattern=r'\s', repl='').find('waddledee') != -1:
            prompt = re.sub(string=prompt, pattern=r'waddledee|Waddledee', repl='waddle dee')
        
        return prompt 