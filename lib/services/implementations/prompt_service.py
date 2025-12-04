"""提示詞生成服務實現"""
import os
import random
import re
from typing import Optional
import datetime
from lib.services.interfaces.prompt_service import IPromptService
from lib.repositories.news_repository import INewsRepository
from lib.repositories.character_repository import ICharacterRepository
from lib.media_auto.models.vision.vision_manager import VisionManagerBuilder
from utils.logger import setup_logger


class PromptService(IPromptService):
    """提示詞生成服務實現"""
    
    def __init__(self, news_repository: INewsRepository = None, character_repository: ICharacterRepository = None, vision_manager=None):
        self.news_repository = news_repository
        self.character_repository = character_repository
        self.logger = setup_logger(__name__)
        self.vision_manager = vision_manager
    
    def _get_vision_manager(self, temperature: float = 1.0):
        """獲取或創建 VisionManager"""
        if self.vision_manager is None:
            # 如果沒有從外部傳入，則創建默認的 VisionManager
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
        """生成提示詞"""
        self.logger.info(f'開始為角色生成提示詞: {character}')
        self.logger.info(f'角色生成方式採用: {method}')
        if method == 'arbitrary':
            prompt = self.generate_by_arbitrary(character, temperature)
        elif method == 'news':
            prompt = self.generate_by_news(character, temperature)
        elif method == 'two_character_interaction':
            prompt = self.generate_by_two_character_interaction(character, temperature, extra_info, character_config)
        else:
            raise ValueError(f"未知的生成方法: {method}")
        
        self.logger.info(f'生成的提示詞: {prompt}')
        
        # 處理特殊角色名稱調整（例如：waddledee -> waddle dee）
        prompt = self._process_prompt_adjustments(prompt)
        
        return prompt.lower()
    
    def generate_by_arbitrary(self, character: str, temperature: float = 1.0) -> str:
        """使用默認方法生成提示詞"""
        vision_manager = self._get_vision_manager(temperature)
        
        # 生成第一層提示詞
        prompt = vision_manager.generate_input_prompt(
            character=character
        )
        prompt = prompt.replace("'", '').replace('"', '')
        
        return prompt
    
    def generate_by_news(self, character: str, temperature: float = 1.0) -> str:
        """使用新聞資訊生成提示詞"""
        if self.news_repository is None:
            raise ValueError("新聞資料庫存取層未設定")
        
        # 獲取隨機新聞
        now_date = datetime.datetime.now().strftime('%Y-%m-%d')
        news_info = self.news_repository.get_random_news(now_date)
        
        if news_info is None:
            self.logger.warning("無法獲取新聞資訊，切換到默認方法")
            return self.generate_by_arbitrary(character, temperature)
        
        vision_manager = self._get_vision_manager(temperature)
        
        info = f"""additional reference information : news_title: {news_info['title']} ; news_keyword: {news_info['keyword']}""".strip()
        
        # 生成提示詞
        prompt = vision_manager.generate_input_prompt(
            character=character,
            extra=info,
            prompt_type='fill_missing_details_system_prompt'
        )
        prompt = prompt.replace("'", '').replace('"', '')
        
        return prompt
    
    def generate_by_two_character_interaction(self, character: str, temperature: float = 1.0, extra_info: Optional[str] = None, character_config: Optional[object] = None) -> str:
        """使用雙角色互動生成提示詞"""
        if self.character_repository is None:
            raise ValueError("角色資料庫存取層未設定")
        
        vision_manager = self._get_vision_manager(temperature)
        
        # 從資料庫中獲取另一個角色作為 Secondary Role
        secondary_character = extra_info if extra_info else self._get_random_secondary_character_with_config(character, character_config)
        
        if not secondary_character:
            self.logger.warning("無法獲取 Secondary Role，切換到默認方法")
            return self.generate_by_arbitrary(character, temperature)
        
        self.logger.info(f'雙角色互動：Main Role: {character}, Secondary Role: {secondary_character}')
        
        # 使用雙角色互動系統提示詞生成
        prompt = vision_manager.generate_two_character_interaction_prompt(
            main_character=character,
            secondary_character=secondary_character
        )
        prompt = prompt.replace("'", '').replace('"', '')
        
        return prompt
    
    def _get_random_secondary_character_with_config(self, main_character: str, character_config: Optional[object]) -> Optional[str]:
        """使用角色配置獲取隨機的 Secondary Role"""
        try:
            if character_config:
                # 從角色配置中獲取 group_name 和 workflow
                group_name = getattr(character_config, 'group_name', '')
                workflow_path = getattr(character_config, 'workflow_path', '')
                
                # 從 workflow_path 中提取 workflow 名稱
                workflow_name = os.path.splitext(os.path.basename(workflow_path))[0] if workflow_path else ''
                
                self.logger.info(f"嘗試從群組 '{group_name}' 和工作流 '{workflow_name}' 中獲取角色")
                
                if group_name and workflow_name:
                    # 從資料庫中獲取同群組的角色
                    characters = self.character_repository.get_characters_by_group(group_name, workflow_name)
                    
                    # 過濾掉主角色
                    available_characters = [char for char in characters if char.lower() != main_character.lower()]
                    
                    if available_characters:
                        selected_character = random.choice(available_characters)
                        self.logger.info(f"從資料庫獲取到 Secondary Role: {selected_character}")
                        return selected_character
                    else:
                        self.logger.info(f"群組 '{group_name}' 中沒有其他可用角色")
                else:
                    self.logger.warning(f"角色配置中缺少 group_name 或 workflow_path")
            
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
        # 特例處理 waddledee
        if re.sub(string=prompt.lower(), pattern=r'\s', repl='').find('waddledee') != -1:
            prompt = re.sub(string=prompt, pattern=r'waddledee|Waddledee', repl='waddle dee')
        
        return prompt 