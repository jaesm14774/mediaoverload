"""提示詞生成服務實現"""
from typing import Optional
from datetime import datetime
from lib.services.interfaces.prompt_service import IPromptService
from lib.repositories.news_repository import INewsRepository
from lib.content_generation.image_content_generator import VisionManagerBuilder
from utils.logger import setup_logger


class PromptService(IPromptService):
    """提示詞生成服務實現"""
    
    def __init__(self, news_repository: INewsRepository = None):
        self.news_repository = news_repository
        self.logger = setup_logger(__name__)
        self.vision_manager = None
    
    def _get_vision_manager(self, temperature: float = 1.0):
        """獲取或創建 VisionManager"""
        if self.vision_manager is None:
            self.vision_manager = VisionManagerBuilder() \
                .with_vision_model('ollama', model_name='llama3.2-vision') \
                .with_text_model('ollama', model_name='qwen3:8b', temperature=temperature) \
                .build()
        return self.vision_manager
    
    def generate_prompt(self, 
                       character: str, 
                       method: str = 'default',
                       temperature: float = 1.0,
                       extra_info: Optional[str] = None) -> str:
        """生成提示詞"""
        self.logger.info(f'開始為角色生成提示詞: {character}')
        self.logger.info(f'角色生成方式採用: {method}')
        
        if method == 'default':
            prompt = self.generate_by_default(character, temperature)
        elif method == 'news':
            prompt = self.generate_by_news(character, temperature)
        else:
            raise ValueError(f"未知的生成方法: {method}")
        
        self.logger.info(f'生成的提示詞: {prompt}')
        return prompt.lower()
    
    def generate_by_default(self, character: str, temperature: float = 1.0) -> str:
        """使用默認方法生成提示詞"""
        vision_manager = self._get_vision_manager(temperature)
        
        # 生成第一層提示詞
        prompt = vision_manager.generate_arbitrary_input(
            character=character,
            extra=f"current time : {datetime.now().strftime('%Y-%m-%d')}",
            prompt_type='default'
        )
        prompt = prompt.replace("'", '').replace('"', '')
        
        # 生成第二層提示詞
        prompt = vision_manager.generate_arbitrary_input(
            character=character,
            extra=f'Act as a Chaos Narrative Forge: roll an imaginary 12-sided die to pick a genre, flip a coin for narrative voice, and draw a tarot card for emotional color. Retain at most **one** micro-detail (≤ 5 words) from "{prompt}" as a hidden Easter egg—do NOT mention which. Everything else must be explosively original: collide clashing eras, graft surreal physics, twist clichés inside-out, and end on an unforeseen pivot. Output 1-3 lush, sensory sentences rich in metaphor and motion, with no meta commentary or game mechanics exposed.'
        )
        return prompt
    
    def generate_by_news(self, character: str, temperature: float = 1.0) -> str:
        """使用新聞資訊生成提示詞"""
        if self.news_repository is None:
            raise ValueError("新聞資料庫存取層未設定")
        
        # 獲取隨機新聞
        now_date = datetime.now().strftime('%Y-%m-%d')
        news_info = self.news_repository.get_random_news(now_date)
        
        if news_info is None:
            self.logger.warning("無法獲取新聞資訊，切換到默認方法")
            return self.generate_by_default(character, temperature)
        
        vision_manager = self._get_vision_manager(temperature)
        
        info = f"""extra_info : {news_info['title']} ; {news_info['keyword']}""".strip()
        
        # 生成提示詞
        prompt = vision_manager.generate_arbitrary_input(
            character=character,
            extra=info,
            prompt_type='default'  # 這裡可能需要根據角色配置調整
        )
        prompt = prompt.replace("'", '').replace('"', '')
        
        return prompt 