"""內容生成服務實現"""
from typing import Dict, Any, List
from lib.services.interfaces.content_generation_service import IContentGenerationService
from lib.media_auto.strategies.base_strategy import GenerationConfig
from lib.media_auto.factory.strategy_factory import StrategyFactory
from utils.logger import setup_logger
import glob


class ContentGenerationService(IContentGenerationService):
    """內容生成服務實現"""
    
    def __init__(self, character_repository=None):
        self.logger = setup_logger(__name__)
        self.strategy = None
        self.character_repository = character_repository
    
    def generate_content(self, config: GenerationConfig) -> Dict[str, Any]:
        """生成內容"""
        self.logger.info("開始內容生成流程")
        
        # 獲取對應的策略
        generation_type = config.get_all_attributes().get('generation_type', 'text2img')
        self.strategy = StrategyFactory.get_strategy(generation_type, character_repository=self.character_repository)
        self.logger.info(f"使用策略: {generation_type}")
        
        # 載入配置
        self.strategy.load_config(config)
        self.logger.info("策略配置載入完成")
        
        # 生成描述
        descriptions = self.generate_descriptions(config)
        
        # 檢查描述是否為空
        if not descriptions:
            self.logger.warning("沒有生成任何描述，終止流程")
            return {
                'descriptions': [],
                'media_files': [],
                'filter_results': [],
                'article_content': ''
            }
        
        # 生成圖片或視頻
        media_files = self.generate_media(config)
        
        # 分析圖文匹配度
        similarity_threshold = config.get_all_attributes().get('similarity_threshold', 0.9)
        filter_results = self.analyze_media_text_match(media_files, descriptions, similarity_threshold)
        
        # 生成文章內容
        article_content = self.generate_article(config, filter_results)
        
        return {
            'descriptions': descriptions,
            'media_files': media_files,
            'filter_results': filter_results,
            'article_content': article_content
        }
    
    def generate_descriptions(self, config: GenerationConfig) -> List[str]:
        """生成描述文字"""
        self.logger.info("開始生成描述")
        self.logger.info(f"採用圖片生成策略 : {config.image_system_prompt}")
        self.strategy.generate_description()
        descriptions = self.strategy.descriptions
        self.logger.info(f"描述生成完成，共 {len(descriptions)} 個描述")
        for desc in descriptions:
            self.logger.info(f"描述: {desc}")
        return descriptions
    
    def generate_media(self, config: GenerationConfig) -> List[str]:
        """根據描述生成圖片或視頻"""
        generation_type = config.get_all_attributes().get('generation_type', 'text2img')
        
        if generation_type in ['text2video', 't2v']:
            self.logger.info("開始生成視頻")
            self.strategy.generate_media()  #enerate_video
            
            # 獲取生成的視頻路徑
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.gif', '*.webm']
            media_files = []
            for ext in video_extensions:
                media_files.extend(glob.glob(f'{config.output_dir}/{ext}'))
            
            self.logger.info(f"視頻生成完成，共生成 {len(media_files)} 個視頻")
            return media_files
        else:
            self.logger.info("開始生成圖片")
            self.strategy.generate_media()
            
            # 獲取生成的圖片路徑
            images = glob.glob(f'{config.output_dir}/*png')
            
            self.logger.info(f"圖片生成完成，共生成 {len(images)} 張圖片")
            return images
    
    def analyze_media_text_match(self, 
                               images: List[str], 
                               descriptions: List[str],
                               similarity_threshold: float = 0.9) -> List[Dict[str, Any]]:
        """分析圖文匹配度"""
        generation_type = self.strategy.config.get_all_attributes().get('generation_type', 'text2img')
        
        if generation_type in ['text2video', 't2v']:
            self.logger.info("開始分析視頻內容")
        else:
            self.logger.info("開始分析圖文匹配度")
            
        self.strategy.analyze_media_text_match(similarity_threshold)
        
        filter_results = []
        if hasattr(self.strategy, 'filter_results'):
            filter_results = self.strategy.filter_results
        
        if generation_type in ['text2video', 't2v']:
            self.logger.info(f"視頻內容分析完成，篩選出 {len(filter_results)} 個視頻")
        else:
            self.logger.info(f"圖文匹配分析完成，篩選出 {len(filter_results)} 張圖片")
        
        return filter_results
    
    def generate_article(self, config: GenerationConfig, filter_results: List[Dict[str, Any]]) -> str:
        """生成文章內容"""
        self.logger.info("開始生成文章內容")
        self.strategy.generate_article_content()
        
        article_content = ""
        if hasattr(self.strategy, 'article_content'):
            article_content = self.strategy.article_content
        
        self.logger.info(f"文章內容生成完成，長度: {len(article_content)}")
        return article_content 