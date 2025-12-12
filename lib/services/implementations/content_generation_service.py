"""內容生成服務實現"""
from typing import Dict, Any, List
from lib.services.interfaces.content_generation_service import IContentGenerationService
from lib.media_auto.strategies.base_strategy import GenerationConfig
from utils.logger import setup_logger
import glob


class ContentGenerationService(IContentGenerationService):
    """內容生成服務
    
    協調完整的媒體生成流程：描述生成、媒體創建、品質分析、文章生成。
    """
    
    def __init__(self, character_repository=None, vision_manager=None):
        """初始化內容生成服務
        
        Args:
            character_repository: 角色資料庫存取（可選）
            vision_manager: 視覺模型管理器（可選）
        """
        self.logger = setup_logger(__name__)
        self.strategy = None
        self.character_repository = character_repository
        self.vision_manager = vision_manager

    def generate_content(self, config: GenerationConfig) -> Dict[str, Any]:
        """執行完整內容生成流程
        
        Args:
            config: 生成配置
            
        Returns:
            包含 descriptions, media_files, filter_results, article_content 的字典
        """
        self.logger.info("開始內容生成流程")
        from lib.media_auto.factory.strategy_factory import StrategyFactory
        generation_type = config.get_all_attributes().get('generation_type', 'text2img')
        self.strategy = StrategyFactory.get_strategy(
            generation_type,
            character_repository=self.character_repository,
            vision_manager=self.vision_manager
        )
        strategy_name = getattr(self.strategy, 'name', None) or self.strategy.__class__.__name__
        self.logger.info(f"使用策略: {strategy_name}")
        
        self.strategy.load_config(config)
        
        descriptions = self.generate_descriptions(config)
        
        if not descriptions:
            self.logger.warning("沒有生成任何描述，終止流程")
            return {
                'descriptions': [],
                'media_files': [],
                'filter_results': [],
                'article_content': ''
            }

        media_files = self.generate_media(config)
        
        similarity_threshold = config.get_all_attributes().get('similarity_threshold', 0.7)
        filter_results = self.analyze_media_text_match(media_files, descriptions, similarity_threshold)

        article_content = ''
        if self.strategy.should_generate_article_now():
            article_content = self.generate_article(config, filter_results)

        return {
            'descriptions': descriptions,
            'media_files': media_files,
            'filter_results': filter_results,
            'article_content': article_content
        }

    def generate_descriptions(self, config: GenerationConfig) -> List[str]:
        """生成文字描述
        
        Args:
            config: 生成配置
            
        Returns:
            描述字串列表
        """
        self.strategy.generate_description()
        descriptions = self.strategy.descriptions
        self.logger.info(f"生成 {len(descriptions)} 個描述")
        return descriptions

    def generate_media(self, config: GenerationConfig) -> List[str]:
        """生成媒體檔案
        
        Args:
            config: 生成配置
            
        Returns:
            生成的媒體檔案路徑列表
        """
        self.strategy.generate_media()

        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp']
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.gif', '*.webm']
        media_files = []
        
        for ext in image_extensions + video_extensions:
            media_files.extend(glob.glob(f'{config.output_dir}/{ext}'))
            media_files.extend(glob.glob(f'{config.output_dir}/*/{ext}'))

        self.logger.info(f"生成 {len(media_files)} 個媒體檔案")
        return media_files

    def analyze_media_text_match(self,
                               images: List[str],
                               descriptions: List[str],
                               similarity_threshold: float = 0.9) -> List[Dict[str, Any]]:
        """分析媒體文字匹配度
        
        Args:
            images: 媒體檔案路徑列表（注意：策略會自行收集）
            descriptions: 原始文字描述列表
            similarity_threshold: 相似度閾值（0.0-1.0）
            
        Returns:
            符合閾值的媒體結果列表
        """
        if hasattr(self.strategy, 'descriptions') and descriptions != self.strategy.descriptions:
            self.logger.warning("傳入的 descriptions 與策略中的不一致，使用策略中的 descriptions")
            descriptions = self.strategy.descriptions
        
        self.strategy.analyze_media_text_match(similarity_threshold)

        filter_results = []
        if hasattr(self.strategy, 'filter_results'):
            filter_results = self.strategy.filter_results
        else:
            self.logger.warning("策略沒有 filter_results 屬性")

        self.logger.info(f"篩選出 {len(filter_results)} 個媒體檔案")
        return filter_results

    def generate_article(self, config: GenerationConfig, filter_results: List[Dict[str, Any]]) -> str:
        """生成文章內容
        
        Args:
            config: 生成配置
            filter_results: 篩選後的媒體結果
            
        Returns:
            文章內容字串
        """
        if hasattr(self.strategy, 'filter_results'):
            self.strategy.filter_results = filter_results
        self.strategy.generate_article_content()

        article_content = ""
        if hasattr(self.strategy, 'article_content'):
            article_content = self.strategy.article_content
        if len(article_content) > 4000:
            article_content = '#ai #video #unbelievable #world #humor #interesting #funny #creative'
        return article_content
