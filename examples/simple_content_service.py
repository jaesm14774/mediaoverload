"""簡化的內容生成服務

專門用於範例，跳過耗時的分析和文章生成步驟
"""
from typing import Dict, Any, List
from lib.media_auto.strategies.base_strategy import GenerationConfig
from lib.media_auto.factory.strategy_factory import StrategyFactory
from utils.logger import setup_logger
import glob


class SimpleContentGenerationService:
    """簡化的內容生成服務
    
    只執行描述生成和媒體生成，跳過以下耗時操作：
    - analyze_media_text_match (圖文匹配分析)
    - generate_article (文章內容生成)
    
    適用於快速範例和人工審核的情況
    """
    
    def __init__(self, character_repository=None, vision_manager=None):
        self.logger = setup_logger(__name__)
        self.strategy = None
        self.character_repository = character_repository
        self.vision_manager = vision_manager
    
    def generate_content(self, config: GenerationConfig) -> Dict[str, Any]:
        """生成內容（簡化版）
        
        Args:
            config: 生成配置
            
        Returns:
            包含以下鍵值的字典：
            - descriptions: 生成的描述列表
            - media_files: 生成的媒體文件路徑列表
            - filter_results: 空列表（跳過分析步驟）
            - article_content: 空字串（跳過文章生成）
        """
        self.logger.info("開始簡化內容生成流程（跳過分析和文章生成）")
        
        # 獲取對應的策略
        generation_type = config.get_all_attributes().get('generation_type', 'text2img')
        self.strategy = StrategyFactory.get_strategy(
            generation_type, 
            character_repository=self.character_repository,
            vision_manager=self.vision_manager
        )
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
        
        # 跳過分析和文章生成步驟
        self.logger.info("跳過圖文匹配分析（範例模式）")
        self.logger.info("跳過文章內容生成（範例模式）")
        
        return {
            'descriptions': descriptions,
            'media_files': media_files,
            'filter_results': [],  # 空列表，不進行分析
            'article_content': ''   # 空字串，不生成文章
        }
    
    def generate_descriptions(self, config: GenerationConfig) -> List[str]:
        """生成描述文字"""
        self.logger.info("開始生成描述")
        self.logger.info(f"採用圖片生成策略 : {config.image_system_prompt}")
        self.strategy.generate_description()
        descriptions = self.strategy.descriptions
        self.logger.info(f"描述生成完成，共 {len(descriptions)} 個描述")
        for i, desc in enumerate(descriptions, 1):
            self.logger.info(f"描述 {i}: {desc}")
        return descriptions
    
    def generate_media(self, config: GenerationConfig) -> List[str]:
        """根據描述生成圖片或視頻"""
        generation_type = config.get_all_attributes().get('generation_type', 'text2img')
        
        if generation_type in ['text2video', 't2v']:
            self.logger.info("開始生成視頻")
            self.strategy.generate_media()
            
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

