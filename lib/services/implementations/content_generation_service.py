"""Content generation service implementation.

This module orchestrates the full media generation pipeline from description generation
through media creation to quality analysis. Supports multiple generation types including
text-to-image, image-to-image, and text-to-video workflows.
"""
from typing import Dict, Any, List
from lib.services.interfaces.content_generation_service import IContentGenerationService
from lib.media_auto.strategies.base_strategy import GenerationConfig
from lib.media_auto.factory.strategy_factory import StrategyFactory
from utils.logger import setup_logger
import glob


class ContentGenerationService(IContentGenerationService):
    """Media generation pipeline orchestrator.

    Coordinates the complete content generation workflow:
    1. Generate text descriptions
    2. Create media (images/videos) from descriptions
    3. Analyze media-text similarity
    4. Generate article content for social media

    Different generation strategies handle specifics for each generation type
    (text2img, image2image, text2video).

    Attributes:
        logger: Logger instance for tracking generation progress
        strategy: Current generation strategy instance
        character_repository: Optional character database access
        vision_manager: Optional vision model manager for similarity analysis
    """

    def __init__(self, character_repository=None, vision_manager=None):
        """Initialize content generation service.

        Args:
            character_repository: Character database access (optional)
            vision_manager: Vision model manager for image analysis (optional)
        """
        self.logger = setup_logger(__name__)
        self.strategy = None
        self.character_repository = character_repository
        self.vision_manager = vision_manager

    def generate_content(self, config: GenerationConfig) -> Dict[str, Any]:
        """Run complete content generation pipeline.

        Executes full workflow from description generation to article creation.
        Returns all intermediate and final outputs.

        Pipeline steps:
        1. Load appropriate strategy for generation type
        2. Generate text descriptions
        3. Create media files from descriptions
        4. Analyze media-text matching quality
        5. Generate article content for posting

        Args:
            config: Generation configuration containing all parameters

        Returns:
            Dictionary with keys:
                - descriptions: List of generated text descriptions
                - media_files: List of generated media file paths
                - filter_results: List of quality-filtered media with scores
                - article_content: Social media caption text
        """
        self.logger.info("開始內容生成流程")

        # Select and load generation strategy
        generation_type = config.get_all_attributes().get('generation_type', 'text2img')
        self.strategy = StrategyFactory.get_strategy(
            generation_type,
            character_repository=self.character_repository,
            vision_manager=self.vision_manager
        )
        strategy_name = getattr(self.strategy, 'name', None) or self.strategy.__class__.__name__
        self.logger.info(f"使用策略: {strategy_name}")

        # Load configuration into strategy
        self.strategy.load_config(config)
        self.logger.info("策略配置載入完成")

        # Generate text descriptions
        descriptions = self.generate_descriptions(config)

        # Early exit if no descriptions generated
        if not descriptions:
            self.logger.warning("沒有生成任何描述，終止流程")
            return {
                'descriptions': [],
                'media_files': [],
                'filter_results': [],
                'article_content': ''
            }

        # Generate media files
        media_files = self.generate_media(config)

        # Analyze and filter by quality
        similarity_threshold = config.get_all_attributes().get('similarity_threshold', 0.9)
        filter_results = self.analyze_media_text_match(media_files, descriptions, similarity_threshold)

        # Generate social media caption（如果策略允許）
        article_content = ''
        if self.strategy.should_generate_article_now():
            article_content = self.generate_article(config, filter_results)
        else:
            self.logger.info("策略延遲生成文章內容，將在後續階段生成")

        return {
            'descriptions': descriptions,
            'media_files': media_files,
            'filter_results': filter_results,
            'article_content': article_content
        }

    def generate_descriptions(self, config: GenerationConfig) -> List[str]:
        """Generate text descriptions for media.

        Creates text prompts that will be used to generate images or videos.
        Number of descriptions depends on strategy and configuration.

        Args:
            config: Generation configuration

        Returns:
            List of description strings
        """
        self.logger.info("開始生成描述")
        self.logger.info(f"採用圖片生成策略 : {config.image_system_prompt}")
        self.logger.info(f"採用風格 : {config.style}")
        self.strategy.generate_description()
        descriptions = self.strategy.descriptions
        self.logger.info(f"描述生成完成，共 {len(descriptions)} 個描述")
        for desc in descriptions:
            self.logger.info(f"描述: {desc}")
        return descriptions

    def generate_media(self, config: GenerationConfig) -> List[str]:
        """Create media files from descriptions.

        Generates images or videos based on configuration. Media type determined
        by the strategy implementation.

        Args:
            config: Generation configuration

        Returns:
            List of generated media file paths (PNG for images, MP4/GIF for videos)
        """
        self.logger.info("開始生成媒體")
        self.strategy.generate_media()

        # Collect media files (both images and videos)
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp']
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.gif', '*.webm']
        media_files = []
        
        for ext in image_extensions + video_extensions:
            media_files.extend(glob.glob(f'{config.output_dir}/{ext}'))
            # Also check subdirectories (e.g., first_stage, videos)
            media_files.extend(glob.glob(f'{config.output_dir}/*/{ext}'))

        self.logger.info(f"媒體生成完成，共生成 {len(media_files)} 個媒體文件")
        return media_files

    def analyze_media_text_match(self,
                               images: List[str],
                               descriptions: List[str],
                               similarity_threshold: float = 0.9) -> List[Dict[str, Any]]:
        """Analyze media-text matching quality.

        Compares generated media against original descriptions using vision models.
        Filters out low-quality results below similarity threshold.

        Args:
            images: List of media file paths
            descriptions: List of original text descriptions
            similarity_threshold: Minimum similarity score (0.0-1.0) to keep media

        Returns:
            List of dictionaries with media paths and similarity scores.
            Only includes media meeting the similarity threshold.
        """
        self.logger.info("開始分析媒體內容匹配度")
        self.strategy.analyze_media_text_match(similarity_threshold)

        filter_results = []
        if hasattr(self.strategy, 'filter_results'):
            filter_results = self.strategy.filter_results

        self.logger.info(f"媒體內容分析完成，篩選出 {len(filter_results)} 個媒體文件")

        return filter_results

    def generate_article(self, config: GenerationConfig, filter_results: List[Dict[str, Any]]) -> str:
        """Generate social media article content.

        Creates caption text for social media posts. Content includes descriptions,
        hashtags, and other metadata formatted for platforms like Instagram.

        Args:
            config: Generation configuration
            filter_results: Filtered media results with quality scores

        Returns:
            Article content string ready for social media posting.
            Truncates to fallback hashtags if exceeds 4000 characters.
        """
        self.logger.info("開始生成文章內容")
        # 確保策略的 filter_results 已設置為最新的結果
        if hasattr(self.strategy, 'filter_results'):
            self.strategy.filter_results = filter_results
        self.strategy.generate_article_content()

        article_content = ""
        if hasattr(self.strategy, 'article_content'):
            article_content = self.strategy.article_content
        if len(article_content) > 4000:
            article_content = '#ai #video #unbelievable #world #humor #interesting #funny #creative'
        self.logger.info(f"文章內容生成完成，長度: {len(article_content)}")
        return article_content
