"""彈性生成器

提供簡單直覺的 API 來生成圖片和影片
使用 system_prompt + keywords 的架構：
- system_prompt: 從 configs/prompt/image_system_guide.py 選擇（如 'stable_diffusion_prompt'）
- keywords: 用戶提供的關鍵詞，會被送到 system_prompt 去生成描述
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv
import pandas as pd

# 確保可以導入 mediaoverload 模組
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from lib.media_auto.strategies.base_strategy import GenerationConfig
from lib.repositories.character_repository import CharacterRepository
from lib.media_auto.models.vision.vision_manager import VisionManagerBuilder
from lib.database import db_pool
from examples.simple_content_service import SimpleContentGenerationService
from examples.quick_draw.helpers.config_builder import ConfigBuilder


class FlexibleGenerator:
    """彈性內容生成器

    提供簡單的 API 來生成圖片和影片，無需深入了解內部架構

    使用範例：
        >>> generator = FlexibleGenerator()
        >>>
        >>> # 生成圖片 - keywords 會被送到 system_prompt 去生成描述
        >>> result = generator.generate_images(
        ...     keywords=["cat", "cherry blossoms", "spring"],
        ...     system_prompt="stable_diffusion_prompt",  # 使用標準 SD 提示詞
        ...     character="kirby",
        ...     num_images=4
        ... )
        >>>
        >>> # 生成黑色幽默風格圖片
        >>> result = generator.generate_images(
        ...     keywords="sleeping peacefully",
        ...     system_prompt="black_humor_system_prompt",
        ...     character="kirby",
        ...     num_images=2
        ... )
        >>>
        >>> # 生成影片
        >>> result = generator.generate_videos(
        ...     keywords=["flying", "stars", "night sky"],
        ...     character="kirby",
        ...     num_videos=2
        ... )
        >>>
        >>> # 批次生成
        >>> prompts = [
        ...     {"keywords": ["morning", "sunrise"]},
        ...     {"keywords": ["night", "stars"]}
        ... ]
        >>> results = generator.batch_generate(prompts, media_type="image")
    """

    def __init__(self,
                 workflow_folder: Optional[str] = None,
                 output_folder: Optional[str] = None,
                 env_path: Optional[str] = None,
                 default_image_workflow: str = 'nova-anime-xl',
                 default_video_workflow: str = 'video-workflow',
                 verbose: bool = True):
        """初始化彈性生成器

        Args:
            workflow_folder: ComfyUI 工作流存放資料夾
            output_folder: 輸出資料夾
            env_path: 環境變數檔案路徑
            default_image_workflow: 預設圖片工作流名稱
            default_video_workflow: 預設影片工作流名稱
            verbose: 是否顯示詳細訊息
        """
        self.project_root = project_root
        self.workflow_folder = workflow_folder or str(self.project_root / 'configs' / 'workflow')
        self.output_folder = output_folder or str(self.project_root / 'output_media')
        self.env_path = env_path or str(self.project_root / 'media_overload.env')
        self.default_image_workflow = default_image_workflow
        self.default_video_workflow = default_video_workflow
        self.verbose = verbose

        # 確保輸出目錄存在
        os.makedirs(self.output_folder, exist_ok=True)

        # 初始化
        self._init_environment()
        self._init_database()
        self._init_services()

    def _init_environment(self):
        """載入環境變數"""
        if self.verbose:
            print(f"正在載入環境變數: {self.env_path}")
        loaded = load_dotenv(self.env_path)
        if self.verbose:
            print(f"環境變數載入{'成功' if loaded else '失敗'}")

        if not os.environ.get('mysql_host'):
            raise EnvironmentError(
                f"環境變數載入失敗！請確認檔案存在: {self.env_path}"
            )

    def _init_database(self):
        """初始化資料庫連接"""
        db_pool.initialize('mysql',
                          host=os.environ['mysql_host'],
                          port=int(os.environ['mysql_port']),
                          user=os.environ['mysql_user'],
                          password=os.environ['mysql_password'],
                          db_name=os.environ['mysql_db_name'])

        mysql_conn = db_pool.get_connection('mysql')
        self.engine = mysql_conn.engine

    def _init_services(self):
        """初始化服務層"""
        # 初始化 character repository
        mysql_conn = db_pool.get_connection('mysql')
        self.character_repository = CharacterRepository(mysql_conn)

        # 初始化 vision manager
        self.vision_manager = VisionManagerBuilder() \
            .with_vision_model('openrouter') \
            .with_text_model('openrouter') \
            .with_random_models(True) \
            .build()

        # 使用簡化的內容生成服務
        self.content_service = SimpleContentGenerationService(
            character_repository=self.character_repository,
            vision_manager=self.vision_manager
        )

        if self.verbose:
            print("✓ 服務初始化完成")

    def _load_workflow_path(self, workflow_name: str) -> str:
        """載入工作流完整路徑

        Args:
            workflow_name: 工作流名稱（可含或不含 .json）

        Returns:
            完整的工作流路徑
        """
        if not workflow_name.endswith('.json'):
            workflow_name = f'{workflow_name}.json'
        return f'{self.workflow_folder}/{workflow_name}'

    def generate_images(self,
                       keywords: Union[str, List[str]],
                       system_prompt: str = 'stable_diffusion_prompt',
                       character: Optional[str] = None,
                       secondary_character: Optional[str] = None,
                       style: str = '',
                       num_images: int = 4,
                       workflow: Optional[str] = None,
                       output_subdir: Optional[str] = None,
                       **kwargs) -> Dict[str, Any]:
        """生成圖片

        Args:
            keywords: 關鍵字（字串或列表），會被送到 system_prompt 去生成描述
            system_prompt: 系統提示詞名稱，從 configs/prompt/image_system_guide.py 選擇
                可選值: 'stable_diffusion_prompt', 'black_humor_system_prompt',
                       'buddhist_combined_image_system_prompt', 'cinematic_stable_diffusion_prompt',
                       'two_character_interaction_generate_system_prompt' 等
            character: 主角色名稱（可選）
            secondary_character: 次要角色名稱（可選）
            style: 風格描述（可選）
            num_images: 要生成的圖片數量
            workflow: 工作流名稱（可選，預設使用 default_image_workflow）
            output_subdir: 輸出子目錄（可選）
            **kwargs: 其他傳遞給 ConfigBuilder 的參數

        Returns:
            包含生成結果的字典
        """
        # 轉換 keywords 為字串（如果是列表）
        if isinstance(keywords, list):
            keywords_str = ', '.join(keywords)
        else:
            keywords_str = keywords

        # 確定工作流
        workflow_name = workflow or self.default_image_workflow
        workflow_path = self._load_workflow_path(workflow_name)

        # 確定輸出目錄
        output_dir = self.output_folder
        if output_subdir:
            output_dir = os.path.join(output_dir, output_subdir)
            os.makedirs(output_dir, exist_ok=True)

        # 建立配置
        # prompt 現在是 keywords，會被送到 system_prompt 去生成描述
        builder = ConfigBuilder() \
            .with_workflow(workflow_path) \
            .with_output_dir(output_dir) \
            .with_prompt(keywords_str) \
            .with_generation_type('text2img') \
            .with_images_per_description(num_images) \
            .with_image_system_prompt(system_prompt)

        # 添加可選參數
        if character:
            builder.with_character(character)
        if secondary_character:
            builder.with_secondary_character(secondary_character)
        if style:
            builder.with_style(style)

        # 添加額外參數
        for key, value in kwargs.items():
            if hasattr(builder, f'with_{key}'):
                getattr(builder, f'with_{key}')(value)

        config = builder.build()

        # 執行生成
        if self.verbose:
            print(f"\n🎨 開始生成圖片...")
            print(f"🔖 Keywords: {keywords_str}")
            print(f"📝 System Prompt: {system_prompt}")
            if character:
                print(f"👤 Character: {character}")
            if style:
                print(f"🎭 Style: {style}")
            print(f"📊 數量: {num_images}")

        result = self.content_service.generate_content(config)

        if self.verbose:
            print(f"✅ 完成！生成了 {len(result['media_files'])} 張圖片")
            print(f"📂 保存位置: {output_dir}")

        return result

    def generate_videos(self,
                       keywords: Union[str, List[str]],
                       system_prompt: str = 'stable_diffusion_prompt',
                       character: Optional[str] = None,
                       style: str = '',
                       num_videos: int = 2,
                       workflow: Optional[str] = None,
                       output_subdir: Optional[str] = None,
                       **kwargs) -> Dict[str, Any]:
        """生成影片

        Args:
            keywords: 關鍵字（字串或列表），會被送到 system_prompt 去生成描述
            system_prompt: 系統提示詞名稱（目前影片也使用 image 的 system_prompt）
            character: 角色名稱（可選）
            style: 風格描述（可選）
            num_videos: 要生成的影片數量
            workflow: 工作流名稱（可選，預設使用 default_video_workflow）
            output_subdir: 輸出子目錄（可選）
            **kwargs: 其他傳遞給 ConfigBuilder 的參數

        Returns:
            包含生成結果的字典
        """
        # 轉換 keywords 為字串（如果是列表）
        if isinstance(keywords, list):
            keywords_str = ', '.join(keywords)
        else:
            keywords_str = keywords

        # 確定工作流
        workflow_name = workflow or self.default_video_workflow
        workflow_path = self._load_workflow_path(workflow_name)

        # 確定輸出目錄
        output_dir = self.output_folder
        if output_subdir:
            output_dir = os.path.join(output_dir, output_subdir)
            os.makedirs(output_dir, exist_ok=True)

        # 建立配置
        builder = ConfigBuilder() \
            .with_video_workflow(workflow_path) \
            .with_output_dir(output_dir) \
            .with_prompt(keywords_str) \
            .with_videos_per_description(num_videos) \
            .with_image_system_prompt(system_prompt)

        # 添加可選參數
        if character:
            builder.with_character(character)
        if style:
            builder.with_style(style)

        # 添加額外參數
        for key, value in kwargs.items():
            if hasattr(builder, f'with_{key}'):
                getattr(builder, f'with_{key}')(value)

        config = builder.build()

        # 執行生成
        if self.verbose:
            print(f"\n🎬 開始生成影片...")
            print(f"🔖 Keywords: {keywords_str}")
            print(f"📝 System Prompt: {system_prompt}")
            if character:
                print(f"👤 Character: {character}")
            if style:
                print(f"🎭 Style: {style}")
            print(f"📊 數量: {num_videos}")

        result = self.content_service.generate_content(config)

        if self.verbose:
            print(f"✅ 完成！生成了 {len(result['media_files'])} 個影片")
            print(f"📂 保存位置: {output_dir}")

        return result

    def generate_text2image2video(self,
                                 keywords: Union[str, List[str]],
                                 system_prompt: str = 'stable_diffusion_prompt',
                                 character: Optional[str] = None,
                                 style: str = '',
                                 num_images: int = 1,
                                 num_videos_per_image: int = 1,
                                 t2i_workflow: Optional[str] = None,
                                 i2v_workflow: Optional[str] = None,
                                 output_subdir: Optional[str] = None,
                                 **kwargs) -> Dict[str, Any]:
        """生成 Text2Image2Video (文生圖 -> 圖生影片)
        
        Args:
            keywords: 關鍵字（字串或列表）
            system_prompt: 系統提示詞名稱
            character: 角色名稱
            style: 風格描述
            num_images: 第一階段生成的圖片數量
            num_videos_per_image: 第二階段每張圖片生成的影片數量
            t2i_workflow: 文生圖工作流名稱 (預設使用 default_image_workflow)
            i2v_workflow: 圖生影片工作流名稱 (預設使用 wan2.2_gguf_i2v_audio)
            output_subdir: 輸出子目錄
            **kwargs: 其他參數
            
        Returns:
            包含生成結果的字典
        """
        # 轉換 keywords 為字串
        if isinstance(keywords, list):
            keywords_str = ', '.join(keywords)
        else:
            keywords_str = keywords
            
        # 確定工作流
        t2i_workflow_name = t2i_workflow or self.default_image_workflow
        t2i_workflow_path = self._load_workflow_path(t2i_workflow_name)
        
        i2v_workflow_name = i2v_workflow or 'wan2.2_gguf_i2v_audio'
        i2v_workflow_path = self._load_workflow_path(i2v_workflow_name)
        
        # 確定輸出目錄
        output_dir = self.output_folder
        if output_subdir:
            output_dir = os.path.join(output_dir, output_subdir)
            os.makedirs(output_dir, exist_ok=True)
            
        # 構建策略參數
        additional_params = {
            'strategies': {
                'text2image2video': {
                    'first_stage': {
                        'images_per_description': num_images,
                        't2i_workflow_path': t2i_workflow_path,
                        'style': style,
                        'image_system_prompt': system_prompt
                    },
                    'video': {
                        'videos_per_image': num_videos_per_image,
                        'i2v_workflow_path': i2v_workflow_path
                    }
                }
            }
        }
        
        # 建立配置
        builder = ConfigBuilder() \
            .with_workflow(t2i_workflow_path) \
            .with_output_dir(output_dir) \
            .with_prompt(keywords_str) \
            .with_generation_type('text2image2video') \
            .with_image_system_prompt(system_prompt) \
            .with_additional_params(**additional_params)
            
        if character:
            builder.with_character(character)
            
        # 添加額外參數
        for key, value in kwargs.items():
            if hasattr(builder, f'with_{key}'):
                getattr(builder, f'with_{key}')(value)
                
        config = builder.build()
        
        # 執行生成
        if self.verbose:
            print(f"\n🎬 開始 Text2Image2Video 生成...")
            print(f"🔖 Keywords: {keywords_str}")
            print(f"📝 System Prompt: {system_prompt}")
            print(f"📊 圖片數量: {num_images}, 影片/圖: {num_videos_per_image}")
            
        result = self.content_service.generate_content(config)
        
        if self.verbose:
            print(f"✅ 完成！生成了 {len(result['media_files'])} 個影片")
            print(f"📂 保存位置: {output_dir}/videos")
            
        return result

    def batch_generate(self,
                      prompts: List[Dict[str, Any]],
                      media_type: str = 'image',
                      base_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """批次生成圖片或影片

        Args:
            prompts: 提示詞列表，每個元素為包含 'keywords' 和可選 'system_prompt' 的字典
            media_type: 媒體類型，'image' 或 'video'
            base_config: 基礎配置參數，應用於所有生成

        Returns:
            生成結果列表

        範例:
            >>> prompts = [
            ...     {"keywords": ["morning", "sunrise"]},
            ...     {"keywords": ["night", "moon"], "style": "dark"}
            ... ]
            >>> results = generator.batch_generate(prompts, media_type="image")
        """
        results = []
        base_config = base_config or {}

        if self.verbose:
            print(f"\n📦 批次生成模式")
            print(f"📊 總數: {len(prompts)} 組")
            print(f"🎯 類型: {media_type}")
            print("="*60)

        for i, prompt_config in enumerate(prompts, 1):
            if self.verbose:
                print(f"\n[{i}/{len(prompts)}] 處理中...")

            # 合併基礎配置和當前配置
            config = {**base_config, **prompt_config}
            keywords = config.pop('keywords')  # keywords 是必須的

            # 根據類型生成
            if media_type.lower() == 'image':
                result = self.generate_images(
                    keywords=keywords,
                    output_subdir=f'batch_{i}',
                    **config
                )
            elif media_type.lower() == 'video':
                result = self.generate_videos(
                    keywords=keywords,
                    output_subdir=f'batch_{i}',
                    **config
                )
            else:
                raise ValueError(f"不支援的媒體類型: {media_type}")

            results.append({
                'index': i,
                'keywords': keywords,
                'result': result
            })

        if self.verbose:
            print("\n" + "="*60)
            print(f"✅ 批次生成完成！")
            total_files = sum(len(r['result']['media_files']) for r in results)
            print(f"📊 總共生成: {total_files} 個檔案")

        return results

    def generate_from_config(self, config: GenerationConfig) -> Dict[str, Any]:
        """使用自訂配置生成（進階用法）

        Args:
            config: GenerationConfig 實例

        Returns:
            生成結果
        """
        return self.content_service.generate_content(config)
