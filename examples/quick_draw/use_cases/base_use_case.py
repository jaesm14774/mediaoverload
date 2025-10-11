"""使用案例基類（範例專用）

提供所有使用案例的共同接口和功能，使用簡化的內容生成服務
"""

import sys
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

# 確保可以導入 mediaoverload 模組
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from lib.media_auto.strategies.base_strategy import GenerationConfig
from lib.repositories.character_repository import CharacterRepository
from lib.media_auto.models.vision.vision_manager import VisionManagerBuilder
from lib.database import db_pool
from examples.simple_content_service import SimpleContentGenerationService


class BaseUseCase(ABC):
    """使用案例基類（範例專用）

    所有使用案例都應該繼承此類，並實現必要的抽象方法。
    
    使用簡化的內容生成服務，跳過耗時的圖文匹配分析和文章生成步驟。
    適合用於：
    - 快速範例展示
    - 需要人工審核的情況
    - 測試和開發
    """

    def __init__(self,
                 workflow_folder: Optional[str] = None,
                 output_folder: Optional[str] = None,
                 env_path: Optional[str] = None):
        """初始化使用案例

        Args:
            workflow_folder: ComfyUI 工作流存放資料夾（預設使用 configs/workflow）
            output_folder: 圖片輸出資料夾（預設使用 output_media）
            env_path: 環境變數檔案路徑（預設使用 media_overload.env）
        """
        # 設定路徑（使用專案相對路徑）
        self.project_root = project_root
        self.workflow_folder = workflow_folder or str(self.project_root / 'configs' / 'workflow')
        self.output_folder = output_folder or str(self.project_root / 'output_media')
        self.env_path = env_path or str(self.project_root / 'media_overload.env')

        # 確保輸出目錄存在
        os.makedirs(self.output_folder, exist_ok=True)

        # 載入環境變數
        print(f"正在載入環境變數: {self.env_path}")
        loaded = load_dotenv(self.env_path)
        print(f"環境變數載入{'成功' if loaded else '失敗'}")

        # 驗證環境變數
        if not os.environ.get('mysql_host'):
            raise EnvironmentError(
                f"環境變數載入失敗！請確認檔案存在: {self.env_path}\n"
                f"當前工作目錄: {os.getcwd()}"
            )

        print(f"MySQL 主機: {os.environ.get('mysql_host')}")

        # 初始化資料庫和服務
        self._init_database()
        self._init_services()

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
        self.cursor = mysql_conn.cursor
        self.conn = mysql_conn.conn

    def _init_services(self):
        """初始化服務層"""
        # 獲取角色資料
        self.character_df = pd.read_sql_query(
            'select * from anime.anime_roles where status=1 and weight > 0',
            self.engine
        )

        # 初始化 character repository
        mysql_conn = db_pool.get_connection('mysql')
        self.character_repository = CharacterRepository(mysql_conn)

        # 初始化 vision manager（使用 OpenRouter 作為預設）
        self.vision_manager = VisionManagerBuilder() \
            .with_vision_model('openrouter') \
            .with_text_model('openrouter') \
            .with_random_models(True) \
            .build()

        # 使用簡化的內容生成服務（跳過分析和文章生成）
        self.content_service = SimpleContentGenerationService(
            character_repository=self.character_repository,
            vision_manager=self.vision_manager
        )
        print("✓ 使用簡化內容生成服務（跳過圖文匹配分析和文章生成）")

    def load_workflow(self, workflow_name: str) -> str:
        """載入工作流路徑

        Args:
            workflow_name: 工作流名稱（不含副檔名）

        Returns:
            完整的工作流路徑
        """
        return f'{self.workflow_folder}/{workflow_name}.json'

    @abstractmethod
    def build_config(self, **kwargs) -> GenerationConfig:
        """建立生成配置

        每個使用案例必須實現此方法來定義其特定的配置。

        Args:
            **kwargs: 使用案例特定的參數

        Returns:
            GenerationConfig 實例
        """
        pass

    def execute(self, **kwargs) -> Dict[str, Any]:
        """執行使用案例

        這是主要的執行方法，會自動處理整個生成流程。

        Args:
            **kwargs: 傳遞給 build_config 的參數

        Returns:
            包含生成結果的字典:
            {
                'descriptions': List[str],      # 生成的描述
                'media_files': List[str],       # 生成的媒體檔案路徑
                'filter_results': List[Dict],   # 空列表（跳過分析）
                'article_content': str          # 空字串（跳過文章生成）
            }
        """
        # 建立配置
        config = self.build_config(**kwargs)

        # 執行生成（使用簡化服務）
        result = self.content_service.generate_content(config)

        return result

    def get_random_character(self, group_name: Optional[str] = None) -> str:
        """從資料庫中獲取隨機角色

        Args:
            group_name: 群組名稱，如果為 None 則從所有角色中選擇

        Returns:
            角色名稱（英文）
        """
        import random

        if group_name:
            df = self.character_df[self.character_df['group_name'] == group_name]
        else:
            df = self.character_df

        if len(df) == 0:
            raise ValueError(f"找不到角色，group_name: {group_name}")

        return random.choice(df['role_name_en'].tolist())

    def get_characters_by_group(self, group_name: str) -> List[str]:
        """獲取指定群組的所有角色

        Args:
            group_name: 群組名稱

        Returns:
            角色名稱列表
        """
        df = self.character_df[self.character_df['group_name'] == group_name]
        return df['role_name_en'].tolist()

    def get_latest_news(self, limit: int = 10000, exclude_categories: Optional[List[str]] = None) -> pd.DataFrame:
        """獲取最新新聞

        Args:
            limit: 返回的新聞數量上限
            exclude_categories: 要排除的新聞類別

        Returns:
            新聞 DataFrame
        """
        import datetime

        # 預設排除的類別
        if exclude_categories is None:
            exclude_categories = ['政治', '兩岸', '美股雷達', 'A股港股', '財經雲', '股市', '台股新聞']

        now_date = datetime.datetime.now().strftime('%Y-%m-%d')

        df = pd.read_sql_query(
            f"select title, keyword, created_at, category from news_ch.news order by id desc limit {limit}",
            self.engine
        )

        # 過濾空白關鍵字
        df = df[df.keyword.map(str.strip) != '']

        # 排除特定類別
        df = df[~df.category.isin(exclude_categories)]

        # 只保留今天的新聞
        df = df[df['created_at'] >= now_date].reset_index(drop=True)

        return df

