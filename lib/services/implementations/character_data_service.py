"""角色資料服務實現"""
import random
from typing import List, Optional
from lib.services.interfaces.character_data_service import ICharacterDataService


class CharacterDataService(ICharacterDataService):
    """MySQL 角色資料服務
    
    從 MySQL 資料庫取得角色資料，包含自動重試邏輯。
    
    資料庫結構：
        anime.anime_roles:
            - role_name_en: 角色英文名
            - group_name: 角色群組識別符
            - status: 啟用標記（1 = 啟用）
            - weight: 選取權重（>0 為可選角色）
    """

    def __init__(self, db_connection):
        """初始化服務
        
        Args:
            db_connection: 資料庫連接物件
        """
        self.db_connection = db_connection

    def get_characters_by_group(self, group_name: str, workflow_name: str) -> List[str]:
        """取得群組內的角色清單（含重試）"""
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                cursor = self.db_connection.cursor
                query = """
                    SELECT role_name_en
                    FROM anime.anime_roles
                    WHERE group_name = %s AND status = 1 AND weight > 0
                """.strip()

                cursor.execute(query, (group_name,))
                results = cursor.fetchall()
                return [row[0] for row in results]

            except Exception:
                retry_count += 1
                if retry_count == max_retries:
                    raise

                from lib.database import db_pool
                self.db_connection = db_pool.get_connection('mysql')

    def get_random_character_from_group(self, group_name: str, workflow_name: str) -> Optional[str]:
        """從群組中隨機選取一個角色"""
        characters = self.get_characters_by_group(group_name, workflow_name)
        if not characters:
            return None
        return random.choice(characters)

    def get_characters_outside_group(self, group_name: str) -> List[str]:
        """取得其他群組的角色清單"""
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                cursor = self.db_connection.cursor
                query = """
                    SELECT role_name_en
                    FROM anime.anime_roles
                    WHERE group_name != %s AND status = 1 AND weight > 0
                """.strip()

                cursor.execute(query, (group_name,))
                results = cursor.fetchall()
                return [row[0] for row in results]

            except Exception:
                retry_count += 1
                if retry_count == max_retries:
                    raise

                from lib.database import db_pool
                self.db_connection = db_pool.get_connection('mysql')

