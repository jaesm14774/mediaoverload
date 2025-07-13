"""角色資料庫存取層"""
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod


class ICharacterRepository(ABC):
    """角色資料庫存取介面"""
    
    @abstractmethod
    def get_characters_by_group(self, group_name: str, workflow_name: str) -> List[str]:
        """根據群組獲取角色列表
        
        Args:
            group_name: 群組名稱
            workflow_name: 工作流名稱
            
        Returns:
            角色名稱列表
        """
        pass
    
    @abstractmethod
    def get_random_character_from_group(self, group_name: str, workflow_name: str) -> Optional[str]:
        """從群組中隨機獲取一個角色
        
        Args:
            group_name: 群組名稱
            workflow_name: 工作流名稱
            
        Returns:
            隨機選擇的角色名稱，如果沒有則返回 None
        """
        pass


class CharacterRepository(ICharacterRepository):
    """角色資料庫存取實現"""
    
    def __init__(self, db_connection):
        self.db_connection = db_connection
    
    def get_characters_by_group(self, group_name: str, workflow_name: str) -> List[str]:
        """根據群組獲取角色列表"""
        max_retries = 3
        retry_count = 0
        json_workflow_name = f'"{workflow_name}"'
        while retry_count < max_retries:
            try:
                cursor = self.db_connection.cursor
                query = """
                    SELECT role_name_en 
                    FROM anime.anime_roles
                    WHERE group_name = %s AND status = 1 AND json_contains(workflow_list, %s) AND weight > 0
                """.strip()
                
                cursor.execute(query, (group_name, json_workflow_name))
                results = cursor.fetchall()
                return [row[0] for row in results]
                
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    raise
                
                # 重新獲取連接
                from lib.database import db_pool
                self.db_connection = db_pool.get_connection('mysql')
                continue
    
    def get_random_character_from_group(self, group_name: str, workflow_name: str) -> Optional[str]:
        """從群組中隨機獲取一個角色"""
        import random
        characters = self.get_characters_by_group(group_name, workflow_name)
        if not characters:
            return None
        return random.choice(characters) 