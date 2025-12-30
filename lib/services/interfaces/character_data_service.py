"""角色資料服務介面"""
from typing import List, Optional
from abc import ABC, abstractmethod


class ICharacterDataService(ABC):
    """角色資料服務介面
    
    定義角色資料存取的契約，實現類別負責實際的資料庫互動邏輯。
    """

    @abstractmethod
    def get_characters_by_group(self, group_name: str, workflow_name: str) -> List[str]:
        """取得群組內的角色清單
        
        Args:
            group_name: 角色群組識別符（如 'Kirby', 'Pokemon'）
            workflow_name: 工作流類型識別符
            
        Returns:
            角色名稱字串清單（英文名）
        """
        pass

    @abstractmethod
    def get_random_character_from_group(self, group_name: str, workflow_name: str) -> Optional[str]:
        """從群組中隨機選取一個角色
        
        Args:
            group_name: 角色群組識別符
            workflow_name: 工作流類型識別符
            
        Returns:
            角色名稱字串，若群組為空則返回 None
        """
        pass

    @abstractmethod
    def get_characters_outside_group(self, group_name: str) -> List[str]:
        """取得其他群組的角色清單
        
        Args:
            group_name: 要排除的群組
            
        Returns:
            其他群組的角色英文名清單
        """
        pass

