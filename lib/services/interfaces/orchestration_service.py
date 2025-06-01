"""協調服務介面"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from lib.media_auto.process import BaseCharacter


class IOrchestrationService(ABC):
    """協調服務介面
    
    新的 ContentProcessor，作為協調者負責調用各個服務，組織整個 ETL 流程
    """
    
    @abstractmethod
    async def execute_workflow(self,
                             character: BaseCharacter,
                             prompt: Optional[str] = None,
                             temperature: float = 1.0) -> Dict[str, Any]:
        """執行完整的工作流程
        
        Args:
            character: 角色實例
            prompt: 提示詞（可選）
            temperature: LLM 溫度參數
            
        Returns:
            執行結果字典
        """
        pass
    
    @abstractmethod
    def configure_services(self,
                         prompt_service,
                         content_service,
                         review_service,
                         publishing_service,
                         notification_service) -> None:
        """配置所需的服務
        
        Args:
            prompt_service: 提示詞生成服務
            content_service: 內容生成服務
            review_service: 審核服務
            publishing_service: 發布服務
            notification_service: 通知服務
        """
        pass
    
    @abstractmethod
    def cleanup(self, output_dir: str) -> None:
        """清理資源
        
        Args:
            output_dir: 要清理的輸出目錄
        """
        pass 