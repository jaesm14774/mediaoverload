"""工作流載入器

提供便捷的工作流管理功能
"""

import json
import os
from typing import Dict, Any, List
from pathlib import Path


class WorkflowLoader:
    """工作流載入器

    管理和載入 ComfyUI 工作流
    """

    def __init__(self, workflow_folder: str = None):
        """初始化工作流載入器

        Args:
            workflow_folder: 工作流資料夾路徑（預設使用 configs/workflow）
        """
        if workflow_folder is None:
            # 使用專案相對路徑
            project_root = Path(__file__).parent.parent.parent.parent
            workflow_folder = str(project_root / 'configs' / 'workflow')
        
        self.workflow_folder = workflow_folder

    def load(self, workflow_name: str) -> Dict[str, Any]:
        """載入工作流

        Args:
            workflow_name: 工作流名稱 (不含 .json 副檔名)

        Returns:
            工作流字典

        Raises:
            FileNotFoundError: 找不到工作流檔案
            json.JSONDecodeError: 工作流格式錯誤
        """
        workflow_path = self.get_path(workflow_name)

        if not os.path.exists(workflow_path):
            raise FileNotFoundError(f"找不到工作流: {workflow_path}")

        with open(workflow_path, "r", encoding='utf-8') as f:
            return json.load(f)

    def get_path(self, workflow_name: str) -> str:
        """獲取工作流完整路徑

        Args:
            workflow_name: 工作流名稱 (不含 .json 副檔名)

        Returns:
            完整路徑
        """
        # 移除可能已經存在的 .json 副檔名
        if workflow_name.endswith('.json'):
            workflow_name = workflow_name[:-5]

        return f'{self.workflow_folder}/{workflow_name}.json'

    def list_workflows(self) -> List[str]:
        """列出所有可用的工作流

        Returns:
            工作流名稱列表 (不含副檔名)
        """
        if not os.path.exists(self.workflow_folder):
            return []

        workflows = []
        for filename in os.listdir(self.workflow_folder):
            if filename.endswith('.json'):
                workflows.append(filename[:-5])  # 移除 .json 副檔名

        return sorted(workflows)

    def exists(self, workflow_name: str) -> bool:
        """檢查工作流是否存在

        Args:
            workflow_name: 工作流名稱

        Returns:
            是否存在
        """
        return os.path.exists(self.get_path(workflow_name))

    def validate(self, workflow_name: str) -> tuple[bool, str]:
        """驗證工作流是否有效

        Args:
            workflow_name: 工作流名稱

        Returns:
            (是否有效, 錯誤訊息)
        """
        try:
            workflow = self.load(workflow_name)

            # 基本驗證
            if not isinstance(workflow, dict):
                return False, "工作流必須是字典格式"

            if len(workflow) == 0:
                return False, "工作流不能為空"

            return True, ""

        except FileNotFoundError as e:
            return False, f"找不到檔案: {e}"
        except json.JSONDecodeError as e:
            return False, f"JSON 格式錯誤: {e}"
        except Exception as e:
            return False, f"未知錯誤: {e}"

    @classmethod
    def create_default(cls) -> 'WorkflowLoader':
        """創建使用預設路徑的載入器

        Returns:
            WorkflowLoader 實例
        """
        return cls()

