import os
import json
from typing import Dict, List, Any, Optional
from lib.comfyui.websockets_api import ComfyUICommunicator

class MediaGenerator:
    """媒體生成服務"""
    def __init__(self, host: str = None, port: int = None):
        self.communicator = ComfyUICommunicator(host, port)
        self.communicator.connect_websocket()

    def generate(self, 
                 workflow_path: str, 
                 updates: List[Dict[str, Any]], 
                 output_dir: str, 
                 file_prefix: str = "media") -> List[str]:
        """生成媒體"""
        if not os.path.exists(workflow_path):
            raise FileNotFoundError(f"Workflow file not found: {workflow_path}")

        with open(workflow_path, "r", encoding='utf-8') as f:
            workflow = json.loads(f.read())

        success, saved_files = self.communicator.process_workflow(
            workflow=workflow,
            updates=updates,
            output_path=output_dir,
            file_name=file_prefix,
            auto_close=False
        )

        if not success:
            error_msg = saved_files[0] if isinstance(saved_files, list) and saved_files and isinstance(saved_files[0], str) else "Unknown error"
            raise RuntimeError(f"Media generation failed for {workflow_path}: {error_msg}")

        return saved_files

    def upload_image(self, image_path: str) -> str:
        """上傳圖片到 ComfyUI"""
        return self.communicator.upload_image(image_path)
