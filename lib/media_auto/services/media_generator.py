import os
import json
from typing import Dict, List, Any, Optional
from lib.comfyui.websockets_api import ComfyUICommunicator

class MediaGenerator:
    """
    Unified service for generating media (images/videos) using ComfyUI.
    Abstracts away the low-level details of workflow manipulation and WebSocket communication.
    """
    def __init__(self, host: str = None, port: int = None):
        self.communicator = ComfyUICommunicator(host, port)
        self.communicator.connect_websocket()

    def generate(self, 
                 workflow_path: str, 
                 updates: List[Dict[str, Any]], 
                 output_dir: str, 
                 file_prefix: str = "media") -> List[str]:
        """
        Generic generation method.
        
        Args:
            workflow_path: Path to the ComfyUI workflow JSON file.
            updates: List of updates to apply to the workflow.
            output_dir: Directory to save the generated files.
            file_prefix: Prefix for the output filenames.
            
        Returns:
            List of generated file paths.
        """
        if not os.path.exists(workflow_path):
            raise FileNotFoundError(f"Workflow file not found: {workflow_path}")

        with open(workflow_path, "r", encoding='utf-8') as f:
            workflow = json.loads(f.read())

        success, saved_files = self.communicator.process_workflow(
            workflow=workflow,
            updates=updates,
            output_path=output_dir,
            file_name=file_prefix,
            auto_close=False # Keep connection open for performance
        )

        if not success:
            raise RuntimeError(f"Media generation failed for {workflow_path}")

        return saved_files

    def upload_image(self, image_path: str) -> str:
        """Uploads an image to ComfyUI and returns the filename."""
        return self.communicator.upload_image(image_path)
