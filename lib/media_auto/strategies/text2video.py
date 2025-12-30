import time
import random
import glob
import os
import numpy as np
from typing import List, Dict, Any, Optional

from lib.media_auto.strategies.base_strategy import ContentStrategy, GenerationConfig
from lib.media_auto.services.media_generator import MediaGenerator
from lib.media_auto.models.vision.vision_manager import VisionManagerBuilder
from lib.comfyui.node_manager import NodeManager

class Text2VideoStrategy(ContentStrategy):
    """
    Text-to-Video generation strategy.
    Refactored to use composition.
    """
    def __init__(self, character_data_service=None, vision_manager=None):
        self.character_data_service = character_data_service
        
        if vision_manager is None:
            vision_manager = VisionManagerBuilder() \
                .with_vision_model('ollama', model_name='qwen2.5vl:7b') \
                .with_text_model('ollama', model_name='gemma3:4b') \
                .build()
        self.vision_manager = vision_manager
        
        self.media_generator = MediaGenerator()
        self.node_manager = NodeManager()
        
        self.config = None
        self.descriptions: List[str] = []
        self.filter_results: List[Dict[str, Any]] = []
        self._reviewed = False

    def load_config(self, config: GenerationConfig):
        self.config = config

    def generate_description(self):
        """Generates video descriptions."""
        start_time = time.time()
        
        # Get strategy config with proper merging
        video_config = self._get_strategy_config('text2video')
        
        # Get style: support weights or single value
        style = self._get_style(video_config)
        # Get image_system_prompt: support weights or single value
        image_system_prompt = self._get_system_prompt(video_config)
        
        prompt = self.config.prompt
        if style:
            prompt = f"{prompt}\nstyle:{style}".strip()
            
        if self.config.character:
            char_lower = self.config.character.lower()
            if char_lower not in prompt.lower() and "main character" not in prompt.lower():
                prompt = f"Main character: {self.config.character}\n{prompt}"
        
        # Two-stage description generation
        # 1. Character description
        character_description = self.vision_manager.generate_image_prompts(
            user_input=self.config.character,
            system_prompt_key=image_system_prompt
        )
        
        # 2. Video description
        video_description = self.vision_manager.generate_video_prompts(
            user_input=character_description,
            system_prompt_key='video_description_system_prompt'
        )
        
        self.descriptions = [video_description]
        
        print(f'Final video descriptions: {self.descriptions}')
        print(f'生成描述花費: {time.time() - start_time}')
        return self
    
    def _get_system_prompt(self, stage_config):
        """覆寫基類方法以使用不同的默認值"""
        return super()._get_system_prompt(stage_config, default='unbelievable_world_system_prompt')

    def generate_media(self):
        """Generates videos."""
        start_time = time.time()
        
        if not self.descriptions:
            print("沒有描述可供生成視頻")
            return self
        
        # Get strategy config with proper merging
        video_config = self._get_strategy_config('text2video')
        
        # Get workflow path: video_config.workflow_path -> config.workflow_path -> default
        workflow_path = video_config.get('workflow_path') or getattr(self.config, 'workflow_path', 'configs/workflow/txt2video.json')
        output_dir = getattr(self.config, 'output_dir', 'output')
        # Get videos_per_description: video_config -> general -> default
        videos_per_desc = video_config.get('videos_per_description', 2)
        
        # Load workflow
        import json
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
            
        for idx, description in enumerate(self.descriptions):
            for i in range(videos_per_desc):
                seed = random.randint(1, 999999999999)
                
                # Default updates for video
                default_updates = [
                    {"node_type": "PrimitiveInt", "inputs": {"value": 512}},
                    {"node_type": "EmptyHunyuanLatentVideo", "inputs": {"length": 97}}
                ]
                
                # Merge additional_params with video_config for node_manager
                merged_params = self._merge_node_manager_params(video_config)
                updates = self.node_manager.generate_updates(
                    workflow=workflow,
                    updates_config=video_config.get('custom_node_updates', default_updates),
                    description=description,
                    seed=seed,
                    **merged_params
                )
                
                self.media_generator.generate(
                    workflow_path=workflow_path,
                    updates=updates,
                    output_dir=output_dir,
                    file_prefix=f"{getattr(self.config, 'character', 'char')}_video_d{idx}_{i}"
                )
                
        print(f'✅ 生成視頻總耗時: {time.time() - start_time:.2f} 秒')
        return self

    def analyze_media_text_match(self, similarity_threshold):
        # Simplified: just return all videos
        output_dir = getattr(self.config, 'output_dir', 'output')
        media_paths = glob.glob(f'{output_dir}/*')
        
        self.filter_results = []
        for path in media_paths:
            if any(path.endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.gif']):
                self.filter_results.append({
                    'media_path': path,
                    'description': self.descriptions[0] if self.descriptions else '',
                    'similarity': 1.0
                })
        return self

    def needs_user_review(self) -> bool:
        """檢查是否需要使用者審核
        
        Text2Video 只需要一次審核（選擇最終要發布的影片）
        """
        if self._reviewed:
            return False
        return hasattr(self, 'filter_results') and len(self.filter_results) > 0
    
    def get_review_items(self, max_items: int = 10) -> List[Dict[str, Any]]:
        """獲取需要審核的項目
        
        返回 filter_results 中的項目，最多 max_items 個
        """
        if hasattr(self, 'filter_results') and self.filter_results:
            return self.filter_results[:max_items]
        return []
    
    def handle_review_result(self, selected_indices: List[int], output_dir: str, selected_paths: List[str] = None) -> bool:
        """處理使用者審核結果
        
        對於 Text2VideoStrategy，審核後標記為已審核，不需要第二次審核
        """
        self._reviewed = True
        return True
    
    def generate_article_content(self):
        # 使用基類的完整實現
        return super().generate_article_content()
