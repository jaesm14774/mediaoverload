import time
import random
import glob
import os
import numpy as np
from typing import Dict, Any, List, Optional

from lib.media_auto.strategies.base_strategy import ContentStrategy, GenerationConfig
from lib.media_auto.services.media_generator import MediaGenerator
from lib.media_auto.models.vision.vision_manager import VisionManagerBuilder
from lib.comfyui.node_manager import NodeManager

class Text2ImageStrategy(ContentStrategy):
    """
    Text-to-Image generation strategy.
    Refactored to use composition.
    """

    def __init__(self, character_data_service=None, vision_manager=None):
        self.character_data_service = character_data_service
        
        if vision_manager is None:
            # Default to Gemini as per original code
            vision_manager = VisionManagerBuilder() \
                .with_vision_model('gemini', model_name='gemini-flash-lite-latest') \
                .with_text_model('gemini', model_name='gemini-flash-lite-latest') \
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
        """Generates image descriptions/prompts."""
        start_time = time.time()
        
        # Get strategy config with proper merging
        image_config = self._get_strategy_config('text2img')
        
        # Get style: support weights or single value
        style = self._get_style(image_config)
        image_system_prompt = self._get_system_prompt(image_config)
        
        # 獲取 prompt（關鍵詞）
        prompt = getattr(self.config, 'prompt', None)
        
        # 如果 prompt 為 None，嘗試從 _attributes 獲取
        if prompt is None and hasattr(self.config, '_attributes'):
            prompt = self.config._attributes.get('prompt')
        
        # 驗證 prompt 不為空
        if not prompt or not str(prompt).strip():
            error_msg = f"prompt (關鍵詞) 不能為空！請確保在 GenerationConfig 中設置了 prompt 屬性。"
            error_msg += f"\n當前 config.prompt = {repr(prompt)}"
            if hasattr(self.config, '_attributes'):
                error_msg += f"\nconfig._attributes = {self.config._attributes}"
            raise ValueError(error_msg)
        
        prompt = str(prompt).strip()
        print(f'📝 使用的 prompt (關鍵詞): {prompt}')
        
        if style and style.strip():
            prompt = f"{prompt}\nstyle: {style}".strip()
            
        # Add character info if needed
        if self.config.character:
            char_lower = self.config.character.lower()
            if prompt and char_lower not in prompt.lower() and "main character" not in prompt.lower():
                prompt = f"Main character: {self.config.character}\n{prompt}"

        try:
            # 檢查是否使用雙角色互動系統提示詞
            if image_system_prompt == 'two_character_interaction_generate_system_prompt':
                descriptions = self._generate_two_character_interaction_description(prompt, style)
            else:
                descriptions = self.vision_manager.generate_image_prompts(prompt, image_system_prompt)
            
            # 確保 descriptions 不為空
            if not descriptions or not descriptions.strip():
                print('⚠️  警告：API 返回空描述，使用原始 prompt 作為回退')
                descriptions = prompt
        except Exception as e:
            print(f'⚠️  警告：生成描述時發生錯誤: {e}')
            print(f'   使用原始 prompt 作為回退: {prompt}')
            descriptions = prompt
            
        # Filter descriptions based on character name (simple check)
        if self.config.character:
            char = self.config.character.lower()
            if descriptions and char in descriptions.lower():
                self.descriptions = [descriptions]
            else:
                # 如果字符檢查失敗，仍然使用描述（而不是設為空）
                # 因為字符名稱可能以不同形式出現
                print(f'⚠️  警告：描述中未找到角色名稱 "{self.config.character}"，但仍使用該描述')
                self.descriptions = [descriptions] if descriptions else [prompt]
        else:
             self.descriptions = [descriptions] if descriptions else [prompt]

        print(f'Image descriptions : {self.descriptions}')
        print(f'生成描述花費 : {time.time() - start_time:.2f} 秒')
        return self

    def generate_media(self):
        start_time = time.time()
        
        # Get strategy config with proper merging
        image_config = self._get_strategy_config('text2img')
        
        # Get workflow path: image_config.workflow_path -> config.workflow_path -> default
        workflow_path = image_config.get('workflow_path') or getattr(self.config, 'workflow_path', 'configs/workflow/txt2img.json')
        # Get images_per_description: image_config -> general -> default
        images_per_desc = image_config.get('images_per_description', 4)
        output_dir = getattr(self.config, 'output_dir', 'output')
        
        for idx, description in enumerate(self.descriptions):
            for i in range(images_per_desc):
                seed = random.randint(1, 999999999999)
                
                import json
                with open(workflow_path, 'r', encoding='utf-8') as f:
                    workflow = json.load(f)
                
                # Merge additional_params with image_config for node_manager
                merged_params = self._merge_node_manager_params(image_config)
                updates = self.node_manager.generate_updates(
                    workflow=workflow,
                    updates_config=image_config.get('custom_node_updates', []),
                    description=description,
                    seed=seed,
                    **merged_params
                )
                
                self.media_generator.generate(
                    workflow_path=workflow_path,
                    updates=updates,
                    output_dir=output_dir,
                    file_prefix=f"{getattr(self.config, 'character', 'char')}_d{idx}_{i}"
                )
                
        print(f'✅ 生成圖片總耗時: {time.time() - start_time:.2f} 秒')
        return self

    def analyze_media_text_match(self, similarity_threshold):
        """分析媒體與文本的匹配度
        
        Args:
            similarity_threshold: 相似度閾值（0.0-1.0）
        """
        output_dir = getattr(self.config, 'output_dir', 'output')
        
        # 遞歸搜索所有圖片文件（包括子目錄）
        media_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp']:
            pattern = os.path.join(output_dir, '**', ext)
            found = glob.glob(pattern, recursive=True)
            media_paths.extend(found)
        
        # 過濾出實際存在的圖片文件並去重
        media_paths = list(set([
            p for p in media_paths 
            if os.path.isfile(p) and any(p.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp'])
        ]))
        
        # 按文件名排序，確保順序一致
        media_paths.sort()
        
        print(f'找到 {len(media_paths)} 個媒體文件進行分析')
        if len(media_paths) == 0:
            print(f'⚠️  警告：在 {output_dir} 中沒有找到任何圖片文件')
            self.filter_results = []
            return self
        
        # 確保有 descriptions
        if not self.descriptions:
            print(f'⚠️  警告：沒有描述可用於分析')
            self.filter_results = []
            return self
        
        print(f'使用 {len(self.descriptions)} 個描述進行匹配分析')
        print(f'相似度閾值: {similarity_threshold}')
        
        self.filter_results = self.vision_manager.analyze_media_text_match(
            media_paths=media_paths,
            descriptions=self.descriptions,
            main_character=getattr(self.config, 'character', ''),
            similarity_threshold=similarity_threshold
        )
        return self

    def post_process_media(self, media_paths: List[str], output_dir: str) -> List[str]:
        image_config = self._get_strategy_config('text2img')
        
        # Check config for upscale: image_config -> general
        enable_upscale = image_config.get('enable_upscale', False)
        if not enable_upscale:
            return media_paths
            
        # Get upscale workflow path: image_config -> general -> default
        upscale_workflow = image_config.get('upscale_workflow_path', 'configs/workflow/Tile Upscaler SDXL.json')
        upscaled_paths = []
        
        for path in media_paths:
            if not any(path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                upscaled_paths.append(path)
                continue
                
            # Upload image first
            filename = self.media_generator.upload_image(path)
            
            # Update workflow
            updates = [{
                "type": "direct_update",
                "node_id": "225", # Assuming fixed node ID for loader in this specific workflow
                "inputs": {"image": filename}
            }]
            
            generated = self.media_generator.generate(
                workflow_path=upscale_workflow,
                updates=updates,
                output_dir=os.path.join(output_dir, 'upscaled'),
                file_prefix=f"upscaled_{os.path.basename(path)}"
            )
            upscaled_paths.extend(generated)
            
        return upscaled_paths

    def needs_user_review(self) -> bool:
        """檢查是否需要使用者審核
        
        Text2Image 只需要一次審核（選擇最終要發布的圖片）
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
        
        對於 Text2ImageStrategy，審核後標記為已審核，不需要第二次審核
        圖片放大在 post_process_media 中處理
        """
        self._reviewed = True
        return True
    
    def generate_article_content(self):
        # 使用基類的完整實現
        return super().generate_article_content()