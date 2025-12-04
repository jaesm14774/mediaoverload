import time
import random
import glob
import os
from typing import List, Dict, Any, Optional

from lib.media_auto.strategies.base_strategy import ContentStrategy, GenerationConfig
from lib.media_auto.services.media_generator import MediaGenerator
from lib.media_auto.models.vision.vision_manager import VisionManagerBuilder
from lib.comfyui.node_manager import NodeManager

class Text2Image2ImageStrategy(ContentStrategy):
    """
    Text-to-Image-to-Image generation strategy.
    Refactored to use composition.
    """
    def __init__(self, character_repository=None, vision_manager=None):
        self.character_repository = character_repository
        
        if vision_manager is None:
            vision_manager = VisionManagerBuilder() \
                .with_vision_model('gemini', model_name='gemini-flash-lite-latest') \
                .with_text_model('gemini', model_name='gemini-flash-lite-latest') \
                .build()
        self.vision_manager = vision_manager
        
        self.media_generator = MediaGenerator()
        self.node_manager = NodeManager()
        
        self.config = None
        self.descriptions: List[str] = []
        self.first_stage_images: List[str] = []
        self.filter_results: List[Dict[str, Any]] = []

    def load_config(self, config: GenerationConfig):
        self.config = config

    def generate_description(self):
        """Generates image descriptions."""
        start_time = time.time()
        
        # Get strategy config with proper merging
        first_stage_config = self._get_strategy_config('text2image2image', 'first_stage')
        
        # Get style: first_stage -> general -> config.style
        style = self._get_config_value(first_stage_config, 'style', '')
        # Get image_system_prompt: first_stage -> general -> config.image_system_prompt
        image_system_prompt = self._get_config_value(first_stage_config, 'image_system_prompt', 'stable_diffusion_prompt')
        
        prompt = self.config.prompt
        if style:
            prompt = f"{prompt}\nstyle: {style}".strip()
            
        if self.config.character:
            char_lower = self.config.character.lower()
            if char_lower not in prompt.lower() and "main character" not in prompt.lower():
                prompt = f"Main character: {self.config.character}\n{prompt}"

        # 檢查是否使用雙角色互動系統提示詞
        if image_system_prompt == 'two_character_interaction_generate_system_prompt':
            descriptions = self._generate_two_character_interaction_description(prompt, style)
        else:
            descriptions = self.vision_manager.generate_image_prompts(prompt, image_system_prompt)
        self.descriptions = [descriptions] if descriptions else []
        
        print(f'Image descriptions : {self.descriptions}')
        print(f'生成描述花費 : {time.time() - start_time}')
        return self

    def generate_media(self):
        """Generates media in two stages."""
        start_time = time.time()
        print("=" * 60)
        print("第一階段：Text to Image 生成")
        print("=" * 60)
        
        # Get strategy config with proper merging
        first_stage_config = self._get_strategy_config('text2image2image', 'first_stage')
        
        # Get workflow path: first_stage.t2i_workflow_path -> config.workflow_path -> default
        t2i_workflow_path = first_stage_config.get('workflow_path') or getattr(self.config, 'workflow_path', '')
        # Get images_per_description: first_stage -> general -> default
        images_per_desc = first_stage_config.get('images_per_description', 4)
        output_dir = os.path.join(getattr(self.config, 'output_dir', 'output'), 'first_stage')
        
        # Load workflow
        import json
        with open(t2i_workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
            
        generated_paths = []
        for idx, description in enumerate(self.descriptions):
            for i in range(images_per_desc):
                seed = random.randint(1, 999999999999)
                
                # Merge additional_params with first_stage_config for node_manager
                merged_params = self._merge_node_manager_params(first_stage_config)
                updates = self.node_manager.generate_updates(
                    workflow=workflow,
                    updates_config=first_stage_config.get('custom_node_updates', []),
                    description=description,
                    seed=seed,
                    **merged_params
                )
                
                paths = self.media_generator.generate(
                    workflow_path=t2i_workflow_path,
                    updates=updates,
                    output_dir=output_dir,
                    file_prefix=f"{getattr(self.config, 'character', 'char')}_d{idx}_{i}"
                )
                generated_paths.extend(paths)
        
        if not generated_paths:
            print("警告：第一階段沒有生成任何圖片")
            return self
            
        print("\n第一階段生成完成，開始第二階段 Image to Image 生成...")
        
        # Filter best images
        print(f"正在從 {len(generated_paths)} 張圖片中篩選最佳圖片...")
        filter_results = self.vision_manager.analyze_media_text_match(
            media_paths=generated_paths,
            descriptions=self.descriptions,
            main_character=getattr(self.config, 'character', ''),
            similarity_threshold=0.0,
            temperature=0.3
        )
        
        # Get second_stage config with proper merging
        second_stage_config = self._get_strategy_config('text2image2image', 'second_stage')
        
        # Second Stage: I2I
        print("\n" + "=" * 60)
        print("第二階段：Image to Image 生成")
        print("=" * 60)
        
        i2i_workflow_path = second_stage_config.get('workflow_path') or getattr(self.config.workflows, 'image2image', '')
        images_per_input = second_stage_config.get('images_per_input', 1)
        second_stage_output_dir = os.path.join(getattr(self.config, 'output_dir', 'output'), 'second_stage')
        
        with open(i2i_workflow_path, 'r', encoding='utf-8') as f:
            i2i_workflow = json.load(f)
            
        for img_idx, row in enumerate(filter_results):
            input_image_path = row['media_path']
            description = row['description']
            image_filename = self.media_generator.upload_image(input_image_path)
            
            for i in range(images_per_input):
                seed = random.randint(1, 999999999999)
                
                custom_updates = second_stage_config.get('custom_node_updates', []).copy()
                custom_updates.append({"node_type": "LoadImage", "node_index": 0, "inputs": {"image": image_filename}})
                
                # Merge additional_params with second_stage_config for node_manager
                merged_params = self._merge_node_manager_params(second_stage_config)
                updates = self.node_manager.generate_updates(
                    workflow=i2i_workflow,
                    updates_config=custom_updates,
                    description=description,
                    seed=seed,
                    **merged_params
                )
                
                self.media_generator.generate(
                    workflow_path=i2i_workflow_path,
                    updates=updates,
                    output_dir=second_stage_output_dir,
                    file_prefix=f"{getattr(self.config, 'character', 'char')}_i2i_{img_idx}_{i}"
                )
                
        print(f'\n✅ Text2Image2Image 生成總耗時: {time.time() - start_time:.2f} 秒')
        return self

    def analyze_media_text_match(self, similarity_threshold):
        output_dir = os.path.join(getattr(self.config, 'output_dir', 'output'), 'second_stage')
        media_paths = glob.glob(f'{output_dir}/*')
        image_paths = [p for p in media_paths if any(p.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp'])]
        
        self.filter_results = self.vision_manager.analyze_media_text_match(
            media_paths=image_paths,
            descriptions=self.descriptions,
            main_character=getattr(self.config, 'character', ''),
            similarity_threshold=similarity_threshold
        )
        return self

    def needs_user_review(self) -> bool:
        """檢查是否需要使用者審核
        
        當有 filter_results 時，需要使用者審核選擇最終要發布的圖片
        """
        return hasattr(self, 'filter_results') and len(self.filter_results) > 0
    
    def get_review_items(self, max_items: int = 10) -> List[Dict[str, Any]]:
        """獲取需要審核的項目
        
        返回 filter_results 中的項目，最多 max_items 個
        """
        if hasattr(self, 'filter_results') and self.filter_results:
            return self.filter_results[:max_items]
        return []
    
    def handle_review_result(self, selected_indices: List[int], output_dir: str) -> bool:
        """處理使用者審核結果
        
        對於 Text2Image2ImageStrategy，審核後不需要特殊處理
        返回 True 表示成功處理
        """
        return True
    
    def generate_article_content(self):
        # 使用基類的完整實現
        return super().generate_article_content()
