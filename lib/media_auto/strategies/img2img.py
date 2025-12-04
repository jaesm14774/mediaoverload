import time
import random
import glob
import os
from typing import List, Dict, Any, Optional

from lib.media_auto.strategies.base_strategy import ContentStrategy, GenerationConfig
from lib.media_auto.services.media_generator import MediaGenerator
from lib.media_auto.models.vision.vision_manager import VisionManagerBuilder
from lib.comfyui.node_manager import NodeManager

class Image2ImageStrategy(ContentStrategy):
    """
    Image-to-Image generation strategy.
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
        self.input_images: List[str] = []
        self.filter_results: List[Dict[str, Any]] = []

    def load_config(self, config: GenerationConfig):
        self.config = config

    def generate_description(self):
        """Generates or extracts description."""
        start_time = time.time()
        
        # Get strategy config with proper merging
        i2i_config = self._get_strategy_config('image2image')
        
        extract_description = getattr(self.config, 'extract_description', False)
        input_image_path = getattr(self.config, 'input_image_path', None)
        
        # Collect input images
        if input_image_path:
            if os.path.isfile(input_image_path):
                self.input_images = [input_image_path]
            elif os.path.isdir(input_image_path):
                image_paths = glob.glob(f'{input_image_path}/*')
                self.input_images = [p for p in image_paths if any(p.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp'])]
            else:
                print(f"警告：輸入圖片路徑不存在: {input_image_path}")
                self.input_images = []
        else:
            print("警告：未提供 input_image_path")
            self.input_images = []
            
        if extract_description and self.input_images:
            # Extract from first image
            print(f"正在從圖片中提取描述: {self.input_images[0]}")
            image_content = self.vision_manager.extract_image_content(self.input_images[0])
            
            # Get image_system_prompt: i2i_config -> general -> config.image_system_prompt
            image_system_prompt = self._get_config_value(i2i_config, 'image_system_prompt', 'stable_diffusion_prompt')
            # 注意：img2img 從圖片提取內容，不適合使用雙角色互動
            descriptions = self.vision_manager.generate_image_prompts(image_content, image_system_prompt)
            self.descriptions = [descriptions] if descriptions else []
        else:
            prompt = self.config.prompt or ''
            # Get style: i2i_config -> general -> config.style
            style = self._get_config_value(i2i_config, 'style', '')
            if style and style.strip():
                prompt = f"{prompt}\nstyle: {style}".strip()
                
            if self.config.character:
                char_lower = self.config.character.lower()
                if char_lower not in prompt.lower() and "main character" not in prompt.lower():
                    prompt = f"Main character: {self.config.character}\n{prompt}"
            
            if not prompt:
                prompt = "Enhance and improve the image with more details"
                
            self.descriptions = [prompt]
            
        print(f'Image descriptions : {self.descriptions}')
        print(f'生成描述花費 : {time.time() - start_time}')
        return self

    def generate_media(self):
        """Generates media using I2I."""
        start_time = time.time()
        
        if not self.input_images:
            print("警告：沒有輸入圖片可供生成")
            return self
        
        # Get strategy config with proper merging
        i2i_config = self._get_strategy_config('image2image')
        
        # Get workflow path: i2i_config.workflow_path -> config.workflow_path -> default
        workflow_path = i2i_config.get('workflow_path') or getattr(self.config, 'workflow_path', 'configs/workflow/flux_dev_i2i.json')
        output_dir = getattr(self.config, 'output_dir', 'output')
        # Get images_per_input: i2i_config -> general -> default
        images_per_input = i2i_config.get('images_per_input', 1)
        
        # Load workflow
        import json
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
            
        for img_idx, input_image_path in enumerate(self.input_images):
            image_filename = self.media_generator.upload_image(input_image_path)
            
            desc_index = img_idx % len(self.descriptions) if self.descriptions else 0
            description = self.descriptions[desc_index] if self.descriptions else ''
            
            for i in range(images_per_input):
                seed = random.randint(1, 999999999999)
                
                custom_updates = i2i_config.get('custom_node_updates', []).copy()
                custom_updates.append({"node_type": "LoadImage", "node_index": 0, "inputs": {"image": image_filename}})
                
                # Merge additional_params with i2i_config for node_manager
                merged_params = self._merge_node_manager_params(i2i_config)
                updates = self.node_manager.generate_updates(
                    workflow=workflow,
                    updates_config=custom_updates,
                    description=description,
                    seed=seed,
                    **merged_params
                )
                
                self.media_generator.generate(
                    workflow_path=workflow_path,
                    updates=updates,
                    output_dir=output_dir,
                    file_prefix=f"{getattr(self.config, 'character', 'char')}_i2i_{img_idx}_{i}"
                )
                
        print(f'\n✅ Image to Image 生成總耗時: {time.time() - start_time:.2f} 秒')
        return self

    def analyze_media_text_match(self, similarity_threshold):
        output_dir = getattr(self.config, 'output_dir', 'output')
        media_paths = glob.glob(f'{output_dir}/*')
        image_paths = [p for p in media_paths if any(p.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp'])]
        
        self.filter_results = self.vision_manager.analyze_media_text_match(
            media_paths=image_paths,
            descriptions=self.descriptions,
            main_character=getattr(self.config, 'character', ''),
            similarity_threshold=similarity_threshold
        )
        return self

    def generate_article_content(self):
        # 使用基類的完整實現
        return super().generate_article_content()
