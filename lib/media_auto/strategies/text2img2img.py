import time
import random
import glob
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional

from lib.media_auto.strategies.base_strategy import ContentStrategy, GenerationConfig
from lib.media_auto.services.media_generator import MediaGenerator
from lib.media_auto.models.vision.vision_manager import VisionManagerBuilder
from lib.comfyui.node_manager import NodeManager
from utils.logger import setup_logger

class Text2Image2ImageStrategy(ContentStrategy):
    """
    Text-to-Image-to-Image generation strategy.
    Refactored to use composition.
    """
    def __init__(self, character_data_service=None, vision_manager=None):
        self.character_data_service = character_data_service
        
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
        self._first_stage_reviewed = False
        self._second_stage_generated = False
        self._second_stage_reviewed = False
        self.logger = setup_logger('mediaoverload')

    def load_config(self, config: GenerationConfig):
        self.config = config

    def generate_description(self):
        """Generates image descriptions."""
        start_time = time.time()
        
        # Get strategy config with proper merging
        first_stage_config = self._get_strategy_config('text2image2image', 'first_stage')
        
        # Get style: support weights or single value
        style = self._get_style(first_stage_config)
        # Get image_system_prompt: support weights or single value
        image_system_prompt = self._get_system_prompt(first_stage_config)
        
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
        self.logger.info(f'最終生成的描述: {self.descriptions}')
        print(f'生成描述花費 : {time.time() - start_time}')
        self.logger.info(f'生成描述花費 : {time.time() - start_time} 秒')
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
        
        # 保存第一階段的圖片，等待第一次審核
        self.first_stage_images = sorted(generated_paths)
        
        # 為第一次審核準備 filter_results
        filter_results = self.vision_manager.analyze_media_text_match(
            media_paths=generated_paths,
            descriptions=self.descriptions,
            main_character=getattr(self.config, 'character', ''),
            similarity_threshold=0.0,
            temperature=0.3
        )
        self.filter_results = filter_results
        
        print(f'\n✅ Text2Image2Image 第一階段完成，生成 {len(generated_paths)} 張圖片')
        print(f'等待使用者審核選擇要進行 I2I 的圖片...')
        return self
    
    def _generate_second_stage(self, selected_image_paths: List[str]):
        """生成第二階段 I2I 圖片
        
        Args:
            selected_image_paths: 使用者選擇的第一階段圖片路徑列表
        """
        start_time = time.time()
        print("\n" + "=" * 60)
        print("第二階段：Image to Image 生成")
        print("=" * 60)
        
        # Get second_stage config with proper merging
        second_stage_config = self._get_strategy_config('text2image2image', 'second_stage')
        
        i2i_workflow_path = second_stage_config.get('workflow_path') or getattr(self.config.workflows, 'image2image', '')
        images_per_input = second_stage_config.get('images_per_input', 1)
        second_stage_output_dir = os.path.join(getattr(self.config, 'output_dir', 'output'), 'second_stage')
        
        with open(i2i_workflow_path, 'r', encoding='utf-8') as f:
            i2i_workflow = json.load(f)
        
        # 找到選中圖片對應的描述
        selected_descriptions = []
        for img_path in selected_image_paths:
            for row in self.filter_results:
                if row['media_path'] == img_path:
                    selected_descriptions.append(row['description'])
                    break
            else:
                # 如果找不到對應描述，使用第一個描述
                selected_descriptions.append(self.descriptions[0] if self.descriptions else '')
        
        for img_idx, (input_image_path, description) in enumerate(zip(selected_image_paths, selected_descriptions)):
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
        
        self._second_stage_generated = True
        print(f'\n✅ Text2Image2Image 第二階段完成，耗時: {time.time() - start_time:.2f} 秒')

    def analyze_media_text_match(self, similarity_threshold):
        """分析媒體與文本的匹配度
        
        如果第二階段已生成，分析第二階段的圖片
        否則分析第一階段的圖片（用於第一次審核）
        """
        if self._second_stage_generated:
            # 第二階段已生成，分析第二階段的圖片
            output_dir = os.path.join(getattr(self.config, 'output_dir', 'output'), 'second_stage')
            media_paths = glob.glob(f'{output_dir}/*')
            image_paths = [p for p in media_paths if any(p.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp'])]
            
            self.filter_results = self.vision_manager.analyze_media_text_match(
                media_paths=image_paths,
                descriptions=self.descriptions,
                main_character=getattr(self.config, 'character', ''),
                similarity_threshold=similarity_threshold
            )
        else:
            # 第一階段，filter_results 已在 generate_media 中設置
            pass
        return self
    
    def should_generate_article_now(self) -> bool:
        """判斷是否應該現在生成文章內容
        
        應該在第二階段完成後才生成文章內容
        """
        return self._second_stage_generated

    def needs_user_review(self) -> bool:
        """檢查是否需要使用者審核
        
        Text2Image2Image 需要兩次審核：
        1. 第一次：選擇第一階段的圖片（用於第二階段的 I2I）
        2. 第二次：選擇第二階段的最終圖片（用於發布）
        """
        if not self._first_stage_reviewed:
            # 第一次審核：第一階段圖片已生成，尚未審核
            return hasattr(self, 'first_stage_images') and len(self.first_stage_images) > 0
        if self._second_stage_generated and not self._second_stage_reviewed:
            # 第二次審核：第二階段圖片已生成，尚未審核
            return True
        return False
    
    def get_review_items(self, max_items: int = 10) -> List[Dict[str, Any]]:
        """獲取需要審核的項目
        
        第一次審核：返回第一階段的圖片
        第二次審核：返回第二階段的圖片
        """
        if self._second_stage_generated and not self._second_stage_reviewed:
            # 第二次審核：返回第二階段的圖片
            output_dir = os.path.join(getattr(self.config, 'output_dir', 'output'), 'second_stage')
            media_paths = glob.glob(f'{output_dir}/*')
            image_paths = [p for p in media_paths if any(p.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp'])]
            return [{'media_path': p, 'similarity': 1.0} for p in sorted(image_paths)[:max_items]]
        
        # 第一次審核：返回第一階段的圖片
        if hasattr(self, 'filter_results') and self.filter_results:
            return self.filter_results[:max_items]
        return []
    
    def handle_review_result(self, selected_indices: List[int], output_dir: str, selected_paths: List[str] = None) -> bool:
        """處理使用者審核結果
        
        第一次審核：觸發第二階段 I2I 生成
        第二次審核：標記為完成
        """
        if not self._first_stage_reviewed:
            # 第一次審核：獲取選中的圖片路徑
            if selected_paths:
                selected_image_paths = selected_paths
            else:
                review_items = self.get_review_items(max_items=10)
                selected_image_paths = [review_items[i]['media_path'] for i in selected_indices if i < len(review_items)]
            
            if not selected_image_paths:
                self.logger.warning("沒有選擇任何圖片，無法繼續生成第二階段")
                return False
            
            # 觸發第二階段生成
            self._generate_second_stage(selected_image_paths)
            self._first_stage_reviewed = True
            return True
        
        # 第二次審核完成
        self._second_stage_reviewed = True
        return True
    
    def generate_article_content(self):
        # 使用基類的完整實現
        return super().generate_article_content()
