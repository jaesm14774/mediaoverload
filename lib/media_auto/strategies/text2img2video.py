import time
import random
import glob
import re
import os
import numpy as np
from typing import List, Dict, Any, Optional

from lib.media_auto.strategies.base_strategy import ContentStrategy, GenerationConfig
from lib.media_auto.services.media_generator import MediaGenerator
from lib.media_auto.models.vision.vision_manager import VisionManagerBuilder
from lib.comfyui.node_manager import NodeManager

class Text2Image2VideoStrategy(ContentStrategy):
    """
    Text-to-Image-to-Video generation strategy.
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
        self.original_images: List[str] = []
        self.video_descriptions: Dict[str, str] = {}
        self.audio_descriptions: Dict[str, str] = {}
        self._videos_generated = False
        self._videos_reviewed = False

    def load_config(self, config: GenerationConfig):
        self.config = config

    def generate_description(self):
        """Generates image descriptions for the first stage."""
        start_time = time.time()
        
        # Get strategy config with proper merging
        first_stage_config = self._get_strategy_config('text2image2video', 'first_stage')
        
        # Get style: support weights or single value
        style = self._get_style(first_stage_config)
        # Get image_system_prompt: support weights or single value
        image_system_prompt = self._get_system_prompt(first_stage_config)

        prompt = self.config.prompt
        if style and style.strip():
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
        """First stage: Generate images."""
        start_time = time.time()
        print("=" * 60)
        print("第一階段：Text to Image 生成")
        print("=" * 60)
        
        # Get strategy config with proper merging
        first_stage_config = self._get_strategy_config('text2image2video', 'first_stage')
        
        # Get workflow path: first_stage.t2i_workflow_path -> config.workflow_path -> default
        t2i_workflow_path = first_stage_config.get('t2i_workflow_path') or getattr(self.config, 'workflow_path', 'configs/workflow/txt2img.json')
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
                
        self.first_stage_images = sorted(generated_paths)
        self.original_images = sorted(generated_paths).copy()
        
        # 不在此處生成文章內容，因為 should_generate_article_now() 返回 False
        # 文章內容將在影片生成後由 orchestration_service 生成
        
        print(f'\n✅ Text2Image2Video 第一階段耗時: {time.time() - start_time:.2f} 秒')
        return self

    def needs_user_review(self) -> bool:
        if len(self.first_stage_images) > 0 and not self._videos_generated:
            return True
        if self._videos_generated and not self._videos_reviewed:
            return True
        return False

    def get_review_items(self, max_items: int = 10) -> List[Dict[str, Any]]:
        if self._videos_generated and not self._videos_reviewed:
            # Return videos
            video_dir = os.path.join(getattr(self.config, 'output_dir', 'output'), 'videos')
            if os.path.exists(video_dir):
                videos = glob.glob(f'{video_dir}/*.mp4') # Simplified glob
                return [{'media_path': p, 'similarity': 1.0} for p in videos[:max_items]]
        
        # Return images
        return [{'media_path': p, 'similarity': 1.0} for p in self.first_stage_images[:max_items]]

    def handle_review_result(self, selected_indices: List[int], output_dir: str, selected_paths: List[str] = None) -> bool:
        if not selected_indices and not selected_paths:
            return False
        
        # 優先使用傳入的 selected_paths，避免 get_review_items 順序不一致的問題
        if selected_paths is None:
            review_items = self.get_review_items(max_items=10)
            selected_paths = [review_items[i]['media_path'] for i in selected_indices if i < len(review_items)]
        
        if self._videos_generated:
            # Reviewing videos, just confirm
            self._videos_reviewed = True
            return True
            
        # Reviewing images -> Upscale -> Generate Videos
        # 檢查是否需要 upscale
        first_stage_config = self._get_strategy_config('text2image2video', 'first_stage')
        enable_upscale = first_stage_config.get('enable_upscale', False)
        
        if enable_upscale:
            print("=" * 60)
            print("圖片放大處理")
            print("=" * 60)
            selected_paths = self._upscale_images(selected_paths, output_dir)
        
        # Generate Videos using upscaled images
        self._generate_videos_from_images(selected_paths, output_dir)
        return True

    def _upscale_images(self, image_paths: List[str], output_dir: str) -> List[str]:
        """放大圖片
        
        Args:
            image_paths: 圖片路徑列表
            output_dir: 輸出路徑
            
        Returns:
            放大後的圖片路徑列表
        """
        first_stage_config = self._get_strategy_config('text2image2video', 'first_stage')
        upscale_workflow = first_stage_config.get('upscale_workflow_path', 'configs/workflow/Tile Upscaler SDXL.json')
        upscaled_paths = []
        
        for path in image_paths:
            if not any(path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp']):
                upscaled_paths.append(path)
                continue
                
            print(f"放大圖片: {path}")
            filename = self.media_generator.upload_image(path)
            
            updates = [{
                "type": "direct_update",
                "node_id": "225",
                "inputs": {"image": filename}
            }]
            
            generated = self.media_generator.generate(
                workflow_path=upscale_workflow,
                updates=updates,
                output_dir=os.path.join(output_dir, 'upscaled'),
                file_prefix=f"upscaled_{os.path.basename(path)}"
            )
            if generated:
                upscaled_paths.extend(generated)
            else:
                upscaled_paths.append(path)
        
        print(f"✅ 圖片放大完成，共 {len(upscaled_paths)} 張")
        return upscaled_paths

    def _generate_videos_from_images(self, image_paths: List[str], output_dir: str):
        print(f"開始使用 {len(image_paths)} 張圖片生成影片")
        # Generate descriptions
        for img_path in image_paths:
            content = self.vision_manager.extract_image_content(img_path)
            vid_desc = self.vision_manager.generate_video_prompts(content)
            self.video_descriptions[img_path] = vid_desc
            
            audio_desc = self.vision_manager.generate_audio_description(img_path, vid_desc)
            self.audio_descriptions[img_path] = audio_desc
            
        # Generate Videos
        # Get strategy config with proper merging
        video_config = self._get_strategy_config('text2image2video', 'video')
        
        # Get workflow path: video.i2v_workflow_path -> default
        i2v_workflow_path = video_config.get('i2v_workflow_path', 'configs/workflow/wan2.2_gguf_i2v_audio.json')
        # Get videos_per_image: video -> general -> default
        videos_per_image = video_config.get('videos_per_image', 1)
        
        # Load workflow
        import json
        with open(i2v_workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
            
        video_output_dir = os.path.join(output_dir, 'videos')
        
        for idx, img_path in enumerate(image_paths):
            # Upload image
            img_filename = self.media_generator.upload_image(img_path)
            vid_desc = self.video_descriptions.get(img_path, '')
            audio_desc = self.audio_descriptions.get(img_path, '')
            
            for i in range(videos_per_image):
                seed = random.randint(1, 999999999999)
                
                # Custom updates for I2V
                custom_updates = video_config.get('custom_node_updates', []).copy()
                custom_updates.append({"node_type": "LoadImage", "node_index": 0, "inputs": {"image": img_filename}})
                custom_updates.append({"node_id": "70", "inputs": {"value": vid_desc}}) # Positive prompt
                custom_updates.append({"node_id": "94", "inputs": {"value": audio_desc}}) # Audio prompt
                
                # Merge additional_params with video_config for node_manager
                merged_params = self._merge_node_manager_params(video_config)
                updates = self.node_manager.generate_updates(
                    workflow=workflow,
                    updates_config=custom_updates,
                    description=vid_desc,
                    seed=seed,
                    **merged_params
                )
                
                self.media_generator.generate(
                    workflow_path=i2v_workflow_path,
                    updates=updates,
                    output_dir=video_output_dir,
                    file_prefix=f"{getattr(self.config, 'character', 'char')}_i2v_{idx}_{i}"
                )
                
        self._videos_generated = True

    def analyze_media_text_match(self, similarity_threshold):
        """分析媒體與文本的匹配度
        
        Text2Image2Video 策略的篩選邏輯：
        - 第一階段（圖片）：不使用 LLM 篩選，直接返回所有圖片讓用戶篩選
        - 第二階段（影片）：直接返回所有影片讓用戶篩選
        
        這樣可以避免浪費 LLM 篩選時間，讓用戶直接選擇想要的圖片/影片。
        
        Args:
            similarity_threshold: 相似度閾值（此策略不使用）
        """
        output_dir = getattr(self.config, 'output_dir', 'output')
        
        if self._videos_generated:
            # 影片已生成，直接返回所有影片讓用戶篩選（不用 LLM 篩選）
            video_dir = os.path.join(output_dir, 'videos')
            if not os.path.exists(video_dir):
                print(f'⚠️ 警告：影片目錄不存在: {video_dir}')
                self.filter_results = []
                return self
            
            video_extensions = ['*.mp4', '*.avi', '*.mov', '*.gif', '*.webm']
            video_paths = []
            for ext in video_extensions:
                pattern = os.path.join(video_dir, ext)
                found = glob.glob(pattern)
                video_paths.extend(found)
            
            video_paths = sorted(list(set(video_paths)))
            print(f'找到 {len(video_paths)} 個影片，直接交由用戶篩選（不使用 LLM）')
            
            if len(video_paths) == 0:
                print(f'⚠️ 警告：在 {video_dir} 中沒有找到任何影片文件')
                self.filter_results = []
                return self
            
            # 為每個影片創建 filter_result，使用對應的影片描述
            self.filter_results = []
            for video_path in video_paths:
                match = re.search(r'_i2v_(\d+)_\d+\.', video_path)
                if match:
                    img_idx = int(match.group(1))
                    if img_idx < len(self.first_stage_images):
                        img_path = self.first_stage_images[img_idx]
                        video_desc = self.video_descriptions.get(img_path, '')
                        if not video_desc:
                            video_desc = list(self.video_descriptions.values())[0] if self.video_descriptions else ''
                    else:
                        video_desc = list(self.video_descriptions.values())[0] if self.video_descriptions else ''
                else:
                    video_desc = list(self.video_descriptions.values())[0] if self.video_descriptions else ''
                
                self.filter_results.append({
                    'media_path': video_path,
                    'description': video_desc if video_desc else (self.descriptions[0] if self.descriptions else ''),
                    'similarity': 1.0
                })
        else:
            # 第一階段（圖片）：直接返回所有圖片讓用戶篩選，不使用 LLM 篩選
            first_stage_dir = os.path.join(output_dir, 'first_stage')
            if not os.path.exists(first_stage_dir):
                first_stage_dir = output_dir
            
            image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp']
            image_paths = []
            for ext in image_extensions:
                pattern = os.path.join(first_stage_dir, ext)
                found = glob.glob(pattern)
                image_paths.extend(found)
            
            image_paths = sorted(list(set(image_paths)))
            print(f'找到 {len(image_paths)} 個圖片，直接交由用戶篩選（不使用 LLM）')
            
            if len(image_paths) == 0:
                print(f'⚠️ 警告：在 {first_stage_dir} 中沒有找到任何圖片文件')
                self.filter_results = []
                return self
            
            # 直接返回所有圖片，不進行 LLM 篩選
            description = self.descriptions[0] if self.descriptions else ''
            self.filter_results = [
                {
                    'media_path': img_path,
                    'description': description,
                    'similarity': 1.0  # 不篩選，全部返回
                }
                for img_path in image_paths
            ]
        
        return self

    def generate_article_content(self):
        # 如果影片已生成，應該使用影片描述來生成文章內容
        # 但基類的實現會使用 filter_results 中的 description
        # 所以只要 filter_results 正確設置了影片描述，基類實現就能正常工作
        return super().generate_article_content()

    def should_generate_article_now(self) -> bool:
        return self._videos_generated
