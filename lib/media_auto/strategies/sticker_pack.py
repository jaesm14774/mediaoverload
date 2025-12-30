import time
import random
import glob
import json
import os
import numpy as np
from typing import List, Dict, Any, Optional

from lib.media_auto.strategies.base_strategy import ContentStrategy, GenerationConfig
from lib.media_auto.services.media_generator import MediaGenerator
from lib.media_auto.models.vision.vision_manager import VisionManagerBuilder
from lib.comfyui.node_manager import NodeManager
from lib.services.implementations.ffmpeg_service import FFmpegService
from utils.logger import setup_logger
from configs.prompt.image_system_guide import sticker_expression_system_prompt


class StickerPackStrategy(ContentStrategy):
    """
    Sticker Pack generation strategy.
    Generates multiple expression stickers for a character, with optional animated GIF support.
    """
    
    def __init__(self, character_data_service=None, vision_manager=None):
        self.character_data_service = character_data_service
        self.logger = setup_logger(__name__)
        
        if vision_manager is None:
            vision_manager = VisionManagerBuilder() \
                .with_vision_model('gemini', model_name='gemini-flash-lite-latest') \
                .with_text_model('openrouter') \
                .with_random_models(True) \
                .build()
        self.vision_manager = vision_manager
        
        self.media_generator = MediaGenerator()
        self.node_manager = NodeManager()
        self.ffmpeg_service = FFmpegService()
        
        self.config = None
        self.expressions: List[str] = []
        self.descriptions: List[str] = []
        self.filter_results: List[Dict[str, Any]] = []
        self.static_stickers: List[str] = []
        self.animated_stickers: List[str] = []
        self._stickers_generated = False
        self._gifs_generated = False

    def load_config(self, config: GenerationConfig):
        self.config = config

    def generate_description(self):
        """Generate sticker expressions using LLM."""
        start_time = time.time()
        self.logger.info("Generating sticker expressions using LLM...")
        
        sticker_config = self._get_strategy_config('sticker_pack')
        character = getattr(self.config, 'character', '')
        prompt = getattr(self.config, 'prompt', '')
        
        # Build user input for LLM
        user_input = f"Character: {character}"
        if prompt:
            user_input += f"\nTheme/Context: {prompt}"
        user_input += "\n\nGenerate 8 unique sticker expressions for this character."
        
        # Use LLM to generate expressions
        messages = [
            {'role': 'system', 'content': sticker_expression_system_prompt},
            {'role': 'user', 'content': user_input}
        ]
        
        response = self.vision_manager.text_model.chat_completion(messages=messages)
        
        # Clean response
        if '</think>' in response:
            response = response.split('</think>')[-1].strip()
        
        # Parse JSON
        if '```json' in response:
            response = response.split('```json')[1].split('```')[0]
        elif '```' in response:
            response = response.split('```')[1].split('```')[0]
        
        self.expressions = json.loads(response.strip())
        
        if not isinstance(self.expressions, list) or len(self.expressions) == 0:
            raise ValueError("Invalid expressions format")
            
        self.logger.info(f"Generated {len(self.expressions)} expressions")
        
        # Build full descriptions with character and style
        style = self._get_style(sticker_config)
        
        self.descriptions = []
        for expr in self.expressions:
            desc = f"{character}, {expr}, {style}"
            self.descriptions.append(desc)
        
        self.logger.info(f"Sticker descriptions prepared: {len(self.descriptions)}")
        print(f'生成描述花費: {time.time() - start_time:.2f} 秒')
        return self

    def generate_media(self):
        """Generate static sticker images."""
        start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info("Generating static sticker images")
        self.logger.info("=" * 60)
        
        sticker_config = self._get_strategy_config('sticker_pack')
        static_config = sticker_config.get('static_config', {})
        
        workflow_path = static_config.get('workflow_path') or \
                       getattr(self.config, 'workflow_path', 'configs/workflow/nova-anime-xl.json')
        images_per_expression = static_config.get('images_per_expression', 1)
        output_dir = os.path.join(getattr(self.config, 'output_dir', 'output'), 'stickers')
        os.makedirs(output_dir, exist_ok=True)
        
        # Load workflow
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        
        generated_paths = []
        
        for idx, description in enumerate(self.descriptions):
            self.logger.info(f"Generating sticker {idx + 1}/{len(self.descriptions)}: {self.expressions[idx]}")
            
            for i in range(images_per_expression):
                seed = random.randint(1, 999999999999)
                
                merged_params = self._merge_node_manager_params(static_config)
                updates = self.node_manager.generate_updates(
                    workflow=workflow,
                    updates_config=static_config.get('custom_node_updates', []),
                    description=description,
                    seed=seed,
                    **merged_params
                )
                
                paths = self.media_generator.generate(
                    workflow_path=workflow_path,
                    updates=updates,
                    output_dir=output_dir,
                    file_prefix=f"sticker_{idx}_{i}"
                )
                generated_paths.extend(paths)
        
        self.static_stickers = sorted(generated_paths)
        self._stickers_generated = True
        
        self.logger.info(f"✅ Generated {len(self.static_stickers)} static stickers in {time.time() - start_time:.2f}s")
        return self

    def analyze_media_text_match(self, similarity_threshold):
        """Analyze sticker quality."""
        # 如果已經在 handle_review_result 中設置了 filter_results，不要覆蓋
        if self._gifs_generated and self.filter_results:
            self.logger.info(f"使用已選中的 {len(self.filter_results)} 個貼圖")
            return self
        
        output_dir = os.path.join(getattr(self.config, 'output_dir', 'output'), 'stickers')
        
        if self._gifs_generated:
            # Return GIFs
            gif_dir = os.path.join(getattr(self.config, 'output_dir', 'output'), 'animated_stickers')
            gif_paths = glob.glob(f'{gif_dir}/*.gif')
            self.filter_results = [
                {'media_path': p, 'description': '', 'similarity': 1.0}
                for p in sorted(gif_paths)
            ]
        else:
            # Return static stickers
            image_paths = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp']:
                image_paths.extend(glob.glob(f'{output_dir}/{ext}'))
            
            self.filter_results = self.vision_manager.analyze_media_text_match(
                media_paths=sorted(image_paths),
                descriptions=self.descriptions,
                main_character=getattr(self.config, 'character', ''),
                similarity_threshold=similarity_threshold
            )
        
        return self

    def needs_user_review(self) -> bool:
        """Check if user review is needed."""
        if self._stickers_generated and not self._gifs_generated:
            return True
        if self._gifs_generated:
            return True
        return False

    def get_review_items(self, max_items: int = 10) -> List[Dict[str, Any]]:
        """Get items for review."""
        if self._gifs_generated:
            gif_dir = os.path.join(getattr(self.config, 'output_dir', 'output'), 'animated_stickers')
            gif_paths = glob.glob(f'{gif_dir}/*.gif')
            if gif_paths:
                return [{'media_path': p, 'similarity': 1.0} for p in sorted(gif_paths)[:max_items]]
            # 如果沒有 GIF，返回 filter_results 中的靜態貼圖（用於第二次 review）
            if hasattr(self, 'filter_results') and self.filter_results:
                return self.filter_results[:max_items]
        
        return [{'media_path': p, 'similarity': 1.0} for p in self.static_stickers[:max_items]]

    def handle_review_result(self, selected_indices: List[int], output_dir: str, selected_paths: List[str] = None) -> bool:
        """Handle user selection and optionally generate animated GIFs.
        
        Args:
            selected_indices: 用戶選擇的索引（向後兼容）
            output_dir: 輸出目錄
            selected_paths: 用戶選擇的圖片路徑（優先使用，避免重複調用 get_review_items）
        """
        if not selected_indices and not selected_paths:
            return False
        
        # 優先使用傳入的 selected_paths，避免 get_review_items 順序不一致的問題
        if selected_paths is None:
            review_items = self.get_review_items(max_items=10)
            selected_paths = [review_items[i]['media_path'] for i in selected_indices if i < len(review_items)]
            self.logger.warning("handle_review_result: 使用 selected_indices 解析路徑，建議傳入 selected_paths")
        
        self.logger.info(f"handle_review_result: 選中的圖片路徑: {selected_paths}")
        
        if self._gifs_generated:
            # Final selection done
            self.filter_results = [
                {'media_path': p, 'description': '', 'similarity': 1.0}
                for p in selected_paths
            ]
            return True
        
        # 檢查是否要生成動畫 GIF
        sticker_config = self._get_strategy_config('sticker_pack')
        animated_config = sticker_config.get('animated_config', {})
        
        if not animated_config.get('enabled', True):
            self.logger.info("Animated stickers disabled in config")
            self.filter_results = [
                {'media_path': p, 'description': '', 'similarity': 1.0}
                for p in selected_paths
            ]
            self._gifs_generated = True
            return True
        
        # 隨機決定是否生成 GIF（預設 50% 機率）
        gif_probability = animated_config.get('gif_probability', 0.5)
        should_generate_gif = random.random() < gif_probability
        
        if should_generate_gif:
            self.logger.info(f"🎬 決定生成動畫 GIF（機率: {gif_probability:.0%}）")
            self._generate_animated_stickers(selected_paths, output_dir)
        else:
            self.logger.info(f"🖼️ 決定使用靜態貼圖（機率: {1-gif_probability:.0%}）")
            self.filter_results = [
                {'media_path': p, 'description': '', 'similarity': 1.0}
                for p in selected_paths
            ]
            self._gifs_generated = True
        
        return True

    def _generate_animated_stickers(self, image_paths: List[str], output_dir: str):
        """Generate animated GIF stickers from selected images."""
        self.logger.info("=" * 60)
        self.logger.info("Generating animated sticker GIFs")
        self.logger.info("=" * 60)
        
        sticker_config = self._get_strategy_config('sticker_pack')
        animated_config = sticker_config.get('animated_config', {})
        
        i2v_workflow_path = animated_config.get('i2v_workflow_path', 
            'configs/workflow/wan2.2_gguf_i2v.json')
        
        # Sticker 專用參數（短動畫）
        total_frames = animated_config.get('total_frames', 40)
        video_fps = animated_config.get('video_fps', 12)
        gif_fps = animated_config.get('gif_fps', 10)
        gif_max_colors = animated_config.get('gif_max_colors', 256)
        gif_scale_width = animated_config.get('gif_scale_width', 512)
        
        self.logger.info(f"Sticker animation settings: {total_frames} frames @ {video_fps} fps = {total_frames/video_fps:.1f}s")
        
        video_output_dir = os.path.join(output_dir, 'animated_videos')
        gif_output_dir = os.path.join(output_dir, 'animated_stickers')
        os.makedirs(video_output_dir, exist_ok=True)
        os.makedirs(gif_output_dir, exist_ok=True)
        
        # Load I2V workflow
        with open(i2v_workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        
        for idx, img_path in enumerate(image_paths):
            self.logger.info(f"Processing animated sticker {idx + 1}/{len(image_paths)}")
            
            # Upload image
            img_filename = self.media_generator.upload_image(img_path)
            
            # Generate simple sticker-focused video description
            vid_desc = self.vision_manager.generate_video_prompts(
                f"simple subtle loop animation, cute character slight movement, minimal motion",
                "sticker_motion_system_prompt"
            )
            
            seed = random.randint(1, 999999999999)
            
            custom_updates = animated_config.get('custom_node_updates', []).copy()
            custom_updates.append({
                "node_type": "LoadImage", 
                "node_index": 0, 
                "inputs": {"image": img_filename}
            })
            # 設定 sticker 專用的 total_frames（短動畫）
            custom_updates.append({
                "node_type": "PrimitiveInt",
                "filter": {"title": "total_frame"},
                "inputs": {"value": total_frames}
            })
            # 設定 sticker 專用的 fps
            custom_updates.append({
                "node_type": "PrimitiveFloat",
                "filter": {"title": "frame_rate"},
                "inputs": {"value": video_fps}
            })
            
            merged_params = self._merge_node_manager_params(animated_config)
            updates = self.node_manager.generate_updates(
                workflow=workflow,
                updates_config=custom_updates,
                description=vid_desc,
                seed=seed,
                **merged_params
            )
            
            # Generate video
            video_paths = self.media_generator.generate(
                workflow_path=i2v_workflow_path,
                updates=updates,
                output_dir=video_output_dir,
                file_prefix=f"animated_{idx}"
            )
            
            # Convert video to optimized GIF
            for video_path in video_paths:
                if video_path.endswith('.mp4'):
                    gif_path = os.path.join(gif_output_dir, f"sticker_{idx}.gif")
                    try:
                        # Use optimize_gif for smoother animation with frame interpolation
                        self.ffmpeg_service.optimize_gif(
                            video_path=video_path,
                            output_path=gif_path,
                            fps=gif_fps,
                            max_colors=gif_max_colors,
                            scale_width=gif_scale_width,
                            minterpolate=True
                        )
                        self.animated_stickers.append(gif_path)
                        self.logger.info(f"Created optimized animated sticker: {gif_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to create GIF: {e}")
        
        self._gifs_generated = True
        self.logger.info(f"✅ Generated {len(self.animated_stickers)} animated stickers")
    
    def _get_style(self, stage_config):
        """覆寫基類方法以使用不同的默認值"""
        return super()._get_style(stage_config, 
            default='LINE sticker style, chibi proportions, white outline, simple clean background, 2D flat shading')

    def should_generate_article_now(self) -> bool:
        """Generate article after GIFs are created."""
        return self._gifs_generated or not self._get_strategy_config('sticker_pack').get('animated_config', {}).get('enabled', True)

    def generate_article_content(self):
        """Generate article content for sticker pack."""
        return super().generate_article_content()

