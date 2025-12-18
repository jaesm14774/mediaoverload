import os
import time
import random
import json
import numpy as np
from typing import List, Dict, Any, Optional

from lib.media_auto.strategies.base_strategy import ContentStrategy, GenerationConfig
from lib.media_auto.services.script_generator import ScriptGenerator
from lib.media_auto.services.media_generator import MediaGenerator
from lib.media_auto.models.vision.vision_manager import VisionManagerBuilder
from lib.comfyui.node_manager import NodeManager
from lib.services.implementations.ffmpeg_service import FFmpegService
from lib.services.implementations.tts_service import TTSService
from utils.logger import setup_logger


class Text2LongVideoFirstFrameStrategy(ContentStrategy):
    """
    Text-to-Long-Video strategy with First-Frame driven transitions.
    
    Key difference from Text2LongVideoStrategy:
    - Uses I2I to transform the last frame into a new first frame for the next segment
    - This creates more natural transitions as I2V models have better control over first frames
    - The I2I step maintains visual style while adapting to the new scene description
    
    Flow:
    1. Generate first frame candidates
    2. User selects first frame
    3. Generate video from first frame
    4. Extract last frame
    5. **I2I: Transform last frame + new scene description -> new first frame**
    6. Generate next video segment
    7. Repeat until all segments complete
    8. Concatenate all videos
    """
    
    def __init__(self, character_data_service=None, vision_manager=None):
        self.logger = setup_logger(__name__)
        self.character_data_service = character_data_service
        
        if vision_manager is None:
            vision_manager = VisionManagerBuilder() \
                .with_vision_model('gemini', model_name='gemini-flash-lite-latest') \
                .with_text_model('gemini', model_name='gemini-flash-lite-latest') \
                .build()
        
        self.script_generator = ScriptGenerator(vision_manager)
        self.media_generator = MediaGenerator()
        self.node_manager = NodeManager()
        self.ffmpeg_service = FFmpegService()
        self.tts_service = TTSService()
        
        self.config = None
        self.script_segments: List[Dict[str, Any]] = []
        self.generated_media_paths: List[str] = []
        self.article_content: str = ""
        self.vision_manager = vision_manager
        self._final_video_generated = False
        self._videos_reviewed = False
        self.descriptions: List[str] = []
        self.filter_results: List[Dict[str, Any]] = []

    def load_config(self, config: GenerationConfig):
        self.config = config

    def generate_description(self):
        """Generate the first segment script."""
        if not self.config:
            raise RuntimeError("Config not loaded")
        
        start_time = time.time()
        self.logger.info("Generating first segment script...")
        
        longvideo_config = self._get_strategy_config('text2longvideo_firstframe', 'longvideo_config')
        first_stage_config = self._get_strategy_config('text2longvideo_firstframe', 'first_stage')
        
        context_data = self.config.get_all_attributes()
        context_data['segment_duration'] = longvideo_config.get('segment_duration', 5)
        context_data['segment_count'] = longvideo_config.get('segment_count', 5)
        context_data['use_tts'] = longvideo_config.get('use_tts', True)
        context_data['tts_voice'] = longvideo_config.get('tts_voice', 'en-US-AriaNeural')
        context_data['tts_rate'] = longvideo_config.get('tts_rate', '+0%')
        context_data['fps'] = longvideo_config.get('fps', 16)
        context_data['style'] = self._get_style(first_stage_config)
        
        first_segment = self.script_generator.generate_script_segment(context_data)
        self.script_segments = [first_segment]
        self.descriptions = [first_segment['visual']]
        
        self.logger.info(f"First segment script: {first_segment['visual']}")
        print(f'生成描述花費: {time.time() - start_time:.2f} 秒')
        return self

    def generate_media(self):
        """Generate candidate images for first frame selection."""
        self.logger.info("Generating first frame candidates...")
        
        first_stage_config = self._get_strategy_config('text2longvideo_firstframe', 'first_stage')
        workflow_path = first_stage_config.get('workflow_path') or \
                       getattr(self.config, 'workflow_path', 'configs/workflow/nova-anime-xl.json')
        
        style = self._get_style(first_stage_config)
        prompt = self.script_segments[0]['visual']
        if style:
            prompt = f"{prompt}\nstyle: {style}".strip()
        
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        
        batch_size = first_stage_config.get('batch_size', 3)
        output_dir = os.path.join(getattr(self.config, 'output_dir', 'output'), 'candidates')
        os.makedirs(output_dir, exist_ok=True)
        
        generated_files = []
        for i in range(batch_size):
            seed = random.randint(1, 999999999999)
            
            merged_params = self._merge_node_manager_params(first_stage_config)
            updates = self.node_manager.generate_updates(
                workflow=workflow,
                updates_config=first_stage_config.get('custom_node_updates', []),
                description=prompt,
                seed=seed,
                **merged_params
            )
            
            files = self.media_generator.generate(
                workflow_path=workflow_path,
                updates=updates,
                output_dir=output_dir,
                file_prefix=f"candidate_{i}"
            )
            generated_files.extend(files)
        
        self.generated_media_paths = generated_files
        return generated_files

    def needs_user_review(self) -> bool:
        """檢查是否需要使用者審核
        
        需要審核的情況：
        1. 候選圖片已生成，影片尚未生成（選擇第一幀）
        2. 最終影片已生成，尚未審核（選擇最終要發布的影片）
        """
        if len(self.generated_media_paths) > 0 and not self._final_video_generated:
            return True
        if self._final_video_generated and not self._videos_reviewed:
            return True
        return False

    def get_review_items(self, max_items: int = 10) -> List[Dict[str, Any]]:
        """Get items for review."""
        return [{'media_path': p} for p in self.generated_media_paths[:max_items]]

    def handle_review_result(self, selected_indices: List[int], output_dir: str, selected_paths: List[str] = None) -> bool:
        """Handle user selection and start full video generation."""
        if not selected_indices and not selected_paths:
            self.logger.warning("No image selected")
            return False
        
        # 優先使用傳入的 selected_paths
        if selected_paths:
            selected_image = selected_paths[0]
        else:
            selected_index = selected_indices[0]
            selected_image = self.generated_media_paths[selected_index]
        self.logger.info(f"User selected: {selected_image}")
        
        try:
            self._generate_full_video_with_firstframe_transitions(selected_image, output_dir)
            return True
        except Exception as e:
            self.logger.error(f"Video generation failed: {e}", exc_info=True)
            raise

    def _generate_first_frame_from_last(
        self, 
        last_frame_path: str, 
        next_visual_desc: str
    ) -> str:
        """
        Transform the last frame into a new first frame using Image-to-Image.
        
        Key concept:
        - Uses scene/style from last frame (via low denoise ~0.5-0.6)
        - Incorporates new scene description
        - Generates a style-consistent image that fits the new script
        
        Args:
            last_frame_path: Path to the last frame of previous segment
            next_visual_desc: Visual description for the next segment
            
        Returns:
            Path to the generated new first frame
        """
        self.logger.info("=" * 60)
        self.logger.info("Generating new first frame via I2I transition")
        self.logger.info("=" * 60)
        
        transition_config = self._get_strategy_config('text2longvideo_firstframe', 'frame_transition')
        
        i2i_workflow_path = transition_config.get('workflow_path', 'configs/workflow/image_to_image.json')
        denoise = transition_config.get('denoise', 0.55)
        style_prompt = transition_config.get('style_continuity_prompt', 
            'maintain visual style, color palette, and artistic consistency')
        
        # Build transition prompt
        transition_prompt = f"{next_visual_desc}, {style_prompt}"
        self.logger.info(f"Transition prompt: {transition_prompt}")
        self.logger.info(f"Denoise: {denoise}")
        
        # Upload last frame
        img_filename = self.media_generator.upload_image(last_frame_path)
        
        # Load workflow
        with open(i2i_workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        
        # Prepare updates
        custom_updates = transition_config.get('custom_node_updates', []).copy()
        custom_updates.append({
            "node_type": "LoadImage",
            "node_index": 0,
            "inputs": {"image": img_filename}
        })
        
        seed = random.randint(1, 999999999999)
        
        merged_params = self._merge_node_manager_params(transition_config)
        # Override denoise
        merged_params['denoise'] = denoise
        
        updates = self.node_manager.generate_updates(
            workflow=workflow,
            updates_config=custom_updates,
            description=transition_prompt,
            seed=seed,
            **merged_params
        )
        
        # Generate new first frame
        output_dir = os.path.join(getattr(self.config, 'output_dir', 'output'), 'first_frames')
        os.makedirs(output_dir, exist_ok=True)
        
        generated_paths = self.media_generator.generate(
            workflow_path=i2i_workflow_path,
            updates=updates,
            output_dir=output_dir,
            file_prefix=f"first_frame_{len(self.script_segments)}"
        )
        
        if not generated_paths:
            raise RuntimeError("Failed to generate new first frame")
        
        new_first_frame = generated_paths[0]
        self.logger.info(f"Generated new first frame: {new_first_frame}")
        return new_first_frame

    def _generate_full_video_with_firstframe_transitions(self, first_frame_path: str, output_dir: str):
        """
        Generate complete long video with first-frame driven transitions.
        """
        self.logger.info(f"Starting full video generation with first-frame transitions")
        
        longvideo_config = self._get_strategy_config('text2longvideo_firstframe', 'longvideo_config')
        video_config = self._get_strategy_config('text2longvideo_firstframe', 'video_generation')
        first_stage_config = self._get_strategy_config('text2longvideo_firstframe', 'first_stage')
        transition_config = self._get_strategy_config('text2longvideo_firstframe', 'frame_transition')
        
        segment_count = longvideo_config.get('segment_count', 5)
        enable_upscale = first_stage_config.get('enable_upscale', False)
        enable_transition = transition_config.get('enabled', True)
        
        # Upscale first frame if enabled
        current_first_frame = first_frame_path
        if enable_upscale:
            self.logger.info("Upscaling first frame...")
            upscaled = self._upscale_images([current_first_frame], output_dir)
            if upscaled:
                current_first_frame = upscaled[0]
        
        # Get video workflow
        video_workflow_path = video_config.get('workflow_path', 'configs/workflow/wan2.2_gguf_i2v.json')
        with open(video_workflow_path, 'r', encoding='utf-8') as f:
            video_workflow = json.load(f)
        
        # Prepare context
        context_data = self.config.get_all_attributes()
        context_data['segment_duration'] = longvideo_config.get('segment_duration', 5)
        context_data['segment_count'] = segment_count
        context_data['style'] = self._get_style(first_stage_config)
        
        frames_dir = os.path.join(output_dir, 'frames')
        video_output_dir = os.path.join(output_dir, 'videos')
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(video_output_dir, exist_ok=True)
        
        all_generated_videos = []
        
        for i in range(segment_count):
            self.logger.info("=" * 60)
            self.logger.info(f"Processing segment {i+1}/{segment_count}")
            self.logger.info("=" * 60)
            
            # Get script for current segment
            if i == 0:
                # First segment uses pre-generated script
                segment_script = self.script_segments[0]
            else:
                # For subsequent segments, script was generated in previous iteration
                # Use the script that was generated based on last frame
                segment_script = self.script_segments[i]
            
            # Generate video from current first frame
            self.logger.info(f"Generating video from first frame: {current_first_frame}")
            img_filename = self.media_generator.upload_image(current_first_frame)
            vid_desc = segment_script.get('visual', '')
            
            seed = random.randint(1, 999999999999)
            
            custom_updates = video_config.get('custom_node_updates', []).copy()
            custom_updates.append({
                "node_type": "LoadImage",
                "node_index": 0,
                "inputs": {"image": img_filename}
            })
            
            merged_params = self._merge_node_manager_params(video_config)
            updates = self.node_manager.generate_updates(
                workflow=video_workflow,
                updates_config=custom_updates,
                description=vid_desc,
                seed=seed,
                use_noise_seed=video_config.get('use_noise_seed', False),
                **merged_params
            )
            
            generated_videos = self.media_generator.generate(
                workflow_path=video_workflow_path,
                updates=updates,
                output_dir=video_output_dir,
                file_prefix=f"segment_{i}"
            )
            
            if not generated_videos:
                raise RuntimeError(f"Failed to generate video for segment {i+1}")
            
            all_generated_videos.append(generated_videos[0])
            self.logger.info(f"Generated video: {generated_videos[0]}")
            
            # Prepare for next segment (if not last)
            if i < segment_count - 1:
                # Extract last frame from current video
                last_frame_path = os.path.join(frames_dir, f"segment_{i}_last_frame.png")
                self.ffmpeg_service.extract_last_frame(
                    video_path=generated_videos[0],
                    output_path=last_frame_path
                )
                self.logger.info(f"Extracted last frame: {last_frame_path}")
                
                # Generate next segment script based on last frame
                self.logger.info(f"Generating script for segment {i+2} based on last frame...")
                next_segment_script = self.script_generator.generate_script_segment(
                    context=context_data,
                    last_frame_path=last_frame_path,
                    segment_index=i + 1
                )
                
                if not next_segment_script:
                    raise RuntimeError(f"Failed to generate script for segment {i+2}")
                
                self.script_segments.append(next_segment_script)
                self.logger.info(f"Next segment visual: {next_segment_script.get('visual', '')}")
                
                if enable_transition:
                    # Use I2I to transform last frame into new first frame
                    # This is the key difference: we use I2I to create a transition frame
                    next_visual = next_segment_script.get('visual', '')
                    self.logger.info("Using I2I to transform last frame into new first frame...")
                    current_first_frame = self._generate_first_frame_from_last(
                        last_frame_path=last_frame_path,
                        next_visual_desc=next_visual
                    )
                    
                    # Optionally upscale the new first frame
                    if enable_upscale:
                        upscaled = self._upscale_images([current_first_frame], output_dir)
                        if upscaled:
                            current_first_frame = upscaled[0]
                else:
                    # Fallback: use last frame directly (no I2I transition)
                    self.logger.info("Transition disabled, using last frame directly as next first frame")
                    current_first_frame = last_frame_path
                    if enable_upscale:
                        upscaled = self._upscale_images([current_first_frame], output_dir)
                        if upscaled:
                            current_first_frame = upscaled[0]
        
        # Concatenate all videos
        self.logger.info(f"Concatenating {len(all_generated_videos)} video segments...")
        concatenated_path = os.path.join(video_output_dir, 'concatenated_video.mp4')
        final_video = self.ffmpeg_service.concat_videos(
            video_paths=all_generated_videos,
            output_path=concatenated_path,
            method='demuxer'
        )
        self.logger.info(f"Concatenated video: {final_video}")
        
        # Add TTS if enabled
        use_tts = longvideo_config.get('use_tts', True)
        if use_tts:
            final_video = self._add_tts_narration(final_video, output_dir, longvideo_config)
        
        self.generated_media_paths = [final_video]
        self._final_video_generated = True
        self._generate_final_article_content()
        
        self.logger.info(f"✅ Complete! Final video: {final_video}")

    def _add_tts_narration(self, video_path: str, output_dir: str, config: Dict) -> str:
        """Add TTS narration to video."""
        self.logger.info("Adding TTS narration...")
        
        tts_voice = config.get('tts_voice', 'en-US-AriaNeural')
        tts_rate = config.get('tts_rate', '+0%')
        
        audio_dir = os.path.join(output_dir, 'audio')
        os.makedirs(audio_dir, exist_ok=True)
        
        audio_files = []
        for i, segment in enumerate(self.script_segments):
            narration = segment.get('narration', '')
            if narration and narration.strip():
                audio_path = os.path.join(audio_dir, f'segment_{i}_narration.mp3')
                try:
                    self.tts_service.generate_speech_sync(
                        text=narration,
                        output_path=audio_path,
                        voice=tts_voice,
                        rate=tts_rate
                    )
                    audio_files.append(audio_path)
                except Exception as e:
                    self.logger.error(f"TTS failed for segment {i}: {e}")
        
        if audio_files:
            concat_audio_path = os.path.join(audio_dir, 'narration.mp3')
            self.ffmpeg_service.concat_audio(audio_files, concat_audio_path)
            
            final_path = os.path.join(output_dir, 'videos', 'final_with_audio.mp4')
            return self.ffmpeg_service.merge_audio_video(
                video_path=video_path,
                audio_path=concat_audio_path,
                output_path=final_path
            )
        
        return video_path

    def _upscale_images(self, image_paths: List[str], output_dir: str) -> List[str]:
        """Upscale images."""
        first_stage_config = self._get_strategy_config('text2longvideo_firstframe', 'first_stage')
        upscale_workflow = first_stage_config.get('upscale_workflow_path', 
            'configs/workflow/Tile Upscaler SDXL.json')
        
        upscaled = []
        for path in image_paths:
            if not any(path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp']):
                upscaled.append(path)
                continue
            
            self.logger.info(f"Upscaling: {path}")
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
                upscaled.extend(generated)
            else:
                upscaled.append(path)
        
        return upscaled

    def _generate_final_article_content(self):
        """Generate article content based on all segments."""
        if not self.script_segments:
            character = getattr(self.config, 'character', '') if self.config else ''
            self.article_content = f"#{character} #AI #video" if character else "#AI #video"
            return
        
        content_parts = []
        if self.config:
            character = getattr(self.config, 'character', '')
            if character:
                content_parts.append(character)
            prompt = getattr(self.config, 'prompt', '')
            if prompt:
                content_parts.append(prompt)
        
        for i, segment in enumerate(self.script_segments[:2]):
            visual = segment.get('visual', '').strip()
            narration = segment.get('narration', '').strip()
            if visual:
                content_parts.append(f"Scene {i+1}: {visual}")
        
        if content_parts and self.vision_manager:
            try:
                combined = '\n\n'.join(content_parts)
                article = self.vision_manager.generate_seo_hashtags(combined)
                
                if '</think>' in article:
                    article = article.split('</think>')[-1].strip()
                
                article = article.replace('"', '').replace('*', '').lower()
                
                if self.config and hasattr(self.config, 'default_hashtags') and self.config.default_hashtags:
                    tags = ' #' + ' #'.join([t.lstrip('#') for t in self.config.default_hashtags])
                    article += tags
                
                article = self.prevent_hashtag_count_too_more(article)
                self.article_content = article
            except Exception as e:
                self.logger.error(f"Article generation failed: {e}")
                character = getattr(self.config, 'character', '') if self.config else ''
                self.article_content = f"#{character} #AI #video" if character else "#AI #video"
        else:
            character = getattr(self.config, 'character', '') if self.config else ''
            self.article_content = f"#{character} #AI #video" if character else "#AI #video"

    def analyze_media_text_match(self, similarity_threshold):
        """Return final video as filter result."""
        if self._final_video_generated and self.generated_media_paths:
            self.filter_results = [{
                'media_path': self.generated_media_paths[0],
                'description': self.script_segments[0].get('visual', '') if self.script_segments else '',
                'similarity': 1.0
            }]
        else:
            self.filter_results = []
        return self

    def should_generate_article_now(self) -> bool:
        return True

    def generate_article_content(self):
        if not self.article_content and self.script_segments:
            self._generate_final_article_content()
        return self

