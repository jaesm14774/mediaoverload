import os
import time
import random
from typing import List, Dict, Any, Optional

from lib.media_auto.strategies.base_strategy import ContentStrategy, GenerationConfig
from lib.media_auto.services.script_generator import ScriptGenerator
from lib.media_auto.services.media_generator import MediaGenerator
from lib.media_auto.models.vision.vision_manager import VisionManagerBuilder
from lib.media_auto.core.context import GenerationContext
from lib.comfyui.node_manager import NodeManager
from lib.services.implementations.ffmpeg_service import FFmpegService
from lib.services.implementations.tts_service import TTSService
from utils.logger import setup_logger

class Text2LongVideoStrategy(ContentStrategy):
    """
    Text-to-Long-Video generation strategy.
    Refactored to use composition over inheritance.
    """
    def __init__(self, character_repository=None, vision_manager=None):
        
        self.logger = setup_logger(__name__)
        
        # Initialize Services
        if vision_manager is None:
            vision_manager = VisionManagerBuilder()\
                .with_vision_model('gemini', model_name='gemini-flash-lite-latest')\
                .with_text_model('gemini', model_name='gemini-flash-lite-latest')\
                .build()
        
        self.script_generator = ScriptGenerator(vision_manager)
        self.media_generator = MediaGenerator()
        self.node_manager = NodeManager()
        self.ffmpeg_service = FFmpegService()
        self.tts_service = TTSService()
        
        # State
        self.config = None
        self.script_segments: List[Dict[str, Any]] = []
        self.generated_media_paths: List[str] = []
        self.article_content: str = ""
        self.vision_manager = vision_manager  # Store for article content generation
        self._final_video_generated = False  # 標記最終影片是否已生成
        
    def load_config(self, config: GenerationConfig):
        self.config = config
        
    def generate_description(self):
        """Generates the video script (segments)."""
        if not self.config:
            raise RuntimeError("Config not loaded")
            
        start_time = time.time()
        self.logger.info("Starting script generation...")
        
        # Get longvideo_config with proper merging
        longvideo_config = self._get_strategy_config('text2longvideo', 'longvideo_config')
        
        first_stage_config = self._get_strategy_config('text2longvideo', 'first_stage')
        context_data = self.config.get_all_attributes()
        context_data['segment_duration'] = longvideo_config.get('segment_duration', context_data.get('segment_duration', 5))
        context_data['segment_count'] = longvideo_config.get('segment_count', context_data.get('segment_count', 5))
        context_data['use_tts'] = longvideo_config.get('use_tts', context_data.get('use_tts', True))
        context_data['tts_voice'] = longvideo_config.get('tts_voice', context_data.get('tts_voice', 'en-US-AriaNeural'))
        context_data['tts_rate'] = longvideo_config.get('tts_rate', context_data.get('tts_rate', '+0%'))
        context_data['fps'] = longvideo_config.get('fps', context_data.get('fps', 16))
        style_value = self._get_config_value(first_stage_config, 'style', '')
        context_data['style'] = style_value
        first_segment = self.script_generator.generate_script_segment(context_data)
        self.script_segments = [first_segment]
        self.descriptions = [first_segment['visual']]
        
        self.logger.info(f"Script generation started. First segment: {first_segment['visual'][:50]}...")
        return self

    def generate_media(self):
        """Generates the video media from the script."""
        if not self.script_segments:
            raise RuntimeError("No script segments generated. Call generate_description first.")
            
        # This method is called to generate the FIRST stage media (candidates)
        # In the original flow, this generates candidate images for the first frame.
        
        self.logger.info("Generating candidate images for first segment...")
        
        # Get first_stage config with proper merging
        first_stage_config = self._get_strategy_config('text2longvideo', 'first_stage')
        
        # Get workflow path: first_stage.workflow_path -> config.workflow_path -> default
        workflow_path = first_stage_config.get('workflow_path') or getattr(self.config, 'workflow_path', 'configs/workflow/txt2img.json')
        
        # Get style and add to prompt if needed
        style = self._get_config_value(first_stage_config, 'style', '')
        prompt = self.script_segments[0]['visual']
        if style and style.strip():
            prompt = f"{prompt}\nstyle: {style}".strip()
        
        # Load workflow
        import json
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        
        # Get batch_size from config
        batch_size = first_stage_config.get('batch_size', 3)
        
        # Generate multiple candidates
        generated_files = []
        for i in range(batch_size):
            import random
            seed = random.randint(1, 999999999999)
            
            # Merge additional_params with first_stage_config for node_manager
            merged_params = self._merge_node_manager_params(first_stage_config)
            updates = self.node_manager.generate_updates(
                workflow=workflow,
                updates_config=first_stage_config.get('custom_node_updates', []),
                description=prompt,
                seed=seed,
                **merged_params
            )
            
            # Generate
            output_dir = os.path.join(getattr(self.config, 'output_dir', 'output'), 'candidates')
            files = self.media_generator.generate(
                workflow_path=workflow_path,
                updates=updates,
                output_dir=output_dir,
                file_prefix=f"candidate_{i}"
            )
            generated_files.extend(files)
        
        self.generated_media_paths = generated_files
        return generated_files

    def generate_article_content(self):
        """Generate article content based on all video segments."""
        # If article_content is already generated in _generate_full_video_loop, use it
        # Otherwise generate it now
        if not self.article_content and self.script_segments:
            self._generate_final_article_content()
        return self
    
    def _generate_final_article_content(self):
        """Generate final article content based on all script segments."""
        if not self.script_segments:
            character = getattr(self.config, 'character', '') if self.config else ''
            self.article_content = f"#{character} #AI #text2longvideo" if character else "#AI #text2longvideo"
            return
        
        # Collect all visual descriptions and narrations from segments
        content_parts = []
        
        # Add character name
        if self.config:
            character = getattr(self.config, 'character', '')
            if character:
                content_parts.append(character)
        
        # Add original prompt
        if self.config:
            prompt = getattr(self.config, 'prompt', '')
            if prompt:
                content_parts.append(prompt)
        
        # Add all segment descriptions (visual and narration)
        for i, segment in enumerate(self.script_segments[:2]):
            visual_desc = segment.get('visual', '').strip()
            narration_desc = segment.get('narration', '').strip()
            
            if visual_desc:
                content_parts.append(f"段落 {i+1} 視覺: {visual_desc}")
            if narration_desc:
                content_parts.append(f"段落 {i+1} 旁白: {narration_desc}")
        
        # Generate SEO hashtags using vision manager
        if content_parts and self.vision_manager:
            try:
                combined_content = '\n\n'.join(content_parts)
                article_content = self.vision_manager.generate_seo_hashtags(combined_content)
                
                # Clean up the content
                if '</think>' in article_content:
                    article_content = article_content.split('</think>')[-1].strip()
                
                article_content = article_content.replace('"', '').replace('*', '').lower()
                
                # Add default hashtags if configured
                if self.config and hasattr(self.config, 'default_hashtags') and self.config.default_hashtags:
                    default_tags = ' #' + ' #'.join([tag.lstrip('#') for tag in self.config.default_hashtags])
                    article_content += default_tags
                
                # 防止 hashtag 數量過多
                article_content = self.prevent_hashtag_count_too_more(article_content)
                
                self.article_content = article_content
                self.logger.info(f"已生成最終文章內容（基於 {len(self.script_segments)} 個段落）")
            except Exception as e:
                self.logger.error(f"生成文章內容失敗: {e}")
                # Fallback to simple hashtags
                character = getattr(self.config, 'character', '') if self.config else ''
                self.article_content = f"#{character} #AI #video #text2longvideo" if character else "#AI #video #text2longvideo"
        else:
            # Fallback if no vision manager or content
            character = getattr(self.config, 'character', '') if self.config else ''
            self.article_content = f"#{character} #AI #video #text2longvideo" if character else "#AI #video #text2longvideo"

    def needs_user_review(self) -> bool:
        """檢查是否需要使用者審核
        
        需要審核的情況：
        1. 候選圖片已生成（選擇第一幀）
        2. 最終影片已生成（選擇最終要發布的影片）
        """
        return len(self.generated_media_paths) > 0

    def get_review_items(self, max_items: int = 10) -> List[Dict[str, Any]]:
        return [{'media_path': p} for p in self.generated_media_paths[:max_items]]

    def handle_review_result(self, selected_indices: List[int], output_dir: str) -> bool:
        """
        Handles the user's selection of the first frame.
        Triggers the generation of the rest of the video.
        """
        if not selected_indices:
            self.logger.warning("沒有選擇任何圖片，無法繼續生成影片")
            return False
            
        selected_index = selected_indices[0]
        selected_image = self.generated_media_paths[selected_index]
        self.logger.info(f"User selected image: {selected_image}")
        
        # Start the loop for the rest of the segments
        try:
            self.logger.info("開始調用 _generate_full_video_loop 方法...")
            self._generate_full_video_loop(selected_image, output_dir)
            self.logger.info("_generate_full_video_loop 方法執行完成")
            return True
        except Exception as e:
            self.logger.error(f"_generate_full_video_loop 執行失敗: {e}", exc_info=True)
            raise
        
    def _generate_full_video_loop(self, first_frame_path: str, output_dir: str):
        """
        Internal method to generate the full video sequence.
        """
        self.logger.info(f"開始生成完整影片循環，第一幀路徑: {first_frame_path}, 輸出路徑: {output_dir}")
        
        # Get configs with proper merging
        longvideo_config = self._get_strategy_config('text2longvideo', 'longvideo_config')
        video_generation_config = self._get_strategy_config('text2longvideo', 'video_generation')
        first_stage_config = self._get_strategy_config('text2longvideo', 'first_stage')
        
        # 檢查是否需要 upscale 第一幀
        enable_upscale = first_stage_config.get('enable_upscale', False)
        current_frame = first_frame_path
        if enable_upscale:
            self.logger.info("對第一幀進行放大處理")
            upscaled_frames = self._upscale_images([current_frame], output_dir)
            if upscaled_frames:
                current_frame = upscaled_frames[0]
                self.logger.info(f"第一幀放大完成: {current_frame}")
        segment_count = longvideo_config.get('segment_count', getattr(self.config, 'segment_count', 5))
        self.logger.info(f"設定段落數量: {segment_count}")
        
        # Get video workflow path: video_generation.workflow_path -> default
        video_workflow_path = video_generation_config.get('workflow_path', 'configs/workflow/wan2.2_gguf_i2v.json')
        
        # Load workflow
        import json
        with open(video_workflow_path, 'r', encoding='utf-8') as f:
            video_workflow = json.load(f)
        
        # Get first_stage config to get style
        first_stage_config = self._get_strategy_config('text2longvideo', 'first_stage')
        
        # Get context data
        context_data = self.config.get_all_attributes()
        context_data['segment_duration'] = longvideo_config.get('segment_duration', context_data.get('segment_duration', 5))
        context_data['segment_count'] = segment_count
        # Get style: first_stage -> general -> config.style
        context_data['style'] = self._get_config_value(first_stage_config, 'style', '')
        
        # Create frames directory for storing extracted frames
        frames_dir = os.path.join(output_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        # Collect all generated videos for review
        all_generated_videos = []
        
        self.logger.info(f"開始處理 {segment_count} 個段落...")
        for i in range(segment_count):
            self.logger.info(f"=" * 60)
            self.logger.info(f"Processing segment {i+1}/{segment_count}")
            self.logger.info(f"=" * 60)
            
            # If it's not the first segment, we need to generate the script for it first
            if i > 0:
                self.logger.info(f"為段落 {i+1} 生成腳本...")
                self.logger.info(f"  使用上一段的最後一幀: {current_frame}")
                try:
                    segment_script = self.script_generator.generate_script_segment(
                        context=context_data,
                        last_frame_path=current_frame,
                        segment_index=i
                    )
                    self.logger.info(f"  腳本生成返回結果: {segment_script}")
                    
                    if not segment_script:
                        self.logger.error(f"段落 {i+1} 腳本生成返回 None 或空值")
                        raise RuntimeError(f"段落 {i+1} 腳本生成失敗：返回 None 或空值")
                    
                    self.script_segments.append(segment_script)
                    # 記錄完整的描述
                    visual_desc = segment_script.get('visual', '')
                    narration_desc = segment_script.get('narration', '')
                    
                    self.logger.info(f"段落 {i+1} 腳本生成完成")
                    self.logger.info(f"  視覺描述: {visual_desc}")
                    self.logger.info(f"  旁白內容: {narration_desc}")
                    
                    if not visual_desc:
                        self.logger.warning(f"段落 {i+1} 視覺描述為空")
                    if not narration_desc:
                        self.logger.warning(f"段落 {i+1} 旁白內容為空")
                        
                except Exception as e:
                    self.logger.error(f"段落 {i+1} 腳本生成時發生錯誤: {e}", exc_info=True)
                    raise
            else:
                # Use existing script for first segment
                segment_script = self.script_segments[0] if self.script_segments else {'visual': '', 'narration': ''}
                visual_desc = segment_script.get('visual', '')
                narration_desc = segment_script.get('narration', '')
                self.logger.info(f"使用第一段腳本")
                self.logger.info(f"  視覺描述: {visual_desc}")
                self.logger.info(f"  旁白內容: {narration_desc}")
            
            # Generate video for this segment using I2V
            # Upload current frame (must be an image file)
            self.logger.info(f"上傳當前幀圖片: {current_frame}")
            try:
                image_filename = self.media_generator.upload_image(current_frame)
                self.logger.info(f"圖片上傳成功: {image_filename}")
            except Exception as e:
                self.logger.error(f"圖片上傳失敗: {e}")
                raise RuntimeError(f"無法上傳圖片 {current_frame} 作為段落 {i+1} 的輸入: {e}")
            
            # Get video description
            vid_desc = segment_script.get('visual', '')
            
            # Generate video
            seed = random.randint(1, 999999999999)
            
            # Custom updates for I2V
            custom_updates = video_generation_config.get('custom_node_updates', []).copy()
            custom_updates.append({"node_type": "LoadImage", "node_index": 0, "inputs": {"image": image_filename}})
            
            # Merge additional_params with video_generation_config for node_manager
            merged_params = self._merge_node_manager_params(video_generation_config)
            updates = self.node_manager.generate_updates(
                workflow=video_workflow,
                updates_config=custom_updates,
                description=vid_desc,
                seed=seed,
                use_noise_seed=video_generation_config.get('use_noise_seed', False),
                **merged_params
            )
            self.logger.info(f"updates: {updates}")
            # Generate video
            video_output_dir = os.path.join(output_dir, 'videos')
            os.makedirs(video_output_dir, exist_ok=True)
            self.logger.info(f"開始生成段落 {i+1} 影片...")
            generated_videos = self.media_generator.generate(
                workflow_path=video_workflow_path,
                updates=updates,
                output_dir=video_output_dir,
                file_prefix=f"segment_{i}"
            )
            self.logger.info(f"段落 {i+1} 影片生成完成，共 {len(generated_videos)} 個文件")
            
            # Collect generated videos for review (use first video from each segment for concatenation)
            # Each segment should generate one video, but if multiple are generated, use the first one
            if generated_videos:
                # Use the first video for concatenation (main output)
                all_generated_videos.append(generated_videos[0])
                
                # Use the last video to extract frame for next segment (in case multiple outputs)
                last_video = generated_videos[-1]
                self.logger.info(f"從影片提取最後一幀: {last_video}")
                
                # Extract last frame from the video
                last_frame_path = os.path.join(frames_dir, f"segment_{i}_last_frame.png")
                try:
                    extracted_frame = self.ffmpeg_service.extract_last_frame(
                        video_path=last_video,
                        output_path=last_frame_path
                    )
                    current_frame = extracted_frame
                    self.logger.info(f"最後一幀已提取: {current_frame}")
                    
                    # 如果需要 upscale，對最後一幀進行放大
                    if enable_upscale and i < segment_count - 1:
                        self.logger.info(f"對段落 {i+1} 的最後一幀進行放大處理")
                        upscaled_frames = self._upscale_images([current_frame], output_dir)
                        if upscaled_frames:
                            current_frame = upscaled_frames[0]
                            self.logger.info(f"最後一幀放大完成: {current_frame}")
                except Exception as e:
                    self.logger.error(f"提取最後一幀失敗: {e}")
                    # If extraction fails, we cannot continue to next segment
                    raise RuntimeError(f"無法從影片 {last_video} 提取最後一幀，無法繼續生成下一段: {e}")
            else:
                self.logger.error(f"段落 {i+1} 沒有生成任何影片文件")
                raise RuntimeError(f"段落 {i+1} 影片生成失敗，沒有輸出文件")
        
        # Step 1: Concatenate all video segments into one video
        self.logger.info(f"開始合併 {len(all_generated_videos)} 個影片段落...")
        concatenated_video_path = os.path.join(output_dir, 'videos', 'concatenated_video.mp4')
        os.makedirs(os.path.dirname(concatenated_video_path), exist_ok=True)
        final_video = self.ffmpeg_service.concat_videos(
            video_paths=all_generated_videos,
            output_path=concatenated_video_path,
            method='demuxer'
        )
        self.logger.info(f"影片段落合併完成: {final_video}")
        
        # Step 2: Generate TTS audio for each segment and merge with video if enabled
        use_tts = longvideo_config.get('use_tts', True)
        if use_tts:
            self.logger.info("開始生成 TTS 音訊...")
            tts_voice = longvideo_config.get('tts_voice', 'en-US-AriaNeural')
            tts_rate = longvideo_config.get('tts_rate', '+0%')
            
            # Generate TTS audio for each segment
            audio_dir = os.path.join(output_dir, 'audio')
            os.makedirs(audio_dir, exist_ok=True)
            audio_files = []
            
            for i, segment_script in enumerate(self.script_segments):
                narration_text = segment_script.get('narration', '')
                if narration_text and narration_text.strip():
                    audio_path = os.path.join(audio_dir, f'segment_{i}_narration.mp3')
                    try:
                        self.tts_service.generate_speech_sync(
                            text=narration_text,
                            output_path=audio_path,
                            voice=tts_voice,
                            rate=tts_rate
                        )
                        audio_files.append(audio_path)
                        self.logger.info(f"段落 {i+1} TTS 音訊生成完成: {audio_path}")
                    except Exception as e:
                        self.logger.error(f"段落 {i+1} TTS 音訊生成失敗: {e}")
                        # Continue even if TTS fails for one segment
                        continue
            
            if audio_files:
                # Concatenate all audio files
                self.logger.info(f"開始合併 {len(audio_files)} 個 TTS 音訊文件...")
                concatenated_audio_path = os.path.join(audio_dir, 'concatenated_audio.mp3')
                final_audio = self.ffmpeg_service.concat_audio(
                    audio_paths=audio_files,
                    output_path=concatenated_audio_path
                )
                self.logger.info(f"TTS 音訊合併完成: {final_audio}")
                
                # Merge audio with video
                self.logger.info("開始將 TTS 音訊與影片合併...")
                final_video_with_audio = os.path.join(output_dir, 'videos', 'final_video_with_audio.mp4')
                final_video = self.ffmpeg_service.merge_audio_video(
                    video_path=final_video,
                    audio_path=final_audio,
                    output_path=final_video_with_audio
                )
                self.logger.info(f"最終影片（含 TTS 音訊）生成完成: {final_video}")
            else:
                self.logger.warning("沒有生成任何 TTS 音訊文件，使用無音訊的合併影片")
        
        # Update generated_media_paths with final merged video (single file)
        self.generated_media_paths = [final_video]
        self._final_video_generated = True
        self.logger.info(f"所有 {segment_count} 個段落已生成並合併完成，最終影片: {final_video}")
        
        # Generate article content based on all segments
        self._generate_final_article_content()

    def analyze_media_text_match(self, similarity_threshold):
        """分析媒體與文本的匹配度
        
        如果最終影片已生成，返回最終影片的 filter_results
        否則返回空列表（因為候選圖片階段不需要分析）
        """
        if self._final_video_generated and self.generated_media_paths:
            # 最終影片已生成，返回最終影片的 filter_results
            self.filter_results = []
            for video_path in self.generated_media_paths:
                # 使用第一個段落的視覺描述作為描述
                description = ''
                if self.script_segments:
                    description = self.script_segments[0].get('visual', '')
                
                self.filter_results.append({
                    'media_path': video_path,
                    'description': description,
                    'similarity': 1.0  # 最終影片直接使用 1.0
                })
        else:
            # 候選圖片階段，不需要分析（直接使用 generated_media_paths）
            self.filter_results = []
        
        return self
    
    def should_generate_article_now(self) -> bool:
        return True
    
    def _upscale_images(self, image_paths: List[str], output_dir: str) -> List[str]:
        """放大圖片
        
        Args:
            image_paths: 圖片路徑列表
            output_dir: 輸出路徑
            
        Returns:
            放大後的圖片路徑列表
        """
        first_stage_config = self._get_strategy_config('text2longvideo', 'first_stage')
        upscale_workflow = first_stage_config.get('upscale_workflow_path', 'configs/workflow/Tile Upscaler SDXL.json')
        upscaled_paths = []
        
        for path in image_paths:
            if not any(path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp']):
                upscaled_paths.append(path)
                continue
                
            self.logger.info(f"放大圖片: {path}")
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
        
        self.logger.info(f"✅ 圖片放大完成，共 {len(upscaled_paths)} 張")
        return upscaled_paths
