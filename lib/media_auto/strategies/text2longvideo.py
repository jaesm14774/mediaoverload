import os
import time
import random
import json
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
    def __init__(self, character_data_service=None, vision_manager=None):
        
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
        self._final_video_generated = False
        self._videos_reviewed = False
        
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
        segment_count = longvideo_config.get('segment_count', context_data.get('segment_count', 5))
        context_data['segment_count'] = segment_count
        context_data['use_tts'] = longvideo_config.get('use_tts', context_data.get('use_tts', True))
        context_data['tts_voice'] = longvideo_config.get('tts_voice', context_data.get('tts_voice', 'en-US-AriaNeural'))
        context_data['tts_rate'] = longvideo_config.get('tts_rate', context_data.get('tts_rate', '+0%'))
        context_data['fps'] = longvideo_config.get('fps', context_data.get('fps', 16))
        style_value = self._get_style(first_stage_config)
        context_data['style'] = style_value
        
        # 生成第一段腳本
        self.logger.info(f"生成第一段腳本...")
        first_segment = self.script_generator.generate_script_segment(
            context=context_data,
            segment_index=0
        )
        self.script_segments = [first_segment]
        self.descriptions = [first_segment['visual']]
        self.logger.info(f"第一段腳本生成完成: {first_segment['visual'][:50]}...")
        
        # 預先生成所有後續段落腳本
        self.logger.info(f"開始預先生成所有 {segment_count} 個段落腳本...")
        for i in range(1, segment_count):
            self.logger.info(f"生成段落 {i+1}/{segment_count} 腳本...")
            previous_segment = self.script_segments[i-1]
            next_segment = self.script_generator.generate_script_segment(
                context=context_data,
                previous_segment=previous_segment,
                segment_index=i
            )
            self.script_segments.append(next_segment)
            self.descriptions.append(next_segment['visual'])
            self.logger.info(f"段落 {i+1} 腳本生成完成: {next_segment['visual'][:50]}...")
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"所有 {segment_count} 個段落腳本生成完成，耗時 {elapsed_time:.2f} 秒")
        return self

    def generate_media(self):
        """Generates the video media from the script."""
        if not self.script_segments:
            raise RuntimeError("No script segments generated. Call generate_description first.")
            
        longvideo_config = self._get_strategy_config('text2longvideo', 'longvideo_config')
        skip_candidate_stage = longvideo_config.get('skip_candidate_stage', False)
        
        if skip_candidate_stage:
            self.logger.info("跳過候選圖片生成，直接生成完整影片")
            output_dir = getattr(self.config, 'output_dir', 'output')
            self._generate_full_video_direct(output_dir)
            return self.generated_media_paths
        
        self.logger.info("Generating candidate images for first segment...")
        
        # Get first_stage config with proper merging
        first_stage_config = self._get_strategy_config('text2longvideo', 'first_stage')
        
        # Get workflow path: first_stage.workflow_path -> config.workflow_path -> default
        workflow_path = first_stage_config.get('workflow_path') or getattr(self.config, 'workflow_path', 'configs/workflow/txt2img.json')
        
        # Get style and add to prompt if needed
        style = self._get_style(first_stage_config)
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
        1. 候選圖片已生成，影片尚未生成（選擇第一幀）
        2. 最終影片已生成，尚未審核（選擇最終要發布的影片）
        """
        if len(self.generated_media_paths) > 0 and not self._final_video_generated:
            return True
        if self._final_video_generated and not self._videos_reviewed:
            return True
        return False

    def get_review_items(self, max_items: int = 10) -> List[Dict[str, Any]]:
        return [{'media_path': p} for p in self.generated_media_paths[:max_items]]

    def handle_review_result(self, selected_indices: List[int], output_dir: str, selected_paths: List[str] = None) -> bool:
        """
        Handles the user's selection of the first frame.
        Triggers the generation of the rest of the video.
        """
        if not selected_indices and not selected_paths:
            self.logger.warning("沒有選擇任何圖片，無法繼續生成影片")
            return False
        
        # 優先使用傳入的 selected_paths
        if selected_paths:
            selected_image = selected_paths[0]
        else:
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
        # Get style: support weights or single value
        context_data['style'] = self._get_style(first_stage_config)
        
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
            
            # 使用預先生成的腳本
            if len(self.script_segments) <= i:
                self.logger.error(f"段落 {i+1} 腳本不存在，這不應該發生。請確保已調用 generate_description() 預先生成所有腳本")
                raise RuntimeError(f"段落 {i+1} 腳本不存在，請確保已調用 generate_description() 預先生成所有腳本")
            
            segment_script = self.script_segments[i]
            visual_desc = segment_script.get('visual', '')
            narration_desc = segment_script.get('narration', '')
            self.logger.info(f"使用預先生成的段落 {i+1} 腳本")
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
                except Exception as e:
                    self.logger.error(f"提取最後一幀失敗: {e}")
                    # If extraction fails, we cannot continue to next segment
                    raise RuntimeError(f"無法從影片 {last_video} 提取最後一幀，無法繼續生成下一段: {e}")
                
                # 如果不是最後一段，使用 I2I 重新生成高質量的第一幀
                if i < segment_count - 1:
                    # 使用預先生成的下一段腳本
                    if len(self.script_segments) > i + 1:
                        next_segment_script = self.script_segments[i + 1]
                        next_visual_desc = next_segment_script.get('visual', '')
                        
                        # 使用 I2I 重新生成高質量的第一幀（避免崩壞並推進劇情）
                        try:
                            self.logger.info(f"使用 I2I 重新生成段落 {i+2} 的高質量第一幀...")
                            current_frame = self._regenerate_first_frame_with_i2i(
                                last_frame_path=current_frame,
                                next_visual_desc=next_visual_desc,
                                output_dir=output_dir,
                                segment_index=i + 1
                            )
                            self.logger.info(f"✅ 段落 {i+2} 的第一幀已重新生成: {current_frame}")
                        except Exception as e:
                            self.logger.warning(f"I2I 重新生成第一幀失敗，使用原始最後一幀: {e}")
                            # 如果 I2I 失敗，繼續使用最後一幀（降級處理）
                            # 如果需要 upscale，對最後一幀進行放大
                            if enable_upscale:
                                self.logger.info(f"對段落 {i+1} 的最後一幀進行放大處理")
                                upscaled_frames = self._upscale_images([current_frame], output_dir)
                                if upscaled_frames:
                                    current_frame = upscaled_frames[0]
                                    self.logger.info(f"最後一幀放大完成: {current_frame}")
                    else:
                        self.logger.error(f"段落 {i+2} 腳本不存在，這不應該發生")
                        raise RuntimeError(f"段落 {i+2} 腳本不存在，請確保已調用 generate_description() 預先生成所有腳本")
                else:
                    # 最後一段，如果需要 upscale，對最後一幀進行放大
                    if enable_upscale:
                        self.logger.info(f"對段落 {i+1} 的最後一幀進行放大處理")
                        upscaled_frames = self._upscale_images([current_frame], output_dir)
                        if upscaled_frames:
                            current_frame = upscaled_frames[0]
                            self.logger.info(f"最後一幀放大完成: {current_frame}")
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
            self.logger.info("=" * 60)
            self.logger.info("開始生成 TTS 音訊...")
            self.logger.info("=" * 60)
            tts_voice = longvideo_config.get('tts_voice', 'en-US-AriaNeural')
            tts_rate = longvideo_config.get('tts_rate', '+0%')
            
            # Generate TTS audio for each segment
            audio_dir = os.path.join(output_dir, 'audio')
            os.makedirs(audio_dir, exist_ok=True)
            audio_files = []
            
            # 確保 script_segments 數量正確
            if len(self.script_segments) != segment_count:
                self.logger.warning(f"腳本段落數量 ({len(self.script_segments)}) 與設定段落數量 ({segment_count}) 不一致")
            
            for i, segment_script in enumerate(self.script_segments):
                narration_text = segment_script.get('narration', '')
                if narration_text and narration_text.strip():
                    audio_path = os.path.join(audio_dir, f'segment_{i}_narration.mp3')
                    self.logger.info(f"生成段落 {i+1} TTS 音訊...")
                    self.logger.info(f"  旁白內容: {narration_text[:100]}...")
                    self.logger.info(f"  語音: {tts_voice}, 語速: {tts_rate}")
                    self.logger.info(f"  輸出路徑: {audio_path}")
                    
                    try:
                        # 確保目錄存在
                        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
                        
                        # 生成 TTS
                        result_path = self.tts_service.generate_speech_sync(
                            text=narration_text,
                            output_path=audio_path,
                            voice=tts_voice,
                            rate=tts_rate
                        )
                        
                        # 驗證文件
                        if result_path and os.path.exists(result_path):
                            file_size = os.path.getsize(result_path)
                            if file_size > 0:
                                audio_files.append(result_path)
                                self.logger.info(f"✅ 段落 {i+1} TTS 音訊生成完成: {result_path} ({file_size} bytes)")
                            else:
                                self.logger.error(f"❌ 段落 {i+1} TTS 音訊文件為空: {result_path}")
                        elif os.path.exists(audio_path):
                            file_size = os.path.getsize(audio_path)
                            if file_size > 0:
                                audio_files.append(audio_path)
                                self.logger.info(f"✅ 段落 {i+1} TTS 音訊生成完成: {audio_path} ({file_size} bytes)")
                            else:
                                self.logger.error(f"❌ 段落 {i+1} TTS 音訊文件為空: {audio_path}")
                        else:
                            self.logger.error(f"❌ 段落 {i+1} TTS 音訊文件不存在: {audio_path}")
                    except Exception as e:
                        self.logger.error(f"❌ 段落 {i+1} TTS 音訊生成失敗: {e}", exc_info=True)
                        self.logger.error(f"  錯誤類型: {type(e).__name__}")
                        self.logger.error(f"  錯誤訊息: {str(e)}")
                        # Continue even if TTS fails for one segment
                        continue
                else:
                    self.logger.warning(f"⚠️ 段落 {i+1} 沒有旁白內容，跳過 TTS 生成")
                    self.logger.debug(f"  腳本內容: {segment_script}")
            
            if audio_files:
                # Concatenate all audio files
                self.logger.info(f"開始合併 {len(audio_files)} 個 TTS 音訊文件...")
                concatenated_audio_path = os.path.join(audio_dir, 'concatenated_audio.mp3')
                try:
                    final_audio = self.ffmpeg_service.concat_audio(
                        audio_paths=audio_files,
                        output_path=concatenated_audio_path
                    )
                    if os.path.exists(final_audio) and os.path.getsize(final_audio) > 0:
                        self.logger.info(f"✅ TTS 音訊合併完成: {final_audio}")
                        
                        # Merge audio with video
                        self.logger.info("開始將 TTS 音訊與影片合併...")
                        final_video_with_audio = os.path.join(output_dir, 'videos', 'final_video_with_audio.mp4')
                        final_video = self.ffmpeg_service.merge_audio_video(
                            video_path=final_video,
                            audio_path=final_audio,
                            output_path=final_video_with_audio
                        )
                        self.logger.info(f"✅ 最終影片（含 TTS 音訊）生成完成: {final_video}")
                    else:
                        self.logger.error(f"TTS 音訊合併失敗，文件不存在或為空: {final_audio}")
                        self.logger.warning("使用無音訊的合併影片")
                except Exception as e:
                    self.logger.error(f"TTS 音訊合併或與影片合併失敗: {e}", exc_info=True)
                    self.logger.warning("使用無音訊的合併影片")
            else:
                self.logger.warning("沒有生成任何 TTS 音訊文件，使用無音訊的合併影片")
        
        # Update generated_media_paths with final merged video (single file)
        self.generated_media_paths = [final_video]
        self._final_video_generated = True
        self._videos_reviewed = False
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
    
    def _regenerate_first_frame_with_i2i(
        self,
        last_frame_path: str,
        next_visual_desc: str,
        output_dir: str,
        segment_index: int
    ) -> str:
        """
        使用 I2I 重新生成高質量的第一幀
        
        關鍵概念：
        - 使用 nova-anime-xl（有 kirby 知識）來重新生成高質量的第一幀
        - 避免直接使用 wan2.2 生成的最後一幀（可能崩壞）
        - 通過 I2I 轉換，保持視覺連續性的同時推進劇情
        
        Args:
            last_frame_path: 上一段的最後一幀路徑
            next_visual_desc: 下一段的視覺描述
            output_dir: 輸出路徑
            segment_index: 段落索引
            
        Returns:
            重新生成的第一幀路徑
        """
        self.logger.info("=" * 60)
        self.logger.info(f"使用 I2I 重新生成段落 {segment_index + 1} 的高質量第一幀")
        self.logger.info("=" * 60)
        
        # 獲取 frame_transition 配置
        frame_transition_config = self._get_strategy_config('text2longvideo', 'frame_transition')
        if not frame_transition_config or not frame_transition_config.get('enabled', True):
            self.logger.info("I2I 轉換已禁用，使用原始最後一幀")
            return last_frame_path
        
        # 獲取 I2I workflow 路徑（使用 nova-anime-xl v140 的 I2I workflow，與第一幀一致）
        i2i_workflow_path = frame_transition_config.get('workflow_path', 'configs/workflow/image_to_image.json')
        denoise = frame_transition_config.get('denoise', 0.55)
        style_prompt = frame_transition_config.get('style_continuity_prompt', 'maintain visual style, color palette, and artistic consistency')
        
        # 獲取前一段的腳本上下文（用於故事連貫性）
        previous_segment = None
        if segment_index > 0 and len(self.script_segments) > segment_index - 1:
            previous_segment = self.script_segments[segment_index - 1]
        
        # 構建轉換提示詞，加入前一段的敘事上下文
        if previous_segment:
            previous_narration = previous_segment.get('narration', '')
            # 限制長度避免提示詞過長
            story_context = previous_narration[:150] + "..." if len(previous_narration) > 150 else previous_narration
            transition_prompt = f"Story progression from: {story_context}. {next_visual_desc}, {style_prompt}"
            self.logger.info(f"轉換提示詞（含故事上下文）: {transition_prompt}")
        else:
            transition_prompt = f"{next_visual_desc}, {style_prompt}"
            self.logger.info(f"轉換提示詞: {transition_prompt}")
        self.logger.info(f"Denoise: {denoise}")
        
        # 上傳最後一幀
        try:
            img_filename = self.media_generator.upload_image(last_frame_path)
            self.logger.info(f"圖片上傳成功: {img_filename}")
        except Exception as e:
            self.logger.error(f"圖片上傳失敗: {e}")
            raise RuntimeError(f"無法上傳圖片 {last_frame_path}: {e}")
        
        # 載入 workflow
        try:
            with open(i2i_workflow_path, 'r', encoding='utf-8') as f:
                workflow = json.load(f)
        except Exception as e:
            self.logger.error(f"無法載入 I2I workflow: {e}")
            raise RuntimeError(f"無法載入 I2I workflow {i2i_workflow_path}: {e}")
        
        # 準備更新
        custom_updates = frame_transition_config.get('custom_node_updates', []).copy()
        custom_updates.append({
            "node_type": "LoadImage",
            "node_index": 0,
            "inputs": {"image": img_filename}
        })
        
        seed = random.randint(1, 999999999999)
        
        # 合併參數
        merged_params = self._merge_node_manager_params(frame_transition_config)
        merged_params['denoise'] = denoise
        
        # 生成更新
        updates = self.node_manager.generate_updates(
            workflow=workflow,
            updates_config=custom_updates,
            description=transition_prompt,
            seed=seed,
            **merged_params
        )
        
        # 生成新的第一幀
        first_frames_dir = os.path.join(output_dir, 'first_frames')
        os.makedirs(first_frames_dir, exist_ok=True)
        
        try:
            generated_paths = self.media_generator.generate(
                workflow_path=i2i_workflow_path,
                updates=updates,
                output_dir=first_frames_dir,
                file_prefix=f"first_frame_segment_{segment_index}"
            )
            
            if not generated_paths:
                raise RuntimeError("I2I 生成沒有返回任何文件")
            
            new_first_frame = generated_paths[0]
            self.logger.info(f"✅ 新的第一幀已生成: {new_first_frame}")
            
            # 如果需要 upscale
            first_stage_config = self._get_strategy_config('text2longvideo', 'first_stage')
            enable_upscale = first_stage_config.get('enable_upscale', False)
            if enable_upscale:
                self.logger.info("對重新生成的第一幀進行放大處理")
                upscaled_frames = self._upscale_images([new_first_frame], output_dir)
                if upscaled_frames:
                    new_first_frame = upscaled_frames[0]
                    self.logger.info(f"第一幀放大完成: {new_first_frame}")
            
            return new_first_frame
            
        except Exception as e:
            self.logger.error(f"I2I 生成失敗: {e}", exc_info=True)
            raise RuntimeError(f"無法使用 I2I 重新生成第一幀: {e}")

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
    
    def _generate_full_video_direct(self, output_dir: str):
        """直接生成完整影片，不經過候選圖片階段"""
        self.logger.info("開始直接生成完整影片（無候選圖片）")
        
        longvideo_config = self._get_strategy_config('text2longvideo', 'longvideo_config')
        video_generation_config = self._get_strategy_config('text2longvideo', 'video_generation')
        first_stage_config = self._get_strategy_config('text2longvideo', 'first_stage')
        
        segment_count = longvideo_config.get('segment_count', getattr(self.config, 'segment_count', 5))
        
        # 先生成第一幀圖片
        self.logger.info("生成第一幀圖片...")
        workflow_path = first_stage_config.get('workflow_path') or getattr(self.config, 'workflow_path', 'configs/workflow/txt2img.json')
        style = self._get_style(first_stage_config)
        prompt = self.script_segments[0]['visual']
        if style and style.strip():
            prompt = f"{prompt}\nstyle: {style}".strip()
        
        import json
        with open(workflow_path, 'r', encoding='utf-8') as f:
            workflow = json.load(f)
        
        seed = random.randint(1, 999999999999)
        merged_params = self._merge_node_manager_params(first_stage_config)
        updates = self.node_manager.generate_updates(
            workflow=workflow,
            updates_config=first_stage_config.get('custom_node_updates', []),
            description=prompt,
            seed=seed,
            **merged_params
        )
        
        temp_dir = os.path.join(output_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        first_frame_files = self.media_generator.generate(
            workflow_path=workflow_path,
            updates=updates,
            output_dir=temp_dir,
            file_prefix="first_frame"
        )
        
        if not first_frame_files:
            raise RuntimeError("第一幀圖片生成失敗")
        
        first_frame = first_frame_files[0]
        self.logger.info(f"第一幀圖片生成完成: {first_frame}")
        
        self._generate_full_video_loop(first_frame, output_dir)
        
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            self.logger.info("清理臨時檔案完成")
