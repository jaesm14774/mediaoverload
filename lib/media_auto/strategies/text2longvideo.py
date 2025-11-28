"""Text to Long-Form Video Strategy

Long-form video generation workflow:
1. Generate first segment description via LLM (OpenRouter)
2. Generate first image candidates from first segment description
3. User reviews and selects the first image (Discord)
4. Sequential video segment generation (iterative):
   - Generate video for segment N using its description
   - Extract last frame of segment N
   - Analyze last frame and generate description for segment N+1 via LLM
   - Use last frame as conditioning for segment N+1
   - Repeat until all segments are generated
5. Generate TTS narration for all segments
6. Concatenate all video segments
7. Merge audio with video

This strategy follows the Pixelle-Video philosophy of iterative segment-based generation.
Each segment's description is generated dynamically based on the actual last frame of the previous segment,
ensuring visual continuity and story progression that adapts to the actual generated content.
"""

import os
import time
import json
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from lib.comfyui.websockets_api import ComfyUICommunicator
from .generation_base import BaseGenerationStrategy


class Text2LongVideoStrategy(BaseGenerationStrategy):
    """
    Text-to-Long-Video generation strategy.
    
    Workflow:
    1. LLM generates first segment description (based on original prompt)
    2. Generate first image candidates from first segment description
    3. User selects first image via Discord
    4. Sequential segment generation (iterative):
       - Generate video for segment N using its description
       - Extract last frame of segment N
       - Analyze last frame and generate description for segment N+1 via LLM
       - Use last frame as conditioning for segment N+1
       - Repeat until all segments are generated
    5. TTS narration generation (optional)
    6. Video concatenation
    7. Audio-video merge (if TTS enabled)
    
    Key feature: Each segment's description is generated dynamically based on the actual
    last frame of the previous segment, ensuring visual continuity and story progression
    that adapts to the actual generated content.
    
    Configuration (from character YAML):
        longvideo_config:
          segment_count: 5              # Number of video segments
          segment_duration: 5           # Seconds per segment
          use_tts: true                 # Enable TTS narration
          tts_voice: "en-US-AriaNeural" # TTS voice
          tts_rate: "+0%"               # Speech rate
    """
    
    def __init__(self, character_repository=None, vision_manager=None):
        """
        Initialize Text2LongVideo strategy.
        
        Args:
            character_repository: Character database
            vision_manager: External vision manager (optional)
        """
        super().__init__(character_repository, vision_manager)
        
        # Ensure vision manager is available
        if not hasattr(self, 'current_vision_manager'):
             self._initialize_vision_managers()
        
        self.logger = logging.getLogger(__name__)
        
        # Lazy import to avoid circular dependency
        from lib.services.implementations.ffmpeg_service import FFmpegService
        
        # Initialize services
        self.ffmpeg_service = FFmpegService()
        self.tts_service = None  # Initialized on-demand if TTS is enabled
        
        # Stage tracking
        self.current_stage = "initial"  # initial, first_image_review, video_generation, complete
        
        # Data storage
        self.script_segments = []         # List of text segments from LLM
        self.first_stage_images = []      # Candidate images for first segment
        self.selected_first_image = None  # User-selected first image
        self.video_segments = []          # Generated video segment paths
        self.narration_audio = []         # Generated TTS audio paths
        self.final_video_path = None      # Final concatenated video
        
        self.logger.info("Text2LongVideoStrategy initialized")
    
    def _get_longvideo_config(self) -> Dict[str, Any]:
        """
        Get long-video configuration from character config.
        
        Returns:
            Dictionary with longvideo settings
        """
        # Use _get_strategy_config to properly access nested configuration
        strategy_config = self._get_strategy_config('text2longvideo')
        
        # Get longvideo_config from strategy configuration
        longvideo_config = strategy_config.get('longvideo_config', {})
        
        # If longvideo_config exists and is not empty, return it
        if longvideo_config:
            return longvideo_config
        
        # Default configuration
        return {
            'segment_count': 5,
            'segment_duration': 5,
            'use_tts': False,
            'tts_voice': 'en-US-AriaNeural',
            'tts_rate': '+0%',
            'fps': 24,
            'bgm_file': 'default.mp3',  # Default BGM file
            'bgm_volume': 0.2,          # Default BGM volume
        }
    
    def needs_user_review(self) -> bool:
        """
        Check if user review is needed.
        
        Returns:
            True if we have first-stage images and need user selection,
            or if video generation is complete and needs final review
        """
        if self.current_stage == "first_image_review":
            return len(self.first_stage_images) > 0
        
        if self.current_stage == "complete":
            # After video generation, we need user to review the final video
            return self.final_video_path is not None and os.path.exists(self.final_video_path)
        
        return False
    
    def get_review_items(self, max_items: int = 10) -> List[Dict[str, Any]]:
        """
        Get items for user review.
        
        - If in first_image_review stage: return first-stage images
        - If in complete stage: return final video
        
        Args:
            max_items: Maximum number of items (Discord limit: 10)
        
        Returns:
            List of review items with media paths
        """
        review_items = []
        
        if self.current_stage == "first_image_review":
            # Return first-stage images for user review
            for i, image_path in enumerate(self.first_stage_images[:max_items]):
                review_items.append({
                    'index': i,
                    'media_path': image_path,
                    'description': f"First frame candidate {i+1}"
                })
        elif self.current_stage == "complete" and self.final_video_path:
            # Return final video for user review
            if os.path.exists(self.final_video_path):
                review_items.append({
                    'index': 0,
                    'media_path': self.final_video_path,
                    'description': "Final long-form video"
                })
                self.logger.info(f"準備最終影片供審核: {self.final_video_path}")
            else:
                self.logger.warning(f"最終影片不存在: {self.final_video_path}")
        
        return review_items
    
    def handle_review_result(self, selected_indices: List[int], output_dir: str) -> bool:
        """
        Handle user review result.
        
        - If in first_image_review stage: start sequential video generation
        - If in complete stage: video already generated, just confirm selection
        
        Args:
            selected_indices: User-selected media indices
            output_dir: Output directory
        
        Returns:
            True if successful
        """
        if not selected_indices:
            self.logger.error("No media selected by user")
            return False
        
        if self.current_stage == "first_image_review":
            # First stage: user selects first image, start video generation
            selected_index = selected_indices[0]
            
            if selected_index >= len(self.first_stage_images):
                self.logger.error(f"Invalid index: {selected_index}")
                return False
            
            self.selected_first_image = self.first_stage_images[selected_index]
            self.logger.info(f"User selected first image: {self.selected_first_image}")
            
            # Update stage
            self.current_stage = "video_generation"
            
            # Generate all video segments sequentially
            success = self._generate_sequential_video_segments(output_dir)
            
            if success:
                self.current_stage = "complete"
                self.logger.info("Video generation completed, ready for final review")
            
            return success
        
        elif self.current_stage == "complete":
            # Final stage: video already generated, user confirms selection
            self.logger.info(f"User confirmed final video selection (indices: {selected_indices})")
            # Video is already generated, no further action needed
            return True
        
        else:
            self.logger.error(f"Unknown stage for review: {self.current_stage}")
            return False
    
    def should_generate_article_now(self) -> bool:
        """
        Check if article should be generated now.
        
        Returns:
            True only after video generation is complete
        """
        return self.current_stage == "complete"
    
    def generate_description(self):
        """
        Generate first segment description using LLM.
        
        Only generates the first segment description based on the original prompt.
        Subsequent segments will be generated iteratively based on the last frame of previous segments.
        
        Sets self.script_segments to contain only the first segment.
        """
        start_time = time.time()
        config = self._get_longvideo_config()
        segment_duration = config.get('segment_duration', 5)
        
        # Get main character name
        main_character = getattr(self.config, 'character', '')
        character_info = f"\n\nMAIN CHARACTER: The main character in this story is '{main_character}'. " \
                        f"You MUST include '{main_character}' by name in both the narration and visual description " \
                        f"to maintain character consistency throughout the video." if main_character else ""
        
        log_msg = f"開始生成第一個 segment 描述"
        if main_character:
            log_msg += f" (主角: {main_character})"
        self.logger.info(log_msg)
        print(f"[Text2LongVideo] {log_msg}")
        
        system_prompt = f"""
        You are an expert video script writer and director. Create the opening segment of a video story.

        STORY REQUIREMENTS:
        - Create an engaging opening scene that introduces the story.
        - The visual description should be vivid and suitable for video generation.
        - The narration should be natural and engaging, suitable for voiceover.
        {character_info}

        OUTPUT FORMAT:
        Return ONLY a valid JSON object with this EXACT structure (no markdown, no code blocks):
        {{
        "narration": "The exact text for the voiceover to read. Make it natural and engaging, introducing the story.",
        "visual": "A detailed, vivid prompt for an AI video generator describing the opening scene. Include the subject, what's happening, the setting, mood, and visual style."
        }}

        The segment should be approximately {segment_duration} seconds long.
        
        IMPORTANT: 
        - Create an engaging opening that captures attention and sets up the story.
        {f"- You MUST explicitly mention '{main_character}' by name in both narration and visual description." if main_character else ""}
        """
        
        user_prompt = f"Topic: {self.config.prompt}\n\nCreate an engaging opening segment for this video story. Make it captivating and suitable for social media."
        if main_character:
            user_prompt += f"\n\nRemember: The main character is '{main_character}' and must be mentioned by name in the segment."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Use common LLM call with retry
        response = self._call_llm_with_retry(messages, context_name="First segment")
        
        # Parse and validate segment
        first_segment = self._parse_segment_from_response(
            response, 
            context_name="first segment",
            fallback_visual=self.config.prompt
        )
        
        # Initialize script_segments with first segment
        self.script_segments = [first_segment]
        self.descriptions = [first_segment['visual']]
        
        duration = time.time() - start_time
        self.logger.info(f"✅ 第一個 segment 描述生成完成，耗時: {duration:.2f} 秒")
        print(f"[Text2LongVideo] ✅ 第一個 segment 描述生成完成，耗時: {duration:.2f} 秒")
        
        # Log segment details
        self.logger.info(f"Segment 1 旁白: {first_segment['narration']}")
        print(f"[Text2LongVideo] Segment 1 旁白: {first_segment['narration']}")
        self.logger.info(f"Segment 1 視覺描述: {first_segment['visual']}")
        print(f"[Text2LongVideo] Segment 1 視覺描述: {first_segment['visual']}")

        return self
    
    def _extract_json_from_response(self, response: str) -> str:
        """
        Extract JSON string from LLM response, handling markdown code blocks.
        
        Args:
            response: Raw response from LLM
        
        Returns:
            Cleaned JSON string ready for parsing
        """
        response_clean = response.strip()
        
        if response_clean.startswith('```'):
            lines = response_clean.split('\n')
            json_start = None
            json_end = None
            
            for i, line in enumerate(lines):
                if line.strip().startswith('```') and json_start is None:
                    json_start = i + 1
                elif line.strip().startswith('```') and json_start is not None:
                    json_end = i
                    break
            
            if json_start is not None and json_end is not None:
                response_clean = '\n'.join(lines[json_start:json_end])
            elif json_start is not None:
                response_clean = '\n'.join(lines[json_start:])
        
        return response_clean
    
    def _call_llm_with_retry(self, messages: List[Dict[str, str]], context_name: str = "LLM call", max_retries: int = 2) -> str:
        """
        Call LLM with retry mechanism and error handling.
        
        Args:
            messages: List of message dictionaries for LLM
            context_name: Name for logging context (e.g., "first segment", "segment 2")
            max_retries: Maximum number of retry attempts
        
        Returns:
            LLM response string
        
        Raises:
            ValueError: If JSON parsing fails after all retries
            RuntimeError: If LLM call fails after all retries
        """
        for attempt in range(max_retries):
            try:
                response = self.current_vision_manager.text_model.chat_completion(messages=messages)
                self.logger.debug(f"{context_name} LLM response (attempt {attempt + 1}): {response[:500]}...")
                return response
                
            except Exception as e:
                error_msg = (
                    f"{context_name} generation failed after {attempt + 1} attempt(s). "
                    f"Error: {e}"
                )
                self.logger.error(error_msg, exc_info=True)
                
                if attempt == max_retries - 1:
                    raise RuntimeError(error_msg) from e
                continue
        
        raise RuntimeError(f"{context_name} generation failed after all retry attempts")
    
    def _parse_segment_from_response(self, response: str, context_name: str = "segment", fallback_visual: str = "") -> Dict[str, str]:
        """
        Parse segment JSON from LLM response and validate.
        
        Args:
            response: Raw LLM response
            context_name: Name for logging context
            fallback_visual: Fallback visual description if parsing fails
        
        Returns:
            Dictionary with 'narration' and 'visual' keys
        
        Raises:
            ValueError: If JSON parsing fails or segment is invalid
        """
        try:
            response_clean = self._extract_json_from_response(response)
            script_data = json.loads(response_clean)
            
            segment = {
                "narration": script_data.get('narration', '').strip(),
                "visual": script_data.get('visual', fallback_visual).strip()
            }
            
            if not segment['narration'] or not segment['visual']:
                error_msg = (
                    f"LLM did not generate valid {context_name}. "
                    f"Response: {response[:500]}..."
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            return segment
            
        except json.JSONDecodeError as e:
            error_msg = (
                f"Failed to parse JSON from LLM response for {context_name}. "
                f"JSON Error: {e}. Response: {response[:1000]}"
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg) from e
    
    def _generate_next_segment_description(self, last_frame_path: str, segment_index: int) -> Dict[str, str]:
        """
        Generate next segment description based on the last frame of previous segment.
        
        This method uses vision model to analyze the last frame and text model to generate
        the next segment's narration and visual description.
        
        Args:
            last_frame_path: Path to the last frame image of previous segment
            segment_index: Current segment index (0-based, so next segment is segment_index + 1)
        
        Returns:
            Dictionary with 'narration' and 'visual' keys
        """
        config = self._get_longvideo_config()
        segment_duration = config.get('segment_duration', 5)
        segment_count = config.get('segment_count', 5)
        next_segment_num = segment_index + 2
        
        # Get main character name
        main_character = getattr(self.config, 'character', '')
        character_info = f"\n\nMAIN CHARACTER: The main character in this story is '{main_character}'. " \
                        f"You MUST include '{main_character}' by name in both the narration and visual description " \
                        f"to maintain character consistency throughout the video." if main_character else ""
        
        log_msg = f"開始生成 Segment {next_segment_num} 描述 (基於上一個 segment 的最後一幀)"
        if main_character:
            log_msg += f" (主角: {main_character})"
        self.logger.info(log_msg)
        print(f"[Text2LongVideo] {log_msg}")
        
        # Extract image content from last frame
        try:
            self.logger.info(f"Segment {next_segment_num}: 提取最後一幀的圖像內容...")
            print(f"[Text2LongVideo] Segment {next_segment_num}: 提取最後一幀的圖像內容...")
            image_content = self.current_vision_manager.extract_image_content(last_frame_path)
            self.logger.info(f"Segment {next_segment_num}: 圖像內容提取完成: {image_content[:200]}...")
            print(f"[Text2LongVideo] Segment {next_segment_num}: 圖像內容提取完成: {image_content[:200]}...")
        except Exception as e:
            self.logger.warning(f"Segment {next_segment_num}: 提取圖像內容失敗，使用備用描述: {e}")
            print(f"[Text2LongVideo] Segment {next_segment_num}: 提取圖像內容失敗，使用備用描述: {e}")
            image_content = "A scene from the previous video segment"
        
        system_prompt = f"""
        You are an expert video script writer and director. Create the next segment of a video story based on the previous scene.

        CONTEXT:
        - This is segment {next_segment_num} of {segment_count} total segments
        - The previous segment ended with the scene described in the image analysis
        - You need to create a natural continuation that progresses the story
        {character_info}

        STORY REQUIREMENTS:
        - Create an engaging continuation that naturally follows from the previous scene
        - The visual description should be vivid and suitable for video generation
        - The narration should be natural and engaging, suitable for voiceover
        - Ensure smooth visual transition from the previous scene
        - Advance the story: show new actions, situations, or plot developments

        OUTPUT FORMAT:
        Return ONLY a valid JSON object with this EXACT structure (no markdown, no code blocks):
        {{
        "narration": "The exact text for the voiceover to read. Make it natural and engaging, continuing the story.",
        "visual": "A detailed, vivid prompt for an AI video generator describing this next scene. Include the subject, what's happening, the setting, mood, and visual style. This should naturally continue from the previous scene."
        }}

        The segment should be approximately {segment_duration} seconds long.
        
        IMPORTANT: 
        - Create a natural continuation that progresses the story while maintaining visual continuity.
        {f"- You MUST explicitly mention '{main_character}' by name in both narration and visual description to maintain character consistency." if main_character else ""}
        """
        
        user_prompt = f"""
        Original story topic: {self.config.prompt}
        
        Previous scene analysis: {image_content}
        
        Previous segments so far: {len(self.script_segments)} segments completed
        
        Create the next segment that naturally continues the story. The visual should smoothly transition from the previous scene while advancing the narrative."""
        
        if main_character:
            user_prompt += f"\n\nRemember: The main character is '{main_character}' and must be mentioned by name in this segment to maintain character consistency."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Use common LLM call with retry
        response = self._call_llm_with_retry(
            messages, 
            context_name=f"Segment {next_segment_num}"
        )
        
        # Parse and validate segment
        next_segment = self._parse_segment_from_response(
            response,
            context_name=f"segment {next_segment_num}"
        )
        
        self.logger.info(f"✅ Segment {next_segment_num} 描述生成完成")
        print(f"[Text2LongVideo] ✅ Segment {next_segment_num} 描述生成完成")
        
        # Log segment details
        self.logger.info(f"Segment {next_segment_num} 旁白: {next_segment['narration']}")
        print(f"[Text2LongVideo] Segment {next_segment_num} 旁白: {next_segment['narration']}")
        self.logger.info(f"Segment {next_segment_num} 視覺描述: {next_segment['visual']}")
        print(f"[Text2LongVideo] Segment {next_segment_num} 視覺描述: {next_segment['visual']}")
        

        return next_segment
    
    def _generate_single_video_segment(self, workflow_json: Dict, current_frame: str, 
                                       segment: Dict[str, str], segment_num: int, 
                                       segments_dir: str) -> str:
        """
        Generate a single video segment.
        
        Args:
            workflow_json: ComfyUI workflow JSON
            current_frame: Path to conditioning image
            segment: Segment dictionary with 'visual' description
            segment_num: Segment number (1-based)
            segments_dir: Output directory for segments
        
        Returns:
            Path to generated video file
        
        Raises:
            RuntimeError: If video generation fails
        """
        import random
        
        self.logger.info(f"Segment {segment_num}: 上傳條件圖片...")
        # Upload conditioning image
        uploaded_filename = self._upload_image_to_comfyui(current_frame)
        self.logger.info(f"Segment {segment_num}: 條件圖片已上傳: {uploaded_filename}")
        
        # Prepare custom updates for LoadImage
        load_image_indices = self.node_manager.get_node_indices(workflow_json, "LoadImage")
        if not load_image_indices:
            self.logger.warning("Segment {segment_num}: 工作流中未找到 LoadImage 節點！條件設定可能會失敗。")
            custom_updates = []
        else:
            self.logger.debug(f"Segment {segment_num}: 找到 LoadImage 節點，相對索引: {load_image_indices}")
            custom_updates = [
                {
                    "node_type": "LoadImage",
                    "node_index": 0,
                    "inputs": {"image": uploaded_filename}
                }
            ]
        
        # Generate updates with random seed
        seed = random.randint(1, 999999999999)
        self.logger.info(f"Segment {segment_num}: 使用隨機種子 {seed}")
        updates = self.node_manager.generate_updates(
            workflow=workflow_json,
            updates_config=custom_updates,
            description=segment['visual'],
            seed=seed,
            use_noise_seed=True  # Wan2.2 usually uses noise_seed
        )
        
        # Debug logging
        self.logger.debug(f"Segment {segment_num}: 更新配置: {json.dumps(updates, indent=2)}")
        
        # Generate video
        self.logger.info(f"Segment {segment_num}: 開始處理工作流...")
        video_start_time = time.time()
        success, results = self.communicator.process_workflow(
            workflow=workflow_json,
            updates=updates,
            output_path=segments_dir,
            file_name=f"segment_{segment_num}",
            auto_close=False
        )
        video_duration = time.time() - video_start_time
        
        if not success or not results:
            error_msg = f"Segment {segment_num} 生成失敗: success={success}, results={results}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        video_path = results[0]
        self.logger.info(f"Segment {segment_num}: 影片生成完成，耗時: {video_duration:.2f} 秒")
        self.logger.info(f"Segment {segment_num}: 影片路徑: {video_path}")
        
        return video_path
    
    def generate_media(self) -> List[str]:
        """
        Generate first-stage images (candidates for first frame).
        
        User will select one via Discord before video generation begins.
        
        Returns:
            List of generated image paths
        """
        self.logger.info("Generating first-stage images")
        
        # Get text2img workflow configuration
        strategy_config = self._get_strategy_config('text2longvideo', 'first_stage')
        workflow_path = strategy_config.get('workflow_path')
        batch_size = strategy_config.get('batch_size', 4)
        
        if not workflow_path or not os.path.exists(workflow_path):
            raise FileNotFoundError(f"Workflow not found: {workflow_path}")
        
        workflow_json = self._load_workflow(workflow_path)
        visual_description = self.script_segments[0]['visual'] if self.script_segments else self.config.prompt
        
        # Apply style to visual description if configured
        style = strategy_config.get('style') or getattr(self.config, 'style', '')
        if style and style.strip():
            visual_description = f"""{visual_description}\nstyle: {style}""".strip()
            self.logger.info(f"Applied first stage style: {style}")
            print(f"[Text2LongVideo] Applied first stage style: {style}")
        
        # Generate images
        self.communicator = ComfyUICommunicator()
        self.communicator.connect_websocket()
        
        try:
            import random
            self.first_stage_images = []
            
            for i in range(batch_size):
                seed = random.randint(1, 999999999999)
                updates = self.node_manager.generate_updates(
                    workflow=workflow_json,
                    description=visual_description,
                    seed=seed
                )
                
                success, results = self.communicator.process_workflow(
                    workflow=workflow_json,
                    updates=updates,
                    output_path=self.config.output_dir,
                    file_name=f"first_stage_{i}",
                    auto_close=False
                )
                
                if success and results:
                    self.first_stage_images.extend(results)
            
            self.current_stage = "first_image_review"
            self.logger.info(f"Generated {len(self.first_stage_images)} candidate images")
            return self.first_stage_images
            
        except Exception as e:
            self.logger.error(f"First-stage image generation failed: {e}")
            raise
        finally:
            if self.communicator and self.communicator.ws:
                self.communicator.ws.close()
    
    def _generate_sequential_video_segments(self, output_dir: str) -> bool:
        """
        Generate video segments sequentially, each conditioned on previous frame.
        
        This is the core iterative generation logic following Pixelle-Video philosophy.
        Each segment's description is generated based on the last frame of the previous segment.
        
        Args:
            output_dir: Output directory for video segments
        
        Returns:
            True if successful
        """
        config = self._get_longvideo_config()
        segment_count = config.get('segment_count', 5)
        
        self.logger.info(f"Starting sequential video generation: {segment_count} segments")
        self.logger.info("Using iterative generation: each segment description is generated based on previous segment's last frame")
        
        # Get img2video workflow
        strategy_config = self._get_strategy_config('text2longvideo', 'video_generation')
        workflow_path = strategy_config.get('workflow_path')
        
        if not workflow_path or not os.path.exists(workflow_path):
            raise FileNotFoundError(f"Video workflow not found: {workflow_path}")
        
        workflow_json = self._load_workflow(workflow_path)
        self.communicator = ComfyUICommunicator()
        self.communicator.connect_websocket()
        
        # Create segments directory
        segments_dir = os.path.join(output_dir, 'segments')
        os.makedirs(segments_dir, exist_ok=True)
        
        # Start with the selected first image
        current_frame = self.selected_first_image
        
        # Ensure we have at least the first segment
        if not self.script_segments:
            raise RuntimeError("No script segments available. Call generate_description() first.")
        
        try:
            # Generate segments iteratively
            for i in range(segment_count):
                segment_num = i + 1
                segment_start_time = time.time()
                
                # Get main character name for logging
                main_character = getattr(self.config, 'character', '')
                
                self.logger.info(f"=" * 80)
                print(f"[Text2LongVideo] {'=' * 80}")
                self.logger.info(f"開始生成 Segment {segment_num}/{segment_count}")
                print(f"[Text2LongVideo] 開始生成 Segment {segment_num}/{segment_count}")
                if main_character:
                    self.logger.info(f"主角: {main_character}")
                    print(f"[Text2LongVideo] 主角: {main_character}")
                self.logger.info(f"=" * 80)
                print(f"[Text2LongVideo] {'=' * 80}")
                
                # Get current segment (should exist for first segment, generated for subsequent ones)
                if i < len(self.script_segments):
                    segment = self.script_segments[i]
                    self.logger.info(f"Segment {segment_num} 描述:")
                    print(f"[Text2LongVideo] Segment {segment_num} 描述:")
                    self.logger.info(f"  - 旁白: {segment['narration']}")
                    print(f"[Text2LongVideo]   - 旁白: {segment['narration']}")
                    self.logger.info(f"  - 視覺: {segment['visual']}")
                    print(f"[Text2LongVideo]   - 視覺: {segment['visual']}")
                else:
                    # This shouldn't happen if logic is correct, but handle it
                    raise RuntimeError(f"Segment {segment_num} description not found")
                
                try:
                    # Generate video segment
                    self.logger.info(f"Segment {segment_num}: 開始生成影片...")
                    print(f"[Text2LongVideo] Segment {segment_num}: 開始生成影片...")
                    video_path = self._generate_single_video_segment(
                        workflow_json=workflow_json,
                        current_frame=current_frame,
                        segment=segment,
                        segment_num=segment_num,
                        segments_dir=segments_dir
                    )
                    
                    segment_duration = time.time() - segment_start_time
                    self.video_segments.append(video_path)
                    self.logger.info(f"✅ Segment {segment_num} 生成完成!")
                    print(f"[Text2LongVideo] ✅ Segment {segment_num} 生成完成!")
                    self.logger.info(f"  - 影片路徑: {video_path}")
                    print(f"[Text2LongVideo]   - 影片路徑: {video_path}")
                    self.logger.info(f"  - 生成耗時: {segment_duration:.2f} 秒")
                    print(f"[Text2LongVideo]   - 生成耗時: {segment_duration:.2f} 秒")
                    
                    # Extract last frame and generate next segment description (except for last segment)
                    if segment_num < segment_count:
                        frame_path = os.path.join(segments_dir, f'segment_{segment_num}_last_frame.png')
                        self.ffmpeg_service.extract_last_frame(video_path, frame_path)
                        current_frame = frame_path
                        
                        # Generate next segment description based on last frame
                        next_segment_start_time = time.time()
                        next_segment = self._generate_next_segment_description(frame_path, i)
                        next_segment_duration = time.time() - next_segment_start_time
                        self.script_segments.append(next_segment)
                        self.logger.info(f"✅ Segment {segment_num + 1} 描述生成完成!")
                        print(f"[Text2LongVideo] ✅ Segment {segment_num + 1} 描述生成完成!")
                        self.logger.info(f"  - 旁白: {next_segment['narration']}")
                        print(f"[Text2LongVideo]   - 旁白: {next_segment['narration']}")
                        self.logger.info(f"  - 視覺: {next_segment['visual']}")
                        print(f"[Text2LongVideo]   - 視覺: {next_segment['visual']}")
                        self.logger.info(f"  - 描述生成耗時: {next_segment_duration:.2f} 秒")
                        print(f"[Text2LongVideo]   - 描述生成耗時: {next_segment_duration:.2f} 秒")
                    else:
                        self.logger.info(f"Segment {segment_num} 是最後一個 segment，無需生成下一個描述")
                        print(f"[Text2LongVideo] Segment {segment_num} 是最後一個 segment，無需生成下一個描述")
                    
                    total_duration = time.time() - segment_start_time
                    self.logger.info(f"Segment {segment_num} 總耗時: {total_duration:.2f} 秒")
                    print(f"[Text2LongVideo] Segment {segment_num} 總耗時: {total_duration:.2f} 秒")
                    self.logger.info(f"=" * 80)
                    print(f"[Text2LongVideo] {'=' * 80}")
                    
                except Exception as e:
                    error_msg = f"❌ Segment {segment_num} 生成失敗: {e}"
                    self.logger.error(error_msg, exc_info=True)
                    print(f"[Text2LongVideo] {error_msg}")
                    return False
        finally:
            if self.communicator and self.communicator.ws:
                self.communicator.ws.close()
        
        self.logger.info(f"=" * 80)
        print(f"[Text2LongVideo] {'=' * 80}")
        self.logger.info(f"✅ 所有 {len(self.video_segments)} 個 segments 生成成功!")
        print(f"[Text2LongVideo] ✅ 所有 {len(self.video_segments)} 個 segments 生成成功!")
        self.logger.info(f"總共生成 {len(self.script_segments)} 個腳本 segments")
        print(f"[Text2LongVideo] 總共生成 {len(self.script_segments)} 個腳本 segments")
        self.logger.info(f"=" * 80)
        print(f"[Text2LongVideo] {'=' * 80}")
        
        # Generate TTS if enabled
        if config.get('use_tts', False):
            self.logger.info("開始生成 TTS 旁白...")
            print(f"[Text2LongVideo] 開始生成 TTS 旁白...")
            tts_start_time = time.time()
            self._generate_tts_narration(segments_dir, config)
            tts_duration = time.time() - tts_start_time
            self.logger.info(f"TTS 生成完成，耗時: {tts_duration:.2f} 秒")
            print(f"[Text2LongVideo] TTS 生成完成，耗時: {tts_duration:.2f} 秒")
            self.logger.info(f"生成了 {len(self.narration_audio)} 個音訊檔案")
            print(f"[Text2LongVideo] 生成了 {len(self.narration_audio)} 個音訊檔案")
        else:
            self.logger.info("TTS 未啟用，跳過音訊生成")
            print(f"[Text2LongVideo] TTS 未啟用，跳過音訊生成")
        
        # Concatenate video segments
        self.logger.info("開始合併影片 segments...")
        print(f"[Text2LongVideo] 開始合併影片 segments...")
        concat_start_time = time.time()
        self._concatenate_segments(output_dir)
        concat_duration = time.time() - concat_start_time
        self.logger.info(f"影片合併完成，耗時: {concat_duration:.2f} 秒")
        print(f"[Text2LongVideo] 影片合併完成，耗時: {concat_duration:.2f} 秒")
        
        # Finalize audio (Narration + BGM)
        self.logger.info("開始處理音訊 (旁白 + BGM)...")
        print(f"[Text2LongVideo] 開始處理音訊 (旁白 + BGM)...")
        merge_start_time = time.time()
        self._finalize_video_audio(output_dir)
        merge_duration = time.time() - merge_start_time
        self.logger.info(f"音訊處理完成，耗時: {merge_duration:.2f} 秒")
        print(f"[Text2LongVideo] 音訊處理完成，耗時: {merge_duration:.2f} 秒")
        
        # Log final video path
        if self.final_video_path:
            self.logger.info(f"✅ 最終影片路徑: {self.final_video_path}")
            print(f"[Text2LongVideo] ✅ 最終影片路徑: {self.final_video_path}")
            if os.path.exists(self.final_video_path):
                file_size = os.path.getsize(self.final_video_path) / (1024 * 1024)  # MB
                self.logger.info(f"最終影片大小: {file_size:.2f} MB")
                print(f"[Text2LongVideo] 最終影片大小: {file_size:.2f} MB")
            else:
                self.logger.warning(f"最終影片檔案不存在: {self.final_video_path}")
                print(f"[Text2LongVideo] ⚠️ 最終影片檔案不存在: {self.final_video_path}")
        
        return True
    
    def _run_async_safely(self, coro):
        """
        Run async coroutine safely, handling both sync and async contexts.
        
        Args:
            coro: Async coroutine to run
        
        Returns:
            Coroutine result
        """
        try:
            # Check if we're in an existing event loop
            loop = asyncio.get_running_loop()
            # We're in an async context, run in a separate thread with new event loop
            import concurrent.futures
            
            def run_in_thread():
                """Run async code in a new event loop in this thread"""
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()
                
        except RuntimeError:
            # No running event loop, safe to use asyncio.run()
            return asyncio.run(coro)
    
    def _generate_tts_narration(self, output_dir: str, config: Dict[str, Any]):
        """
        Generate TTS narration for all segments.
        
        Args:
            output_dir: Output directory
            config: Long-video configuration
        """
        self.logger.info("Generating TTS narration")
        
        # Lazy import to avoid circular dependency
        from lib.services.implementations.tts_service import TTSService
        
        # Initialize TTS service if not already done
        if self.tts_service is None:
            voice = config.get('tts_voice', 'en-US-AriaNeural')
            self.tts_service = TTSService(default_voice=voice)
        
        narrations = [seg['narration'] for seg in self.script_segments]
        rate = config.get('tts_rate', '+0%')
        audio_dir = os.path.join(output_dir, 'audio')
        
        try:
            # Use async batch generation with safe async runner
            coro = self.tts_service.generate_multiple_speeches(
                narrations,
                audio_dir,
                rate=rate,
                filename_prefix='narration'
            )
            self.narration_audio = self._run_async_safely(coro)
            self.logger.info(f"Generated {len(self.narration_audio)} TTS audio files")
            
        except Exception as e:
            self.logger.error(f"TTS generation failed: {e}")
            self.narration_audio = []
    
    def _concatenate_segments(self, output_dir: str):
        """
        Concatenate all video segments into final video.
        
        Args:
            output_dir: Output directory
        """
        self.logger.info("Concatenating video segments")
        
        output_path = os.path.join(output_dir, 'long_video_no_audio.mp4')
        
        try:
            self.ffmpeg_service.concat_videos(
                self.video_segments,
                output_path,
                method='demuxer'  # Use demuxer method for faster concatenation and to avoid audio stream issues
            )
            
            self.final_video_path = output_path
            self.logger.info(f"Video concatenated: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Video concatenation failed: {e}")
            raise
    
    def _finalize_video_audio(self, output_dir: str):
        """
        Finalize video audio: Merge narration and add BGM.
        
        Args:
            output_dir: Output directory
        """
        config = self._get_longvideo_config()
        current_video = self.final_video_path
        
        if not current_video or not os.path.exists(current_video):
            self.logger.error("No video to finalize audio for")
            return

        # 1. Merge Narration (if any)
        if self.narration_audio and config.get('use_tts', False):
            self.logger.info("Merging narration with video")
            
            try:
                # Concatenate audio using FFmpeg service (handles re-encoding to fix duration issues)
                self.ffmpeg_service.concat_audio(
                    self.narration_audio,
                    combined_audio
                )
                
                # Merge with video
                video_with_narration = os.path.join(output_dir, 'video_with_narration.mp4')
                self.ffmpeg_service.merge_audio_video(
                    current_video,
                    combined_audio,
                    video_with_narration
                )
                current_video = video_with_narration
                self.logger.info(f"Video with narration: {current_video}")
                
            except Exception as e:
                self.logger.error(f"Narration merge failed: {e}")
        
        # 2. Add BGM (if configured)
        bgm_file = config.get('bgm_file')
        if bgm_file:
            # Resolve BGM path
            bgm_path = os.path.abspath(bgm_file)
            if not os.path.exists(bgm_path):
                # Try bgm/ folder in current working directory
                bgm_path = os.path.abspath(os.path.join('bgm', bgm_file))
            
            if os.path.exists(bgm_path):
                self.logger.info(f"Adding BGM: {bgm_path}")
                video_with_bgm = os.path.join(output_dir, 'video_with_bgm.mp4')
                try:
                    self.ffmpeg_service.add_bgm(
                        current_video,
                        bgm_path,
                        video_with_bgm,
                        volume=config.get('bgm_volume', 0.2),
                        loop=True
                    )
                    current_video = video_with_bgm
                    self.logger.info(f"Video with BGM: {current_video}")
                except Exception as e:
                    self.logger.error(f"Failed to add BGM: {e}")
            else:
                self.logger.warning(f"BGM file not found: {bgm_file} (checked {bgm_path})")
        
        # 3. Finalize
        final_output = os.path.join(output_dir, 'long_video_final.mp4')
        if current_video != final_output:
            if os.path.exists(final_output):
                try:
                    os.remove(final_output)
                except OSError:
                    pass
            
            import shutil
            try:
                shutil.copy2(current_video, final_output)
                self.final_video_path = final_output
                self.logger.info(f"Final video finalized: {final_output}")
            except Exception as e:
                self.logger.error(f"Failed to copy final video: {e}")
                # Fallback to current video if copy fails
                self.final_video_path = current_video
        else:
            self.final_video_path = final_output
    
    def analyze_media_text_match(self, similarity_threshold) -> List[Dict[str, Any]]:
        """
        Return media for review/publishing.
        
        For long-video:
        - If in first_image_review stage: return candidate images
        - If complete: return final video
        
        Args:
            similarity_threshold: Not used for long-video
        
        Returns:
            List of media items
        """
        if self.current_stage == "first_image_review":
            # Return first-stage images for user review
            self.filter_results = [
                {'media_path': img, 'score': 1.0, 'index': i}
                for i, img in enumerate(self.first_stage_images)
            ]
            self.logger.info(f"Returning {len(self.filter_results)} first-stage images for review")
            return self.filter_results
        
        elif self.current_stage == "complete":
            # Return final video
            if self.final_video_path and os.path.exists(self.final_video_path):
                self.filter_results = [
                    {'media_path': self.final_video_path, 'score': 1.0}
                ]
                self.logger.info(f"Returning final video for review: {self.final_video_path}")
                return self.filter_results
            else:
                self.logger.warning(f"Final video path not found: {self.final_video_path}")
                self.filter_results = []
                return []
        
        self.logger.warning(f"Unknown stage: {self.current_stage}, returning empty results")
        self.filter_results = []
        return []
    
    def post_process_media(self, media_paths: List[str], output_dir: str) -> List[str]:
        """
        No post-processing needed for long-video.
        
        Args:
            media_paths: Media file paths
            output_dir: Output directory
        
        Returns:
            Original media paths
        """
        return media_paths
    
    def generate_article_content(self) -> str:
        """
        Generate article/post content for the long video.
        
        Returns:
            Generated article text
        """
        self.logger.info("開始生成文章內容...")
        
        if not self.script_segments:
            self.logger.warning("沒有腳本 segments，使用父類方法生成文章")
            return super().generate_article_content()
        
        # Use the vision manager to generate post content
        summary = "\n".join([seg['narration'] for seg in self.script_segments])
        self.logger.debug(f"使用 {len(self.script_segments)} 個 segments 的旁白生成文章")
        
        system_prompt = """你是一個專業的社群媒體內容創作者。為這個影片創建一個吸引人的社群媒體貼文。

要求：
- 使用繁體中文
- 內容要生動有趣，能夠吸引觀眾
- 包含相關的主題標籤（hashtags）
- 長度適中，適合社群媒體發布
- 可以包含表情符號來增加吸引力"""
        
        user_prompt = f"""影片旁白內容摘要：
{summary}

請為這個影片創建一個吸引人的社群媒體貼文，包含主題標籤。"""
        
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.logger.info(f"嘗試生成文章內容 (嘗試 {attempt + 1}/{max_retries})...")
                # Use text_model for article generation (text-based task)
                article = self.current_vision_manager.text_model.chat_completion(messages=messages)
                
                if article and article.strip():
                    self.article_content = article.strip()
                    self.logger.info(f"✅ 文章內容生成成功，長度: {len(self.article_content)} 字元")
                    self.logger.debug(f"文章內容預覽: {self.article_content[:200]}...")
                    return self.article_content
                else:
                    self.logger.warning(f"文章內容為空 (嘗試 {attempt + 1}/{max_retries})")
                    
            except Exception as e:
                error_msg = f"文章生成失敗 (嘗試 {attempt + 1}/{max_retries}): {e}"
                self.logger.error(error_msg, exc_info=True)
                
                if attempt == max_retries - 1:
                    # Last attempt failed, use fallback
                    self.logger.warning("所有重試都失敗，使用備用方法生成文章")
                    try:
                        fallback_content = super().generate_article_content()
                        if fallback_content:
                            self.article_content = fallback_content
                            return fallback_content
                    except Exception as fallback_error:
                        self.logger.error(f"備用方法也失敗: {fallback_error}")
                
                # Wait before retry (exponential backoff)
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    self.logger.info(f"等待 {wait_time} 秒後重試...")
                    time.sleep(wait_time)
                    continue
        
        # If all attempts failed, return a basic fallback
        fallback_hashtags = '#ai #video #unbelievable #world #humor #interesting #funny #creative'
        self.logger.warning(f"文章生成完全失敗，使用預設標籤: {fallback_hashtags}")
        self.article_content = fallback_hashtags
        return self.article_content
