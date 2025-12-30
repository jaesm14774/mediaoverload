import json
import time
from typing import Dict, List, Any, Optional
from lib.media_auto.models.vision.vision_manager import VisionContentManager

class ScriptGenerator:
    """
    Service for generating video scripts and content using LLMs.
    Focuses on narrative coherence, character consistency, and visual continuity.
    """
    def __init__(self, vision_manager: VisionContentManager):
        self.vision_manager = vision_manager

    def generate_script_segment(self, 
                              context: Dict[str, Any], 
                              previous_segment: Optional[Dict] = None, 
                              last_frame_path: Optional[str] = None,
                              segment_index: int = 0) -> Dict[str, Any]:
        """
        Orchestrates the generation of a script segment ensuring continuity.
        """
        # 1. Analyze Context & Narrative Stage
        segment_count = context.get('segment_count', 5)
        # 1-based index for prompting (easier for LLM to understand)
        current_step = segment_index + 1 
        
        narrative_stage = self._determine_narrative_stage(current_step, segment_count)
        
        # 2. Get Visual Context - 優先使用 previous_segment 的完整上下文
        visual_context = ""
        narration_context = ""
        if previous_segment:
            # 優先使用前一段的完整腳本上下文
            visual_context = previous_segment.get('visual', '')
            narration_context = previous_segment.get('narration', '')
        elif last_frame_path and segment_index > 0:
            # 降級：僅使用視覺描述
            try:
                visual_context = self.vision_manager.extract_image_content(last_frame_path)
            except Exception as e:
                print(f"Warning: Failed to extract image content: {e}")
                visual_context = "Previous scene context unavailable."

        # 3. Build Prompts
        system_prompt = self._build_system_prompt(context)
        user_prompt = self._build_user_prompt(
            context=context,
            visual_context=visual_context,
            narration_context=narration_context,
            narrative_stage=narrative_stage,
            current_step=current_step,
            total_steps=segment_count
        )

        # 4. Call LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self._call_llm_with_retry(messages, context_name=f"Segment {current_step}")

    def _determine_narrative_stage(self, current: int, total: int) -> str:
        """Determine the role of this segment in the story arc."""
        if current == 1:
            return "OPENING (Hook the audience, introduce character/setting)"
        elif current == total:
            return "CONCLUSION (Resolve the story, meaningful ending)"
        else:
            return "DEVELOPMENT (Advance the plot, maintain momentum, bridge previous scene to next)"

    def _build_system_prompt(self, context: Dict[str, Any]) -> str:
        """
        Constructs the core system instruction focusing on format and character rules.
        """
        main_character = context.get('character', '')
        segment_duration = context.get('segment_duration', 6)

        # Base instructions
        prompt = f"""
        You are an expert Director and Screenwriter for AI-generated videos. 
        Your goal is to write a cohesive, engaging video script segment that maintains strict continuity and provides SUBSTANTIAL CONTENT.

        ### CRITICAL REQUIREMENTS FOR {segment_duration}-SECOND SEGMENTS:
        1. **Substantial Action Content**: Each segment MUST contain meaningful actions and visual changes that can sustain {segment_duration} seconds of video. 
           - AVOID repetitive small movements like head nodding, slight body swaying, or minor facial expressions (these are only 1-3 seconds)
           - INCLUDE significant actions: walking, running, jumping, interacting with objects, changing locations, performing tasks, etc.
           - Each segment should show clear progression and movement
        
        2. **Visual Richness**: The visual description must include:
           - Clear action sequences that unfold over time
           - Background/environment changes or camera movement
           - Multiple visual elements that create depth and interest
           - Dynamic elements that can animate naturally over {segment_duration} seconds
        
        3. **Character Consistency**: The Main Character is '{main_character}'. You MUST explicitly name them in BOTH the 'narration' and 'visual' prompt every time to ensure the AI video generator keeps the same person.
        
        4. **Visual Continuity**: The 'visual' description must logically evolve from the previous frame (if provided). Do not jump cut randomly.
        
        5. **Engagement**: The narration should be conversational, pacing well for a {segment_duration}-second clip and matching the visual action.

        ### FORBIDDEN CONTENT:
        - Simple repetitive movements (head nodding, slight swaying, minor gestures)
        - Static poses with minimal movement
        - Actions that cannot sustain {segment_duration} seconds of meaningful video
        
        ### OUTPUT FORMAT:
        Return ONLY a valid JSON object with this EXACT structure:
        {{
            "narration": "Natural, engaging voiceover text suitable for social media that matches the visual action.",
            "visual": "Highly detailed AI image prompt with SUBSTANTIAL ACTION CONTENT. Format: [Subject + Meaningful Action Sequence], [Setting/Background with movement potential], [Lighting/Mood], [Style]. MUST include '{main_character}' performing actions that can sustain {segment_duration} seconds."
        }}
        """
        return prompt

    def _build_user_prompt(self, context: Dict[str, Any], visual_context: str, 
                          narration_context: str = "",
                          narrative_stage: str = "",
                          current_step: int = 1, 
                          total_steps: int = 5) -> str:
        """
        Constructs the dynamic user prompt based on the story state.
        """
        topic = context.get('prompt', '')
        style = context.get('style', '')
        main_character = context.get('character', '')

        # Part 1: Task Definition
        prompt = f"""
        ### TASK: Generate Video Segment {current_step} of {total_steps}
        **Topic**: {topic}
        **Visual Style**: {style}
        **Narrative Stage**: {narrative_stage}
        **Main Character**: {main_character}
        """

        segment_duration = context.get('segment_duration', 6)
        
        # Part 2: Continuity Context (The "Core Essence" Logic)
        if current_step > 1:
            if narration_context:
                # 使用前一段的完整腳本上下文（視覺 + 旁白）
                prompt += f"""
            
            ### PREVIOUS SEGMENT CONTEXT:
            **Previous Visual**: {visual_context}
            **Previous Narration**: {narration_context}

            ### CRITICAL INSTRUCTIONS for this {segment_duration}-second segment:
            1. **Substantial Action Required**: Create visual descriptions with MEANINGFUL ACTIONS that can sustain {segment_duration} seconds:
               - Include actions like: walking, running, jumping, climbing, interacting with objects, changing locations, performing tasks
               - Show clear progression: character moving from one place to another, completing an action sequence, interacting with environment
               - AVOID: simple head movements, slight body swaying, minor gestures (these are too short!)
            
            2. **Connect & Evolve**: Start from the previous visual context and show clear progression:
               - If character was in location A, show them moving to location B or performing a new action
               - Continue the story thread from previous narration with new developments
               - Ensure smooth transition but with clear visual and narrative progression
            
            3. **Visual Richness**: Include multiple visual elements:
               - Background changes or camera movement
               - Dynamic elements (objects, environment, lighting changes)
               - Action sequences that unfold over time
            
            4. **Consistency**: Ensure {main_character} looks and acts consistent with the previous description.
            """
            else:
                # 僅有視覺上下文（降級情況）
                prompt += f"""
            
            ### PREVIOUS VISUAL CONTEXT (The "Anchor"):
            The previous segment ended with this visual: 
            "{visual_context}"

            ### CRITICAL INSTRUCTIONS for this {segment_duration}-second segment:
            1. **Substantial Action Required**: Create visual descriptions with MEANINGFUL ACTIONS that can sustain {segment_duration} seconds:
               - Include actions like: walking, running, jumping, climbing, interacting with objects, changing locations
               - AVOID: simple head movements, slight body swaying, minor gestures
            
            2. **Connect & Evolve**: Start from the previous visual context and show clear progression with new actions or location changes.
            
            3. **Consistency**: Ensure {main_character} looks and acts consistent with the previous description.
            """
        else:
            prompt += f"""
            
            ### CRITICAL INSTRUCTIONS for this {segment_duration}-second opening segment:
            1. **Substantial Opening Action**: Create a visually striking opening with MEANINGFUL ACTIONS:
               - Show {main_character} performing a clear action sequence (walking into scene, starting a task, interacting with environment)
               - Include dynamic visual elements that can animate over {segment_duration} seconds
               - AVOID: static poses or simple repetitive movements
            
            2. **Establish**: Clearly show {main_character} in the setting with engaging action content.
            """

        prompt += "\nGenerate the JSON output now."
        return prompt

    def _call_llm_with_retry(self, messages: List[Dict[str, str]], context_name: str = "LLM call", max_retries: int = 2) -> Dict[str, Any]:
        """Executes the LLM call with retry logic."""
        for attempt in range(max_retries + 1):
            try:
                response = self.vision_manager.text_model.chat_completion(messages=messages)
                return self.parse_json_response(response)
            except Exception as e:
                if attempt == max_retries:
                    raise RuntimeError(f"Failed to generate {context_name} after {max_retries} retries: {e}")
                print(f"Retry {attempt + 1}/{max_retries} for {context_name}: {e}")
                time.sleep(1)

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """Clean and parse JSON response."""
        try:
            # Handle standard markdown blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            # Handle <think> tags (Common in reasoning models like DeepSeek)
            if '</think>' in response:
                response = response.split('</think>')[-1]
            
            return json.loads(response.strip())
        except Exception as e:
            # Fallback strategy: simple heuristic repair or logging raw response
            print(f"JSON Parse Error. Raw response: {response}")
            raise ValueError(f"Failed to parse JSON: {e}")

