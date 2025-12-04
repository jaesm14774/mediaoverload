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
        
        # 2. Get Visual Context (if not the first segment)
        visual_context = ""
        if last_frame_path and segment_index > 0:
            try:
                # 這裡假設 extract_image_content 回傳的是對圖片的詳細文字描述
                visual_context = self.vision_manager.extract_image_content(last_frame_path)
            except Exception as e:
                print(f"Warning: Failed to extract image content: {e}")
                visual_context = "Previous scene context unavailable."

        # 3. Build Prompts
        system_prompt = self._build_system_prompt(context)
        user_prompt = self._build_user_prompt(
            context=context,
            visual_context=visual_context,
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
        segment_duration = context.get('segment_duration', 5)

        # Base instructions
        prompt = f"""
        You are an expert Director and Screenwriter for AI-generated videos. 
        Your goal is to write a cohesive, engaging video script segment that maintains strict continuity.

        ### CORE GUIDELINES:
        1. **Character Consistency**: The Main Character is '{main_character}'. You MUST explicitly name them in BOTH the 'narration' and 'visual' prompt every time to ensure the AI video generator keeps the same person.
        2. **Visual Continuity**: The 'visual' description must logically evolve from the previous frame (if provided). Do not jump cut randomly.
        3. **Engagement**: The narration should be conversational, pacing well for a {segment_duration}-second clip.
        
        ### OUTPUT FORMAT:
        Return ONLY a valid JSON object with this EXACT structure:
        {{
            "narration": "Natural, engaging voiceover text suitable for social media.",
            "visual": "Highly detailed AI image prompt. Format: [Subject + Action], [Setting/Background], [Lighting/Mood], [Style]. MUST include '{main_character}'."
        }}
        """
        return prompt

    def _build_user_prompt(self, context: Dict[str, Any], visual_context: str, 
                          narrative_stage: str, current_step: int, total_steps: int) -> str:
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

        # Part 2: Continuity Context (The "Core Essence" Logic)
        if current_step > 1:
            prompt += f"""
            
            ### PREVIOUS VISUAL CONTEXT (The "Anchor"):
            The previous segment ended with this visual: 
            "{visual_context}"

            ### INSTRUCTIONS for this segment:
            1. **Connect**: Start the visual description for THIS segment based on the PREVIOUS visual context. If the character was running, are they still running or stopping? 
            2. **Evolve**: Move the story forward. Don't just repeat the previous scene, but ensure the transition is smooth.
            3. **Consistency**: Ensure {main_character} looks and acts consistent with the previous description.
            """
        else:
            prompt += f"""
            
            ### INSTRUCTIONS for this segment:
            1. **Hook**: Create a visually striking opening that immediately grabs attention.
            2. **Establish**: Clearly show {main_character} in the setting.
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
