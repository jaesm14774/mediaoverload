from __future__ import annotations

from typing import Any, Callable
from agentic.runtime.contracts import GoalRequest
from agentic.runtime.prompt_requests import GenerationRoutingRequest
from agentic.runtime.observability import RunRecorder
from agentic.runtime.llm_client import PromptClient
from agentic.runtime.prompt_tasks import (
    routing,
    native_h3,
    scenes,
    graphics,
    story_cards,
    publishing,
)


class LLMPromptEngine:
    """The application task API; writers share one client and own their contracts."""

    def __init__(
        self,
        mode: str = "auto",
        manager: Any | None = None,
        recorder: RunRecorder | None = None,
        *,
        manager_factory: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self.client = PromptClient(
            mode=mode,
            manager=manager,
            recorder=recorder,
            manager_factory=manager_factory,
        )
        self.recorder = recorder

    def backend_info(self) -> dict[str, Any]:
        return self.client.backend_info()

    def route_generation_strategy(
        self,
        request: GenerationRoutingRequest,
    ) -> dict[str, Any]:
        return routing.route_generation_strategy(self.client, request)

    def generate_native_h3_storyboard(
        self,
        *,
        character: str,
        subject_context: dict[str, Any] | None = None,
        style: str,
        duration_seconds: int,
        base_storyboard: dict[str, Any],
        news_context: dict[str, Any],
        creative_brief: str = "",
        arc_guidance: str = "",
        reference_analysis: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return native_h3.generate_native_h3_storyboard(
            self.client,
            character=character,
            subject_context=subject_context,
            style=style,
            duration_seconds=duration_seconds,
            base_storyboard=base_storyboard,
            news_context=news_context,
            creative_brief=creative_brief,
            arc_guidance=arc_guidance,
            reference_analysis=reference_analysis,
        )

    def generate_autonomous_scene_prompt(
        self,
        *,
        character: str,
        style: str,
        media_type: str,
        news_context: dict[str, Any] | None = None,
        news_grounding_required: bool = False,
    ) -> dict[str, Any]:
        return scenes.generate_autonomous_scene_prompt(
            self.client,
            character=character,
            style=style,
            media_type=media_type,
            news_context=news_context,
            news_grounding_required=news_grounding_required,
        )

    def expand_goal(
        self,
        goal: GoalRequest,
        selected_style: str,
        idea_variants: list[dict[str, Any]],
        reference_analysis: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return scenes.expand_goal(
            self.client, goal, selected_style, idea_variants, reference_analysis
        )

    def compose_prompt(
        self,
        goal: GoalRequest,
        prompt: str,
        style: str,
        prefix: str = "",
        suffix: str = "",
        negative_prompt: str = "",
    ) -> dict[str, Any]:
        return scenes.compose_prompt(
            self.client, goal, prompt, style, prefix, suffix, negative_prompt
        )

    def segment_story(
        self,
        goal: GoalRequest,
        creative_brief: str,
        segment_count: int,
        tone: str,
        reference_analysis: dict[str, Any] | None = None,
        *,
        production_profile: str = "",
    ) -> list[dict[str, Any]]:
        return scenes.segment_story(
            self.client,
            goal,
            creative_brief,
            segment_count,
            tone,
            reference_analysis,
            production_profile=production_profile,
        )

    def prepare_segment(
        self,
        goal: GoalRequest,
        segment: dict[str, Any],
        negative_prompt: str,
        previous_segment: dict[str, Any] | None = None,
        prior_frame: str | None = None,
    ) -> dict[str, Any]:
        return scenes.prepare_segment(
            self.client, goal, segment, negative_prompt, previous_segment, prior_frame
        )

    def refine_prompt_from_review(
        self,
        goal: GoalRequest,
        original_prompt: str,
        review_notes: str,
        media_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        return scenes.refine_prompt_from_review(
            self.client, goal, original_prompt, review_notes, media_paths
        )

    def sticker_expressions(
        self, goal: GoalRequest, prompt: str, character: str, expression_count: int
    ) -> list[str]:
        return graphics.sticker_expressions(
            self.client, goal, prompt, character, expression_count
        )

    def build_sticker_prompt_set(
        self,
        goal: GoalRequest,
        expressions: list[str],
        character: str,
        prompt_prefix: str,
        style: str,
    ) -> dict[str, Any]:
        return graphics.build_sticker_prompt_set(
            self.client, goal, expressions, character, prompt_prefix, style
        )

    def build_sticker_motion_prompt(
        self,
        goal: GoalRequest,
        base_prompt: str,
        character: str,
        selected_expression: str = "",
    ) -> dict[str, Any]:
        return graphics.build_sticker_motion_prompt(
            self.client, goal, base_prompt, character, selected_expression
        )

    def build_motion_graphics_plan(
        self,
        goal: GoalRequest,
        *,
        duration_seconds: float,
        fps: float,
    ) -> dict[str, Any]:
        return graphics.build_motion_graphics_plan(
            self.client, goal, duration_seconds=duration_seconds, fps=fps
        )

    def review_motion_graphics_plan(
        self,
        goal: GoalRequest,
        *,
        plan: dict[str, Any],
        contact_sheet_path: str,
        round_number: int,
    ) -> dict[str, Any]:
        return graphics.review_motion_graphics_plan(
            self.client,
            goal,
            plan=plan,
            contact_sheet_path=contact_sheet_path,
            round_number=round_number,
        )

    def build_dynamic_sprite_motion_plan(self, goal: GoalRequest) -> dict[str, Any]:
        return graphics.build_dynamic_sprite_motion_plan(self.client, goal)

    def build_carousel_prompt_set(
        self,
        goal: GoalRequest,
        segments: list[dict[str, Any]],
        style: str,
    ) -> dict[str, Any]:
        return graphics.build_carousel_prompt_set(self.client, goal, segments, style)

    def build_story_card(self, goal: GoalRequest) -> dict[str, Any]:
        return story_cards.build_story_card(self.client, goal)

    def prepare_publish_caption(
        self,
        goal: GoalRequest,
        prefix: str,
        hashtags: list[str],
        platforms: list[str],
        media_paths: list[str] | None = None,
        review_notes: str = "",
        visual_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        return publishing.prepare_publish_caption(
            self.client,
            goal,
            prefix,
            hashtags,
            platforms,
            media_paths,
            review_notes,
            visual_paths,
        )
