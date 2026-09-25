from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic.runtime.contracts import GoalRequest
from agentic.runtime.llm_engine import LLMPromptEngine
from agentic.runtime.platform_content import PLATFORM_STRATEGY_VERSION, build_platform_bundle
from agentic.runtime.post_strategy import resolve_post_strategy
from agentic.runtime.prompt_requests import GenerationRoutingRequest


class PromptEngine:
    def __init__(self, llm_engine: LLMPromptEngine | None = None) -> None:
        self.llm_engine = llm_engine or LLMPromptEngine(mode="template")

    def backend_info(self) -> dict[str, Any]:
        return self.llm_engine.backend_info()

    def route_generation_strategy(
        self,
        request: GenerationRoutingRequest,
    ) -> dict[str, Any]:
        return self.llm_engine.route_generation_strategy(request)

    def expand_goal(
        self,
        goal: GoalRequest,
        selected_style: str,
        idea_variants: list[dict[str, Any]],
        reference_analysis: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.llm_engine.expand_goal(
            goal,
            selected_style,
            idea_variants,
            reference_analysis=reference_analysis,
        )

    def compose_prompt(
        self,
        goal: GoalRequest,
        prompt: str,
        style: str,
        prefix: str = "",
        suffix: str = "",
        negative_prompt: str = "ugly, blurry, low quality, bad anatomy, deformed, duplicate, watermark, text",
    ) -> dict[str, Any]:
        return self.llm_engine.compose_prompt(
            goal,
            prompt=prompt,
            style=style,
            prefix=prefix,
            suffix=suffix,
            negative_prompt=negative_prompt,
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
        return self.llm_engine.segment_story(
            goal,
            creative_brief,
            segment_count,
            tone,
            reference_analysis=reference_analysis,
            production_profile=production_profile,
        )

    def sticker_expressions(self, goal: GoalRequest, prompt: str, character: str, expression_count: int) -> list[str]:
        return self.llm_engine.sticker_expressions(goal, prompt, character, expression_count)

    def build_sticker_prompt_set(
        self,
        goal: GoalRequest,
        expressions: list[str],
        character: str,
        prompt_prefix: str,
        style: str,
    ) -> dict[str, Any]:
        return self.llm_engine.build_sticker_prompt_set(
            goal,
            expressions=expressions,
            character=character,
            prompt_prefix=prompt_prefix,
            style=style,
        )

    def prepare_segment(
        self,
        goal: GoalRequest,
        segment: dict[str, Any],
        negative_prompt: str,
        previous_segment: dict[str, Any] | None = None,
        prior_frame: str | None = None,
    ) -> dict[str, Any]:
        return self.llm_engine.prepare_segment(
            goal,
            segment,
            negative_prompt,
            previous_segment=previous_segment,
            prior_frame=prior_frame,
        )

    def refine_prompt_from_review(
        self,
        goal: GoalRequest,
        original_prompt: str,
        review_notes: str,
        media_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        bundle = self.llm_engine.refine_prompt_from_review(
            goal,
            original_prompt=original_prompt,
            review_notes=review_notes,
            media_paths=media_paths,
        )
        return bundle
    def build_sticker_motion_prompt(
        self,
        goal: GoalRequest,
        base_prompt: str,
        character: str,
        selected_expression: str = "",
    ) -> dict[str, Any]:
        return self.llm_engine.build_sticker_motion_prompt(
            goal,
            base_prompt=base_prompt,
            character=character,
            selected_expression=selected_expression,
        )

    def build_dynamic_sprite_motion_plan(self, goal: GoalRequest) -> dict[str, Any]:
        return self.llm_engine.build_dynamic_sprite_motion_plan(goal)

    def build_carousel_prompt_set(
        self,
        goal: GoalRequest,
        segments: list[dict[str, Any]],
        style: str,
    ) -> dict[str, Any]:
        return self.llm_engine.build_carousel_prompt_set(goal, segments, style)

    def build_story_card(self, goal: GoalRequest) -> dict[str, Any]:
        return self.llm_engine.build_story_card(goal)

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
        bundle = self.llm_engine.prepare_publish_caption(
            goal,
            prefix=prefix,
            hashtags=hashtags,
            platforms=platforms,
            media_paths=media_paths,
            review_notes=review_notes,
            visual_paths=visual_paths,
        )
        normalized_hashtags = str(bundle.get("hashtags", "") or "").strip()
        post_strategy = resolve_post_strategy(goal, media_paths)
        platform_captions = bundle.get("platform_captions", {})
        if not isinstance(platform_captions, dict):
            platform_captions = {}
        effective_platforms = platforms or list(platform_captions.keys())
        platform_bundle = build_platform_bundle(
            goal=goal,
            caption=str(bundle.get("caption", "") or "").strip(),
            hashtags=normalized_hashtags,
            platform_captions={
                str(platform): str(caption)
                for platform, caption in platform_captions.items()
            },
            platforms=[str(platform) for platform in effective_platforms],
            media_paths=media_paths,
            post_strategy=post_strategy,
        )
        bundle["platform_bundle"] = platform_bundle
        bundle["post_strategy"] = post_strategy
        bundle["caption_strategy"] = "platform_adapted" if platform_bundle else "generic"
        bundle["platform_strategy_version"] = PLATFORM_STRATEGY_VERSION
        bundle["dispatch_ready"] = bool(media_paths) and bool(bundle.get("caption"))
        return bundle
