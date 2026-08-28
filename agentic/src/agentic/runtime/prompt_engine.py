from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic.runtime.contracts import GoalRequest
from agentic.runtime.llm_engine import LLMPromptEngine
from agentic.runtime.platform_content import PLATFORM_STRATEGY_VERSION, build_platform_bundle
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
    ) -> list[dict[str, Any]]:
        return self.llm_engine.segment_story(
            goal,
            creative_brief,
            segment_count,
            tone,
            reference_analysis=reference_analysis,
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
        failure_tags = self._derive_failure_tags(review_notes, media_paths or [])
        bundle["failure_tags"] = failure_tags
        bundle["retry_direction"] = self._derive_retry_direction(review_notes, failure_tags)
        bundle["retry_intensity"] = self._derive_retry_intensity(review_notes, failure_tags)
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

    def build_carousel_prompt_set(
        self,
        goal: GoalRequest,
        segments: list[dict[str, Any]],
        style: str,
    ) -> dict[str, Any]:
        return self.llm_engine.build_carousel_prompt_set(goal, segments, style)

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
        )
        bundle["platform_bundle"] = platform_bundle
        bundle["caption_strategy"] = "platform_adapted" if platform_bundle else "generic"
        bundle["platform_strategy_version"] = PLATFORM_STRATEGY_VERSION
        bundle["dispatch_ready"] = bool(media_paths) and bool(bundle.get("caption"))
        return bundle

    def evaluate_video_contact_sheet(
        self,
        *,
        contact_sheet_path: str,
        character: str,
        subject_context: dict[str, Any] | None = None,
        story_spine: dict[str, Any],
        native_shots: list[dict[str, Any]],
        news_context: dict[str, Any],
        rendered_prompt: str,
        news_anchor_terms: list[str] | None = None,
        duration_seconds: int | float | None = None,
    ) -> dict[str, Any]:
        return self.llm_engine.evaluate_video_contact_sheet(
            contact_sheet_path=contact_sheet_path,
            character=character,
            subject_context=dict(subject_context or {}),
            story_spine=story_spine,
            native_shots=native_shots,
            news_context=news_context,
            rendered_prompt=rendered_prompt,
            news_anchor_terms=news_anchor_terms,
            duration_seconds=duration_seconds,
        )

    def review_asset_candidates(
        self,
        goal: GoalRequest,
        media_paths: list[str],
        review_notes: str,
        selection_limit: int,
    ) -> dict[str, Any]:
        bundle = self.llm_engine.review_asset_candidates(
            goal,
            media_paths=media_paths,
            review_notes=review_notes,
            selection_limit=selection_limit,
        )
        selected_assets = [str(path) for path in bundle.get("selected_assets", []) if str(path)]
        rejected_assets = [path for path in media_paths if path not in selected_assets]
        failure_tags = self._derive_failure_tags(review_notes, rejected_assets)
        ranked = bundle.get("ranked_candidates", [])
        rejected_details: list[dict[str, Any]] = []
        for path in rejected_assets:
            rationale = "Rejected after shortlist ranking."
            for item in ranked:
                if isinstance(item, dict) and str(item.get("media_path", "")) == path:
                    rationale = str(item.get("rationale") or rationale)
                    break
            rejected_details.append(
                {
                    "media_path": path,
                    "reason": rationale,
                    "failure_tags": failure_tags,
                }
            )
        bundle["rejected_assets"] = rejected_assets
        bundle["rejected_asset_details"] = rejected_details
        bundle["failure_tags"] = failure_tags
        bundle["retry_direction"] = self._derive_retry_direction(review_notes, failure_tags)
        bundle["retry_intensity"] = self._derive_retry_intensity(review_notes, failure_tags)
        bundle["publish_ready"] = bool(selected_assets)
        return bundle

    @staticmethod
    def _derive_failure_tags(review_notes: str, rejected_assets: list[str]) -> list[str]:
        note_text = (review_notes or "").lower()
        tags: list[str] = []
        tag_rules = {
            "motion_weak": ("motion", "action", "static", "stiff"),
            "composition_weak": ("composition", "framing", "empty space", "crop"),
            "subject_unclear": ("clarity", "readability", "subject", "identity"),
            "publish_risk": ("publish", "suitable", "thumbnail", "platform"),
        }
        for tag, keywords in tag_rules.items():
            if any(keyword in note_text for keyword in keywords):
                tags.append(tag)
        if rejected_assets and "publish_risk" not in tags:
            tags.append("publish_risk")
        return tags or ["generic_quality"]

    @staticmethod
    def _derive_retry_direction(review_notes: str, failure_tags: list[str]) -> str:
        directions: list[str] = []
        note_text = (review_notes or "").strip()
        if "motion_weak" in failure_tags:
            directions.append("increase visible action and kinetic staging")
        if "composition_weak" in failure_tags:
            directions.append("tighten framing and strengthen focal hierarchy")
        if "subject_unclear" in failure_tags:
            directions.append("reinforce character identity and readability")
        if not directions and note_text:
            directions.append(note_text)
        return "; ".join(directions) or "raise overall quality while preserving intent"

    @staticmethod
    def _derive_retry_intensity(review_notes: str, failure_tags: list[str]) -> str:
        note_text = (review_notes or "").lower()
        if any(token in note_text for token in ("major", "stronger", "dramatic", "aggressive")):
            return "high"
        if "generic_quality" not in failure_tags or note_text:
            return "medium"
        return "low"
