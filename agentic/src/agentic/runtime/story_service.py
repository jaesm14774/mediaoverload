from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from agentic.runtime.llm_engine import LLMPromptEngine
from agentic.storyboard import merge_native_h3_storyboard
from agentic.tools.context_services import NewsContextService


class NativeH3StoryService:
    """Orchestrate news selection and LLM story generation around pure storyboard rules."""

    def __init__(
        self,
        llm_engine: LLMPromptEngine | None = None,
        news_service: NewsContextService | None = None,
        storyboard_merger: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]] = merge_native_h3_storyboard,
    ) -> None:
        self.llm_engine = llm_engine or LLMPromptEngine(mode=os.environ.get("AGENTIC_LLM_MODE", "llm"))
        self.news_service = news_service or NewsContextService()
        self.storyboard_merger = storyboard_merger

    def resolve(
        self,
        base_storyboard: dict[str, Any],
        *,
        character: str,
        subject_context: dict[str, Any] | None = None,
        style: str,
        duration_seconds: int,
        news_context: dict[str, Any] | None = None,
        creative_brief: str = "",
        arc_guidance: str = "",
        reference_analysis: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        resolved_news = dict(news_context or {})
        if resolved_news.get("error"):
            raise RuntimeError(f"Native H3 news context failed: {resolved_news['error']}")
        if not NewsContextService.is_usable_selection(
            str(resolved_news.get("title") or ""),
            str(resolved_news.get("keyword") or ""),
        ) or not NewsContextService.is_brand_safe_selection(
            str(resolved_news.get("title") or ""),
            str(resolved_news.get("keyword") or ""),
        ):
            selected_news = self.news_service.get_random_news()
            if selected_news is None:
                raise RuntimeError("Native H3 requires a selectable news item; no news context was available")
            resolved_news = selected_news.to_dict()
        recorder = getattr(self.llm_engine, "recorder", None)
        if recorder is not None:
            recorder.record_event("news.selected", news_context=resolved_news)

        resolved_character = str(character or base_storyboard.get("character") or "").strip()
        resolved_subject_context = dict(
            subject_context
            or base_storyboard.get("subject_context")
            or {}
        )
        resolved_base_storyboard = dict(base_storyboard)
        if resolved_character:
            resolved_base_storyboard["character"] = resolved_character
        if resolved_subject_context:
            resolved_base_storyboard["subject_context"] = resolved_subject_context

        payload = self.llm_engine.generate_native_h3_storyboard(
            character=resolved_character,
            subject_context=resolved_subject_context,
            style=style,
            duration_seconds=duration_seconds,
            base_storyboard=resolved_base_storyboard,
            news_context=resolved_news,
            creative_brief=creative_brief,
            arc_guidance=arc_guidance,
            reference_analysis=dict(reference_analysis or {}),
        )
        return self.storyboard_merger(resolved_base_storyboard, payload["story"]), payload
