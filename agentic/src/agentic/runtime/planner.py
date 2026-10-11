from __future__ import annotations

from agentic.assets.registry import AssetRegistry
from agentic.runtime.contracts import ExecutionPlan, GoalRequest
from agentic.runtime.creativity import IdeaDirector
from agentic.runtime.planning.common import PlanningContext
from agentic.runtime.planning.native_h3 import (
    build_native_h3_story,
    build_native_h3_t2v_story,
    build_native_h3_l2va_story,
    build_native_h3_ref2va,
)
from agentic.runtime.planning.long_video import build_long_video
from agentic.runtime.planning.short_video import build_text2img2video, build_text2video
from agentic.runtime.planning.images import (
    build_image,
    build_comfy_primitive,
    build_text2img2img,
    build_story_card,
    build_carousel,
)
from agentic.runtime.planning.stickers import (
    build_sticker_pack,
    build_game_sprite,
    build_animated_sticker,
)
from agentic.runtime.planning.editing import (
    build_video_narrate,
    build_image_sequence_edit,
)
from agentic.runtime.planning.publishing import build_publish_review
from agentic.runtime.planning.motion_comic import build_motion_comic


PLAN_BUILDERS = {
    "publish_review": build_publish_review,
    "game_sprite": build_game_sprite,
    "animated_sticker": build_animated_sticker,
    "carousel": build_carousel,
    "story_card": build_story_card,
    "motion_comic": build_motion_comic,
    "sticker_pack": build_sticker_pack,
    "text2img2img": build_text2img2img,
    "text2video": build_text2video,
    "text2img2video": build_text2img2video,
    "native_h3_story": build_native_h3_story,
    "native_h3_t2v_story": build_native_h3_t2v_story,
    "native_h3_fl2va_story": build_native_h3_story,
    "native_h3_l2va_story": build_native_h3_l2va_story,
    "native_h3_ref2va": build_native_h3_ref2va,
    "text2image2native_h3_ref2va": build_native_h3_ref2va,
    "video_narrate": build_video_narrate,
    "image_sequence_edit": build_image_sequence_edit,
    "image": build_image,
    "image_refine": build_comfy_primitive,
    "image_upscale": build_comfy_primitive,
    "image_to_video": build_comfy_primitive,
    "long_video": build_long_video,
}


class TaskPlanner:
    """Select a graph builder; execution and model transport belong elsewhere."""

    def __init__(
        self, asset_registry: AssetRegistry, idea_director: IdeaDirector | None = None
    ) -> None:
        self.asset_registry = asset_registry
        self.context = PlanningContext(asset_registry, idea_director)

    def create_goal(
        self,
        prompt: str,
        media_type: str,
        duration_seconds: int,
        style: str,
        constraints: dict | None = None,
    ) -> GoalRequest:
        return GoalRequest(
            prompt=prompt,
            media_type=media_type,
            duration_seconds=duration_seconds,
            style=style,
            constraints=constraints or {},
        )

    def build_plan(self, goal: GoalRequest) -> ExecutionPlan:
        try:
            builder = PLAN_BUILDERS[goal.media_type]
        except KeyError as exc:
            raise ValueError(f"Unsupported media type: {goal.media_type}") from exc
        return builder(self.context, goal)
