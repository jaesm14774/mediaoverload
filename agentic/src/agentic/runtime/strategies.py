"""Product strategy contracts, independent of planners and model providers.

Routing YAML chooses available candidates and weights. This catalog defines
what each strategy means; it does not select workflows or execute anything.
"""

from dataclasses import dataclass
from typing import Any


WORKFLOW_STAGE_KEYS = (
    "image_workflow_name",
    "video_workflow_name",
    "refine_workflow_name",
    "transition_workflow_name",
    "upscale_workflow_name",
)


@dataclass(frozen=True, slots=True)
class GenerationStrategy:
    media_type: str
    primary_workflow_keys: tuple[str, ...]
    image_only: bool = False
    news_grounded: bool = False
    native_story: bool = False


_IMAGE = ("image_workflow_name",)
_NATIVE = ("video_workflow_name", "image_workflow_name", "refine_workflow_name")
GENERATION_STRATEGIES = {
    "text2img": GenerationStrategy("image", _IMAGE, image_only=True),
    "text2video": GenerationStrategy(
        "text2video", ("video_workflow_name", "image_workflow_name")
    ),
    "text2image2video": GenerationStrategy(
        "text2img2video", (*_IMAGE, "video_workflow_name", "upscale_workflow_name")
    ),
    "text2image2image": GenerationStrategy(
        "text2img2img", (*_IMAGE, "refine_workflow_name")
    ),
    # Strategy selection dispatches this route to the serial application flow.
    "serial_video": GenerationStrategy("long_video", ()),
    "native_h3_story": GenerationStrategy(
        "native_h3_story",
        ("video_workflow_name", "image_workflow_name"),
        news_grounded=True,
        native_story=True,
    ),
    "native_h3_t2v_story": GenerationStrategy(
        "native_h3_t2v_story",
        ("video_workflow_name",),
        news_grounded=True,
        native_story=True,
    ),
    "native_h3_fl2va_story": GenerationStrategy(
        "native_h3_fl2va_story", _NATIVE, news_grounded=True, native_story=True
    ),
    "native_h3_l2va_story": GenerationStrategy(
        "native_h3_l2va_story",
        ("video_workflow_name", "image_workflow_name", "refine_workflow_name"),
        news_grounded=True,
        native_story=True,
    ),
    "native_h3_ref2va": GenerationStrategy(
        "native_h3_ref2va",
        ("video_workflow_name",),
        news_grounded=True,
        native_story=True,
    ),
    "text2image2native_h3_ref2va": GenerationStrategy(
        "text2image2native_h3_ref2va",
        ("video_workflow_name", "image_workflow_name"),
        news_grounded=True,
        native_story=True,
    ),
    "sticker_pack": GenerationStrategy(
        "sticker_pack", (*_IMAGE, "video_workflow_name"), image_only=True
    ),
    "game_sprite": GenerationStrategy(
        "game_sprite", ("video_workflow_name", "image_workflow_name")
    ),
    "story_card": GenerationStrategy(
        "story_card", _IMAGE, image_only=True, news_grounded=True
    ),
    "motion_comic": GenerationStrategy("motion_comic", _IMAGE),
}
CONFIG_MEDIA_TYPE_MAP = {
    name: strategy.media_type for name, strategy in GENERATION_STRATEGIES.items()
}
NATIVE_STORY_GENERATION_TYPES = frozenset(
    name for name, strategy in GENERATION_STRATEGIES.items() if strategy.native_story
)
NEWS_GROUNDED_GENERATION_TYPES = frozenset(
    name for name, strategy in GENERATION_STRATEGIES.items() if strategy.news_grounded
)
IMAGE_ONLY_GENERATION_TYPES = frozenset(
    name for name, strategy in GENERATION_STRATEGIES.items() if strategy.image_only
)


def media_type_for_generation(generation_type: str) -> str:
    """Resolve a configured generation strategy without guessing a fallback."""
    try:
        return GENERATION_STRATEGIES[generation_type].media_type
    except KeyError as exc:
        raise ValueError(
            f"Unsupported generation strategy: {generation_type}"
        ) from exc


def active_count_keys(
    generation_type: str, stage_candidates: dict[str, list[str]]
) -> set[str]:
    """Counts with an observable consumer in the selected strategy's graph."""
    if generation_type not in GENERATION_STRATEGIES:
        raise ValueError(f"Unsupported generation strategy: {generation_type}")
    active = {"review_selection_limit"}
    if stage_candidates.get("image_workflow_name"):
        active.add("image_count")
    if (
        stage_candidates.get("video_workflow_name")
        and generation_type != "sticker_pack"
    ):
        active.add("video_count")
    if generation_type == "sticker_pack":
        active.update({"sticker_expression_count", "images_per_prompt"})
    return active


def active_count_policy(
    generation_type: str, stage_candidates: dict[str, list[str]], policy: dict[str, Any]
) -> dict[str, Any]:
    active = active_count_keys(generation_type, stage_candidates)
    return {key: value for key, value in (policy or {}).items() if key in active}


def primary_workflow_name(generation_type: str, workflow_plan: dict[str, str]) -> str:
    for key in GENERATION_STRATEGIES[generation_type].primary_workflow_keys:
        value = str(workflow_plan.get(key) or "").strip()
        if value:
            return value
    return ""
