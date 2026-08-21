from __future__ import annotations

import json
import os
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agentic.assets.registry import AssetRegistry
from agentic.app.main import _build_prompt_summary, build_runtime
from agentic.h3_reference import normalize_reference_manifest
from agentic.runtime.llm_engine import LLMPromptEngine
from agentic.runtime.observability import RunRecorder
from agentic.runtime.route_selection import select_weighted_route
from agentic.runtime.step_logger import create_run_logger
from agentic.tools.context_services import DiscordRunNotificationService, NewsContextService

SUPPORTED_PUBLISH_PLATFORMS = {"twitter", "facebook", "instagram_graph", "youtube"}
MEDIA_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}
IMAGE_ONLY_GENERATION_TYPES = {"text2img", "sticker_pack"}

# A bare character invocation is the autonomous production mode. It must
# exercise the complete news-to-media path instead of allowing the strategy
# LLM to stop at an image-only strategy.
NEWS_GROUNDED_GENERATION_TYPES = frozenset(
    {
        "native_h3_story",
        "native_h3_t2v_story",
        "native_h3_fl2va_story",
        "native_h3_l2va_story",
        "native_h3_ref2va",
        "text2image2native_h3_ref2va",
        "text2longvideo",
    }
)
AUTONOMOUS_E2E_ROUTE_PREFERENCE = (
    "native_h3_story",
    "text2image2native_h3_ref2va",
    "text2longvideo",
)

CONFIG_MEDIA_TYPE_MAP = {
    "text2img": "image",
    "text2video": "text2video",
    "text2image2video": "text2img2video",
    "text2longvideo": "long_video",
    "native_h3_story": "native_h3_story",
    "native_h3_t2v_story": "native_h3_t2v_story",
    "native_h3_fl2va_story": "native_h3_fl2va_story",
    "native_h3_l2va_story": "native_h3_l2va_story",
    "native_h3_ref2va": "native_h3_ref2va",
    "text2image2native_h3_ref2va": "text2image2native_h3_ref2va",
    "image2image": "image",
    "text2image2image": "text2img2img",
    "sticker_pack": "sticker_pack",
}

def _normalize_generation_type(
    value: str | None,
    aliases: dict[str, Any] | None = None,
) -> str | None:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    alias_map = {
        str(key).strip(): str(target).strip()
        for key, target in dict(aliases or {}).items()
        if str(key).strip() and str(target).strip()
    }
    return alias_map.get(normalized, normalized)


def load_character_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_global_social_config(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "configs" / "social_media" / "platforms.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(data) if isinstance(data, dict) else {}


def _extract_failure_details(result: dict[str, Any] | None) -> dict[str, str]:
    """Promote the first failed node's diagnostic into the run manifest."""
    if not isinstance(result, dict):
        return {}
    records = result.get("records")
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            if str(record.get("status") or "").lower() == "success":
                continue
            logs = record.get("logs")
            reason = ""
            if isinstance(logs, list):
                reason = next((str(item).strip() for item in logs if str(item).strip()), "")
            if not reason:
                reason = (
                    f"{record.get('skill_name') or record.get('node_id') or 'workflow node'} "
                    f"finished with status {record.get('status') or 'failed'}"
                )
            details = {
                "failure_reason": reason,
                "failure_node": str(record.get("node_id") or ""),
                "failure_skill": str(record.get("skill_name") or ""),
            }
            return {key: value for key, value in details.items() if value}
    if str(result.get("status") or "").lower() not in {"", "success"}:
        return {"failure_reason": f"Workflow finished with status {result.get('status')}"}
    return {}


def choose_media_type(
    config: dict[str, Any],
    preferred_generation_type: str | None = None,
    rng: random.Random | None = None,
) -> tuple[str, str]:
    generation = dict(config.get("generation", {}) or {})
    weights = dict(generation.get("generation_type_weights", {}) or {})
    normalized_preference = _normalize_generation_type(preferred_generation_type)
    if normalized_preference:
        config_generation_type = normalized_preference
    else:
        config_generation_type = _weighted_choice(weights, rng=rng)
    agentic_media_type = CONFIG_MEDIA_TYPE_MAP.get(config_generation_type, "long_video")
    return config_generation_type, agentic_media_type


def build_goal_payload_from_character_config(
    repo_root: Path,
    config_path: str | Path,
    *,
    prompt: str = "",
    temperature: float = 1.0,
    preferred_generation_type: str | None = None,
    duration_seconds: int | None = None,
    dry_run_publish: bool = False,
    publish_mode: str = "",
    publish_platforms: list[str] | None = None,
    publish_after_generate: bool = True,
    output_dir: str | None = None,
    enable_review_loop: bool = False,
    review_notes: str = "",
    no_review: bool = False,
    stage_probe: bool = False,
    news_driven: bool = False,
    news_history_path: str | None = None,
    routing_history_path: str | None = None,
    asset_root: Path | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = load_character_config(path)
    character = dict(config.get("character", {}) or {})
    generation = dict(config.get("generation", {}) or {})
    additional_params = dict(config.get("additional_params", {}) or {})
    strategies = dict(additional_params.get("strategies", {}) or {})
    requested_duration_seconds = duration_seconds
    if requested_duration_seconds is not None and int(requested_duration_seconds) <= 0:
        raise ValueError("duration_seconds must be a positive integer")
    social_media = _merge_social_media_config(
        dict(load_global_social_config(repo_root).get("social_media", {}) or {}),
        dict(config.get("social_media", {}) or {}),
    )

    style = _pick_primary_style(generation.get("style_weights"))
    character_name = str(character.get("name") or character.get("group_name") or path.stem)
    routing = _route_generation_from_character_config(
        repo_root,
        config,
        character_name=character_name,
        style=style,
        prompt=prompt,
        preferred_generation_type=preferred_generation_type,
        requested_duration_seconds=requested_duration_seconds,
        news_driven=news_driven,
        rng=rng,
        routing_history_path=Path(routing_history_path).expanduser() if routing_history_path else None,
        asset_root=asset_root,
    )
    config_generation_type = str(routing["generation_type"])
    effective_news_driven = bool(
        news_driven
        or (
            not str(prompt).strip()
            and routing.get("selection_source") == "autonomous_e2e_default"
            and config_generation_type in NEWS_GROUNDED_GENERATION_TYPES
        )
    )
    agentic_media_type = CONFIG_MEDIA_TYPE_MAP.get(config_generation_type, "long_video")
    native_recipe = dict(_native_recipe_for_generation(generation, config_generation_type))
    configured_reference_values = any(
        native_recipe.get(key) or generation.get(key)
        for key in ("reference_manifest", "reference_image_paths", "reference_video_paths")
    )
    auto_reference_generation = (
        config_generation_type == "text2image2native_h3_ref2va"
        or (config_generation_type == "native_h3_ref2va" and not configured_reference_values)
    )
    pre_video_review_config = dict(routing.get("pre_video_review", {}) or {})
    # Ref2VA uses configured references when present. With an empty manifest,
    # it expands into the six-candidate T2I plus Discord reference gate.
    pre_video_review_enabled = bool(
        pre_video_review_config.get("enabled", False)
        and config_generation_type not in {"text2video", "native_h3_t2v_story"}
        and not (no_review and config_generation_type == "text2image2video")
        and (
            config_generation_type != "native_h3_ref2va"
            or auto_reference_generation
        )
    )
    pre_video_candidate_count = max(1, int(pre_video_review_config.get("candidate_count") or 6))
    pre_video_selection_limit = max(
        1,
        int(
            pre_video_review_config.get(
                "ref2va_selection_limit"
                if config_generation_type in {"native_h3_ref2va", "text2image2native_h3_ref2va"}
                else "default_selection_limit",
            )
            or (4 if config_generation_type == "text2image2native_h3_ref2va" else 1)
        ),
    )
    if no_review and config_generation_type == "text2image2video":
        # Without a review gate there is no consumer for the six-candidate
        # selection contract. Keep the fast path to one keyframe so it can
        # actually reach the I2V stage on a low-VRAM ComfyUI host.
        routing["count_plan"]["image_count"] = 1
        routing["count_plan"]["review_selection_limit"] = 1
    if stage_probe:
        # Stage probes must exercise the configured candidate contract for
        # every image-producing strategy, not just Native H3. Use the policy
        # maximum so the probe can evaluate alternatives without changing
        # production defaults.
        stage_probe_policy = dict(
            routing.get("count_policies", {}).get(config_generation_type, {}) or {}
        )
        if config_generation_type in {"text2img", "text2image2image", "text2video"}:
            image_policy = dict(stage_probe_policy.get("image_count", {}) or {})
            if image_policy:
                routing["count_plan"]["image_count"] = int(image_policy.get("max") or 1)
            routing["count_plan"]["review_selection_limit"] = 1
        elif config_generation_type == "sticker_pack":
            expression_policy = dict(stage_probe_policy.get("sticker_expression_count", {}) or {})
            images_policy = dict(stage_probe_policy.get("images_per_prompt", {}) or {})
            if expression_policy:
                routing["count_plan"]["sticker_expression_count"] = int(expression_policy.get("max") or 4)
            if images_policy:
                routing["count_plan"]["images_per_prompt"] = int(images_policy.get("max") or 1)
    autonomous_prompt = _resolve_autonomous_prompt(
        prompt=prompt,
        character_name=character_name,
        style=style,
        media_type=agentic_media_type,
        generation=generation,
        generation_type=config_generation_type,
        news_driven=effective_news_driven,
        news_history_path=news_history_path or _default_news_history_path(repo_root, character_name),
    )
    if config_generation_type in {"native_h3_story", "native_h3_t2v_story", "native_h3_fl2va_story", "native_h3_l2va_story", "native_h3_ref2va", "text2image2native_h3_ref2va"} and (effective_news_driven or not str(prompt).strip()):
        # Native H3 owns the news-to-story prompt contract. Do not create a
        # second autonomous scene prompt here and then carry it through the
        # routing summary as if it were a user brief.
        resolved_prompt = ""
    else:
        resolved_prompt = str(autonomous_prompt["prompt"]).strip() or _build_default_prompt(character_name, agentic_media_type, style)
    resolved_output_dir = _resolve_output_dir(
        repo_root,
        output_dir or generation.get("output_dir"),
        character_name,
    )
    duration_seconds = _resolve_duration_seconds(
        config_generation_type,
        strategies,
        generation=generation,
        routed_segment_count=routing.get("count_plan", {}).get("segment_count"),
        requested_duration_seconds=requested_duration_seconds,
    )
    platform_configs, platform_aliases, skipped_platforms = _normalize_platform_configs(
        repo_root,
        social_media.get("platforms"),
    )
    if publish_platforms:
        selected_platforms: list[str] = []
        for requested in publish_platforms:
            normalized = str(requested).lower().strip()
            if normalized == "instagram":
                normalized = "instagram_graph"
            if normalized not in SUPPORTED_PUBLISH_PLATFORMS:
                raise ValueError(f"Unsupported publish platform: {requested}")
            if normalized not in selected_platforms:
                selected_platforms.append(normalized)
        platform_configs = {
            name: config for name, config in platform_configs.items() if name in selected_platforms
        }
    if config_generation_type in IMAGE_ONLY_GENERATION_TYPES and "youtube" in platform_configs:
        platform_configs.pop("youtube", None)
        skipped_platforms.append("youtube:image_only_route")
    hashtags = [str(tag) for tag in (social_media.get("default_hashtags") or []) if tag]
    longvideo_config = dict(routing.get("longvideo_config", {}) or {})
    longvideo_config.update(
        dict(dict(strategies.get("text2longvideo", {}) or {}).get("longvideo_config", {}) or {})
    )
    use_tts = bool(longvideo_config.get("use_tts", False))

    native_keyframe_candidate_count = max(1, int(native_recipe.get("keyframe_candidate_count") or 1))
    if (
        pre_video_review_enabled
        and config_generation_type in {
            "native_h3_story",
            "native_h3_fl2va_story",
            "native_h3_l2va_story",
        }
        and (not no_review or stage_probe)
    ):
        native_keyframe_candidate_count = pre_video_candidate_count
    native_reference_candidate_count = int(
        native_recipe.get("reference_candidate_count")
        or routing.get("count_plan", {}).get("image_count")
        or 4
    )
    if auto_reference_generation and pre_video_review_enabled:
        native_reference_candidate_count = pre_video_candidate_count
    if no_review and not stage_probe and config_generation_type in {"native_h3_story", "native_h3_t2v_story", "native_h3_fl2va_story", "native_h3_l2va_story", "native_h3_ref2va", "text2image2native_h3_ref2va"}:
        native_recipe["keyframe_candidate_count"] = 1
        native_keyframe_candidate_count = 1
        native_recipe["require_human_review"] = False
        if auto_reference_generation:
            native_recipe["reference_selection_limit"] = 1
    if config_generation_type != "native_h3_fl2va_story":
        native_recipe["use_last_frame"] = False
    native_h3_quality = dict(generation.get("native_h3_quality") or {})
    native_h3_creative_brief = str(
        native_recipe.get("creative_brief")
        or native_h3_quality.get("creative_brief")
        or generation.get("native_h3_creative_brief")
        or ""
    ).strip()
    native_h3_visual_style_contract = str(
        generation.get("visual_style_contract")
        or native_recipe.get("visual_style_contract")
        or native_h3_quality.get("visual_style_contract")
        or generation.get("native_h3_visual_style_contract")
        or ""
    ).strip()
    native_h3_semantic_qa_blocking = bool(
        native_recipe.get(
            "semantic_qa_blocking",
            native_h3_quality.get("semantic_qa_blocking", False),
        )
    )
    constraints = {
        "character": character_name,
        "h3_profile": str(generation.get("h3_profile") or "balanced-lowvram"),
        "h3_video_defaults": dict(generation.get("video_defaults") or {}),
        "keyframe_workflow_name": str(generation.get("keyframe_workflow_name") or ""),
        "identity_refine_workflow_name": str(generation.get("identity_refine_workflow_name") or ""),
        "storyboard_path": str(generation.get("storyboard_path") or ""),
        "native_h3_storyboard_path": str(
            native_recipe.get("storyboard_path")
            or generation.get("storyboard_path")
            or ""
        ),
        "native_h3_recipe": native_recipe,
        "native_h3_reference_manifest": list(native_recipe.get("reference_manifest") or generation.get("reference_manifest") or []),
        "native_h3_reference_image_paths": list(native_recipe.get("reference_image_paths") or generation.get("reference_image_paths") or []),
        "native_h3_reference_video_paths": list(native_recipe.get("reference_video_paths") or generation.get("reference_video_paths") or []),
        "native_h3_reference_image_size": str(native_recipe.get("reference_image_size") or generation.get("reference_image_size") or "match"),
        "native_h3_reference_max_images": int(native_recipe.get("reference_max_images") or generation.get("reference_max_images") or 9),
        "native_h3_reference_max_videos": int(native_recipe.get("reference_max_videos") or generation.get("reference_max_videos") or 3),
        "native_h3_reference_selection_limit": int(native_recipe.get("reference_selection_limit") or generation.get("reference_selection_limit") or 4),
        "native_h3_reference_candidate_count": int(
            native_reference_candidate_count
        ),
        "auto_reference_generation": auto_reference_generation,
        "native_h3_model_profile": str(
            native_recipe.get("model_profile")
            or generation.get("model_profile")
            or (
                "q2"
                if (
                    config_generation_type.startswith("native_h3_")
                    or config_generation_type == "text2image2native_h3_ref2va"
                )
                and str(generation.get("h3_profile") or "balanced-lowvram").strip().lower() == "balanced-lowvram"
                else {"ultra-lowvram": "q2", "native-quality": "native"}.get(
                    str(generation.get("h3_profile") or "balanced-lowvram").strip().lower(),
                    "q4",
                )
            )
        ),
        "input_gate": str(generation.get("input_gate") or ""),
        "hashtags": hashtags,
        "platforms": list(platform_configs.keys()),
        "platform_configs": platform_configs,
        "dry_run": dry_run_publish,
        "publish_mode": str(publish_mode or "").strip().lower(),
        "selection_limit": routing.get("count_plan", {}).get("review_selection_limit"),
        "enable_review_loop": bool(enable_review_loop and not no_review),
        "review_notes": review_notes,
        "news_driven": effective_news_driven,
        "news_history_path": str(news_history_path or _default_news_history_path(repo_root, character_name)),
        "native_h3_keyframe_candidate_count": native_keyframe_candidate_count,
        "pre_video_review_enabled": pre_video_review_enabled,
        "pre_video_candidate_count": pre_video_candidate_count,
        "pre_video_selection_limit": pre_video_selection_limit,
        "pre_video_review_require_human": bool(pre_video_review_enabled and not no_review and not stage_probe),
        "stage_probe_auto_select": bool(stage_probe),
        "pre_video_review_reject_stops_workflow": bool(
            pre_video_review_config.get("reject_stops_workflow", True)
        ),
        "pre_video_review_failure_policy": str(
            pre_video_review_config.get("failure_policy", "block") or "block"
        ).strip().lower(),
        "require_human_review": bool(
            (
                native_recipe.get("require_human_review", True)
                if config_generation_type
                in {
                    "native_h3_story",
                    "native_h3_t2v_story",
                    "native_h3_fl2va_story",
                    "native_h3_l2va_story",
                    "native_h3_ref2va",
                    "text2image2native_h3_ref2va",
                }
                else False
            )
            if not no_review and not stage_probe
            else False
        ),
        "native_h3_semantic_qa_required": bool(native_recipe.get("semantic_qa_required", False)),
        "native_h3_semantic_qa_blocking": native_h3_semantic_qa_blocking,
        "native_h3_creative_brief": native_h3_creative_brief,
        "native_h3_visual_style_contract": native_h3_visual_style_contract,
        "visual_style_contract": native_h3_visual_style_contract,
        "native_h3_use_last_frame": bool(native_recipe.get("use_last_frame", False)),
        # The pre-video gate selects from the raw keyframes. Upscaling before
        # that gate creates an artifact that the selected I2V input never
        # consumes, so keep the route's authoritative input to one raw frame.
        "skip_upscale_for_i2v": config_generation_type == "text2image2video",
        "output_dir": str(resolved_output_dir),
        "use_tts": use_tts if agentic_media_type == "long_video" else False,
        "source_temperature": temperature,
        "source_config_path": str(path),
        "source_generation_type": config_generation_type,
        "duration_override_seconds": (
            max(1, int(requested_duration_seconds)) if requested_duration_seconds is not None else None
        ),
        "duration_profile": _duration_profile(requested_duration_seconds),
        "video_frame_rate": int(dict(generation.get("video_defaults", {}) or {}).get("frame_rate", 24)),
        "workflow_name": routing.get("workflow_name", ""),
        "image_workflow_name": routing.get("workflow_plan", {}).get("image_workflow_name", ""),
        "video_workflow_name": routing.get("workflow_plan", {}).get("video_workflow_name", ""),
        "refine_workflow_name": routing.get("workflow_plan", {}).get("refine_workflow_name", ""),
        "transition_workflow_name": routing.get("workflow_plan", {}).get("transition_workflow_name", ""),
        "upscale_workflow_name": routing.get("workflow_plan", {}).get("upscale_workflow_name", ""),
        "image_count": routing.get("count_plan", {}).get("image_count"),
        "video_count": routing.get("count_plan", {}).get("video_count"),
        "segment_count": routing.get("count_plan", {}).get("segment_count"),
        "review_selection_limit": routing.get("count_plan", {}).get("review_selection_limit"),
        "sticker_expression_count": routing.get("count_plan", {}).get("sticker_expression_count"),
        "images_per_prompt": routing.get("count_plan", {}).get("images_per_prompt"),
        "routing_reason": routing.get("reason", ""),
        "routing_selection_source": routing.get("selection_source", ""),
        "routing_runtime_context": dict(routing.get("routing_runtime_context", {}) or {}),
        "routing_prompt_mode": routing.get("prompt_mode", ""),
        "workflow_stage_candidates": routing.get("workflow_stage_candidates", {}),
        "generation_type_candidates": routing.get("generation_type_candidates", []),
        "count_policies": routing.get("count_policies", {}),
        "publish_after_generate": publish_after_generate,
        "platform_aliases": platform_aliases,
        "skipped_platforms": skipped_platforms,
        "prompt_source": autonomous_prompt.get("source", "user"),
        "prompt_mode": autonomous_prompt.get("prompt_mode", "user"),
        "creative_seed": autonomous_prompt.get("creative_seed", ""),
        "news_context": autonomous_prompt.get("news_context", {}),
        "fallback_reason": autonomous_prompt.get("fallback_reason", ""),
        "enable_stage_review": bool(
            (enable_review_loop or stage_probe)
            and not no_review
            and
            (stage_probe or (os.getenv("discord_review_bot_token") and os.getenv("discord_review_channel_id")))
            and config_generation_type in {"text2video", "text2image2video", "text2longvideo", "native_h3_story", "native_h3_fl2va_story", "native_h3_l2va_story", "native_h3_ref2va", "text2image2native_h3_ref2va"}
        ),
    }
    constraints.update(
        {
            "native_h3_workflow_name": str(native_recipe.get("workflow_name") or ""),
            "native_h3_keyframe_workflow_name": str(
                native_recipe.get("keyframe_workflow_name")
                or generation.get("keyframe_workflow_name")
                or ""
            ),
            "native_h3_refine_workflow_name": str(
                native_recipe.get("identity_refine_workflow_name")
                or generation.get("identity_refine_workflow_name")
                or ""
            ),
            "native_h3_duration_seconds": int(native_recipe.get("duration_seconds") or duration_seconds),
            "native_h3_width": int(native_recipe.get("width") or dict(generation.get("video_defaults", {}) or {}).get("width", 608)),
            "native_h3_height": int(native_recipe.get("height") or dict(generation.get("video_defaults", {}) or {}).get("height", 352)),
            "native_h3_length": int(native_recipe.get("length") or dict(generation.get("video_defaults", {}) or {}).get("length", 362)),
            "native_h3_steps": int(native_recipe.get("steps") or dict(generation.get("video_defaults", {}) or {}).get("steps", 16)),
            "native_h3_frame_rate": int(native_recipe.get("frame_rate") or dict(generation.get("video_defaults", {}) or {}).get("frame_rate", 24)),
            "native_h3_render_mode": str(
                native_recipe.get("render_mode")
                or (
                    "text_to_video"
                    if config_generation_type == "native_h3_t2v_story"
                    else "first_last_to_video"
                    if config_generation_type == "native_h3_fl2va_story"
                    else "last_frame_to_video"
                    if config_generation_type == "native_h3_l2va_story"
                    else "reference_to_video"
                    if config_generation_type in {"native_h3_ref2va", "text2image2native_h3_ref2va"}
                    else "image_to_video"
                )
            ),
        }
    )
    if agentic_media_type == "long_video":
        # Keep long-video policy in the shared strategy config.  The planner
        # consumes provider-neutral names; legacy H3 aliases are normalized at
        # that boundary without creating an H3-only planner path.
        longvideo_config_keys = {
            "mix_weights": "longvideo_mix_weights",
            "mix_seed": "longvideo_mix_seed",
            "review_policy": "longvideo_review_policy",
            "workflow_names": "longvideo_workflow_names",
            "frame_candidate_count": "longvideo_frame_candidate_count",
            "reference_candidate_count": "longvideo_reference_candidate_count",
            "reference_selection_limit": "longvideo_reference_selection_limit",
            "width": "longvideo_width",
            "height": "longvideo_height",
            "length": "longvideo_length",
            "steps": "longvideo_steps",
            "model_profile": "longvideo_model_profile",
        }
        for source_key, target_key in longvideo_config_keys.items():
            value = longvideo_config.get(source_key)
            if value not in (None, "", []):
                constraints[target_key] = (
                    dict(value)
                    if isinstance(value, dict)
                    else list(value)
                    if isinstance(value, list)
                    else value
                )
    return {
        "prompt": resolved_prompt,
        "media_type": agentic_media_type,
        "duration_seconds": duration_seconds,
        "style": style,
        "auto_download_assets": False,
        "constraints": constraints,
        "source_generation_type": config_generation_type,
        "selected_workflow_name": routing.get("workflow_name", ""),
        "character_name": character_name,
        "resolved_output_dir": str(resolved_output_dir),
        "routing": routing,
        "prompt_generation": autonomous_prompt,
        "character_config_summary": _summarize_character_config(
            path=path,
            character=character,
            generation=generation,
            social_media=social_media,
            strategies=strategies,
            platform_configs=platform_configs,
        ),
        "routing_summary": _build_routing_summary(
            character_name=character_name,
            prompt=resolved_prompt,
            source_generation_type=config_generation_type,
            agentic_media_type=agentic_media_type,
            duration_seconds=duration_seconds,
            output_dir=str(resolved_output_dir),
            routing=routing,
        ),
    }


def run_character_workflow(
    repo_root: Path,
    config_path: str | Path,
    *,
    prompt: str = "",
    temperature: float = 1.0,
    preferred_generation_type: str | None = None,
    duration_seconds: int | None = None,
    dry_run_publish: bool = False,
    publish_mode: str = "",
    publish_platforms: list[str] | None = None,
    publish_after_generate: bool = True,
    output_dir: str | None = None,
    enable_review_loop: bool = False,
    review_notes: str = "",
    no_review: bool = False,
    stage_probe: bool = False,
    news_driven: bool = False,
    news_history_path: str | None = None,
    routing_history_path: str | None = None,
    comfy_host: str | None = None,
    comfy_port: int | None = None,
    comfy_root: str | None = None,
    auto_download_assets: bool = False,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    effective_publish_after_generate = _effective_publish_after_generate(
        requested=publish_after_generate,
        stage_probe=stage_probe,
    )
    run_id = uuid.uuid4().hex[:12]
    recorder = RunRecorder(repo_root / "agentic" / "logs" / "runs", run_id)
    logger, log_path = create_run_logger(repo_root / "agentic" / "logs" / "runs", run_id, recorder=recorder)
    os.environ["AGENTIC_RUN_LOGGER_NAME"] = logger.name
    logger.info("workflow.start | run_id=%s | config=%s", run_id, Path(config_path).resolve())
    logger.info("character.config.load | path=%s", Path(config_path).resolve())
    payload = build_goal_payload_from_character_config(
        repo_root,
        config_path,
        prompt=prompt,
        temperature=temperature,
        preferred_generation_type=preferred_generation_type,
        duration_seconds=duration_seconds,
        dry_run_publish=dry_run_publish,
        publish_mode=publish_mode,
        publish_platforms=publish_platforms,
        publish_after_generate=effective_publish_after_generate,
        output_dir=output_dir,
        enable_review_loop=enable_review_loop,
        review_notes=review_notes,
        no_review=no_review,
        stage_probe=stage_probe,
        news_driven=news_driven,
        news_history_path=news_history_path,
        routing_history_path=routing_history_path,
        asset_root=_resolve_routing_asset_root(repo_root, comfy_root),
        rng=rng,
    )
    logger.info(
        "routing.selected | strategy=%s | media_type=%s | workflow=%s | prompt_source=%s",
        payload["source_generation_type"],
        payload["media_type"],
        payload["selected_workflow_name"],
        payload["constraints"].get("prompt_source", ""),
    )
    logger.info("character.config.summary | payload=%s", json.dumps(payload.get("character_config_summary", {}), ensure_ascii=False))
    if payload["constraints"].get("news_context"):
        logger.info("news.context | payload=%s", json.dumps(payload["constraints"]["news_context"], ensure_ascii=False))
    planner, runner, run_memory = build_runtime(
        repo_root / "agentic",
        output_root=Path(payload["resolved_output_dir"]),
        comfy_host=comfy_host,
        comfy_port=comfy_port,
        comfy_root=Path(comfy_root).resolve() if comfy_root else None,
        run_id=run_id,
        logger=logger,
        recorder=recorder,
    )
    goal = planner.create_goal(
        prompt=payload["prompt"],
        media_type=payload["media_type"],
        duration_seconds=int(payload["duration_seconds"]),
        style=str(payload["style"]),
        auto_download_assets=auto_download_assets,
        constraints=dict(payload["constraints"]),
    )
    plan = planner.build_plan(goal)
    logger.info("plan.built | workflow=%s | node_count=%s", plan.workflow_name, len(plan.nodes))
    generation_result = runner.run(plan)
    generation_dict = generation_result.to_dict()
    generation_media_paths = collect_media_paths_from_run_result(generation_dict)
    logger.info("generation.finished | status=%s", generation_result.status)

    publish_dict: dict[str, Any] | None = None
    if effective_publish_after_generate and generation_result.status == "success":
        media_paths = generation_media_paths
        platform_configs = payload["constraints"].get("platform_configs", {})
        if media_paths and isinstance(platform_configs, dict) and platform_configs:
            logger.info("publish.start | media_count=%s | platforms=%s", len(media_paths), list(platform_configs.keys()))
            publish_prompt, publish_prompt_source = _resolve_publish_prompt(
                generation_dict,
                fallback_prompt=str(payload["prompt"]),
            )
            publish_goal = planner.create_goal(
                prompt=publish_prompt,
                media_type="publish_review",
                duration_seconds=0,
                style=str(payload["style"]),
                auto_download_assets=False,
                constraints={
                    "media_paths": media_paths,
                    "platforms": list(platform_configs.keys()),
                    "platform_configs": dict(platform_configs),
                    "hashtags": list(payload["constraints"].get("hashtags", [])),
                    "social_post_format": True,
                    "character": payload["character_name"],
                    "dry_run": dry_run_publish,
                    "publish_mode": str(payload["constraints"].get("publish_mode") or ""),
                    "output_dir": str(Path(payload["resolved_output_dir"]) / "publish_ready"),
                    "selection_limit": _publish_selection_limit(
                        media_paths,
                        payload["constraints"].get("review_selection_limit"),
                    ),
                    "review_notes": review_notes,
                    # Every publishable media artifact must be explicitly
                    # approved in Discord before any platform dispatch.
                    "require_human_review": True,
                    "review_scope": _publish_review_scope(media_paths),
                    "review_all_candidates": _publish_review_scope(media_paths) == "final_media",
                    "publish_prompt_source": publish_prompt_source,
                    "visual_grounding": _extract_publish_visual_grounding(generation_dict),
                    "news_context": dict(payload["constraints"].get("news_context") or {}),
                    "news_grounding_required": bool(
                        payload["constraints"].get("news_driven")
                        or payload["source_generation_type"] in NEWS_GROUNDED_GENERATION_TYPES
                    ),
                    "news_trace_contract": (
                        "source context -> active mechanism -> visible consequence"
                    ),
                },
            )
            publish_plan = planner.build_plan(publish_goal)
            publish_result = runner.run(publish_plan)
            publish_result_dict = publish_result.to_dict()
            publish_state_data = publish_result_dict.get("state") or {}
            publish_node_outputs = (
                publish_state_data.get("node_outputs", {})
                if isinstance(publish_state_data, dict)
                else {}
            )
            publish_outputs = (
                publish_node_outputs.get("dispatch-publish", {})
                if isinstance(publish_node_outputs, dict)
                else {}
            )
            publish_state = str(publish_outputs.get("publication_state") or "unknown")
            publicly_visible = bool(publish_outputs.get("publicly_visible", False))
            logger.info(
                "publish.finished | status=%s | publication_state=%s | publicly_visible=%s",
                publish_result.status,
                publish_state,
                publicly_visible,
            )
            publish_dict = {
                "plan": publish_plan.to_dict(),
                "result": publish_result_dict,
                "prompt_summary": _build_prompt_summary(publish_result_dict.get("state", {})),
            }
    failure_details = _extract_failure_details(generation_dict)
    overall_status = "success" if generation_result.status == "success" else "failed"
    if publish_dict:
        publish_result_status = str((publish_dict.get("result") or {}).get("status") or "").lower()
        if publish_result_status not in {"success"}:
            overall_status = "failed"
            publish_failure_details = _extract_failure_details(publish_dict.get("result"))
            if not failure_details:
                failure_details = publish_failure_details
            if not failure_details:
                publish_result_data = publish_dict.get("result")
                publish_state_data = (
                    publish_result_data.get("state", {})
                    if isinstance(publish_result_data, dict)
                    else {}
                )
                publish_node_outputs = (
                    publish_state_data.get("node_outputs", {})
                    if isinstance(publish_state_data, dict)
                    else {}
                )
                dispatch_outputs = (
                    publish_node_outputs.get("dispatch-publish", {})
                    if isinstance(publish_node_outputs, dict)
                    else {}
                )
                dispatch_errors = (
                    dispatch_outputs.get("errors", {})
                    if isinstance(dispatch_outputs, dict)
                    else {}
                )
                if dispatch_errors:
                    failure_details = {"failure_reason": f"Publish dispatch errors: {dispatch_errors}"}
    logger.info(
        "workflow.end | run_id=%s | status=%s | log_path=%s | failure_reason=%s",
        run_id,
        overall_status,
        log_path,
        failure_details.get("failure_reason", ""),
    )

    discord_notification: dict[str, Any] = {}
    if publish_dict:
        publish_result = publish_dict.get("result") or {}
        publish_state_data = publish_result.get("state") if isinstance(publish_result, dict) else {}
        publish_node_outputs = (
            publish_state_data.get("node_outputs", {})
            if isinstance(publish_state_data, dict)
            else {}
        )
        dispatch_outputs = (
            publish_node_outputs.get("dispatch-publish", {})
            if isinstance(publish_node_outputs, dict)
            else {}
        )
        platform_results = dispatch_outputs.get("results", {}) if isinstance(dispatch_outputs, dict) else {}
        errors = dispatch_outputs.get("errors", {}) if isinstance(dispatch_outputs, dict) else {}
        discord_lines = [
            f"MediaOverload | {payload['character_name']} | run {run_id}",
            f"Publish: {str(dispatch_outputs.get('status') or publish_result.get('status') or overall_status)} | "
            f"state={str(dispatch_outputs.get('publication_state') or 'unknown')} | "
            f"public={bool(dispatch_outputs.get('publicly_visible', False))}",
            f"Platforms: {json.dumps(platform_results, ensure_ascii=False)}",
        ]
        if errors:
            discord_lines.append(f"Errors: {json.dumps(errors, ensure_ascii=False)[:700]}")
        discord_lines.append(f"Log: {log_path}")
        discord_notification = DiscordRunNotificationService().notify("\n".join(discord_lines))
        logger.info(
            "discord.notification.end | status=%s | message_id=%s",
            str(discord_notification.get("status") or "unknown"),
            str(discord_notification.get("message_id") or ""),
        )

    review_status = _review_stage_status(generation_dict)
    publish_status = _publish_stage_status(
        publish_dict,
        requested=effective_publish_after_generate,
        generation_status=str(generation_result.status),
        media_paths=generation_media_paths,
        platform_configs=payload["constraints"].get("platform_configs", {}),
    )
    result_payload = {
        "status": overall_status,
        "failure_reason": failure_details.get("failure_reason"),
        "failure_node": failure_details.get("failure_node"),
        "failure_skill": failure_details.get("failure_skill"),
        "run_id": run_id,
        "log_path": str(log_path),
        "config_path": str(Path(config_path).resolve()),
        "source_generation_type": payload["source_generation_type"],
        "agentic_media_type": payload["media_type"],
        "character": payload["character_name"],
        "routing_summary": payload.get("routing_summary", {}),
        "plan": plan.to_dict(),
        "generation": {
            "result": generation_dict,
            "prompt_summary": _build_prompt_summary(generation_dict.get("state", {})),
        },
        "routing": payload.get("routing", {}),
        "character_config_summary": payload.get("character_config_summary", {}),
        "publish": publish_dict,
        "stage_status": {
            "render": {
                "status": str(generation_result.status),
                "artifact_count": len(generation_media_paths),
            },
            "review": {"status": review_status},
            "publish": {"status": publish_status},
        },
        "artifacts": {
            "media_paths": generation_media_paths,
            "video_paths": [
                path for path in generation_media_paths if Path(path).suffix.lower() in VIDEO_EXTENSIONS
            ],
            "image_paths": [
                path for path in generation_media_paths if Path(path).suffix.lower() not in VIDEO_EXTENSIONS
            ],
        },
        "discord_notification": discord_notification,
        "memory": run_memory.as_serializable(),
    }
    recorder.finalize(result_payload)
    return result_payload


def collect_media_paths_from_run_result(run_result: dict[str, Any]) -> list[str]:
    """Return final media artifacts for the publish review boundary.

    Generation nodes intentionally expose intermediate assets (opening/ending
    frames, segment clips, and previews) for downstream workflow steps. Those
    are not publish candidates. Prefer explicit package-node media paths and
    only fall back to media artifacts when an older plan has no package node
    output. Images and videos are both valid final media.
    """
    state = dict(run_result.get("state", {}) or {})
    node_outputs = dict(state.get("node_outputs", {}) or {})
    collected: list[str] = []

    def append_media(value: object) -> None:
        values = value if isinstance(value, list) else [value]
        for item in values:
            path = str(item or "")
            if Path(path).suffix.lower() in MEDIA_EXTENSIONS and path not in collected:
                collected.append(path)

    preferred_package_nodes = (
        "native-h3-package",
        "package-outputs",
        "collect-longvideo-outputs",
        "collect-text2img2video-outputs",
        "collect-text2video-outputs",
    )
    for node_id in preferred_package_nodes:
        outputs = node_outputs.get(node_id)
        if not isinstance(outputs, dict):
            continue
        # Explicit final paths take precedence over package saved_files. A
        # video package may also expose opening/ending PNGs in saved_files;
        # those are continuity artifacts, not publish candidates.
        for key in ("media_paths", "video_path", "final_video_path", "image_path", "final_image_path"):
            append_media(outputs.get(key))
        if collected:
            return collected

    for outputs in node_outputs.values():
        if not isinstance(outputs, dict):
            continue
        for key in ("video_path", "final_video_path", "saved_files", "media_paths"):
            append_media(outputs.get(key))
    # Older plans do not have an explicit package node. Once a real video is
    # present, opening frames and GIF previews are continuity artifacts, not
    # publish candidates. If the workflow only produced a GIF, keep that as
    # the final animated asset instead of mixing it with source frames.
    video_paths = [path for path in collected if Path(path).suffix.lower() in VIDEO_EXTENSIONS]
    if video_paths:
        return video_paths
    gif_paths = [path for path in collected if Path(path).suffix.lower() == ".gif"]
    if gif_paths:
        return gif_paths
    return collected


def _publish_review_scope(media_paths: list[str]) -> str:
    """Choose a review contract based on the final media, not the producer."""
    if media_paths and all(Path(path).suffix.lower() in VIDEO_EXTENSIONS for path in media_paths):
        return "final_video"
    return "final_media"


def _publish_selection_limit(media_paths: list[str], configured_limit: Any) -> int:
    """Keep video publish review single-select while allowing image media sets."""
    if _publish_review_scope(media_paths) == "final_video":
        return 1
    try:
        limit = int(configured_limit or 0)
    except (TypeError, ValueError):
        limit = 0
    return max(1, min(len(media_paths), limit or len(media_paths)))


def _review_stage_status(run_result: dict[str, Any]) -> str:
    records = run_result.get("records") if isinstance(run_result, dict) else None
    review_records = [
        record
        for record in (records if isinstance(records, list) else [])
        if isinstance(record, dict)
        and (
            str(record.get("skill_name") or "").startswith("review.")
            or "review" in str(record.get("node_id") or "").lower()
        )
    ]
    if not review_records:
        return "not_required"
    if any(str(record.get("status") or "").lower() not in {"success", "skipped"} for record in review_records):
        return "failed"
    return "success"


def _publish_stage_status(
    publish_result: dict[str, Any] | None,
    *,
    requested: bool,
    generation_status: str,
    media_paths: list[str],
    platform_configs: Any,
) -> str:
    if not requested:
        return "not_requested"
    if generation_status != "success":
        return "not_run_generation_failed"
    if not media_paths:
        return "skipped_no_media"
    if not isinstance(platform_configs, dict) or not platform_configs:
        return "skipped_no_compatible_platform"
    if not publish_result:
        return "not_run"
    result = publish_result.get("result") if isinstance(publish_result, dict) else None
    if isinstance(result, dict):
        if str(result.get("dispatch_status") or "").lower() == "skipped":
            return "skipped_no_compatible_platform"
        return str(result.get("status") or "unknown")
    return "unknown"


def _effective_publish_after_generate(*, requested: bool, stage_probe: bool) -> bool:
    """Never publish an automatically selected stage-probe artifact."""
    return bool(requested and not stage_probe)


def _resolve_routing_asset_root(repo_root: Path, explicit_root: str | None) -> Path | None:
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()
    configured_root = os.environ.get("COMFYUI_ROOT", "").strip()
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    container_root = Path("/comfyui")
    if container_root.is_dir():
        return container_root.resolve()
    portable_root = Path(r"D:\ComfyUI_windows_portable")
    if portable_root.is_dir():
        return portable_root.resolve()
    return None


def _resolve_publish_prompt(generation_result: dict[str, Any], *, fallback_prompt: str) -> tuple[str, str]:
    """Use the story actually rendered by generation when preparing publish copy."""
    state = generation_result.get("state") or {}
    node_outputs = state.get("node_outputs") if isinstance(state, dict) else {}
    if isinstance(node_outputs, dict):
        native_story = node_outputs.get("native-story-prompt")
        if isinstance(native_story, dict):
            generated_storyboard = native_story.get("generated_storyboard")
            if isinstance(generated_storyboard, dict):
                compact_context = _build_publish_story_context(
                    generated_storyboard,
                    news_context=native_story.get("news_context"),
                )
                if compact_context:
                    return compact_context, "native_h3_story"
            native_prompt = str(native_story.get("prompt") or "").strip()
            if native_prompt:
                return native_prompt, "native_h3_story"
        native_render = node_outputs.get("native-h3-render")
        if isinstance(native_render, dict):
            native_prompt = str(native_render.get("native_h3_prompt") or "").strip()
            if native_prompt:
                return native_prompt, "native_h3_render"
    return str(fallback_prompt or "").strip(), "goal_prompt"


def _extract_publish_visual_grounding(run_result: dict[str, Any]) -> dict[str, Any]:
    """Expose only compact, visible-content QA evidence to caption generation."""
    state = run_result.get("state") or {}
    node_outputs = state.get("node_outputs") if isinstance(state, dict) else {}
    qa = node_outputs.get("native-h3-qa") if isinstance(node_outputs, dict) else None
    contact_sheet_path = str(qa.get("contact_sheet_path") or "").strip() if isinstance(qa, dict) else ""
    semantic_qa = qa.get("semantic_qa") if isinstance(qa, dict) else None
    if not isinstance(semantic_qa, dict) or not semantic_qa.get("enabled"):
        return {"contact_sheet_path": contact_sheet_path} if contact_sheet_path else {}
    return {
        "contact_sheet_path": contact_sheet_path,
        "status": str(semantic_qa.get("status") or "unknown"),
        "passed": semantic_qa.get("passed"),
        "observed_story": str(semantic_qa.get("observed_story") or ""),
        "caption_guidance": str(semantic_qa.get("caption_guidance") or ""),
        "issues": [str(item) for item in (semantic_qa.get("issues") or []) if str(item)],
        "checks": dict(semantic_qa.get("checks") or {}) if isinstance(semantic_qa.get("checks"), dict) else {},
    }


def _build_publish_story_context(
    storyboard: dict[str, Any],
    *,
    news_context: Any = None,
) -> str:
    """Give caption generation the story/news contract, not the full render prompt."""
    spine = storyboard.get("story_spine") if isinstance(storyboard.get("story_spine"), dict) else {}
    trace = storyboard.get("news_trace") if isinstance(storyboard.get("news_trace"), dict) else {}
    news = news_context if isinstance(news_context, dict) else {}
    lines = [
        f"Video story: {str(storyboard.get('name') or '').strip()}",
        f"Premise: {str(spine.get('premise') or '').strip()}",
        f"Objective: {str(spine.get('objective') or '').strip()}",
        f"Resolution: {str(spine.get('resolution') or '').strip()}",
        f"News headline: {str(news.get('title') or trace.get('source_title') or '').strip()}",
        f"News visual translation: {str(trace.get('visual_translation') or '').strip()}",
        f"News visual anchors: {', '.join(str(item).strip() for item in trace.get('visual_anchors', []) if str(item).strip())}",
        "Write publish copy about the rendered video. Do not paste the full production prompt or unrelated news text.",
    ]
    return "\n".join(line for line in lines if line.split(": ", 1)[-1].strip())


def dumps_result(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _pick_primary_style(style_weights: Any) -> str:
    if not isinstance(style_weights, dict) or not style_weights:
        return "cinematic surreal"
    ranked = [
        (str(style), float(weight))
        for style, weight in style_weights.items()
        if str(style).strip()
    ]
    if not ranked:
        return "cinematic surreal"
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[0][0]


def _weighted_choice(weights: dict[str, Any], rng: random.Random | None = None) -> str:
    filtered = [(str(name), float(weight)) for name, weight in weights.items() if float(weight) > 0]
    if not filtered:
        return "text2longvideo"
    chooser = rng or random
    total = sum(weight for _, weight in filtered)
    threshold = chooser.uniform(0, total)
    cumulative = 0.0
    for name, weight in filtered:
        cumulative += weight
        if threshold <= cumulative:
            return name
    return filtered[-1][0]


def _weighted_candidate_choice(
    weights: dict[str, Any],
    candidates: list[str],
    *,
    aliases: dict[str, Any] | None = None,
    rng: random.Random | None = None,
    state_path: Path | None = None,
    diversity_config: dict[str, Any] | None = None,
) -> str:
    """Pick a configured strategy without selecting an unavailable route."""
    allowed = {str(candidate).strip() for candidate in candidates if str(candidate).strip()}
    normalized_weights: dict[str, float] = {}
    for name, raw_weight in weights.items():
        normalized = _normalize_generation_type(str(name), aliases)
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            continue
        if normalized and normalized in allowed and weight > 0:
            normalized_weights[normalized] = weight
    if not normalized_weights:
        normalized_weights = {candidate: 1.0 for candidate in candidates if str(candidate).strip()}
    return select_weighted_route(
        normalized_weights,
        list(normalized_weights),
        rng=rng,
        state_path=state_path,
        diversity_config=diversity_config,
    )


def _select_workflow_candidate(
    candidates: list[str],
    *,
    weights: dict[str, Any],
    rng: random.Random | None,
    asset_registry: AssetRegistry | None,
) -> str:
    if not candidates:
        return ""
    qualified = _asset_qualified_workflow_candidates(candidates, asset_registry)
    if rng is None:
        return qualified[0]
    positive_weights: dict[str, Any] = {
        candidate: weights.get(candidate, 1)
        for candidate in qualified
        if _positive_weight(weights.get(candidate, 1))
    }
    return _weighted_choice(positive_weights or {candidate: 1 for candidate in qualified}, rng=rng)


def _positive_weight(value: Any) -> bool:
    try:
        return float(value or 0) > 0
    except (TypeError, ValueError):
        return False


def _asset_qualified_workflow_candidates(
    candidates: list[str],
    asset_registry: AssetRegistry | None,
) -> list[str]:
    if asset_registry is None:
        return candidates
    qualified: list[str] = []
    for candidate in candidates:
        try:
            statuses = asset_registry.ensure_workflow_ready(candidate, auto_download=False).get("asset_status", [])
        except (KeyError, OSError, ValueError):
            continue
        # An empty asset manifest means "unverified", not "ready". Treating
        # it as ready made stage weights select workflow JSONs whose model
        # files are not installed on the active ComfyUI host.
        if not isinstance(statuses, list) or not statuses:
            continue
        if all(
            isinstance(item, dict) and str(item.get("status") or "").lower() == "ready"
            for item in statuses
        ):
            qualified.append(candidate)
    # If no candidate is verified, keep one deterministic fallback so the
    # runtime asset-check node can emit the precise missing-asset diagnostic;
    # do not let an unverified list re-enter weighted selection.
    return qualified or candidates[:1]


def _route_generation_from_character_config(
    repo_root: Path,
    config: dict[str, Any],
    *,
    character_name: str,
    style: str,
    prompt: str,
    preferred_generation_type: str | None,
    requested_duration_seconds: int | None,
    rng: random.Random | None,
    news_driven: bool = False,
    routing_history_path: Path | None = None,
    asset_root: Path | None = None,
) -> dict[str, Any]:
    generation = dict(config.get("generation", {}) or {})
    additional_params = dict(config.get("additional_params", {}) or {})
    strategies = dict(additional_params.get("strategies", {}) or {})
    routing_config = _load_global_routing_config(repo_root)
    preferred_generation_type = _normalize_generation_type(
        preferred_generation_type,
        dict(routing_config.get("strategy_aliases", {}) or {}),
    )
    routing_overrides = dict(generation.get("routing_overrides", {}) or {})
    merged_routing = _merge_routing_config(routing_config, routing_overrides)
    if merged_routing.get("enabled", True) is False:
        raise ValueError("Global routing is disabled. LLM routing is required for this workflow.")
    routing_runtime_context = _build_h3_reference_runtime_context(repo_root, generation)
    generation_type_candidates = _collect_generation_type_candidates(
        merged_routing,
        preferred_generation_type,
        routing_runtime_context=routing_runtime_context,
    )
    workflow_stage_candidates = _collect_workflow_stage_candidates(
        repo_root,
        generation,
        strategies,
        merged_routing,
        generation_type_candidates,
    )
    count_policies = _collect_count_policies(
        merged_routing,
        generation_type_candidates,
        workflow_stage_candidates=workflow_stage_candidates,
    )
    asset_registry = (
        AssetRegistry(repo_root / "agentic", asset_root=asset_root)
        if asset_root is not None
        else None
    )
    workflow_selection_weights = dict(merged_routing.get("workflow_selection_weights", {}) or {})

    def build_fixed_route(
        selected_generation_type: str,
        reason: str,
        *,
        selection_source: str,
        prompt_mode: str,
    ) -> dict[str, Any]:
        workflow_plan: dict[str, str] = {}
        for stage_key in (
            "image_workflow_name",
            "video_workflow_name",
            "refine_workflow_name",
            "transition_workflow_name",
            "upscale_workflow_name",
        ):
            candidates = list(
                workflow_stage_candidates.get(selected_generation_type, {}).get(stage_key, [])
            )
            workflow_plan[stage_key] = _select_workflow_candidate(
                candidates,
                weights=dict(workflow_selection_weights.get(stage_key, {}) or {}),
                # An explicit generation-type override fixes only the route
                # family. Keep stage workflow selection weighted when the
                # scheduler supplied an RNG; otherwise every explicit I2V
                # request silently falls back to the first candidate (Krea).
                rng=rng,
                asset_registry=asset_registry,
            )
        count_plan = {
            count_key: _pick_policy_value(policy)
            for count_key, policy in dict(count_policies.get(selected_generation_type, {}) or {}).items()
        }
        return {
            "generation_type": selected_generation_type,
            "workflow_name": _primary_workflow_name(selected_generation_type, workflow_plan),
            "workflow_plan": workflow_plan,
            "count_plan": count_plan,
            "reason": reason,
            "prompt_mode": prompt_mode,
            "selection_source": selection_source,
            "workflow_selection_mode": "weighted_random" if rng is not None else "first_qualified",
            "llm_backend": {},
            "workflow_stage_candidates": workflow_stage_candidates,
            "generation_type_candidates": generation_type_candidates,
            "count_policies": count_policies,
            "routing_runtime_context": routing_runtime_context,
            "pre_video_review": dict(merged_routing.get("pre_video_review", {}) or {}),
            "longvideo_config": dict(merged_routing.get("longvideo_config", {}) or {}),
        }

    if preferred_generation_type:
        selected_generation_type = str(preferred_generation_type).strip()
        return build_fixed_route(
            selected_generation_type,
            "preferred_generation_type override",
            selection_source="explicit_override",
            prompt_mode="override",
        )

    duration_strategy = _duration_strategy_override(
        requested_duration_seconds,
        generation_type_candidates,
    )
    if duration_strategy:
        return build_fixed_route(
            duration_strategy,
            f"duration policy selected {duration_strategy} for {int(requested_duration_seconds)} seconds",
            selection_source="duration_policy",
            prompt_mode="duration_policy",
        )

    # The exact public command `python run_media_interface.py --character X`
    # is intentionally deterministic: it is an autonomous E2E production run,
    # not an invitation for the strategy LLM to choose a shortcut. Keep
    # explicit prompts, generation types, and durations on their existing
    # paths; only the empty bare invocation gets this policy.
    if (
        not str(prompt).strip()
        and not news_driven
        and requested_duration_seconds is None
    ):
        selected_generation_type = next(
            (
                candidate
                for candidate in AUTONOMOUS_E2E_ROUTE_PREFERENCE
                if candidate in generation_type_candidates
            ),
            generation_type_candidates[0],
        )
        return build_fixed_route(
            selected_generation_type,
            "bare character invocation requires the complete news-to-story-to-media workflow",
            selection_source="autonomous_e2e_default",
            prompt_mode="autonomous_e2e",
        )

    if rng is not None:
        selected_generation_type = _weighted_candidate_choice(
            dict(generation.get("generation_type_weights", {}) or {}),
            generation_type_candidates,
            aliases=dict(routing_config.get("strategy_aliases", {}) or {}),
            rng=rng,
            state_path=routing_history_path,
            diversity_config=dict(routing_config.get("route_diversity", {}) or {}),
        )
        return build_fixed_route(
            selected_generation_type,
            f"weighted random selected {selected_generation_type} from character generation_type_weights",
            selection_source="weighted_random",
            prompt_mode="weighted_random",
        )

    route_prompt = str(prompt).strip() or character_name
    engine = LLMPromptEngine(mode=os.environ.get("AGENTIC_LLM_MODE", "llm"))
    raw_routing_hints = merged_routing.get("routing_hints", {})
    routing_hints = dict(raw_routing_hints) if isinstance(raw_routing_hints, dict) else {}
    routing_hints["runtime_context"] = routing_runtime_context
    result = engine.route_generation_strategy(
        prompt=route_prompt,
        character=character_name,
        style=style,
        generation_type_candidates=generation_type_candidates,
        workflow_stage_candidates=workflow_stage_candidates,
        count_policies=count_policies,
        routing_hints=routing_hints if isinstance(routing_hints, dict) else {},
        preferred_generation_type=preferred_generation_type,
    )
    selected_generation_type = str(result["generation_type"]).strip()
    workflow_plan = {str(key): str(value or "").strip() for key, value in dict(result["workflow_plan"]).items()}
    count_plan = {str(key): int(value) for key, value in dict(result["count_plan"]).items()}
    return {
        "generation_type": selected_generation_type,
        "workflow_name": _primary_workflow_name(selected_generation_type, workflow_plan),
        "workflow_plan": workflow_plan,
        "count_plan": count_plan,
        "reason": str(result.get("reason") or "").strip(),
        "prompt_mode": str(result["prompt_mode"]),
        "selection_source": "llm",
        "llm_backend": result.get("llm_backend") or engine.backend_info(),
        "workflow_stage_candidates": workflow_stage_candidates,
        "generation_type_candidates": generation_type_candidates,
        "count_policies": count_policies,
        "routing_runtime_context": routing_runtime_context,
        "pre_video_review": dict(merged_routing.get("pre_video_review", {}) or {}),
        "longvideo_config": dict(merged_routing.get("longvideo_config", {}) or {}),
    }


def _build_h3_reference_runtime_context(
    repo_root: Path,
    generation: dict[str, Any],
) -> dict[str, Any]:
    """Expose only safe reference availability facts to the routing LLM.

    Ref2VA can be selected explicitly and receive assets later from the
    planner's review stage, but automatic routing must not treat an empty or
    unreadable character-config manifest as an available reference input.
    """

    recipe = dict(generation.get("native_h3_ref2va", {}) or {})
    manifest = recipe.get("reference_manifest")
    if manifest is None:
        manifest = generation.get("reference_manifest")
    image_paths = recipe.get("reference_image_paths")
    if image_paths is None:
        image_paths = generation.get("reference_image_paths")
    video_paths = recipe.get("reference_video_paths")
    if video_paths is None:
        video_paths = generation.get("reference_video_paths")
    max_images = int(recipe.get("reference_max_images") or generation.get("reference_max_images") or 9)
    max_videos = int(recipe.get("reference_max_videos") or generation.get("reference_max_videos") or 3)

    def resolve_path(value: Any) -> str:
        path = Path(str(value or "")).expanduser()
        if not path.is_absolute():
            path = repo_root / path
        return str(path)

    def resolve_manifest(values: Any) -> Any:
        if values is None:
            return None
        if not isinstance(values, (list, tuple)):
            return values
        resolved: list[Any] = []
        for item in values:
            if isinstance(item, dict):
                record = dict(item)
                key = "path" if record.get("path") is not None else "source_path"
                if record.get(key):
                    record[key] = resolve_path(record[key])
                resolved.append(record)
            else:
                resolved.append(resolve_path(item))
        return resolved

    resolved_manifest = resolve_manifest(manifest)
    resolved_image_paths = [resolve_path(item) for item in (image_paths or [])]
    resolved_video_paths = [resolve_path(item) for item in (video_paths or [])]
    has_configured_values = bool(resolved_manifest or resolved_image_paths or resolved_video_paths)
    base_context = {
        "source": "character_config",
        "reference_manifest_available": False,
        "reference_image_count": 0,
        "reference_video_count": 0,
        # An empty manifest is not a dead end: the automatic Ref2VA route can
        # create six T2I candidates and ask Discord to select references.
        "automatic_ref2va_eligible": not has_configured_values,
        "reference_manifest_error_code": "auto_generation_available" if not has_configured_values else "unverified",
    }
    if not has_configured_values:
        return base_context

    try:
        references = normalize_reference_manifest(
            resolved_manifest,
            image_paths=resolved_image_paths,
            video_paths=resolved_video_paths,
            require_files=True,
            max_images=max_images,
            max_videos=max_videos,
        )
    except FileNotFoundError:
        base_context["reference_manifest_error_code"] = "missing_file"
        return base_context
    except (TypeError, ValueError):
        base_context["reference_manifest_error_code"] = "invalid"
        return base_context

    image_count = sum(1 for reference in references if reference.get("type") == "image")
    video_count = sum(1 for reference in references if reference.get("type") == "video")
    return {
        **base_context,
        "reference_manifest_available": True,
        "reference_image_count": image_count,
        "reference_video_count": video_count,
        "automatic_ref2va_eligible": True,
        "reference_manifest_error_code": "",
    }


def _collect_generation_type_candidates(
    routing_config: dict[str, Any],
    preferred_generation_type: str | None,
    routing_runtime_context: dict[str, Any] | None = None,
) -> list[str]:
    explicit = routing_config.get("strategy_candidates") or routing_config.get("allowed_generation_types") or []
    candidates: list[str] = []
    if preferred_generation_type:
        candidates.append(str(preferred_generation_type).strip())
    elif isinstance(explicit, list):
        candidates.extend(str(item).strip() for item in explicit if str(item).strip())
    if not preferred_generation_type and isinstance(routing_runtime_context, dict):
        if not bool(routing_runtime_context.get("automatic_ref2va_eligible", False)):
            candidates = [candidate for candidate in candidates if candidate != "native_h3_ref2va"]
    ordered: list[str] = []
    for candidate in candidates:
        if not candidate or candidate in ordered:
            continue
        ordered.append(candidate)
    if not ordered:
        raise ValueError("Routing config must provide at least one strategy candidate.")
    return ordered


def _collect_workflow_stage_candidates(
    repo_root: Path,
    generation: dict[str, Any],
    strategies: dict[str, Any],
    routing_config: dict[str, Any],
    generation_type_candidates: list[str],
) -> dict[str, dict[str, list[str]]]:
    workflow_stage_candidates: dict[str, dict[str, list[str]]] = {}
    explicit_candidates = routing_config.get("workflow_stage_candidates", {}) or routing_config.get("workflow_candidates", {})
    if not isinstance(explicit_candidates, dict):
        explicit_candidates = {}
    generation_workflows = dict(generation.get("workflows", {}) or {})

    for generation_type in generation_type_candidates:
        stage_candidates: dict[str, list[str]] = {}
        explicit = explicit_candidates.get(generation_type, {})
        strategy = dict(strategies.get(generation_type, {}) or {})
        if isinstance(explicit, list):
            explicit = {"image_workflow_name": explicit}
        if not isinstance(explicit, dict):
            explicit = {}

        inferred: dict[str, list[Any]] = {}
        if generation_type == "sticker_pack":
            inferred = {
                "image_workflow_name": [
                    _nested_get(strategy, "static_config", "workflow_name"),
                    _nested_get(strategy, "static_config", "workflow_path"),
                    generation_workflows.get("sticker_pack"),
                    generation_workflows.get("text2img"),
                ],
                "video_workflow_name": [
                    _nested_get(strategy, "animated_config", "workflow_name"),
                    _nested_get(strategy, "animated_config", "i2v_workflow_path"),
                ],
            }
        elif generation_type == "text2image2video":
            inferred = {
                "image_workflow_name": [
                    _nested_get(strategy, "first_stage", "workflow_name"),
                    _nested_get(strategy, "first_stage", "t2i_workflow_path"),
                    _nested_get(strategy, "first_stage", "workflow_path"),
                    generation_workflows.get("text2img"),
                ],
                "video_workflow_name": [
                    _nested_get(strategy, "video", "workflow_name"),
                    _nested_get(strategy, "video", "i2v_workflow_path"),
                    generation_workflows.get("text2image2video"),
                ],
                "upscale_workflow_name": [
                    _nested_get(strategy, "first_stage", "upscale_workflow_name"),
                    _nested_get(strategy, "first_stage", "upscale_workflow_path"),
                ],
            }
        elif generation_type == "text2image2image":
            inferred = {
                "image_workflow_name": [
                    _nested_get(strategy, "first_stage", "workflow_name"),
                    _nested_get(strategy, "first_stage", "workflow_path"),
                    generation_workflows.get("text2img"),
                ],
                "refine_workflow_name": [
                    _nested_get(strategy, "second_stage", "workflow_name"),
                    _nested_get(strategy, "second_stage", "workflow_path"),
                ],
            }
        elif generation_type == "text2longvideo":
            inferred = {
                "image_workflow_name": [
                    _nested_get(strategy, "first_stage", "workflow_name"),
                    _nested_get(strategy, "first_stage", "workflow_path"),
                    generation_workflows.get("text2img"),
                ],
                "video_workflow_name": [
                    _nested_get(strategy, "video_generation", "workflow_name"),
                    _nested_get(strategy, "video_generation", "workflow_path"),
                    generation_workflows.get("text2video"),
                ],
                "transition_workflow_name": [
                    _nested_get(strategy, "frame_transition", "workflow_name"),
                    _nested_get(strategy, "frame_transition", "workflow_path"),
                ],
            }
        elif generation_type == "native_h3_story":
            native_recipe = dict(generation.get("native_h3_story", {}) or {})
            inferred = {
                "image_workflow_name": [
                    native_recipe.get("keyframe_workflow_name"),
                    generation.get("keyframe_workflow_name"),
                    generation_workflows.get("text2img"),
                ],
                "video_workflow_name": [
                    native_recipe.get("workflow_name"),
                    generation_workflows.get("native_h3_story"),
                    generation_workflows.get("text2image2video"),
                ],
                "refine_workflow_name": [
                    native_recipe.get("identity_refine_workflow_name"),
                    generation.get("identity_refine_workflow_name"),
                ],
            }
        elif generation_type == "native_h3_t2v_story":
            native_recipe = dict(generation.get("native_h3_t2v_story", {}) or {})
            inferred = {
                "video_workflow_name": [
                    native_recipe.get("workflow_name"),
                    generation_workflows.get("native_h3_t2v_story"),
                    generation_workflows.get("text2video"),
                ],
            }
        elif generation_type == "native_h3_fl2va_story":
            native_recipe = dict(generation.get("native_h3_fl2va_story", {}) or {})
            inferred = {
                "image_workflow_name": [
                    native_recipe.get("keyframe_workflow_name"),
                    generation.get("keyframe_workflow_name"),
                    generation_workflows.get("text2img"),
                ],
                "video_workflow_name": [
                    native_recipe.get("workflow_name"),
                    generation_workflows.get("native_h3_fl2va_story"),
                    generation_workflows.get("text2image2video"),
                ],
                "refine_workflow_name": [
                    native_recipe.get("identity_refine_workflow_name"),
                    generation.get("identity_refine_workflow_name"),
                ],
            }
        elif generation_type == "native_h3_l2va_story":
            native_recipe = dict(generation.get("native_h3_l2va_story", {}) or {})
            inferred = {
                "image_workflow_name": [
                    native_recipe.get("keyframe_workflow_name"),
                    generation.get("keyframe_workflow_name"),
                    generation_workflows.get("text2img"),
                ],
                "video_workflow_name": [
                    native_recipe.get("workflow_name"),
                    generation_workflows.get("native_h3_l2va_story"),
                    generation_workflows.get("text2image2video"),
                ],
            }
        elif generation_type == "native_h3_ref2va":
            native_recipe = dict(generation.get("native_h3_ref2va", {}) or {})
            inferred = {
                "video_workflow_name": [
                    native_recipe.get("workflow_name"),
                    generation_workflows.get("native_h3_ref2va"),
                ],
            }
        elif generation_type == "text2image2native_h3_ref2va":
            native_recipe = dict(generation.get("text2image2native_h3_ref2va", {}) or {})
            inferred = {
                "image_workflow_name": [
                    native_recipe.get("keyframe_workflow_name"),
                    native_recipe.get("image_workflow_name"),
                    generation.get("keyframe_workflow_name"),
                    generation_workflows.get("text2img"),
                ],
                "video_workflow_name": [
                    native_recipe.get("workflow_name"),
                    generation_workflows.get("text2image2native_h3_ref2va"),
                    generation_workflows.get("native_h3_ref2va"),
                ],
            }
        elif generation_type == "image2image":
            inferred = {
                "refine_workflow_name": [
                    _nested_get(strategy, "workflow_name"),
                    _nested_get(strategy, "workflow_path"),
                    generation_workflows.get("image2image"),
                ],
            }
        elif generation_type == "text2video":
            inferred = {
                "image_workflow_name": [
                    generation_workflows.get("text2img"),
                ],
            }
        else:
            inferred = {
                "image_workflow_name": [
                    _nested_get(strategy, "workflow_name"),
                    _nested_get(strategy, "workflow_path"),
                    generation_workflows.get(generation_type),
                ],
            }

        for stage_key in (
            "image_workflow_name",
            "video_workflow_name",
            "refine_workflow_name",
            "transition_workflow_name",
            "upscale_workflow_name",
        ):
            stage_references: list[Any] = []
            explicit_stage = explicit.get(stage_key, [])
            if explicit_stage:
                if isinstance(explicit_stage, list):
                    stage_references.extend(explicit_stage)
                else:
                    stage_references.append(explicit_stage)
            stage_references.extend(inferred.get(stage_key, []))
            resolved = _resolve_workflow_references(repo_root, stage_references)
            if resolved:
                workflow_stage_candidates.setdefault(generation_type, {})[stage_key] = resolved
        workflow_stage_candidates.setdefault(generation_type, {})
    _prioritize_h3_profile(repo_root, generation, workflow_stage_candidates, generation_type_candidates)
    return workflow_stage_candidates


def _prioritize_h3_profile(
    repo_root: Path,
    generation: dict[str, Any],
    workflow_stage_candidates: dict[str, dict[str, list[str]]],
    generation_type_candidates: list[str],
) -> None:
    profile_to_slug = {
        "balanced-lowvram": "lowvram",
        "ultra-lowvram": "lowvram",
        "native-quality": "native",
    }
    profile = str(generation.get("h3_profile") or "balanced-lowvram").strip().lower()
    slug = profile_to_slug.get(profile, profile_to_slug["balanced-lowvram"])
    for generation_type in generation_type_candidates:
        if generation_type not in {"text2video", "text2image2video", "text2longvideo", "native_h3_story", "native_h3_t2v_story", "native_h3_fl2va_story", "native_h3_l2va_story", "native_h3_ref2va", "text2image2native_h3_ref2va", "sticker_pack"}:
            continue
        if generation_type in {"text2video", "native_h3_t2v_story"}:
            suffix = "t2v"
        elif generation_type in {"native_h3_story", "native_h3_fl2va_story", "native_h3_l2va_story"}:
            suffix = "15s_fl2va_i2v"
        elif generation_type in {"native_h3_ref2va", "text2image2native_h3_ref2va"}:
            continue
        else:
            suffix = "i2v"
        preferred = f"minimax_h3_{slug}_{suffix}"
        workflow_path = repo_root / "configs" / "workflow" / f"{preferred}.json"
        if not workflow_path.exists():
            continue
        stage_candidates = workflow_stage_candidates.setdefault(generation_type, {})
        candidates = list(stage_candidates.get("video_workflow_name", []))
        candidates = [candidate for candidate in candidates if candidate != preferred]
        stage_candidates["video_workflow_name"] = [preferred, *candidates]


def _collect_count_policies(
    routing_config: dict[str, Any],
    generation_type_candidates: list[str],
    workflow_stage_candidates: dict[str, dict[str, list[str]]] | None = None,
) -> dict[str, dict[str, dict[str, int]]]:
    raw_policies = routing_config.get("count_policies", {})
    if not isinstance(raw_policies, dict):
        raw_policies = {}
    policies: dict[str, dict[str, dict[str, int]]] = {}
    has_stage_contract = workflow_stage_candidates is not None
    workflow_stage_candidates = workflow_stage_candidates or {}
    for generation_type in generation_type_candidates:
        strategy_policy = dict(raw_policies.get(generation_type, {}) or {})
        normalized: dict[str, dict[str, int]] = {}
        active_count_keys = (
            _active_count_policy_keys(
                generation_type,
                workflow_stage_candidates.get(generation_type, {}),
            )
            if has_stage_contract
            else {
                "image_count",
                "video_count",
                "segment_count",
                "review_selection_limit",
                "sticker_expression_count",
                "images_per_prompt",
            }
        )
        for count_key in (
            "image_count",
            "video_count",
            "segment_count",
            "review_selection_limit",
            "sticker_expression_count",
            "images_per_prompt",
        ):
            if count_key not in active_count_keys:
                continue
            raw = strategy_policy.get(count_key, {})
            if not isinstance(raw, dict):
                continue
            minimum = int(raw.get("min", 1))
            maximum = int(raw.get("max", minimum))
            normalized[count_key] = {"min": minimum, "max": max(minimum, maximum)}
        policies[generation_type] = normalized
    return policies


def _active_count_policy_keys(
    generation_type: str,
    stage_candidates: dict[str, list[str]],
) -> set[str]:
    """Return count fields that the selected strategy can actually consume.

    The routing YAML historically provided one superset of count fields for
    every strategy. That made an image-only route validate unrelated fields
    such as ``video_count`` and fail when an LLM correctly returned zero.
    Keep the filtering close to route construction so the prompt, schema, and
    persisted routing summary describe the same contract.
    """
    known_generation_types = set(CONFIG_MEDIA_TYPE_MAP)
    if generation_type not in known_generation_types:
        return {
            "image_count",
            "video_count",
            "segment_count",
            "review_selection_limit",
            "sticker_expression_count",
            "images_per_prompt",
        }

    active: set[str] = {"review_selection_limit"}
    if stage_candidates.get("image_workflow_name"):
        active.add("image_count")
    if stage_candidates.get("video_workflow_name") and generation_type != "sticker_pack":
        active.add("video_count")
    if generation_type == "text2longvideo":
        active.add("segment_count")
    if generation_type == "sticker_pack":
        active.update({"sticker_expression_count", "images_per_prompt"})
    return active


def _pick_policy_value(policy: dict[str, Any]) -> int:
    if not isinstance(policy, dict):
        return 1
    minimum = int(policy.get("min", 1))
    maximum = int(policy.get("max", minimum))
    return minimum if minimum <= maximum else maximum


def _load_global_routing_config(repo_root: Path) -> dict[str, Any]:
    routing_path = repo_root / "configs" / "routing.yaml"
    if not routing_path.exists():
        raise FileNotFoundError(f"Global routing config not found: {routing_path}")
    data = yaml.safe_load(routing_path.read_text(encoding="utf-8")) or {}
    if "routing" in data and isinstance(data["routing"], dict):
        return dict(data["routing"])
    if isinstance(data, dict):
        return data
    raise ValueError(f"Invalid routing config structure: {routing_path}")


def _merge_routing_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_routing_config(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _workflow_definitions_dir(repo_root: Path) -> Path:
    return repo_root / "configs" / "workflow"


def _list_workflow_stems(repo_root: Path) -> set[str]:
    wf_dir = _workflow_definitions_dir(repo_root)
    if not wf_dir.is_dir():
        return set()
    return {path.stem for path in wf_dir.glob("*.json")}


def _resolve_workflow_references(repo_root: Path, references: list[Any]) -> list[str]:
    stems = _list_workflow_stems(repo_root)
    wf_dir = _workflow_definitions_dir(repo_root)
    resolved: list[str] = []
    for reference in references:
        name = _resolve_workflow_name_from_reference(repo_root, stems, wf_dir, reference)
        if name and name not in resolved:
            resolved.append(name)
    return resolved


def _resolve_workflow_name_from_reference(
    repo_root: Path,
    stems: set[str],
    wf_dir: Path,
    reference: Any,
) -> str | None:
    normalized = str(reference or "").strip()
    if not normalized:
        return None
    if normalized in stems:
        return normalized
    resolved_reference = _resolve_repo_path(repo_root, normalized).resolve()
    if not wf_dir.is_dir():
        return None
    for path in sorted(wf_dir.glob("*.json")):
        if path.resolve() == resolved_reference:
            return path.stem
    return None


def _primary_workflow_name(generation_type: str, workflow_plan: dict[str, str]) -> str:
    stage_priority = {
        "text2img": ("image_workflow_name",),
        "text2video": ("video_workflow_name", "image_workflow_name"),
        "text2image2video": ("image_workflow_name", "video_workflow_name", "upscale_workflow_name"),
        "text2image2image": ("image_workflow_name", "refine_workflow_name"),
        "text2longvideo": ("image_workflow_name", "video_workflow_name", "transition_workflow_name"),
        "native_h3_story": ("video_workflow_name", "image_workflow_name", "refine_workflow_name"),
        "native_h3_t2v_story": ("video_workflow_name",),
        "native_h3_fl2va_story": ("video_workflow_name", "image_workflow_name", "refine_workflow_name"),
        "native_h3_l2va_story": ("video_workflow_name", "image_workflow_name"),
        "native_h3_ref2va": ("video_workflow_name",),
        "text2image2native_h3_ref2va": ("video_workflow_name", "image_workflow_name"),
        "sticker_pack": ("image_workflow_name", "video_workflow_name"),
        "image2image": ("refine_workflow_name",),
    }
    for key in stage_priority.get(generation_type, ("image_workflow_name", "video_workflow_name", "refine_workflow_name")):
        value = str(workflow_plan.get(key) or "").strip()
        if value:
            return value
    return ""


def _native_recipe_for_generation(generation: dict[str, Any], generation_type: str) -> dict[str, Any]:
    if generation_type in {"native_h3_story", "native_h3_t2v_story", "native_h3_fl2va_story", "native_h3_l2va_story", "native_h3_ref2va", "text2image2native_h3_ref2va"}:
        selected = dict(generation.get(generation_type, {}) or {})
        if selected:
            return selected
        return dict(generation.get("native_h3_story", {}) or {})
    return dict(generation.get("native_h3_story", {}) or {})


def _nested_get(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _resolve_duration_seconds(
    config_generation_type: str,
    strategies: dict[str, Any],
    generation: dict[str, Any] | None = None,
    routed_segment_count: Any | None = None,
    requested_duration_seconds: int | None = None,
) -> int:
    requested = max(1, int(requested_duration_seconds)) if requested_duration_seconds is not None else None
    if config_generation_type in {"native_h3_story", "native_h3_t2v_story", "native_h3_fl2va_story", "native_h3_l2va_story", "native_h3_ref2va", "text2image2native_h3_ref2va"}:
        native_recipe = _native_recipe_for_generation(dict(generation or {}), config_generation_type)
        native_duration = max(1, int(native_recipe.get("duration_seconds", 15)))
        if requested is not None and requested != native_duration:
            raise ValueError(
                f"{config_generation_type} supports duration_seconds={native_duration}; "
                "use text2image2video for a 5-second clip."
            )
        return native_duration
    if config_generation_type == "text2image2video":
        # The default I2V workflow is a short 124-frame clip at 24 fps.
        # Keep prompt planning aligned with that effective ~5-second render
        # so the short-action contract is applied unless the caller opts in
        # to a different duration explicitly.
        return requested or 5
    if config_generation_type != "text2longvideo":
        return requested or 30
    longvideo = dict(strategies.get("text2longvideo", {}) or {})
    longvideo_config = dict(longvideo.get("longvideo_config", {}) or {})
    segment_count = int(routed_segment_count or longvideo_config.get("segment_count", 3))
    segment_duration = int(longvideo_config.get("segment_duration", 5))
    return max(10, segment_count * segment_duration)


def _duration_strategy_override(
    requested_duration_seconds: int | None,
    generation_type_candidates: list[str],
) -> str:
    """Keep short clips on a single-action route and 15s clips on native H3."""
    if requested_duration_seconds is None:
        return ""
    duration = max(1, int(requested_duration_seconds))
    candidates = set(generation_type_candidates)
    if duration <= 6 and "text2image2video" in candidates:
        return "text2image2video"
    if duration == 15 and "native_h3_story" in candidates:
        return "native_h3_story"
    return ""


def _duration_profile(duration_seconds: int | None) -> str:
    if duration_seconds is None:
        return "config_default"
    duration = max(1, int(duration_seconds))
    if duration <= 6:
        return "single_action"
    if duration <= 15:
        return "compact_story"
    return "extended_story"


def _build_routing_summary(
    *,
    character_name: str,
    prompt: str,
    source_generation_type: str,
    agentic_media_type: str,
    duration_seconds: int,
    output_dir: str,
    routing: dict[str, Any],
) -> dict[str, Any]:
    workflow_plan = {
        key: str(value).strip()
        for key, value in dict(routing.get("workflow_plan", {}) or {}).items()
        if str(value).strip()
    }
    count_plan = {
        key: int(value)
        for key, value in dict(routing.get("count_plan", {}) or {}).items()
        if value not in {None, ""}
    }
    backend = dict(routing.get("llm_backend", {}) or {})
    backend_summary = {
        "mode": str(backend.get("mode", "")),
        "text_provider": str(backend.get("text_provider", "")),
        "text_model": str(backend.get("text_model", "")),
        "vision_provider": str(backend.get("vision_provider", "")),
        "vision_model": str(backend.get("vision_model", "")),
    }
    return {
        "character": character_name,
        "prompt": prompt,
        "strategy": source_generation_type,
        "agentic_media_type": agentic_media_type,
        "primary_workflow": str(routing.get("workflow_name", "")),
        "workflow_plan": workflow_plan,
        "count_plan": count_plan,
        "reason": str(routing.get("reason", "")),
        "prompt_mode": str(routing.get("prompt_mode", "")),
        "selection_source": str(routing.get("selection_source", "")),
        "routing_runtime_context": dict(routing.get("routing_runtime_context", {}) or {}),
        "llm_backend": backend_summary,
        "duration_seconds": int(duration_seconds),
        "output_dir": output_dir,
    }


def _build_default_prompt(character_name: str, media_type: str, style: str) -> str:
    media_label = media_type.replace("_", " ")
    return f"{character_name} in a {style} {media_label} concept"


def _default_news_history_path(repo_root: Path, character_name: str) -> Path:
    configured = os.environ.get("AGENTIC_NEWS_HISTORY_PATH", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return repo_root / "agentic" / "state" / "news_selection" / f"{character_name.lower()}.json"


def _load_news_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"News history could not be read safely: {path}") from exc
    if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
        raise RuntimeError(f"News history has an invalid format: {path}")
    return [dict(item) for item in payload]


def _select_fresh_news(news_service: NewsContextService, history_path: Path) -> Any:
    history = _load_news_history(history_path)
    exclude_keys = {
        str(item.get("key") or NewsContextService.selection_key(item.get("title", ""), item.get("keyword", "")))
        for item in history
    }
    selected = news_service.get_random_news(exclude_keys=exclude_keys)
    if selected is None:
        raise RuntimeError(
            "News-driven generation requires an unseen news item, but no unseen usable news was available. "
            f"History: {history_path}"
        )
    selected_dict = dict(selected.to_dict() or {})
    history.append(
        {
            **selected_dict,
            "key": NewsContextService.selection_key(
                selected_dict.get("title", ""),
                selected_dict.get("keyword", ""),
            ),
            "selected_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    return selected


def _resolve_autonomous_prompt(
    *,
    prompt: str,
    character_name: str,
    style: str,
    media_type: str,
    generation: dict[str, Any],
    generation_type: str = "",
    news_driven: bool = False,
    news_history_path: str | Path | None = None,
) -> dict[str, Any]:
    explicit_prompt = str(prompt).strip()
    if explicit_prompt and not news_driven:
        return {
            "prompt": explicit_prompt,
            "source": "user",
            "prompt_mode": "user",
            "creative_seed": "",
            "news_context": {},
        }

    news_context: dict[str, Any] = {}
    try:
        news_service = NewsContextService()
        if news_driven:
            selected_news = _select_fresh_news(
                news_service,
                Path(news_history_path or "news_selection_history.json").expanduser().resolve(),
            )
        else:
            selected_news = news_service.get_random_news()
        if selected_news is not None:
            news_context = selected_news.to_dict()
    except Exception as exc:
        if news_driven:
            raise
        news_context = {"error": f"{type(exc).__name__}: {exc}"}

    if news_driven and not news_context:
        raise RuntimeError("News-driven generation did not receive a usable news context.")

    if generation_type in {"native_h3_story", "native_h3_t2v_story", "native_h3_fl2va_story", "native_h3_l2va_story", "native_h3_ref2va", "text2image2native_h3_ref2va"}:
        return {
            "prompt": "",
            "source": "news",
            "prompt_mode": "news",
            "creative_seed": "",
            "news_context": news_context,
        }

    bundle = LLMPromptEngine(mode=os.environ.get("AGENTIC_LLM_MODE", "llm")).generate_autonomous_scene_prompt(
        character=character_name,
        style=style,
        media_type=media_type,
        news_context=news_context,
    )
    bundle.setdefault("prompt_mode", "template")
    bundle.setdefault("source", "autonomous_fallback")
    return bundle


def _normalize_platform_configs(repo_root: Path, raw_platforms: Any) -> tuple[dict[str, dict[str, Any]], dict[str, str], list[str]]:
    if not isinstance(raw_platforms, dict):
        return {}, {}, []
    configs: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    skipped: list[str] = []
    for platform_name, platform_config in raw_platforms.items():
        if not isinstance(platform_config, dict) or not bool(platform_config.get("enabled", True)):
            continue
        normalized = str(platform_name).lower().strip()
        if normalized == "instagram":
            aliases[normalized] = "instagram_graph"
            normalized = "instagram_graph"
        if normalized not in SUPPORTED_PUBLISH_PLATFORMS:
            skipped.append(str(platform_name))
            continue
        resolved = dict(platform_config)
        if "config_folder_path" in resolved:
            resolved["config_folder_path"] = str(_resolve_repo_path(repo_root, str(resolved["config_folder_path"])))
        configs[normalized] = resolved
    return configs, aliases, skipped


def _merge_social_media_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_social_media_config(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _summarize_character_config(
    *,
    path: Path,
    character: dict[str, Any],
    generation: dict[str, Any],
    social_media: dict[str, Any],
    strategies: dict[str, Any],
    platform_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    longvideo = dict(strategies.get("text2longvideo", {}) or {})
    longvideo_config = dict(longvideo.get("longvideo_config", {}) or {})
    return {
        "config_path": str(path),
        "character_name": str(character.get("name") or character.get("group_name") or path.stem),
        "group_name": str(character.get("group_name") or ""),
        "output_dir": str(generation.get("output_dir") or ""),
        "generation_type_weights": dict(generation.get("generation_type_weights", {}) or {}),
        "style_count": len(dict(generation.get("style_weights", {}) or {})),
        "default_hashtags": [str(tag) for tag in (social_media.get("default_hashtags") or []) if tag],
        "enabled_platforms": list(platform_configs.keys()),
        "platform_config_folders": {
            name: str(config.get("config_folder_path", ""))
            for name, config in platform_configs.items()
        },
        "longvideo": {
            "segment_count": int(longvideo_config.get("segment_count", 0) or 0),
            "segment_duration": int(longvideo_config.get("segment_duration", 0) or 0),
            "use_tts": bool(longvideo_config.get("use_tts", False)),
            "tts_voice": str(longvideo_config.get("tts_voice", "")),
        },
    }


def _resolve_output_dir(repo_root: Path, configured_output_dir: Any, character_name: str) -> Path:
    base = _resolve_repo_path(repo_root, str(configured_output_dir or (repo_root / "agentic" / "output")))
    return Path(base) / character_name.lower()


def _resolve_repo_path(repo_root: Path, raw_path: str) -> Path:
    normalized = raw_path.strip()
    if not normalized:
        return repo_root / "agentic" / "output"
    # Character configs are shared with the Windows host and may contain a
    # drive-qualified path such as ``D:/MediaOverload/output``.  Inside the
    # Linux container that string is not an absolute path; without this
    # mapping it becomes ``/app/D:/MediaOverload/output`` and bypasses the
    # mounted ``/app/output`` directory.
    if (
        len(normalized) >= 3
        and normalized[1] == ":"
        and normalized[2] in {"/", "\\"}
        and Path("/comfyui").is_dir()
    ):
        return repo_root / "output"
    if normalized.startswith("/app/"):
        return repo_root / normalized.removeprefix("/app/")
    if normalized == "/app":
        return repo_root
    path = Path(normalized)
    if path.is_absolute():
        return path
    return repo_root / path


def _is_media_file(path: str) -> bool:
    return Path(path).suffix.lower() in MEDIA_EXTENSIONS
