"""Character Payload responsibilities."""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Any
from agentic.app.character_requests import CharacterWorkflowRequest
from agentic.app.creative_profiles import resolve_creative_profile
from agentic.runtime.observability import RunRecorder
from agentic.runtime.strategies import (
    IMAGE_ONLY_GENERATION_TYPES,
    NEWS_GROUNDED_GENERATION_TYPES,
    NATIVE_STORY_GENERATION_TYPES,
    media_type_for_generation,
)
from agentic.runtime.story_cards import (
    STORY_CARD_DEFAULT_HEIGHT,
    STORY_CARD_DEFAULT_WIDTH,
    STORY_CARD_PAGE_COUNT_DEFAULT,
)
from agentic.tools.publishing_adapter import FACEBOOK_PROFILE_HANDOFF_PLATFORM
from agentic.app.character_selection import (
    _fixed_character_selection,
    _replace_character_identity,
    _selection_subject_mode,
    _subject_context_from_selection,
    _subject_mode,
    _subject_mode_weights,
    _subject_prompt_block,
)
from agentic.app.character_requests import (
    SUBJECT_MODE_INTERACTION,
    SUBJECT_MODE_RANDOM,
    SUPPORTED_SUBJECT_MODES,
)
from agentic.app.character_config import (
    SUPPORTED_PUBLISH_PLATFORMS,
    _merge_social_media_config,
    _normalize_platform_configs,
    _normalize_video_speed_config,
    _resolve_output_dir,
    _summarize_character_config,
    load_character_config,
    load_global_social_config,
)
from agentic.app.generation_routing import (
    _build_default_prompt,
    _build_routing_summary,
    _duration_profile,
    _native_recipe_for_generation,
    _resolve_autonomous_prompt,
    _resolve_duration_seconds,
    _route_generation_from_character_config,
)


def build_goal_payload_from_character_config(
    request: CharacterWorkflowRequest,
    *,
    recorder: RunRecorder | None = None,
) -> dict[str, Any]:
    generation = request.generation
    review = request.review
    runtime = request.runtime
    repo_root = request.repo_root
    config_path = request.config_path
    prompt = generation.prompt
    selected_character_name = str(generation.selected_character_name or "").strip()
    character_selection = dict(generation.character_selection or {})
    temperature = generation.temperature
    preferred_generation_type = generation.preferred_generation_type
    duration_seconds = generation.duration_seconds
    motion_graphics_requested = bool(generation.motion_graphics)
    output_dir = generation.output_dir
    news_driven = generation.news_driven
    news_context_override = dict(generation.news_context or {})
    requested_native_h3_creative_brief = str(
        generation.native_h3_creative_brief or ""
    ).strip()
    requested_native_h3_arc_instruction = str(
        generation.native_h3_arc_instruction or ""
    ).strip()
    rng = generation.rng
    requested_seed = generation.seed
    requested_reference_video_source = str(
        generation.reference_video_source or ""
    ).strip()
    requested_reference_video_depth = (
        str(generation.reference_video_depth or "").strip().lower()
    )
    requested_reference_video_max_keyframes = generation.reference_video_max_keyframes
    requested_subject_mode = str(generation.subject_mode or "").strip().lower()
    dry_run_publish = review.dry_run_publish
    publish_mode = review.publish_mode
    publish_platforms = list(review.publish_platforms)
    facebook_profile_share_url = str(review.facebook_profile_share_url or "").strip()
    publish_after_generate = review.publish_after_generate
    enable_review_loop = review.enable_review_loop
    review_notes = review.review_notes
    no_review = review.no_review
    stage_probe = review.stage_probe
    asset_root = runtime.asset_root
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
    video_speed = _normalize_video_speed_config(generation.get("video_speed"))

    configured_character_name = str(character.get("name") or "").strip()
    configured_group_name = str(character.get("group_name") or "").strip()
    configured_prompt_identity = configured_character_name or configured_group_name
    if requested_subject_mode and requested_subject_mode not in SUPPORTED_SUBJECT_MODES:
        raise ValueError(f"Unsupported subject mode: {requested_subject_mode}")
    configured_subject_mode = requested_subject_mode or _subject_mode(config)
    if configured_subject_mode == SUBJECT_MODE_RANDOM:
        _subject_mode_weights(config)
    if configured_group_name and not character_selection:
        raise ValueError(
            "group_name requires a resolved character selection before payload construction"
        )
    if not character_selection and configured_subject_mode == SUBJECT_MODE_RANDOM:
        raise ValueError(
            "random subject_mode requires a resolved selection before payload construction"
        )
    subject_mode = _selection_subject_mode(
        character_selection,
        configured_subject_mode=configured_subject_mode,
    )
    if subject_mode == SUBJECT_MODE_INTERACTION and not selected_character_name:
        selected_subjects = character_selection.get("subjects")
        if isinstance(selected_subjects, list) and selected_subjects:
            selected_character_name = str(
                selected_subjects[0].get("name") or ""
            ).strip()
    character_name = selected_character_name or configured_character_name or path.stem
    if not character_selection:
        if subject_mode == SUBJECT_MODE_INTERACTION:
            raise ValueError(
                "two_character_interaction requires a resolved pair selection before payload construction"
            )
        character_selection = _fixed_character_selection(
            character_name,
            group_name=str(character.get("group_name") or ""),
        )
    subject_context = _subject_context_from_selection(
        subject_mode=subject_mode,
        character_name=character_name,
        character_selection=character_selection,
        configured_group_name=str(character.get("group_name") or "").strip(),
    )
    creative_rng = (
        random.Random(int(requested_seed)) if requested_seed is not None else None
    )
    if creative_rng is None and rng is not None and hasattr(rng, "getstate"):
        creative_rng = random.Random()
        creative_rng.setstate(rng.getstate())
    creative_profile = resolve_creative_profile(
        dict(generation.get("creative_profile") or {}),
        rng=creative_rng,
    )
    canvas_profile = dict(creative_profile["canvas"])
    style_profile = dict(creative_profile["style"])
    style = str(style_profile["label"])
    prompt = _replace_character_identity(
        prompt,
        configured_name=configured_prompt_identity,
        selected_name=character_name,
    )
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
        asset_root=asset_root,
    )
    config_generation_type = str(routing["generation_type"])
    if motion_graphics_requested and config_generation_type != "text2image2video":
        raise ValueError(
            "--motion-graphics is supported only with the text2image2video generation type"
        )
    longvideo_config = dict(routing.get("longvideo_config", {}) or {})
    longvideo_config.update(
        dict(
            dict(strategies.get("text2longvideo", {}) or {}).get("longvideo_config", {})
            or {}
        )
    )
    effective_news_driven = bool(
        news_driven
        or (
            not str(prompt).strip()
            and config_generation_type in NEWS_GROUNDED_GENERATION_TYPES
        )
    )
    agentic_media_type = media_type_for_generation(config_generation_type)
    native_recipe = dict(
        _native_recipe_for_generation(generation, config_generation_type)
    )
    configured_reference_values = any(
        native_recipe.get(key) or generation.get(key)
        for key in (
            "reference_manifest",
            "reference_image_paths",
            "reference_video_paths",
        )
    )
    auto_reference_generation = (
        config_generation_type == "text2image2native_h3_ref2va"
        or (
            config_generation_type == "native_h3_ref2va"
            and not configured_reference_values
        )
    )
    pre_video_review_config = dict(routing.get("pre_video_review", {}) or {})
    # Ref2VA uses configured references when present. With an empty manifest,
    # it expands into the six-candidate T2I plus Discord reference gate.
    pre_video_review_enabled = bool(
        pre_video_review_config.get("enabled", False)
        and config_generation_type != "game_sprite"
        and config_generation_type not in {"text2video", "native_h3_t2v_story"}
        and not (no_review and config_generation_type == "text2image2video")
        and (config_generation_type != "native_h3_ref2va" or auto_reference_generation)
    )
    pre_video_candidate_count = max(
        1, int(pre_video_review_config.get("candidate_count") or 6)
    )
    pre_video_selection_limit = max(
        1,
        int(
            pre_video_review_config.get(
                "ref2va_selection_limit"
                if config_generation_type
                in {"native_h3_ref2va", "text2image2native_h3_ref2va"}
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
            expression_policy = dict(
                stage_probe_policy.get("sticker_expression_count", {}) or {}
            )
            images_policy = dict(stage_probe_policy.get("images_per_prompt", {}) or {})
            if expression_policy:
                routing["count_plan"]["sticker_expression_count"] = int(
                    expression_policy.get("max") or 4
                )
            if images_policy:
                routing["count_plan"]["images_per_prompt"] = int(
                    images_policy.get("max") or 1
                )
    autonomous_prompt = _resolve_autonomous_prompt(
        prompt=prompt,
        character_name=character_name,
        style=style,
        media_type=agentic_media_type,
        generation=generation,
        generation_type=config_generation_type,
        news_driven=effective_news_driven,
        news_context_override=news_context_override,
        recorder=recorder,
    )
    if config_generation_type in ({"story_card"} | NATIVE_STORY_GENERATION_TYPES) and (
        effective_news_driven or not str(prompt).strip()
    ):
        # Native H3 owns the news-to-story prompt contract. Do not create a
        # second autonomous scene prompt here and then carry it through the
        # routing summary as if it were a user brief.
        resolved_prompt = ""
    else:
        resolved_prompt = str(
            autonomous_prompt["prompt"]
        ).strip() or _build_default_prompt(character_name, agentic_media_type, style)
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
        longvideo_config=longvideo_config,
    )
    reference_video_source = (
        requested_reference_video_source
        or str(generation.get("reference_video_source") or "").strip()
    )
    reference_video_depth = (
        requested_reference_video_depth
        or str(generation.get("reference_video_depth") or "standard").strip().lower()
    )
    reference_video_max_keyframes = int(
        requested_reference_video_max_keyframes
        if requested_reference_video_max_keyframes is not None
        else generation.get("reference_video_max_keyframes") or 12
    )
    if reference_video_depth not in {"standard", "deep"}:
        raise ValueError("reference_video_depth must be 'standard' or 'deep'")
    if not 2 <= reference_video_max_keyframes <= 20:
        raise ValueError(
            "reference_video_max_keyframes must be an integer between 2 and 20"
        )
    if config_generation_type == "text2image2video" and reference_video_source:
        # Reference-driven action clips already encode their intended rhythm in
        # the prompt. Do not compress the generated payoff with a speed pass.
        video_speed = {"enabled": False, "factor": 1.0}
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
            name: config
            for name, config in platform_configs.items()
            if name in selected_platforms
        }
        if FACEBOOK_PROFILE_HANDOFF_PLATFORM in selected_platforms:
            platform_configs.setdefault(
                FACEBOOK_PROFILE_HANDOFF_PLATFORM,
                {"enabled": True, "delivery": "discord"},
            )
    if (
        config_generation_type in IMAGE_ONLY_GENERATION_TYPES
        and "youtube" in platform_configs
    ):
        platform_configs.pop("youtube", None)
        skipped_platforms.append("youtube:image_only_route")
    hashtags = [str(tag) for tag in (social_media.get("default_hashtags") or []) if tag]
    use_tts = bool(longvideo_config.get("use_tts", False))
    if (
        agentic_media_type == "long_video"
        and str(longvideo_config.get("production_profile") or "").strip().lower()
        == "text2longvideo"
    ):
        # A publishable 30/45-second story must retain its requested runtime;
        # Kirby's short-form 2x playback transform would otherwise compress it
        # to half length before the publish review.
        video_speed = {"enabled": False, "factor": 1.0}
    if config_generation_type == "native_h3_fl2va_story":
        # FL2VA's native workflow already owns the requested 15-second timing;
        # applying the character-wide 2x post-process would turn it into a
        # short preview before QA and review.
        video_speed = {"enabled": False, "factor": 1.0}

    native_keyframe_candidate_count = max(
        1, int(native_recipe.get("keyframe_candidate_count") or 1)
    )
    if (
        pre_video_review_enabled
        and config_generation_type
        in {
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
    if (
        no_review
        and not stage_probe
        and config_generation_type in NATIVE_STORY_GENERATION_TYPES
    ):
        native_recipe["keyframe_candidate_count"] = 1
        native_keyframe_candidate_count = 1
        native_recipe["require_human_review"] = False
        if auto_reference_generation:
            native_recipe["reference_selection_limit"] = 1
    if config_generation_type != "native_h3_fl2va_story":
        native_recipe["use_last_frame"] = False
    native_h3_creative_brief = _replace_character_identity(
        str(
            native_recipe.get("creative_brief")
            or generation.get("native_h3_creative_brief")
            or ""
        ).strip(),
        configured_name=configured_prompt_identity,
        selected_name=character_name,
    ).strip()
    requested_native_h3_creative_brief = _replace_character_identity(
        requested_native_h3_creative_brief,
        configured_name=configured_prompt_identity,
        selected_name=character_name,
    ).strip()
    if requested_native_h3_creative_brief:
        native_h3_creative_brief = "\n".join(
            item
            for item in (
                native_h3_creative_brief,
                f"Additional Native H3 creative direction: {requested_native_h3_creative_brief}",
            )
            if item
        )
    style_guidance = "; ".join(
        part
        for part in (
            str(style_profile.get("prompt") or "").strip(),
            str(style_profile.get("composition") or "").strip(),
            str(style_profile.get("palette") or "").strip(),
            str(style_profile.get("motion") or "").strip(),
            f"Avoid: {style_profile['avoid']}" if style_profile.get("avoid") else "",
        )
        if part
    )
    native_h3_creative_brief = "; ".join(
        part
        for part in (
            native_h3_creative_brief,
            f"Selected visual style: {style_guidance}",
        )
        if part
    )
    selected_profile = dict(character_selection.get("selected_profile") or {})
    role_description = str(selected_profile.get("role_description") or "").strip()
    role_keywords = str(selected_profile.get("keywords") or "").strip()
    if selected_profile and subject_mode != SUBJECT_MODE_INTERACTION:
        subject_context["character_profile"] = dict(selected_profile)
    if subject_mode == SUBJECT_MODE_INTERACTION:
        profile_block = (
            f"Selected subject configuration: {_subject_prompt_block(subject_context)}"
        )
    elif role_description or role_keywords:
        profile_parts = [f"Selected character: {character_name}"]
        if role_description:
            profile_parts.append(f"Role profile: {role_description}")
        if role_keywords:
            profile_parts.append(f"Keywords: {role_keywords}")
        profile_block = " ".join(profile_parts)
    else:
        profile_block = ""
    if profile_block:
        native_h3_creative_brief = "\n".join(
            item for item in (native_h3_creative_brief, profile_block) if item
        )
    story_card_config = dict(generation.get("story_card", {}) or {})
    story_card_visual = dict(story_card_config.get("visual") or {})
    game_sprite_config = dict(generation.get("game_sprite", {}) or {})
    constraints = {
        "character": character_name,
        "subject_mode": subject_mode,
        "configured_subject_mode": configured_subject_mode,
        "subject_context": subject_context,
        "subjects": list(subject_context.get("subjects") or []),
        "interaction_contract": dict(subject_context.get("interaction_contract") or {}),
        "character_selection": character_selection,
        "character_profile": selected_profile,
        "creative_profile": creative_profile,
        "canvas_profile": canvas_profile,
        "canvas_aspect_ratio": str(canvas_profile["aspect_ratio"]),
        "canvas_width": int(canvas_profile["width"]),
        "canvas_height": int(canvas_profile["height"]),
        "visual_style_profile": style_profile,
        "seed": int(requested_seed) if requested_seed is not None else None,
        "h3_video_defaults": dict(generation.get("video_defaults") or {}),
        "video_speed": video_speed,
        "story_card_page_count": str(
            story_card_config.get("page_count", STORY_CARD_PAGE_COUNT_DEFAULT)
        ).strip()
        or STORY_CARD_PAGE_COUNT_DEFAULT,
        "story_card_width": int(
            story_card_config.get("width", STORY_CARD_DEFAULT_WIDTH)
        ),
        "story_card_height": int(
            story_card_config.get("height", STORY_CARD_DEFAULT_HEIGHT)
        ),
        "story_card_overlay_opacity": int(
            story_card_config.get("overlay_opacity", 202)
        ),
        "story_card_brand_label": str(
            story_card_config.get("brand_label", "STORY NOTE")
        ),
        "story_card_font_path": str(story_card_config.get("font_path", "")),
        "story_card_visual": story_card_visual,
        "keyframe_workflow_name": str(generation.get("keyframe_workflow_name") or ""),
        "identity_refine_workflow_name": str(
            generation.get("identity_refine_workflow_name") or ""
        ),
        "native_h3_storyboard_path": str(
            native_recipe.get("storyboard_path")
            or generation.get("storyboard_path")
            or ""
        ),
        "native_h3_recipe": native_recipe,
        "native_h3_reference_manifest": list(
            native_recipe.get("reference_manifest")
            or generation.get("reference_manifest")
            or []
        ),
        "native_h3_reference_image_paths": list(
            native_recipe.get("reference_image_paths")
            or generation.get("reference_image_paths")
            or []
        ),
        "native_h3_reference_video_paths": list(
            native_recipe.get("reference_video_paths")
            or generation.get("reference_video_paths")
            or []
        ),
        "native_h3_reference_image_size": str(
            native_recipe.get("reference_image_size")
            or generation.get("reference_image_size")
            or "match"
        ),
        "native_h3_reference_max_images": int(
            native_recipe.get("reference_max_images")
            or generation.get("reference_max_images")
            or 9
        ),
        "native_h3_reference_max_videos": int(
            native_recipe.get("reference_max_videos")
            or generation.get("reference_max_videos")
            or 3
        ),
        "native_h3_reference_selection_limit": int(
            native_recipe.get("reference_selection_limit")
            or generation.get("reference_selection_limit")
            or 4
        ),
        "native_h3_reference_candidate_count": int(native_reference_candidate_count),
        "auto_reference_generation": auto_reference_generation,
        "native_h3_model_profile": str(
            native_recipe.get("model_profile")
            or generation.get("model_profile")
            or "q4"
        ),
        "hashtags": hashtags,
        "platforms": list(platform_configs.keys()),
        "platform_configs": platform_configs,
        "dry_run": dry_run_publish,
        "publish_mode": str(publish_mode or "").strip().lower(),
        "facebook_profile_share_url": facebook_profile_share_url,
        "selection_limit": routing.get("count_plan", {}).get("review_selection_limit"),
        "enable_review_loop": bool(enable_review_loop and not no_review),
        "review_notes": review_notes,
        "news_driven": effective_news_driven,
        "native_h3_arc_instruction": requested_native_h3_arc_instruction,
        "native_h3_keyframe_candidate_count": native_keyframe_candidate_count,
        "pre_video_review_enabled": pre_video_review_enabled,
        "pre_video_candidate_count": pre_video_candidate_count,
        "pre_video_selection_limit": pre_video_selection_limit,
        "pre_video_review_require_human": bool(
            pre_video_review_enabled and not no_review and not stage_probe
        ),
        "stage_probe_auto_select": bool(stage_probe),
        "pre_video_review_reject_stops_workflow": bool(
            pre_video_review_config.get("reject_stops_workflow", True)
        ),
        "pre_video_review_failure_policy": str(
            pre_video_review_config.get("failure_policy", "block") or "block"
        )
        .strip()
        .lower(),
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
        "native_h3_creative_brief": native_h3_creative_brief,
        "native_h3_use_last_frame": bool(native_recipe.get("use_last_frame", False)),
        # The pre-video gate selects from the raw keyframes. Upscaling before
        # that gate creates an artifact that the selected I2V input never
        # consumes, so keep the route's authoritative input to one raw frame.
        "skip_upscale_for_i2v": config_generation_type == "text2image2video",
        "output_dir": str(resolved_output_dir),
        "reference_video_source": reference_video_source,
        "reference_video_depth": reference_video_depth,
        "reference_video_max_keyframes": reference_video_max_keyframes,
        "use_tts": use_tts if agentic_media_type == "long_video" else False,
        "source_temperature": temperature,
        "source_config_path": str(path),
        "source_generation_type": config_generation_type,
        "duration_override_seconds": (
            max(1, int(requested_duration_seconds))
            if requested_duration_seconds is not None
            else None
        ),
        "duration_profile": _duration_profile(requested_duration_seconds),
        "video_frame_rate": int(
            dict(generation.get("video_defaults", {}) or {}).get("frame_rate", 24)
        ),
        "max_i2v_frames": int(
            dict(generation.get("video_defaults", {}) or {}).get("max_i2v_frames", 240)
        ),
        "workflow_name": routing.get("workflow_name", ""),
        "image_workflow_name": routing.get("workflow_plan", {}).get(
            "image_workflow_name", ""
        ),
        "video_workflow_name": routing.get("workflow_plan", {}).get(
            "video_workflow_name", ""
        ),
        "refine_workflow_name": routing.get("workflow_plan", {}).get(
            "refine_workflow_name", ""
        ),
        "transition_workflow_name": routing.get("workflow_plan", {}).get(
            "transition_workflow_name", ""
        ),
        "upscale_workflow_name": routing.get("workflow_plan", {}).get(
            "upscale_workflow_name", ""
        ),
        "image_count": routing.get("count_plan", {}).get("image_count"),
        "video_count": routing.get("count_plan", {}).get("video_count"),
        "segment_count": routing.get("count_plan", {}).get("segment_count"),
        "review_selection_limit": routing.get("count_plan", {}).get(
            "review_selection_limit"
        ),
        "sticker_expression_count": routing.get("count_plan", {}).get(
            "sticker_expression_count"
        ),
        "images_per_prompt": routing.get("count_plan", {}).get("images_per_prompt"),
        "routing_reason": routing.get("reason", ""),
        "routing_selection_source": routing.get("selection_source", ""),
        "routing_runtime_context": dict(
            routing.get("routing_runtime_context", {}) or {}
        ),
        "strategy_context": dict(routing.get("strategy_context", {}) or {}),
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
            and (
                stage_probe
                or (
                    os.getenv("discord_review_bot_token")
                    and os.getenv("discord_review_channel_id")
                )
            )
            and config_generation_type
            in {
                "text2video",
                "text2image2video",
                "text2longvideo",
                "native_h3_story",
                "native_h3_fl2va_story",
                "native_h3_l2va_story",
                "native_h3_ref2va",
                "text2image2native_h3_ref2va",
            }
        ),
    }
    if motion_graphics_requested:
        constraints["enable_motion_graphics"] = True
    if config_generation_type == "game_sprite":
        requested_sprite_length = (
            max(1, round(float(duration_seconds) * 24.0))
            if requested_duration_seconds is not None
            else None
        )
        configured_sprite_length = int(
            game_sprite_config.get(
                "video_length", round(float(duration_seconds) * 24.0)
            )
        )
        constraints.update(
            {
                "game_sprite": game_sprite_config,
                "sprite_source_width": int(
                    game_sprite_config.get("source_width", 1024)
                ),
                "sprite_source_height": int(
                    game_sprite_config.get("source_height", 576)
                ),
                "sprite_video_width": int(game_sprite_config.get("video_width", 608)),
                "sprite_video_height": int(game_sprite_config.get("video_height", 352)),
                "sprite_video_length": requested_sprite_length
                or configured_sprite_length,
                "sprite_video_steps": int(game_sprite_config.get("video_steps", 16)),
                "sprite_h3_model_profile": str(
                    game_sprite_config.get("h3_model_profile", "q2")
                ),
                "sprite_fps": float(game_sprite_config.get("fps", 12)),
                "sprite_chroma_color": str(
                    game_sprite_config.get("chroma_color", "random")
                ),
                "sprite_chroma_threshold": int(
                    game_sprite_config.get("chroma_threshold", 52)
                ),
                "sprite_cell_width": int(game_sprite_config.get("cell_width", 64)),
                "sprite_cell_height": int(game_sprite_config.get("cell_height", 64)),
            }
        )
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
            "native_h3_duration_seconds": int(
                native_recipe.get("duration_seconds") or duration_seconds
            ),
            "native_h3_length": int(
                native_recipe.get("length")
                or dict(generation.get("video_defaults", {}) or {}).get("length", 362)
            ),
            "native_h3_steps": int(
                native_recipe.get("steps")
                or dict(generation.get("video_defaults", {}) or {}).get("steps", 16)
            ),
            "native_h3_frame_rate": int(
                native_recipe.get("frame_rate")
                or dict(generation.get("video_defaults", {}) or {}).get(
                    "frame_rate", 24
                )
            ),
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
                    if config_generation_type
                    in {"native_h3_ref2va", "text2image2native_h3_ref2va"}
                    else "image_to_video"
                )
            ),
        }
    )
    if agentic_media_type == "long_video":
        # Keep long-video policy in the shared strategy config. The planner
        # consumes provider-neutral recipe names.
        longvideo_config_keys = {
            "production_profile": "longvideo_production_profile",
            "default_duration_seconds": "longvideo_default_duration_seconds",
            "segment_duration": "longvideo_segment_duration",
            "review_policy": "longvideo_review_policy",
            "continuity_mode": "longvideo_continuity_mode",
            "workflow_names": "longvideo_workflow_names",
            "frame_candidate_count": "longvideo_frame_candidate_count",
            "reference_candidate_count": "longvideo_reference_candidate_count",
            "reference_selection_limit": "longvideo_reference_selection_limit",
            "width": "longvideo_width",
            "height": "longvideo_height",
            "length": "longvideo_length",
            "steps": "longvideo_steps",
            "model_profile": "longvideo_model_profile",
            "reference_manifest": "reference_manifest",
            "reference_image_paths": "reference_image_paths",
            "reference_video_paths": "reference_video_paths",
            "reference_max_images": "longvideo_reference_max_images",
            "reference_max_videos": "longvideo_reference_max_videos",
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
        if (
            str(constraints.get("longvideo_production_profile") or "").strip().lower()
            == "text2longvideo"
        ):
            # The routing count policy is intentionally conservative for LLM
            # schema generation. Production duration, however, must determine
            # the actual number of five-second H3 clips so 30s and 45s do not
            # render four clips and stretch the final image for the remainder.
            if requested_duration_seconds is not None:
                duration_seconds = max(1, int(requested_duration_seconds))
            constraints["segment_count"] = max(
                2, math.ceil(float(duration_seconds) / 5.0)
            )
    character_config_summary = _summarize_character_config(
        path=path,
        character=character,
        selected_character_name=character_name,
        character_selection=character_selection,
        generation=generation,
        social_media=social_media,
        strategies=strategies,
        platform_configs=platform_configs,
        effective_longvideo_config=longvideo_config,
    )
    character_config_summary["subject_context"] = subject_context
    character_config_summary["creative_profile"] = creative_profile
    if config_generation_type == "game_sprite":
        character_config_summary["game_sprite"] = game_sprite_config
    return {
        "prompt": resolved_prompt,
        "media_type": agentic_media_type,
        "duration_seconds": duration_seconds,
        "style": style,
        "constraints": constraints,
        "source_generation_type": config_generation_type,
        "selected_workflow_name": routing.get("workflow_name", ""),
        "character_name": character_name,
        "character_selection": character_selection,
        "resolved_output_dir": str(resolved_output_dir),
        "routing": routing,
        "prompt_generation": autonomous_prompt,
        "character_config_summary": character_config_summary,
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
