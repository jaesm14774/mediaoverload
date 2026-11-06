"""Scenes prompt contracts and response validation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from agentic.minimax_prompting import require_h3_prompt
from agentic.runtime.contracts import GoalRequest
from agentic.runtime.reference_video import (
    format_reference_video_directive,
    reference_keyframe_paths,
)
from agentic.video_directing import (
    VIDEO_MOTION_DIRECTION,
    VIDEO_SEGMENT_DIRECTION_FIELDS,
    apply_video_opening_frame_direction,
    requires_video_motion_direction,
)
from agentic.runtime.prompting import (
    IMAGE_PROMPT_CONTRACT,
    LONG_VIDEO_SYSTEM_PROMPT,
    build_autonomous_scene_prompt,
    build_goal_brief,
    include_image_prompt_quality_clause,
    include_visual_style_profile,
    news_grounding_anchor_clause,
    build_story_segments,
    style_profile_avoidance,
    style_profile_direction,
    validate_story_segments,
)
from agentic.runtime.llm_client import PromptClient
from agentic.runtime.news_grounding import is_usable_news_selection


def _goal_subject_instruction(goal: GoalRequest) -> str:
    """Give the prompt writer the selected cast and its visible interaction contract."""

    context = goal.constraints.get("subject_context")
    context = dict(context) if isinstance(context, dict) else {}
    subjects = [
        item
        for item in (context.get("subjects") or [])
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    ]
    character = str(goal.constraints.get("character") or "").strip()
    if not subjects:
        profile = dict(
            context.get("character_profile")
            or goal.constraints.get("character_profile")
            or {}
        )
        details = "; ".join(
            part
            for part in (
                str(profile.get("role_description") or "").strip(),
                str(profile.get("keywords") or "").strip(),
            )
            if part
        )
        return "; ".join(part for part in (character, details) if part)

    cast = []
    for subject in subjects:
        name = str(subject.get("name") or "").strip()
        role = str(subject.get("role") or "subject").strip()
        profile = dict(subject.get("profile") or {})
        details = "; ".join(
            part
            for part in (
                str(profile.get("role_description") or "").strip(),
                str(profile.get("keywords") or "").strip(),
            )
            if part
        )
        detail_clause = f" ({details})" if details else ""
        cast.append(f"{role}: {name}{detail_clause}")

    contract = dict(context.get("interaction_contract") or {})
    if bool(contract.get("required")):
        return (
            "Selected cast (exactly these required subjects): "
            + "; ".join(cast)
            + ". Keep both in the same readable frame, show their physical interaction, preserve their roles and identities, "
            "and add no other characters."
        )

    selected_names = {
        str(subject.get("name") or "").strip().casefold() for subject in subjects
    }
    extra_cast_allowed = contract.get("unexpected_extra_subjects", True)
    if character and character.casefold() not in selected_names:
        cast.append(f"configured character: {character}")
    cast_policy = (
        "Keep these selected identities as the scene anchors; add other characters only "
        "when the creative brief names them."
        if extra_cast_allowed
        else "This is the full cast; do not add other characters."
    )
    return f"Selected cast: {'; '.join(cast)}. {cast_policy}"


SCENE_RESPONSE_CONTRACT = (
    "The prompt field must contain only the final scene description as one non-empty string. "
    "Describe the subjects, appearance, setting, style, action, visible result, camera and optional sound. "
    "Include no nested JSON, alternatives, markdown, analysis, explanations, source records, "
    "or advice about writing prompts. Apply all writing guidance without copying it into prompt.\n\n"
)

OPENING_FRAME_VARIANT_GUIDANCE = (
    "These are still-image prompts for alternate compositions of one exact video opening frame. "
    "Keep the same frozen story state and initial pose in every variant. The related motion prompt describes what begins after this frame; "
    "use it only to keep the still aligned with that action, and do not depict later moments or write a motion timeline. "
    "Vary only the visual presentation through the layout idea, the viewing angle onto the scene, subject placement, depth staging, or focal path, keeping the character small in a larger world. "
    "Preserve all explicit framing and layout constraints from the user's request."
)

IMAGE_FALLBACK_PRESENTATIONS = (
    (
        "tiny_in_vast_world",
        "tiny character in a vast world",
        "Favor a vast, gently detailed world with the character tiny near one edge and a large area of open sky or paper. If the request fixes the framing, preserve it and express scale through the depth layers inside that frame.",
    ),
    (
        "diagonal_flow",
        "one diagonal path leading the eye",
        "Favor one strong diagonal path such as a river, road, rope, or row of objects that leads from the foreground to the character at its far end, with quiet space on the other side. If the request fixes the layout, preserve it and emphasize the path within that layout.",
    ),
    (
        "nested_in_object",
        "character nested in or on a large familiar object",
        "Favor a large familiar object that fills most of the frame and holds the character, so the object's scale tells the story. If the request fixes the framing, preserve it and let the nearest object carry the scale.",
    ),
    (
        "soft_vignette",
        "soft-edged vignette on open paper",
        "Favor a rounded, soft-edged vignette of the scene floating on open background, with a few small related details scattered just outside its edge. If the request fixes the layout, preserve it and keep the scene's edges soft.",
    ),
    (
        "framed_through_opening",
        "scene seen through a window, doorway, or foliage frame",
        "Favor a layered foreground frame such as a window, doorway, or leaves, with the character small in the lit scene beyond it. If the request fixes the framing, preserve it and create separation through overlap and value contrast.",
    ),
    (
        "tabletop_from_above",
        "tableau seen from above",
        "Favor a viewpoint from above onto one surface where the character stands among a few related objects arranged with generous space between them. If the request fixes the viewpoint, preserve it and arrange the objects within it.",
    ),
)

IMAGE_VARIANT_RESPONSE_CONTRACT = (
    "Return exactly the requested number of image prompt variants. Each prompt must be a complete, standalone scene description. "
    "All variants must preserve the exact same requested and selected subject identities, their defining physical traits, action, "
    "setting, visual medium, and style. Name every required subject in every prompt; never substitute a generic person or mascot. "
    "Give every variant a different layout idea, chosen once each from this list: "
    + "; ".join(presentation for _, presentation, _ in IMAGE_FALLBACK_PRESENTATIONS)
    + ". Name the layout idea in the label and presentation, never a lens or camera-shot term such as low-angle, close-up, or "
    "over-the-shoulder, and write the layout into the prompt so the six images differ at thumbnail size. "
    "Keep every named character small and simple in a larger world with open space unless a close-up is explicitly requested. "
    "Do not change the story or invent props, characters, crowds, or actions. Preserve any explicitly requested "
    "framing or layout; when a dimension is locked, create variation through the dimensions the request leaves open. "
    "Give each item a short unique label and a presentation description that names its compositional approach."
)

H3_SEGMENT_RESPONSE_CONTRACT = (
    "Return a concise H3 scene description in prompt, separate from the two audio fields. "
    "Unless the brief explicitly requests an ensemble or co-leads, choose one primary visual focus "
    "and keep every other required subject subordinate. Preserve each subject's identity, action role, "
    "and stated spatial relationship across the segment; never reverse who leads, follows, pursues, "
    "or flees. When the setting changes, connect it through the subject's route or a motivated cut, "
    "preserving readable screen direction and subject order. "
    "overall_soundscape must give 1–4 concrete sentences about ambience and audible physical actions. "
    "non_diegetic_music must give 1–3 concrete sentences about instrumentation, tempo, rhythm, and dynamics, "
    "or N/A when no score is requested. Do not repeat dialogue or music in overall_soundscape. "
)

H3_INITIAL_ACTION_CONTRACT = (
    "For a video request, write prompt as a motion-only timeline for H3 I2VA, not as another image prompt. "
    "Unless the brief explicitly requests an ensemble or co-leads, keep one primary visual focus; "
    "other required subjects remain subordinate and retain their stated roles and spatial relationship. "
    "Do not reverse who leads, follows, pursues, or flees. "
    "The selected image is the exact opening state and supplies the subject's visible identity, pose, style, "
    "composition, colors, setting, and existing props. Describe only what changes after that state: immediate "
    "action onset, a chronological physical path, follow-through, and "
    "the changed result or reaction. Keep the main character action clear and energetic unless the user explicitly "
    "asks for stillness or slow pacing. Style controls medium, line, surface, and palette; its motion language may "
    "shape secondary texture, but must not slow, replace, or contradict requested subject action. Do not invent "
    "anatomy, props, locations, or a second action. Keep the opening_keyframe_prompt as one frozen loaded pose "
    "that sets up the same action and follows the still-image composition, lighting, and material contract. "
    "Apply that still-image contract only to opening_keyframe_prompt; keep prompt motion-only. "
    "Keep the illustrated look unchanged and the camera to at most one slow gentle push-in. Start the action in the "
    "first tenth and keep the character moving energetically through the whole clip, chaining two or three connected "
    "moves with stretch and squash, still moving in the last frame. Apply this contract without "
    "copying it into prompt."
)


def validate_scene_response(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(
            "prompt response must be an object with a scene description string."
        )
    payload["prompt"] = require_h3_prompt(payload.get("prompt"))
    return payload


def _normalize_prompt_variant_count(value: object) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        count = 6
    return max(4, min(6, count))


def _normalize_composition_variant_count(value: object) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        count = 6
    return max(1, min(6, count))


def _build_fallback_image_prompt_sets(
    prompt: str, variant_count: int
) -> list[dict[str, str]]:
    return [
        {
            "label": label,
            "presentation": presentation,
            "prompt": f"{prompt.rstrip().rstrip('.')}. {direction}",
        }
        for label, presentation, direction in IMAGE_FALLBACK_PRESENTATIONS[
            :variant_count
        ]
    ]


def _required_image_subject_names(goal: GoalRequest) -> list[str]:
    subject_context = goal.constraints.get("subject_context")
    subjects = (
        dict(subject_context or {}).get("subjects") or []
        if isinstance(subject_context, dict)
        else []
    )
    names = [
        str(subject.get("name") or "").strip()
        for subject in subjects
        if isinstance(subject, dict) and str(subject.get("name") or "").strip()
    ]
    if names:
        return names
    return [
        part.split(":", 1)[0].strip()
        for part in str(goal.constraints.get("character") or "").split(";")
        if part.split(":", 1)[0].strip()
    ]


def _validate_image_prompt_set(
    payload: Any,
    expected_count: int,
    required_subject_names: list[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("image prompt response must be an object.")
    prompt_sets = payload.get("prompt_sets")
    if not isinstance(prompt_sets, list) or len(prompt_sets) != expected_count:
        raise ValueError(
            f"image prompt response must contain exactly {expected_count} variants."
        )
    labels: set[str] = set()
    presentations: set[str] = set()
    prompts: set[str] = set()
    for index, prompt_set in enumerate(prompt_sets, start=1):
        if not isinstance(prompt_set, dict):
            raise ValueError(f"image prompt variant {index} must be an object.")
        label = str(prompt_set.get("label") or "").strip()
        presentation = str(prompt_set.get("presentation") or "").strip()
        if not label or not presentation:
            raise ValueError(
                f"image prompt variant {index} needs a label and presentation."
            )
        prompt = require_h3_prompt(prompt_set.get("prompt"))
        missing_subjects = [
            name
            for name in required_subject_names or []
            if name.casefold() not in prompt.casefold()
        ]
        if missing_subjects:
            raise ValueError(
                f"image prompt variant {index} omits required subjects: "
                + ", ".join(missing_subjects)
            )
        label_key = label.casefold()
        presentation_key = re.sub(r"\s+", " ", presentation.casefold())
        prompt_key = re.sub(r"\s+", " ", prompt.casefold())
        if label_key in labels or presentation_key in presentations or prompt_key in prompts:
            raise ValueError("image prompt variants must have unique presentations and prompts.")
        labels.add(label_key)
        presentations.add(presentation_key)
        prompts.add(prompt_key)
        prompt_set.update(
            {"label": label, "presentation": presentation, "prompt": prompt}
        )
    payload["prompt_sets"] = prompt_sets
    return payload


def _preserve_later_episode_action(
    prompt: str, goal: GoalRequest, segment: dict[str, Any]
) -> str:
    """Keep the current episode's planned event primary after vision prompt writing."""
    if not goal.constraints.get("serial_mode"):
        return prompt
    try:
        episode_number = int(goal.constraints.get("serial_episode_number") or 1)
    except (TypeError, ValueError):
        episode_number = 1
    if episode_number <= 1:
        return prompt

    action = str(segment.get("action") or "").strip()
    effect = str(segment.get("effect") or segment.get("end_state") or "").strip()
    visual = str(segment.get("visual") or "").strip()
    if not any((action, effect, visual)):
        return prompt

    required = [
        "Current episode action (required and visually primary): "
        + (action or visual),
    ]
    if effect:
        required.append("Current episode visible outcome (required): " + effect)
    if visual and visual != action:
        required.append("Current episode visual direction: " + visual)
    required.extend(
        [
            "Cross-episode continuity: preserve character identity, world, and palette from the prior reference. Treat that image as the opening continuity point only; develop the current episode's new action immediately instead of repeating the prior pose or composition.",
            "The required action takes priority over reference-aware scene notes if they conflict. Use those notes for compatible appearance and camera details only.",
            "Reference-aware appearance and camera notes: " + prompt,
        ]
    )
    return "\n\n".join(required)


def validate_segment_response(payload: Any) -> dict[str, Any]:
    payload = validate_scene_response(payload)
    if not isinstance(payload.get("narration"), str):
        raise ValueError("narration must be a string, not a nested object.")
    for key in ("overall_soundscape", "non_diegetic_music"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a non-empty string.")
        payload[key] = value.strip()
    return payload


def _normalize_anchor_text(value: Any) -> str:
    return " ".join(re.sub(r"\s+", " ", str(value or "")).split()).casefold()


def validate_news_grounded_scene_response(
    payload: Any, news_context: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("news-grounded scene response must be an object.")
    prompt = require_h3_prompt(payload.get("prompt"))
    source_text = _normalize_anchor_text(
        "\n".join(
            str(news_context.get(key) or "")
            for key in ("title", "keyword", "content")
        )
    )
    raw_anchors = payload.get("source_anchors")
    if not isinstance(raw_anchors, list) or not 1 <= len(raw_anchors) <= 2:
        raise ValueError("source_anchors must contain one or two source-to-scene mappings.")

    validated_anchors: list[dict[str, str]] = []
    normalized_prompt = _normalize_anchor_text(prompt)
    for index, raw_anchor in enumerate(raw_anchors, start=1):
        if not isinstance(raw_anchor, dict):
            raise ValueError(f"source_anchors[{index}] must be an object.")
        source_excerpt = str(raw_anchor.get("source_excerpt") or "").strip()
        visual_mapping = str(raw_anchor.get("visual_mapping") or "").strip()
        normalized_excerpt = _normalize_anchor_text(source_excerpt)
        normalized_mapping = _normalize_anchor_text(visual_mapping)
        if not normalized_excerpt or normalized_excerpt not in source_text:
            raise ValueError(
                f"source_anchors[{index}] source_excerpt must quote the supplied article."
            )
        if not normalized_mapping or normalized_mapping not in normalized_prompt:
            raise ValueError(
                f"source_anchors[{index}] visual_mapping must appear in the scene prompt."
            )
        validated_anchors.append(
            {"source_excerpt": source_excerpt, "visual_mapping": visual_mapping}
        )

    payload["prompt"] = prompt
    payload["source_anchors"] = validated_anchors
    return payload


def generate_autonomous_scene_prompt(
    client: PromptClient,
    *,
    character: str,
    style: str,
    media_type: str,
    news_context: dict[str, Any] | None = None,
    news_grounding_required: bool = False,
) -> dict[str, Any]:
    normalized_news_context = dict(news_context or {})
    news_grounding_required = bool(news_grounding_required)
    fallback = build_autonomous_scene_prompt(
        character=character,
        style=style,
        media_type=media_type,
        news_context=normalized_news_context,
        news_grounding_required=news_grounding_required,
    )
    try:
        manager = client.require_manager()
        user_prompt = "\n".join(
            [
                f"Character: {character}",
                f"Style: {style}",
                f"Media type: {media_type}",
                f"News context JSON: {json.dumps(normalized_news_context, ensure_ascii=False)}",
                "Create a generation prompt that follows the supplied creative brief.",
                (
                    f"Video action direction: {VIDEO_MOTION_DIRECTION}"
                    if requires_video_motion_direction(media_type)
                    else IMAGE_PROMPT_CONTRACT
                    if media_type == "image"
                    else ""
                ),
                (
                    "This is a news-grounded story, not optional mood or style inspiration. Anchor the scene in the article's main documented event or impact. Select one or two concrete people, things, places, or events that the article explicitly names; turn each selected fact into a visible action, object, or consequence that makes its relationship to the article immediately clear. Put each exact short source excerpt and its matching visual_mapping into source_anchors, and include every visual_mapping verbatim in prompt. Preserve the source's actor-event relationship. A character may act as an allegorical stand-in, but a generic mood, a decorative motif, or style language cannot replace the event. Style controls medium, surface, palette, and composition only; it must not change the subject, setting, action, or emotional meaning of the source. Do not draw text, invent source facts, or imply a reported person did something the article does not say."
                    if news_grounding_required
                    else (
                        "The supplied article is optional creative inspiration. Draw on one or two concrete people, things, places, or events it names when shaping the scene, and make their visual relationship to the scene recognizable. A character may act as an imaginative stand-in, but keep reported facts distinct from invented action. Do not add unsupported claims about what a real person did."
                        if is_usable_news_selection(
                            normalized_news_context.get("title", ""),
                            normalized_news_context.get("keyword", ""),
                        )
                        else "No usable news article was supplied. Do not invent or imply a news event; build the scene only from the character, style, and media request."
                    )
                ),
                "Return the requested JSON directly using the supplied information.",
                (
                    "Return JSON with keys: prompt, creative_seed, source, source_anchors. Each source_anchors item has exactly source_excerpt and visual_mapping."
                    if news_grounding_required
                    else "Return JSON with keys: prompt, creative_seed, source."
                ),
            ]
        )
        schema_properties: dict[str, Any] = {
            "prompt": {"type": "string"},
            "creative_seed": {"type": "string"},
            "source": {"type": "string"},
        }
        required_fields = ["prompt", "creative_seed", "source"]
        if news_grounding_required:
            schema_properties["source_anchors"] = {
                "type": "array",
                "minItems": 1,
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "source_excerpt": {"type": "string"},
                        "visual_mapping": {"type": "string"},
                    },
                    "required": ["source_excerpt", "visual_mapping"],
                    "additionalProperties": False,
                },
            }
            required_fields.append("source_anchors")
        if news_grounding_required:
            def validate_response(payload: Any) -> dict[str, Any]:
                return validate_news_grounded_scene_response(
                    payload, normalized_news_context
                )
        else:
            validate_response = validate_scene_response
        payload = client.generate_json(
            manager,
            LONG_VIDEO_SYSTEM_PROMPT,
            user_prompt,
            schema_name="autonomous_scene_prompt",
            schema={
                "type": "object",
                "properties": schema_properties,
                "required": required_fields,
                "additionalProperties": False,
            },
            validate_response=validate_response,
        )
        result = {
            "prompt": str(payload.get("prompt") or fallback["prompt"]),
            "creative_seed": str(
                payload.get("creative_seed") or fallback["creative_seed"]
            ),
            "source": str(payload.get("source") or "autonomous_llm"),
            "news_context": normalized_news_context,
            "news_grounding_required": news_grounding_required,
        }
        if news_grounding_required:
            result["source_anchors"] = list(payload["source_anchors"])
        return client.mark_payload(result)
    except Exception as exc:
        if news_grounding_required:
            raise client.generation_error("autonomous_scene_prompt", exc) from exc
        fallback["prompt_mode"] = "template"
        fallback["fallback_reason"] = "manager_unavailable"
        fallback["manager_error"] = f"{type(exc).__name__}: {exc}"
        fallback["llm_backend"] = client.backend_info()
        fallback["news_context"] = normalized_news_context
        fallback["news_grounding_required"] = False
        return fallback


def expand_goal(
    client: PromptClient,
    goal: GoalRequest,
    selected_style: str,
    idea_variants: list[dict[str, Any]],
    reference_analysis: dict[str, Any] | None = None,
    prompt_variant_count: int = 6,
) -> dict[str, Any]:
    fallback = build_goal_brief(goal, selected_style, idea_variants)
    image_prompt_request = goal.media_type == "image"
    prompt_variant_count = (
        _normalize_prompt_variant_count(prompt_variant_count)
        if image_prompt_request
        else 0
    )
    video_request = requires_video_motion_direction(goal.media_type)
    reference_directive = format_reference_video_directive(
        reference_analysis, max_chars=2200
    )
    reference_images = reference_keyframe_paths(reference_analysis)[:6]
    news_context = dict(goal.constraints.get("news_context") or {})
    news_grounding_required = bool(
        goal.constraints.get("news_driven")
        or goal.constraints.get("news_grounding_required")
    )
    optional_news_context = is_usable_news_selection(
        news_context.get("title", ""), news_context.get("keyword", "")
    )
    news_grounding_contract = (
        f"{news_grounding_anchor_clause(news_context)} "
        f"Carry the validated source-to-scene mappings into the creative brief and prompt: {json.dumps(goal.constraints.get('news_source_anchors') or [], ensure_ascii=False)}. "
        "Keep at least one mapped article fact visible in the story action or consequence; do not let style or generic mood replace it. "
        "News source accuracy: Preserve each documented location, actor, and event relationship; do not merge events from different places or invent source-specific people or actions. "
        "For a multi-place story, select one event the article explicitly ties to that place or use an unlocated metaphor for the shared impact. "
        "Any added cartoon comedy must read as allegory, not as a reported event."
        if news_grounding_required
        else (
            "The supplied article is optional creative inspiration. Draw on one or two concrete people, things, places, or events it names when shaping the scene, and make their visual relationship to the scene recognizable. Keep reported facts distinct from imagined action and do not imply unsupported claims about real people."
            if optional_news_context
            else "No usable news article was supplied. Do not invent or imply one; follow the user's brief and style."
        )
    )
    reference_motion_directive = (
        "Use the supplied reference for visual inspiration while following the current prompt."
        if reference_images or reference_analysis
        else ""
    )
    if reference_directive:
        fallback["creative_brief"] = (
            f"{fallback['creative_brief']}\n{reference_directive}"
        )
        fallback["prompt"] = (
            f"{fallback['prompt']}, {reference_motion_directive or 'use the supplied reference for visual inspiration'}"
        )
    if reference_motion_directive:
        fallback["creative_brief"] = (
            f"{fallback['creative_brief']}\n{reference_motion_directive}"
        )
        fallback["prompt"] = f"{fallback['prompt']}, {reference_motion_directive}"
    still_prompt_contract = ""
    if goal.media_type == "image":
        still_prompt_contract = IMAGE_PROMPT_CONTRACT
    elif video_request and goal.media_type not in {
        "sticker_pack",
        "animated_sticker",
        "game_sprite",
    }:
        still_prompt_contract = (
            "Apply this still-image contract only to opening_keyframe_prompt; keep prompt motion-only. "
            + IMAGE_PROMPT_CONTRACT
        )
    style_treatment = (
        f"Selected video rendering treatment (for prompt only): {style_profile_direction(goal)}"
        if video_request
        else f"Selected rendering treatment: {style_profile_direction(goal, include_motion=False)}"
    )
    keyframe_treatment = (
        f"Still-image rendering treatment for opening_keyframe_prompt only: "
        f"{style_profile_direction(goal, include_motion=False)}"
        if still_prompt_contract and video_request
        else ""
    )
    if image_prompt_request:
        fallback["prompt_sets"] = _build_fallback_image_prompt_sets(
            fallback["prompt"], prompt_variant_count
        )
    try:
        manager = client.require_manager()
        if image_prompt_request:
            response_contract = IMAGE_VARIANT_RESPONSE_CONTRACT
            response_schema = {
                "type": "object",
                "properties": {
                    "creative_brief": {"type": "string"},
                    "prompt_sets": {
                        "type": "array",
                        "minItems": prompt_variant_count,
                        "maxItems": prompt_variant_count,
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "presentation": {"type": "string"},
                                "prompt": {"type": "string"},
                            },
                            "required": ["label", "presentation", "prompt"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["creative_brief", "prompt_sets"],
                "additionalProperties": False,
            }

            required_subject_names = _required_image_subject_names(goal)

            def validate_response(payload: Any) -> dict[str, Any]:
                return _validate_image_prompt_set(
                    payload, prompt_variant_count, required_subject_names
                )

            response_prompt = (
                f"Return JSON with keys: creative_brief and prompt_sets. Include exactly "
                f"{prompt_variant_count} prompt_sets, each with label, presentation, and prompt."
            )
            variant_guidance = (
                "Required cast and identity details: "
                + _goal_subject_instruction(goal)
                + ". Repeat the exact name and distinguishing traits of every required subject in every prompt. Keep the same decisive action, "
                "cast, setting, and style in every prompt. Make the presentations meaningfully different at thumbnail size by changing "
                "composition, camera viewpoint or crop, subject placement, depth staging, or focal path; each prompt must be independently usable. "
                "Do not substitute a human or generic mascot, add spectators, or invent any character or prop. If the user explicitly fixes a "
                "dimension, preserve it and vary other dimensions. Do not create alternate stories."
            )
        else:
            response_contract = (
                H3_INITIAL_ACTION_CONTRACT if video_request else SCENE_RESPONSE_CONTRACT
            )
            response_schema = {
                "type": "object",
                "properties": {
                    "creative_brief": {"type": "string"},
                    "prompt": {"type": "string"},
                    "opening_keyframe_prompt": {"type": "string"},
                },
                "required": ["creative_brief", "prompt"],
                "additionalProperties": False,
            }
            validate_response = validate_scene_response
            response_prompt = (
                "Return JSON with keys: creative_brief, prompt, opening_keyframe_prompt."
            )
            variant_guidance = ""
        user_prompt = "\n".join(
            [
                f"Goal: {goal.prompt}",
                f"Media type: {goal.media_type}",
                f"Character: {goal.constraints.get('character', '')}",
                _goal_subject_instruction(goal),
                still_prompt_contract,
                style_treatment,
                keyframe_treatment,
                f"Style: {selected_style}",
                f"Duration seconds: {goal.duration_seconds}",
                f"News context JSON: {json.dumps(goal.constraints.get('news_context', {}), ensure_ascii=False)}",
                reference_directive,
                reference_motion_directive,
                (
                    "Use attached reference keyframes as optional visual inspiration."
                    if reference_images
                    else "No reference-video keyframes were supplied."
                ),
                response_prompt,
                variant_guidance,
                "Preserve the requested scene, cast, action, setting, text, and explicit framing. "
                "Style controls rendering only; its composition is a fallback only when framing is unspecified.",
                (
                    "Write prompt as a motion-only timeline for H3 I2VA; the selected image will be the exact opening frame. "
                    "Do not restate the image's pose, composition, appearance, colors, or existing objects. Begin with the first visible movement, "
                    "then describe one cause-driven action along a clear path, follow-through, "
                    "and a visible changed result or reaction. Begin the action in the first tenth, reach the resolved endpoint only in the final tenth, "
                    "and keep it moving through most of the shot. Do not settle early or leave a static tail; the character is still moving in the last frame. "
                    "Use energetic motion unless the user's request explicitly asks for stillness or slow pacing. Style controls medium, line, "
                    "surface, and palette; do not let style-profile motion language slow or replace the requested subject action. Do not invent "
                    "anatomy, props, locations, or a second action. The opening_keyframe_prompt describes one frozen loaded pose before this same action. "
                    f"Video direction: {VIDEO_MOTION_DIRECTION}"
                    if video_request
                    else ""
                ),
                news_grounding_contract,
            ]
        )
        payload = client.generate_json(
            manager,
            LONG_VIDEO_SYSTEM_PROMPT,
            user_prompt,
            schema_name="goal_brief",
            output_contract=response_contract,
            validate_response=validate_response,
            schema=response_schema,
            images=reference_images or None,
        )
        fallback["creative_brief"] = str(
            payload.get("creative_brief") or fallback["creative_brief"]
        )
        if image_prompt_request:
            fallback["prompt_sets"] = payload["prompt_sets"]
            fallback["prompt"] = fallback["prompt_sets"][0]["prompt"]
            fallback["opening_keyframe_prompt"] = fallback["prompt"]
        else:
            fallback.update(
                {
                    "prompt": require_h3_prompt(payload.get("prompt")),
                    "opening_keyframe_prompt": apply_video_opening_frame_direction(
                        str(
                            payload.get("opening_keyframe_prompt")
                            or fallback.get("opening_keyframe_prompt")
                            or fallback["prompt"]
                        ),
                        goal.media_type,
                    ),
                }
            )
        fallback["prompt"] = include_image_prompt_quality_clause(
            fallback["prompt"], goal
        )
        return client.mark_payload(fallback)
    except Exception as exc:
        return client.template_fallback(fallback, exc)


def expand_image_prompt_variants(
    client: PromptClient,
    goal: GoalRequest,
    base_prompt: str,
    variant_count: int,
    *,
    prompt_role: str = "opening frame",
    related_motion_prompt: str = "",
    review_guidance: str = "",
) -> dict[str, Any]:
    """Write distinct still compositions without changing the underlying scene."""
    count = _normalize_composition_variant_count(variant_count)
    base_prompt = require_h3_prompt(base_prompt)
    fallback_sets = _build_fallback_image_prompt_sets(base_prompt, count)
    fallback: dict[str, Any] = {
        "prompt": fallback_sets[0]["prompt"],
        "prompt_sets": fallback_sets,
    }
    try:
        manager = client.require_manager()
        schema = {
            "type": "object",
            "properties": {
                "prompt_sets": {
                    "type": "array",
                    "minItems": count,
                    "maxItems": count,
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "presentation": {"type": "string"},
                            "prompt": {"type": "string"},
                        },
                        "required": ["label", "presentation", "prompt"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["prompt_sets"],
            "additionalProperties": False,
        }
        required_subject_names = _required_image_subject_names(goal)

        def validate_response(payload: Any) -> dict[str, Any]:
            return _validate_image_prompt_set(
                payload, count, required_subject_names
            )

        user_prompt = "\n".join(
            [
                f"Goal: {goal.prompt}",
                f"Prompt role: {prompt_role}",
                f"Base still prompt: {base_prompt}",
                (
                    f"Related motion prompt for continuity only: {related_motion_prompt}"
                    if related_motion_prompt
                    else "No separate motion prompt was supplied."
                ),
                f"Selected style: {goal.style}",
                _goal_subject_instruction(goal),
                f"Review guidance for these still compositions: {review_guidance}",
                f"Return exactly {count} complete, standalone image prompts in prompt_sets.",
                IMAGE_VARIANT_RESPONSE_CONTRACT,
                OPENING_FRAME_VARIANT_GUIDANCE,
                "Repeat the exact required subject names and identity details in every prompt. "
                "Do not change the decisive action, cast, setting, medium, or style, and do not add characters, props, or story events.",
            ]
        )
        payload = client.generate_json(
            manager,
            LONG_VIDEO_SYSTEM_PROMPT,
            user_prompt,
            schema_name="opening_frame_prompt_variants",
            output_contract=(
                IMAGE_VARIANT_RESPONSE_CONTRACT
                + " Each prompt describes one frozen image, never video movement."
            ),
            validate_response=validate_response,
            schema=schema,
        )
        result = {
            "prompt_sets": payload["prompt_sets"],
            "prompt": payload["prompt_sets"][0]["prompt"],
        }
        return client.mark_payload(result)
    except Exception as exc:
        return client.template_fallback(fallback, exc)


def compose_prompt(
    client: PromptClient,
    goal: GoalRequest,
    prompt: str,
    style: str,
    prefix: str = "",
    suffix: str = "",
    negative_prompt: str = "",
) -> dict[str, Any]:
    fallback = {
        "prompt": ", ".join(
            part for part in (prefix, prompt, style, suffix) if part
        ),
        "negative_prompt": ", ".join(
            part
            for part in (str(negative_prompt or "").strip(), style_profile_avoidance(goal))
            if part
        ),
    }
    fallback["prompt"] = include_image_prompt_quality_clause(
        include_visual_style_profile(fallback["prompt"], goal), goal
    )
    manager = client.require_manager()
    user_prompt = "\n".join(
        [
            f"Goal: {goal.prompt}",
            f"Media type: {goal.media_type}",
            f"Base prompt: {prompt}",
            _goal_subject_instruction(goal),
            IMAGE_PROMPT_CONTRACT if goal.media_type == "image" else "",
            f"Selected rendering treatment: {style_profile_direction(goal)}",
            f"Style: {style}",
            f"Prefix: {prefix}",
            f"Suffix: {suffix}",
            f"Character: {goal.constraints.get('character', '')}",
            f"News context JSON: {json.dumps(goal.constraints.get('news_context', {}), ensure_ascii=False)}",
            "Return JSON with key: prompt.",
            "Preserve the requested scene, cast, action, setting, and explicit framing. "
            "Style controls rendering only; its composition is a fallback only when framing is unspecified.",
            "Write one natural concise prompt that follows the user's brief and base scene.",
            "When a prior or first frame is supplied, use it as the opening state and describe any requested motion from there.",
            "When using news context, keep factual claims accurate while following the user's requested visual treatment.",
        ]
    )
    try:
        payload = client.generate_json(
            manager,
            LONG_VIDEO_SYSTEM_PROMPT,
            user_prompt,
            schema_name="compose_prompt",
            output_contract=SCENE_RESPONSE_CONTRACT,
            validate_response=validate_scene_response,
            schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
        )
        return client.mark_payload(
            {
                "prompt": include_image_prompt_quality_clause(
                    require_h3_prompt(payload.get("prompt")), goal
                ),
                "negative_prompt": fallback["negative_prompt"],
            }
        )
    except Exception as exc:
        raise client.generation_error("compose_prompt", exc) from exc


def segment_story(
    client: PromptClient,
    goal: GoalRequest,
    creative_brief: str,
    segment_count: int,
    tone: str,
    reference_analysis: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    fallback = build_story_segments(
        goal,
        creative_brief,
        segment_count,
        tone,
    )
    raw_news_context = goal.constraints.get("news_context")
    news_context = dict(raw_news_context) if isinstance(raw_news_context, dict) else {}
    has_news_source = any(
        str(news_context.get(key) or "").strip()
        for key in ("title", "content", "keyword")
    )
    try:
        manager = client.require_manager()
    except Exception:
        if bool(goal.constraints.get("serial_mode")):
            return validate_story_segments(fallback, segment_count)
        raise
    reference_directive = format_reference_video_directive(
        reference_analysis, max_chars=2200
    )
    reference_images = reference_keyframe_paths(reference_analysis)[:6]

    video_story_contract = requires_video_motion_direction(goal.media_type)
    if video_story_contract:
        direction_field_names = ", ".join(
            key for key, _ in VIDEO_SEGMENT_DIRECTION_FIELDS
        )
        camera_contract = (
            "Unless the brief explicitly requests an ensemble or co-leads, choose one primary visual "
            "focus and identify it in every segment; keep other required subjects subordinate and "
            "preserve their identities, roles, and spatial order. Do not reverse who leads, follows, "
            "pursues, or flees. When a setting changes, connect it through the subject's route or a "
            "motivated cut; preserve screen direction, or re-establish it clearly before changing the axis. "
            "Give every segment a distinct consecutive beat and a readable start and result. "
            "For a causal story, keep one visible goal, one obstacle, and one turn that changes the outcome; "
            "each beat changes a visible state and causes the next, and the ending resolves or reframes the opening. "
            "If the brief requests one action, do not invent a second problem or subplot. "
            "The first segment begins its action within its first tenth; the final segment reaches "
            "a clear payoff before a brief ending hold. Keep the subject's body, face, and goal readable. "
            "When the tone supports it, include a compact anticipation, action, follow-through, and charming "
            "reaction such as a blink, delighted glance, or proud bounce. Do not add cute business to solemn scenes. "
            f"Each segment must include {direction_field_names}. The camera field names the starting framing and either a locked camera or "
            "the single slow push-in, and what stays in frame; it never plans a pan, tilt, dolly, tracking move, orbit, zoom, close-up cut, or crane. "
            "Keep the protagonist's face and every story-critical object within frame with clear edge margins for the whole segment, "
            "and show the visible result before any reaction. "
            f"Video action and camera direction: {VIDEO_MOTION_DIRECTION} "
            "Describe the selected style with stable visible rules for medium and surface, palette and light, and motion or transitions; keep them consistent across shots."
        )
    else:
        camera_contract = ""

    user_prompt = "\n".join(
        [
            f"Goal: {goal.prompt}",
            f"Media type: {goal.media_type}",
            f"Character: {goal.constraints.get('character', '')}",
            _goal_subject_instruction(goal),
            f"Style: {goal.style}",
            f"Resolved visual style rules: {style_profile_direction(goal)}",
            f"Creative brief: {creative_brief}",
            f"Tone: {tone}",
            f"Segment count: {segment_count}",
            f"News context JSON: {json.dumps(news_context, ensure_ascii=False)}",
            reference_directive,
            camera_contract,
            (
                "Keep news details consistent with the supplied source; do not present the character as a real participant in the event."
                if has_news_source
                else "Follow the user's brief."
            ),
            (
                "Use the reference images as optional visual inspiration."
                if reference_images
                else "No reference-video keyframes were supplied."
            ),
            "Return one JSON object with a segments array in the requested order. Each segment needs segment_id, visual, and narration; video segments also require action, cause, effect, and camera.",
            "Write creative fields in natural English.",
        ]
    )
    segment_properties: dict[str, Any] = {
        "segment_id": {"type": "string"},
        "visual": {"type": "string"},
        "narration": {"type": "string"},
    }
    segment_required = ["segment_id", "visual", "narration"]
    if video_story_contract:
        segment_properties.update(
            {key: {"type": "string"} for key, _ in VIDEO_SEGMENT_DIRECTION_FIELDS}
        )
        segment_required.extend(key for key, _ in VIDEO_SEGMENT_DIRECTION_FIELDS)
    try:
        payload = client.generate_json(
            manager,
            LONG_VIDEO_SYSTEM_PROMPT,
            user_prompt,
            schema_name="story_segments",
            schema={
                "type": "object",
                "properties": {
                    "segments": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": segment_properties,
                            "required": segment_required,
                            "additionalProperties": True,
                        },
                    }
                },
                "required": ["segments"],
                "additionalProperties": False,
            },
            images=reference_images or None,
        )
        segments = payload.get("segments")
        if isinstance(segments, list) and segments:
            normalized: list[dict[str, Any]] = []
            for index, item in enumerate(segments[:segment_count]):
                item = item if isinstance(item, dict) else {}
                normalized_item: dict[str, Any] = {
                    "segment_id": str(item.get("segment_id") or f"segment-{index + 1}"),
                    "visual": str(item.get("visual") or fallback[index]["visual"]),
                    "narration": str(
                        item.get("narration") or fallback[index]["narration"]
                    ),
                }
                if video_story_contract:
                    for key, _ in VIDEO_SEGMENT_DIRECTION_FIELDS:
                        value = str(item.get(key) or "").strip()
                        if value:
                            normalized_item[key] = value
                normalized.append(normalized_item)
            while len(normalized) < segment_count:
                fallback_item = dict(fallback[len(normalized)])
                normalized.append(fallback_item)
            return validate_story_segments(normalized, segment_count)
    except Exception as exc:
        del exc
    return validate_story_segments(fallback, segment_count)


def prepare_segment(
    client: PromptClient,
    goal: GoalRequest,
    segment: dict[str, Any],
    negative_prompt: str,
    previous_segment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manager = client.require_manager()

    continuity_lines = [
        f"Current segment id: {segment.get('segment_id', '')}",
        f"Current segment visual: {segment.get('visual', '')}",
        (
            "Internal shot beats JSON: "
            + json.dumps(segment.get("shots"), ensure_ascii=False)
            if isinstance(segment.get("shots"), list)
            else ""
        ),
        f"Current segment narration: {segment.get('narration', '')}",
        *(
            f"Current segment {key}: {segment.get(key, '')}"
            for key, _ in VIDEO_SEGMENT_DIRECTION_FIELDS
            if str(segment.get(key) or "").strip()
        ),
        f"Style: {goal.style}",
        f"Resolved visual style rules: {style_profile_direction(goal)}",
        f"Character: {goal.constraints.get('character', '')}",
        _goal_subject_instruction(goal),
        f"News context JSON: {json.dumps(goal.constraints.get('news_context', {}), ensure_ascii=False)}",
    ]
    if previous_segment:
        continuity_lines.extend(
            [
                f"Previous segment visual: {previous_segment.get('visual', '')}",
                f"Previous segment narration: {previous_segment.get('narration', '')}",
            ]
        )
    continuity_lines.extend(
        [
            "Return JSON with keys: prompt, narration, overall_soundscape, non_diegetic_music.",
            "prompt must be one non-empty string containing only the final visual scene description for H3: "
            "subjects and appearance, setting and visual style, visible action and result, camera movement, "
            "and any dialogue or synchronized sound caused by an on-screen action. Incorporate the supplied shot details into that description. "
            "Do not include nested objects, alternative prompts, explanations, analysis, markdown, "
            "source records, or instructions about how to write a prompt. narration must be a string. "
            "overall_soundscape describes only concrete environmental and physical-action sounds; "
            "non_diegetic_music describes an audience-only score or is N/A when no score is requested.",
            f"Writing guidance (apply it to the scene; do not copy it into prompt): {VIDEO_MOTION_DIRECTION}",
            "Follow the user's requested scene, action, framing, style, and pacing.",
            "Keep claims about supplied news context within the source; use its visual treatment as requested.",
        ]
    )
    try:
        payload = client.generate_json(
            manager,
            LONG_VIDEO_SYSTEM_PROMPT,
            "\n".join(continuity_lines),
            schema_name="segment_prompt",
            output_contract=H3_SEGMENT_RESPONSE_CONTRACT,
            validate_response=validate_segment_response,
            schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "narration": {"type": "string"},
                    "overall_soundscape": {"type": "string"},
                    "non_diegetic_music": {"type": "string"},
                },
                "required": ["prompt", "narration", "overall_soundscape", "non_diegetic_music"],
                "additionalProperties": False,
            },
        )
        subject_context = goal.constraints.get("subject_context")
        result = {
            "segment_id": segment["segment_id"],
            "prompt": require_h3_prompt(payload.get("prompt")),
            "narration": payload["narration"],
            "overall_soundscape": payload["overall_soundscape"],
            "non_diegetic_music": payload["non_diegetic_music"],
            "negative_prompt": negative_prompt,
            "subject_context": dict(subject_context)
            if isinstance(subject_context, dict)
            else {},
        }
        return client.mark_payload(result)
    except Exception as exc:
        raise client.generation_error("prepare_segment", exc) from exc


def prepare_h3_segment(
    client: PromptClient,
    goal: GoalRequest,
    segment: dict[str, Any],
    *,
    draft_prompt: str,
    overall_soundscape: str,
    non_diegetic_music: str,
    duration_seconds: float,
    image_references: list[dict[str, str]],
) -> dict[str, Any]:
    """Write the final H3 scene after reviewing its actual conditioning images."""
    manager = client.require_manager()
    image_paths: list[str] = []
    image_context: list[str] = []
    for index, reference in enumerate(image_references, start=1):
        path = str(reference.get("path") or "").strip()
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f"H3 prompt reference image is not readable: {path}")
        image_paths.append(path)
        label = str(reference.get("label") or f"Image {index}").strip()
        role = str(reference.get("role") or "visual reference").strip()
        image_context.append(f"Attached image {index} ({label}) is the {role}.")

    source_prompt = require_h3_prompt(draft_prompt)
    prompt_context = [
        "Write the final scene prompt for MiniMax H3 after inspecting the attached conditioning image(s).",
        "Treat the visible image(s) as the source of truth for character identity and design, palette, art style, location, and important visible objects. The written draft supplies intended action and story meaning. When details conflict, preserve the image's identity and objects and adapt the action to them.",
        "For a comic page, contact sheet, collage, or other image with repeated depictions, interpret those poses as successive views of the same subject. Write one continuous scene with one instance of each requested character; do not turn gutters or repeated poses into extra characters or a persistent split-screen layout unless the brief explicitly asks for that. Keep the visible starting pose and subject-to-object relationship readable.",
        "Keep one continuous shot with one main visible action and at most one slow gentle push-in; move the camera further only when the brief asks for it. Describe the visible starting state, physical movement, and readable ending state. Keep the H3 scene concise and specific; remove redundant style adjectives and invented details. Do not add characters, props, costume changes, scene changes, text, or actions unsupported by the image or the source draft.",
        "Write an ordered chain of positive physical-action beats, not a list of prohibitions: from the first moment name what moves, its target and path, the next connected movement, and the visible resulting state. Use 2–3 concrete action verbs; do not describe a static pose and expect H3 to infer movement. Start the prompt with a motion verb for the character and never describe the picture as it stands; even when the image is calm or the draft has no action, give the character a lively physical action in its own scene, such as bouncing, hopping, swinging its feet, or reaching, chained into connected moves within the visible scene; the character never travels to a place the image does not show, and the camera only pushes in toward it rather than following it. When the draft's action needs a place or object the image does not show, such as a platform, a train, or a street, replace it with a lively action on the visible props, for example hopping on, around, or over the object it stands on.",
        f"Segment duration: {float(duration_seconds):.2f} seconds. Start the action in the first moment and keep the character moving energetically through the whole segment: anticipation, a big action with stretch and squash, then the same action chained into two or three connected moves while props and scene elements respond on every beat, still moving in the last frame. Do not jump directly to the ending frame.",
        "Return only scene description in prompt, concrete environmental and physical-action sound in overall_soundscape, and an audience-only score in non_diegetic_music (or N/A). Do not put prompt-writing instructions, source analysis, or serialized objects inside prompt.",
        f"Character: {goal.constraints.get('character', '')}",
        f"Selected style: {goal.style}",
        f"Scene draft: {source_prompt}",
        f"Segment plan: {json.dumps(segment, ensure_ascii=False)}",
        f"Existing soundscape: {overall_soundscape}",
        f"Existing score direction: {non_diegetic_music}",
        *image_context,
    ]
    try:
        serial_episode_number = int(
            goal.constraints.get("serial_episode_number") or 1
        )
    except (TypeError, ValueError):
        serial_episode_number = 1
    if goal.constraints.get("serial_mode") and serial_episode_number > 1:
        prompt_context.insert(
            2,
            "This is a later serial episode. The prior episode reference is only the opening continuity moment: preserve the character identity and world, then move away from the prior pose and composition as the current action unfolds. The structured current segment plan is authoritative for the new plot event. Explicit characters, props, and locations named in that plan must be shown even when they are absent from the prior reference image.",
        )
    news_context = goal.constraints.get("news_context")
    if isinstance(news_context, dict) and news_context:
        prompt_context.append(
            "Source facts for semantic boundaries (do not depict this metadata): "
            + json.dumps(news_context, ensure_ascii=False)
        )

    def validate_h3_scene(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("H3 scene response must be an object.")
        payload = validate_scene_response(payload)
        for key in ("overall_soundscape", "non_diegetic_music"):
            value = payload.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{key} must be a non-empty string.")
            payload[key] = value.strip()
        return payload

    try:
        payload = client.generate_json(
            manager,
            LONG_VIDEO_SYSTEM_PROMPT,
            "\n".join(prompt_context),
            schema_name="h3_scene_prompt",
            output_contract=H3_SEGMENT_RESPONSE_CONTRACT,
            model="vision",
            validate_response=validate_h3_scene,
            schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "overall_soundscape": {"type": "string"},
                    "non_diegetic_music": {"type": "string"},
                },
                "required": ["prompt", "overall_soundscape", "non_diegetic_music"],
                "additionalProperties": False,
            },
            images=image_paths or None,
        )
        return client.mark_payload(
            {
                "segment_id": str(segment.get("segment_id") or ""),
                "prompt": _preserve_later_episode_action(
                    require_h3_prompt(payload.get("prompt")), goal, segment
                ),
                "overall_soundscape": payload["overall_soundscape"],
                "non_diegetic_music": payload["non_diegetic_music"],
                "image_references": image_references,
            }
        )
    except Exception as exc:
        raise client.generation_error("prepare_h3_segment", exc) from exc


def refine_prompt_from_review(
    client: PromptClient,
    goal: GoalRequest,
    original_prompt: str,
    review_notes: str,
    media_paths: list[str] | None = None,
) -> dict[str, Any]:
    fallback = {
        "prompt": ", ".join(
            part
            for part in (
                original_prompt,
                f"revision notes: {review_notes}" if review_notes else "",
            )
            if part
        ),
        "negative_prompt": str(goal.constraints.get("negative_prompt") or ""),
    }
    try:
        manager = client.require_manager()
        user_prompt = "\n".join(
            [
                f"Goal: {goal.prompt}",
                f"Media type: {goal.media_type}",
                f"Original prompt: {original_prompt}",
                f"Review notes: {review_notes}",
                f"Selected media count: {len(media_paths or [])}",
                "Return JSON with key: prompt.",
                "Apply the review notes while following the original prompt.",
            ]
        )
        payload = client.generate_json(
            manager,
            LONG_VIDEO_SYSTEM_PROMPT,
            user_prompt,
            schema_name="review_prompt_refinement",
            schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
        )
        return client.mark_payload(
            {
                "prompt": str(payload.get("prompt") or fallback["prompt"]),
                "negative_prompt": fallback["negative_prompt"],
            }
        )
    except Exception as exc:
        return client.template_fallback(
            fallback, exc, fallback_reason="json_parse_failed"
        )
