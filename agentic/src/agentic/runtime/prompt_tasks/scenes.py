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
    LONG_VIDEO_SYSTEM_PROMPT,
    build_autonomous_scene_prompt,
    build_goal_brief,
    news_grounding_anchor_clause,
    build_story_segments,
    style_profile_avoidance,
    style_profile_direction,
    validate_story_segments,
)
from agentic.runtime.llm_client import PromptClient
from agentic.runtime.news_grounding import is_usable_news_selection


def _goal_subject_instruction(goal: GoalRequest) -> str:
    """Describe selected subjects as references without turning them into a gate."""

    context = goal.constraints.get("subject_context")
    context = dict(context) if isinstance(context, dict) else {}
    subject_names = [
        str(item.get("name") or "").strip()
        for item in (context.get("subjects") or [])
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    ]
    profile = dict(
        context.get("character_profile")
        or goal.constraints.get("character_profile")
        or {}
    )
    if not profile and subject_names:
        profile = dict(
            next(
                (
                    item.get("profile")
                    for item in (context.get("subjects") or [])
                    if isinstance(item, dict)
                    and str(item.get("name") or "").strip() == subject_names[0]
                ),
                {},
            )
            or {}
        )
    profile_details = "; ".join(
        part
        for part in (
            str(profile.get("role_description") or "").strip(),
            str(profile.get("keywords") or "").strip(),
        )
        if part
    )
    character = str(goal.constraints.get("character") or "").strip()
    return "; ".join(
        part for part in (", ".join(subject_names), character, profile_details) if part
    )


SCENE_RESPONSE_CONTRACT = (
    "The prompt field must contain only the final scene description as one non-empty string. "
    "Describe the subjects, appearance, setting, style, action, visible result, camera and optional sound. "
    "Include no nested JSON, alternatives, markdown, analysis, explanations, source records, "
    "or advice about writing prompts. Apply all writing guidance without copying it into prompt.\n\n"
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
    "action onset, a chronological physical path, camera movement that supports that path, follow-through, and "
    "the changed result or reaction. Keep the main character action clear and energetic unless the user explicitly "
    "asks for stillness or slow pacing. Style controls medium, line, surface, and palette; its motion language may "
    "shape secondary texture, but must not slow, replace, or contradict requested subject action. Do not invent "
    "anatomy, props, locations, or a second action. Keep the opening_keyframe_prompt as one frozen loaded pose "
    "that sets up the same action. Keep meaningful subject or camera motion through most of the segment, reach "
    "the resolved endpoint only in the final tenth, and use only a brief final hold. Do not settle early or leave a "
    "long static tail. Apply this contract without copying it into prompt."
)


def validate_scene_response(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(
            "prompt response must be an object with a scene description string."
        )
    payload["prompt"] = require_h3_prompt(payload.get("prompt"))
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
) -> dict[str, Any]:
    fallback = build_goal_brief(goal, selected_style, idea_variants)
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
    try:
        manager = client.require_manager()
        user_prompt = "\n".join(
            [
                f"Goal: {goal.prompt}",
                f"Media type: {goal.media_type}",
                f"Style: {selected_style}",
                f"Resolved visual style rules: {style_profile_direction(goal)}",
                f"Character: {goal.constraints.get('character', '')}",
                _goal_subject_instruction(goal),
                f"Duration seconds: {goal.duration_seconds}",
                f"News context JSON: {json.dumps(goal.constraints.get('news_context', {}), ensure_ascii=False)}",
                reference_directive,
                reference_motion_directive,
                (
                    "Use attached reference keyframes as optional visual inspiration."
                    if reference_images
                    else "No reference-video keyframes were supplied."
                ),
                "Return JSON with keys: creative_brief, prompt, opening_keyframe_prompt.",
                "Follow the user's requested subject, scene, action, style, composition, and text where specified.",
                (
                    "Write prompt as a motion-only timeline for H3 I2VA; the selected image will be the exact opening frame. "
                    "Do not restate the image's pose, composition, appearance, colors, or existing objects. Begin with the first visible movement, "
                    "then describe one cause-driven action along a clear path, one motivated camera move that supports it, follow-through, "
                    "and a visible changed result or reaction. Begin the action in the first tenth, reach the resolved endpoint only in the final tenth, "
                    "and keep it moving through most of the shot. Do not settle early or leave a long static tail; end on a brief hold. "
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
            output_contract=H3_INITIAL_ACTION_CONTRACT if video_request else SCENE_RESPONSE_CONTRACT,
            validate_response=validate_scene_response,
            schema={
                "type": "object",
                "properties": {
                    "creative_brief": {"type": "string"},
                    "prompt": {"type": "string"},
                    "opening_keyframe_prompt": {"type": "string"},
                },
                "required": ["creative_brief", "prompt"],
                "additionalProperties": False,
            },
            images=reference_images or None,
        )
        fallback.update(
            {
                "creative_brief": str(
                    payload.get("creative_brief") or fallback["creative_brief"]
                ),
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
        return client.mark_payload(fallback)
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
            part
            for part in (prefix, prompt, style, style_profile_direction(goal), suffix)
            if part
        ),
        "negative_prompt": ", ".join(
            part
            for part in (str(negative_prompt or "").strip(), style_profile_avoidance(goal))
            if part
        ),
    }
    manager = client.require_manager()
    user_prompt = "\n".join(
        [
            f"Goal: {goal.prompt}",
            f"Media type: {goal.media_type}",
            f"Base prompt: {prompt}",
            f"Style: {style}",
            f"Resolved visual style rules: {style_profile_direction(goal)}",
            f"Prefix: {prefix}",
            f"Suffix: {suffix}",
            f"Character: {goal.constraints.get('character', '')}",
            _goal_subject_instruction(goal),
            f"News context JSON: {json.dumps(goal.constraints.get('news_context', {}), ensure_ascii=False)}",
            "Return JSON with key: prompt.",
            "Write a natural prompt that follows the user's brief, base prompt, and style.",
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
                "prompt": require_h3_prompt(payload.get("prompt")),
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
            f"Each segment must include {direction_field_names}. The camera field names starting framing, one "
            "motivated move or intentional lock, how it relates to the subject's path, what it reveals, and ending "
            "framing. Keep camera terms distinct: pan/tilt rotates from a fixed point; dolly/truck/arc travels "
            "through space and creates parallax; tracking/follow moves with the subject; zoom changes focal length. "
            "Use one continuous move or a locked shot; name a separate shot if a second camera operation is needed. "
            "Pan and tilt change camera direction, not subject distance or scale. A scale change requires a matching "
            "camera translation, zoom, or physical subject travel, and the ending frame must follow from that move. "
            "Keep the protagonist's expression and the visible result together at the payoff when both matter; "
            "show the result before any close-up reaction. Keep the face and every story-critical object within "
            "frame with clear edge margins during and after the move. If a close-up would crop the reaction or "
            "result, stop wider or cut to a separate shot. Describe the selected style with stable visible rules "
            "for medium and surface, palette and light, and motion or transitions; keep them consistent across shots."
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
        "Keep one continuous shot with one main visible action and at most one motivated camera move. Describe the visible starting state, physical movement, and readable ending state. Keep the H3 scene concise and specific; remove redundant style adjectives and invented details. Do not add characters, props, costume changes, scene changes, text, or actions unsupported by the image or the source draft.",
        "Write an ordered chain of positive physical-action beats, not a list of prohibitions: from the first moment name what moves, its target and path, the next connected movement, and the visible resulting state. Use 2–3 concrete action verbs; do not describe a static pose and expect H3 to infer movement.",
        f"Segment duration: {float(duration_seconds):.2f} seconds. From the first moment, distribute the action and camera progression through the segment, and reach the resolved endpoint only in the final tenth before a brief hold. Do not hold the source pose for most of the clip or jump directly to the ending frame.",
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
