"""Scenes prompt contracts and response validation."""

from __future__ import annotations

import json
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
    validate_story_segments,
)
from agentic.runtime.llm_client import PromptClient


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


def validate_scene_response(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(
            "prompt response must be an object with a scene description string."
        )
    payload["prompt"] = require_h3_prompt(payload.get("prompt"))
    return payload


def validate_segment_response(payload: Any) -> dict[str, Any]:
    payload = validate_scene_response(payload)
    if not isinstance(payload.get("narration"), str):
        raise ValueError("narration must be a string, not a nested object.")
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
    fallback = build_autonomous_scene_prompt(
        character=character,
        style=style,
        media_type=media_type,
        news_context=news_context,
        news_grounding_required=news_grounding_required,
    )
    try:
        manager = client.require_manager()
        user_prompt = "\n".join(
            [
                f"Character: {character}",
                f"Style: {style}",
                f"Media type: {media_type}",
                f"News context JSON: {json.dumps(news_context or {}, ensure_ascii=False)}",
                "Create a generation prompt that follows the supplied creative brief.",
                (
                    f"Video action direction: {VIDEO_MOTION_DIRECTION}"
                    if requires_video_motion_direction(media_type)
                    else ""
                ),
                (
                    "This workflow is news-grounded. Anchor the story in the article's main documented event or impact, then turn one source-supported fact into a visible object or action. Carry that anchor through the scenario and keep each actor, location, cause, and result attached to the source. Frame added cartoon comedy as visual allegory."
                    if news_grounding_required
                    else "Use a few optional news motifs as visual inspiration while keeping reported facts distinct from the imagined scene."
                ),
                "Return the requested JSON directly using the supplied information.",
                "Return JSON with keys: prompt, creative_seed, source.",
            ]
        )
        payload = client.generate_json(
            manager,
            LONG_VIDEO_SYSTEM_PROMPT,
            user_prompt,
            schema_name="autonomous_scene_prompt",
            schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "creative_seed": {"type": "string"},
                    "source": {"type": "string"},
                },
                "required": ["prompt", "creative_seed", "source"],
                "additionalProperties": False,
            },
        )
        return client.mark_payload(
            {
                "prompt": str(payload.get("prompt") or fallback["prompt"]),
                "creative_seed": str(
                    payload.get("creative_seed") or fallback["creative_seed"]
                ),
                "source": str(payload.get("source") or "autonomous_llm"),
                "news_context": dict(news_context or {}),
            }
        )
    except Exception as exc:
        fallback["prompt_mode"] = "template"
        fallback["fallback_reason"] = "manager_unavailable"
        fallback["manager_error"] = f"{type(exc).__name__}: {exc}"
        fallback["llm_backend"] = client.backend_info()
        fallback["news_context"] = dict(news_context or {})
        return fallback


def expand_goal(
    client: PromptClient,
    goal: GoalRequest,
    selected_style: str,
    idea_variants: list[dict[str, Any]],
    reference_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fallback = build_goal_brief(goal, selected_style, idea_variants)
    reference_directive = format_reference_video_directive(
        reference_analysis, max_chars=2200
    )
    reference_images = reference_keyframe_paths(reference_analysis)[:6]
    news_grounding_required = bool(
        goal.constraints.get("news_driven")
        or goal.constraints.get("news_grounding_required")
    )
    news_grounding_contract = (
        f"{news_grounding_anchor_clause(dict(goal.constraints.get('news_context') or {}))} "
        "News source accuracy: Preserve each documented location, actor, and event relationship; do not merge events from different places or invent source-specific people or actions. "
        "For a multi-place story, select one event the article explicitly ties to that place or use an unlocated metaphor for the shared impact. "
        "Any added cartoon comedy must read as allegory, not as a reported event."
        if news_grounding_required
        else "When news context is optional inspiration, do not present invented details as facts from that article."
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
                    "The prompt field describes the full moving action with a concrete cause, protagonist movement, visible result, and motivated camera move. "
                    "The opening_keyframe_prompt describes one frozen instant before action: a loaded pose, a visible goal or obstacle, and a clear travel lane. "
                    f"Video direction: {VIDEO_MOTION_DIRECTION}"
                    if requires_video_motion_direction(goal.media_type)
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
            output_contract=SCENE_RESPONSE_CONTRACT,
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
        "prompt": ", ".join(part for part in (prefix, prompt, style, suffix) if part),
        "negative_prompt": negative_prompt,
    }
    manager = client.require_manager()
    user_prompt = "\n".join(
        [
            f"Goal: {goal.prompt}",
            f"Media type: {goal.media_type}",
            f"Base prompt: {prompt}",
            f"Style: {style}",
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
    *,
    production_profile: str = "",
) -> list[dict[str, Any]]:
    fallback = build_story_segments(
        goal,
        creative_brief,
        segment_count,
        tone,
        production_profile=production_profile,
    )
    production_mode = str(production_profile or "").strip().lower() == "text2longvideo"
    raw_news_context = goal.constraints.get("news_context")
    news_context = dict(raw_news_context) if isinstance(raw_news_context, dict) else {}
    has_news_source = any(
        str(news_context.get(key) or "").strip()
        for key in ("title", "content", "keyword")
    )
    try:
        manager = client.require_manager()
    except Exception:
        if production_mode:
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
    prior_frame: str | None = None,
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
        f"Character: {goal.constraints.get('character', '')}",
        _goal_subject_instruction(goal),
        f"News context JSON: {json.dumps(goal.constraints.get('news_context', {}), ensure_ascii=False)}",
        f"Has prior frame path: {'yes' if prior_frame else 'no'}",
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
            "Return JSON with keys: prompt, narration.",
            "prompt must be one non-empty string containing only the final scene description for H3: "
            "subjects and appearance, setting and visual style, visible action and result, camera movement, "
            "and sound if needed. Incorporate the supplied shot details into that description. "
            "Do not include nested objects, alternative prompts, explanations, analysis, markdown, "
            "source records, or instructions about how to write a prompt. narration must be a string.",
            f"Writing guidance (apply it to the scene; do not copy it into prompt): {VIDEO_MOTION_DIRECTION}",
            "Follow the user's requested scene, action, framing, style, and pacing.",
            "Use any supplied prior frame as visual context while following the requested scene and motion.",
            "Keep claims about supplied news context within the source; use its visual treatment as requested.",
        ]
    )
    try:
        payload = client.generate_json(
            manager,
            LONG_VIDEO_SYSTEM_PROMPT,
            "\n".join(continuity_lines),
            schema_name="segment_prompt",
            output_contract=SCENE_RESPONSE_CONTRACT,
            validate_response=validate_segment_response,
            schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "narration": {"type": "string"},
                },
                "required": ["prompt", "narration"],
                "additionalProperties": False,
            },
        )
        subject_context = goal.constraints.get("subject_context")
        result = {
            "segment_id": segment["segment_id"],
            "prompt": require_h3_prompt(payload.get("prompt")),
            "narration": payload["narration"],
            "negative_prompt": negative_prompt,
            "subject_context": dict(subject_context)
            if isinstance(subject_context, dict)
            else {},
        }
        if prior_frame:
            result["prior_frame_path"] = prior_frame
        return client.mark_payload(result)
    except Exception as exc:
        raise client.generation_error("prepare_segment", exc) from exc


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
