"""Graphics prompt contracts and response validation."""

from __future__ import annotations

import json
from typing import Any
from agentic.runtime.contracts import GoalRequest
from agentic.runtime.prompting import (
    DYNAMIC_SPRITE_SYSTEM_PROMPT,
    LONG_VIDEO_SYSTEM_PROMPT,
    STICKER_SYSTEM_PROMPT,
    build_animated_sticker_motion_prompt,
    build_dynamic_sprite_motion_fallback,
    build_game_sprite_reference_context,
    compile_dynamic_sprite_video_prompt,
    dynamic_sprite_frame_map,
    dynamic_sprite_source_contract,
    normalize_dynamic_sprite_beats,
    resolve_dynamic_sprite_background,
    include_role_description,
    build_sticker_prompt,
    selected_role_description,
)
from agentic.runtime.llm_client import PromptClient


def sticker_expressions(
    client: PromptClient,
    goal: GoalRequest,
    prompt: str,
    character: str,
    expression_count: int,
) -> list[str]:
    target_count = max(1, int(expression_count))

    user_prompt = "\n".join(
        [
            f"Character: {character}",
            f"Theme/Context: {prompt}",
            f"Generate exactly {target_count} unique sticker expressions.",
            'Return JSON object with key "expressions" only.',
        ]
    )
    try:
        manager = client.require_manager()
        payload = client.generate_json(
            manager,
            STICKER_SYSTEM_PROMPT,
            user_prompt,
            schema_name="sticker_expressions",
            schema={
                "type": "object",
                "properties": {
                    "expressions": {
                        "type": "array",
                        "minItems": target_count,
                        "maxItems": target_count,
                        "items": {"type": "string"},
                    }
                },
                "required": ["expressions"],
                "additionalProperties": False,
            },
        )
        expressions = payload.get("expressions")
        if isinstance(expressions, list) and expressions:
            normalized = [
                str(item).strip() for item in expressions if str(item).strip()
            ]
            if len(normalized) == target_count:
                return normalized
    except Exception:
        pass
    fallback = fallback_sticker_expressions(prompt, target_count)
    return fallback


def build_sticker_prompt_set(
    client: PromptClient,
    goal: GoalRequest,
    expressions: list[str],
    character: str,
    prompt_prefix: str,
    style: str,
) -> dict[str, Any]:
    role_description = selected_role_description(goal)

    fallback_prompt_sets = [
        {
            "label": f"sticker_{index + 1:02d}",
            "expression": expression,
            "prompt": include_role_description(
                build_sticker_prompt(character, expression, prompt_prefix, style),
                goal,
            ),
        }
        for index, expression in enumerate(expressions)
    ]
    user_prompt = "\n".join(
        [
            f"Goal: {goal.prompt}",
            f"Character: {character}",
            *(
                [
                    f"Character feature description to preserve in every prompt: {role_description}"
                ]
                if role_description
                else []
            ),
            f"Prompt prefix: {prompt_prefix}",
            f"Style: {style}",
            f"Expressions JSON: {json.dumps(expressions, ensure_ascii=False)}",
            "Return a JSON object with key prompt_sets.",
            "prompt_sets must be an array where each item has keys: label, expression, prompt.",
            "Follow the requested expressions and style.",
        ]
    )
    try:
        manager = client.require_manager()
        payload = client.generate_json(
            manager,
            STICKER_SYSTEM_PROMPT,
            user_prompt,
            schema_name="sticker_prompt_set",
            schema={
                "type": "object",
                "properties": {
                    "prompt_sets": {
                        "type": "array",
                        "minItems": len(fallback_prompt_sets),
                        "maxItems": len(fallback_prompt_sets),
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "expression": {"type": "string"},
                                "prompt": {"type": "string"},
                            },
                            "required": ["label", "expression", "prompt"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["prompt_sets"],
                "additionalProperties": False,
            },
        )
        prompt_items = payload.get("prompt_sets")
        if isinstance(prompt_items, list) and prompt_items:
            prompt_sets: list[dict[str, Any]] = []
            for index, item in enumerate(prompt_items[: len(fallback_prompt_sets)]):
                base_item = fallback_prompt_sets[index]
                prompt_sets.append(
                    {
                        "label": str(item.get("label") or base_item["label"]),
                        "expression": str(
                            item.get("expression") or base_item["expression"]
                        ),
                        "prompt": include_role_description(
                            item.get("prompt") or base_item["prompt"],
                            goal,
                        ),
                    }
                )
            while len(prompt_sets) < len(fallback_prompt_sets):
                prompt_sets.append(fallback_prompt_sets[len(prompt_sets)])
            return client.mark_payload(
                {
                    "prompt_sets": prompt_sets,
                    "prompt_count": len(prompt_sets),
                }
            )
    except Exception as exc:
        return client.template_fallback(
            {
                "prompt_sets": fallback_prompt_sets,
                "prompt_count": len(fallback_prompt_sets),
            },
            exc,
            fallback_reason="json_parse_failed",
        )
    return client.template_fallback(
        {
            "prompt_sets": fallback_prompt_sets,
            "prompt_count": len(fallback_prompt_sets),
        },
        fallback_reason="json_parse_failed",
    )


def build_sticker_motion_prompt(
    client: PromptClient,
    goal: GoalRequest,
    base_prompt: str,
    character: str,
    selected_expression: str = "",
) -> dict[str, Any]:
    role_description = selected_role_description(goal)
    fallback = {
        "prompt": include_role_description(
            build_animated_sticker_motion_prompt(goal), goal
        ),
    }
    manager = client.require_manager()

    user_prompt = "\n".join(
        [
            f"Goal: {goal.prompt}",
            f"Character: {character}",
            *(
                [f"Character feature description to preserve: {role_description}"]
                if role_description
                else []
            ),
            f"Base sticker prompt: {base_prompt}",
            f"Primary expression: {selected_expression}",
            f"Style: {goal.style}",
            "Return JSON with key: prompt.",
            "Create an animated sticker motion prompt from the user's brief.",
        ]
    )
    try:
        payload = client.generate_json(
            manager,
            STICKER_SYSTEM_PROMPT,
            user_prompt,
            schema_name="animated_sticker_motion",
            schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                },
                "required": ["prompt"],
                "additionalProperties": False,
            },
        )
        fallback["prompt"] = include_role_description(
            payload.get("prompt") or fallback["prompt"], goal
        )
        return client.mark_payload(fallback)
    except Exception as exc:
        raise client.generation_error("build_sticker_motion_prompt", exc) from exc


def build_motion_graphics_plan(
    client: PromptClient,
    goal: GoalRequest,
    *,
    duration_seconds: float,
    fps: float,
) -> dict[str, Any]:
    """Produce a validated, declarative overlay plan without executable code."""
    from agentic.runtime.motion_graphics import (
        MotionGraphicsPlan,
        motion_graphics_plan_json_schema,
    )

    plan_schema = motion_graphics_plan_json_schema()
    user_prompt = "\n".join(
        [
            f"Creative request: {goal.prompt}",
            f"Character: {goal.constraints.get('character') or ''}",
            f"Style: {goal.style}",
            f"Source duration: {duration_seconds:.6f} seconds",
            f"Source frame rate: {fps:.6f} fps",
            "Create a compact plan for program-rendered motion graphics layered over the finished character video.",
            "Keep the generated character, face, silhouette, hands, and action unobscured; do not draw or redraw the character.",
            "Use only captions, impact bursts, sparkles, and motion lines. Keep every cue inside the supplied normalized safe area.",
            "Each caption has at most 32 characters. Use precise start/end times that follow the action in the clip.",
            "Return JSON with duration_seconds and cues only. Every cue must contain all schema fields.",
        ]
    )
    try:
        payload = client.generate_json(
            client.require_manager(),
            LONG_VIDEO_SYSTEM_PROMPT,
            user_prompt,
            schema_name="motion_graphics_plan",
            schema=plan_schema,
        )
        normalized = MotionGraphicsPlan.from_dict(payload)
        if (
            abs(normalized.duration_seconds - float(duration_seconds))
            > (1 / float(fps)) + 1e-3
        ):
            raise ValueError(
                "motion graphics plan duration does not match the source video"
            )
        return normalized.to_dict()
    except Exception as exc:
        raise client.generation_error("build_motion_graphics_plan", exc) from exc


def review_motion_graphics_plan(
    client: PromptClient,
    goal: GoalRequest,
    *,
    plan: dict[str, Any],
    contact_sheet_path: str,
    round_number: int,
) -> dict[str, Any]:
    """Review sampled rendered frames and return a bounded declarative revision."""
    from agentic.runtime.motion_graphics import (
        MotionGraphicsPlan,
        motion_graphics_plan_json_schema,
    )

    plan_schema = motion_graphics_plan_json_schema()
    try:
        current_plan = MotionGraphicsPlan.from_dict(plan).to_dict()
        payload = client.generate_json(
            client.require_manager(),
            "You review sampled frames of short character videos with programmatic graphic overlays. Protect character readability and action causality.",
            "\n".join(
                [
                    f"Creative request: {goal.prompt}",
                    f"Character: {goal.constraints.get('character') or ''}",
                    f"Current review round: {int(round_number)} of 2",
                    f"Current declarative plan: {json.dumps(current_plan, ensure_ascii=False)}",
                    "Inspect the attached contact sheet. Decide whether the timing, safe placement, restraint, and relationship to the visible action are ready.",
                    "Do not request changes to the generated character itself. If changes are needed, return a revised declarative plan in the same schema.",
                    "Keep the existing plan when it already works. Return satisfied, critique, and plan.",
                ]
            ),
            schema_name="motion_graphics_visual_review",
            schema={
                "type": "object",
                "properties": {
                    "satisfied": {"type": "boolean"},
                    "critique": {"type": "string", "maxLength": 1200},
                    "plan": plan_schema,
                },
                "required": ["satisfied", "critique", "plan"],
                "additionalProperties": False,
            },
            model="vision",
            images=[contact_sheet_path],
        )
        if not isinstance(payload, dict) or not isinstance(
            payload.get("satisfied"), bool
        ):
            raise ValueError(
                "motion graphics visual review returned an invalid response"
            )
        normalized_plan = MotionGraphicsPlan.from_dict(payload.get("plan"))
        return {
            "satisfied": payload["satisfied"],
            "critique": str(payload.get("critique") or ""),
            "plan": normalized_plan.to_dict(),
        }
    except Exception as exc:
        raise client.generation_error("review_motion_graphics_plan", exc) from exc


def build_dynamic_sprite_motion_plan(
    client: PromptClient, goal: GoalRequest
) -> dict[str, Any]:
    """Build direct sprite prompts with only renderer-required metadata."""
    fallback = build_dynamic_sprite_motion_fallback(goal)
    reference_context = build_game_sprite_reference_context(goal)
    target_duration_seconds = max(
        4.0,
        min(
            8.0,
            float(
                goal.duration_seconds or fallback.get("video_duration_seconds") or 8.0
            ),
        ),
    )
    requested_background = (
        str(goal.constraints.get("sprite_chroma_color") or "").strip().casefold()
    )
    background_color = resolve_dynamic_sprite_background(
        requested_background
        if requested_background not in {"", "random", "auto"}
        else fallback["background_color"]
    )
    user_prompt = "\n".join(
        part
        for part in (
            f"Creative request: {goal.prompt}",
            f"Subject reference: {goal.constraints.get('character') or ''}",
            f"Style: {goal.style}",
            f"Reference notes (optional): {reference_context['prompt_text'] if reference_context['references'] else ''}",
            f"The render uses a {target_duration_seconds:g}-second clip sampled into a 4x4 frame atlas.",
            f"Use a flat, uniform chroma-key background {background_color} for sprite extraction.",
            "Return JSON with image_prompt and video_prompt. Other descriptive fields are optional.",
        )
        if part
    )
    try:
        manager = client.require_manager()
        payload = client.generate_json(
            manager,
            DYNAMIC_SPRITE_SYSTEM_PROMPT,
            user_prompt,
            schema_name="dynamic_game_sprite_prompts",
            schema={
                "type": "object",
                "properties": {
                    "image_prompt": {"type": "string"},
                    "video_prompt": {"type": "string"},
                    "beats": {"type": "array", "items": {"type": "object"}},
                    "animation_kind": {"type": "string"},
                    "fps": {"type": "number"},
                    "background_color": {"type": "string"},
                    "negative_prompt": {"type": "string"},
                    "motion_name": {"type": "string"},
                },
                "required": ["image_prompt", "video_prompt"],
                "additionalProperties": True,
            },
        )
        if not isinstance(payload, dict):
            raise ValueError("dynamic sprite prompt response must be an object")
        background_color = resolve_dynamic_sprite_background(
            requested_background
            if requested_background not in {"", "random", "auto"}
            else payload.get("background_color") or background_color
        )
        beats = normalize_dynamic_sprite_beats(
            payload.get("beats"), list(fallback.get("beats") or [])
        )
        creative_video_prompt = str(
            payload.get("video_prompt") or fallback["creative_video_prompt"]
        ).strip()
        animation_kind = str(
            goal.constraints.get("sprite_animation_kind")
            or payload.get("animation_kind")
            or fallback.get("animation_kind")
        ).strip()
        if animation_kind not in {"periodic", "one_shot"}:
            animation_kind = "one_shot"
        normalized = {
            **fallback,
            "motion_name": str(
                payload.get("motion_name") or fallback["motion_name"]
            ).strip(),
            "motion_plan_mode": "direct_prompt",
            "image_prompt": " ".join(
                part
                for part in (
                    str(
                        payload.get("image_prompt") or fallback["image_prompt"]
                    ).strip(),
                    dynamic_sprite_source_contract(background_color),
                )
                if part
            ),
            "creative_video_prompt": creative_video_prompt,
            "beats": beats,
            "video_prompt": compile_dynamic_sprite_video_prompt(
                creative_video_prompt, background_color
            ),
            "negative_prompt": str(goal.constraints.get("negative_prompt") or ""),
            "animation_kind": animation_kind,
            "fps": max(4.0, min(24.0, float(payload.get("fps", fallback["fps"])))),
            "video_duration_seconds": target_duration_seconds,
            "chroma_color": background_color,
            "background_color": background_color,
            "action_reference_pack": reference_context["pack_version"],
            "action_references": reference_context["references"],
            "frame_map": dynamic_sprite_frame_map(beats),
        }
        return client.mark_payload(normalized)
    except Exception as exc:
        fallback["action_reference_pack"] = reference_context["pack_version"]
        fallback["action_references"] = reference_context["references"]
        fallback["video_duration_seconds"] = target_duration_seconds
        fallback["video_prompt"] = compile_dynamic_sprite_video_prompt(
            str(fallback.get("creative_video_prompt") or goal.prompt),
            str(fallback["background_color"]),
        )
        return client.template_fallback(
            fallback, exc, fallback_reason="json_parse_failed"
        )


def build_carousel_prompt_set(
    client: PromptClient,
    goal: GoalRequest,
    segments: list[dict[str, Any]],
    style: str,
) -> dict[str, Any]:
    fallback_prompt_sets = [
        {
            "label": f"slide_{index + 1:02d}",
            "prompt": ", ".join(
                part
                for part in (
                    str(segment.get("visual", "")),
                    style,
                )
                if part
            ),
            "narration": str(segment.get("narration", "")),
        }
        for index, segment in enumerate(segments)
    ]
    fallback = {
        "prompt_sets": fallback_prompt_sets,
        "prompt_count": len(fallback_prompt_sets),
    }
    compact_segments = [
        {
            "segment_id": str(segment.get("segment_id", f"segment-{index + 1}")),
            "visual": str(segment.get("visual", "")),
            "narration": str(segment.get("narration", "")),
        }
        for index, segment in enumerate(segments)
    ]
    user_prompt = "\n".join(
        [
            f"Goal: {goal.prompt}",
            f"Style: {style}",
            f"Segment count: {len(compact_segments)}",
            f"Segments JSON: {json.dumps(compact_segments, ensure_ascii=False)}",
            "Return a JSON object with key prompt_sets.",
            "prompt_sets must be an array where each item has keys: label, prompt, narration.",
            "Follow the requested subject, sequence, and style.",
        ]
    )
    try:
        manager = client.require_manager()
        payload = client.generate_json(
            manager,
            LONG_VIDEO_SYSTEM_PROMPT,
            user_prompt,
            schema_name="carousel_prompt_set",
            schema={
                "type": "object",
                "properties": {
                    "prompt_sets": {
                        "type": "array",
                        "minItems": len(fallback_prompt_sets),
                        "maxItems": len(fallback_prompt_sets),
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "prompt": {"type": "string"},
                                "narration": {"type": "string"},
                            },
                            "required": ["label", "prompt", "narration"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["prompt_sets"],
                "additionalProperties": False,
            },
        )
        prompt_items = payload.get("prompt_sets")
        if isinstance(prompt_items, list) and prompt_items:
            prompt_sets: list[dict[str, Any]] = []
            for index, item in enumerate(prompt_items[: len(fallback_prompt_sets)]):
                base_item = fallback_prompt_sets[index]
                prompt_sets.append(
                    {
                        "label": str(item.get("label") or base_item["label"]),
                        "prompt": str(item.get("prompt") or base_item["prompt"]),
                        "narration": str(
                            item.get("narration") or base_item["narration"]
                        ),
                    }
                )
            while len(prompt_sets) < len(fallback_prompt_sets):
                prompt_sets.append(fallback_prompt_sets[len(prompt_sets)])
            return client.mark_payload(
                {
                    "prompt_sets": prompt_sets,
                    "prompt_count": len(prompt_sets),
                }
            )
    except Exception as exc:
        return client.template_fallback(fallback, exc)
    return client.template_fallback(fallback)


def fallback_sticker_expressions(prompt: str, target_count: int) -> list[str]:
    prompt_tail = prompt.split(":", 1)[-1]
    candidates = [item.strip() for item in prompt_tail.split(",") if item.strip()]
    fallback = candidates[:target_count]
    default_pool = [
        "happy",
        "angry",
        "crying",
        "sleepy",
        "surprised",
        "celebrating",
        "confused",
        "love struck",
        "thinking",
        "cheering",
    ]
    for item in default_pool:
        if len(fallback) >= target_count:
            break
        if item not in fallback:
            fallback.append(item)
    return fallback[:target_count]
