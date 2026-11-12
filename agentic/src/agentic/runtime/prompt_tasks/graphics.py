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
    compile_dynamic_sprite_image_prompt,
    compile_dynamic_sprite_video_prompt,
    dynamic_sprite_frame_map,
    normalize_dynamic_sprite_beats,
    include_role_description,
    selected_role_description,
    style_profile_direction,
    texts_are_alike,
)
from agentic.runtime.llm_client import PromptClient, PromptGenerationError
from agentic.runtime.series_carousel import series_prompt_sets


STICKER_ACTING = (
    "A sticker is a reaction someone sends in a chat, so the acting is bigger than life: the whole body squashes, "
    "stretches, tilts, or flops, the eyes and mouth are exaggerated, and where it helps the feeling read, a simple "
    "drawn emotion mark that fits that one feeling sits beside the character. Cuteness comes first: the character is "
    "drawn at its most adorable, with soft round shapes, big shining eyes, and a feeling that reads at thumbnail size."
)

STICKER_FRAMING = (
    "Cute chat sticker: one isolated subject with a thick white die-cut outline that stays fully inside the frame, "
    "on a plain, flat, pale backdrop with no scenery, ground, crowd, or writing."
)



# Typographic quotes and dashes are ordinary English; anything still outside
# ASCII after this is an emoji or symbol the renderer cannot draw.
_PLAIN_PUNCTUATION = str.maketrans(
    {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"}
)


def _sticker_reaction(payload: Any, target_count: int) -> list[str]:
    """Check the writer's reactions and join each into one drawable sentence."""
    reactions = payload.get("reactions") if isinstance(payload, dict) else None
    if not isinstance(reactions, list) or len(reactions) != target_count:
        raise ValueError(f"sticker writer must return exactly {target_count} reactions.")
    feelings: set[str] = set()
    framings: list[str] = []
    sentences: list[str] = []
    for index, reaction in enumerate(reactions, start=1):
        fields = {
            key: " ".join(
                str((reaction or {}).get(key) or "").translate(_PLAIN_PUNCTUATION).split()
            ).rstrip(".")
            for key in ("framing", "feeling", "body", "eyes", "mark")
        }
        if not all(fields.values()):
            raise ValueError(f"sticker reaction {index} needs framing, feeling, body, eyes, and mark.")
        if not all(character.isascii() for character in "".join(fields.values())):
            raise ValueError(f"sticker reaction {index} must be plain English words without symbols.")
        if fields["feeling"].casefold() in feelings:
            raise ValueError("sticker reactions must each carry a different feeling.")
        if any(texts_are_alike(fields["framing"], earlier) for earlier in framings):
            raise ValueError("sticker reactions must each be drawn at a different distance.")
        feelings.add(fields["feeling"].casefold())
        framings.append(fields["framing"])
        sentences.append(
            f"{fields['framing'][:1].upper()}{fields['framing'][1:]}. {fields['body']}. "
            f"For this feeling the eyes are drawn as {fields['eyes']}. "
            f"{fields['mark'][:1].upper()}{fields['mark'][1:]} floats beside the head"
        )
    return sentences


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
            f"Character features: {selected_role_description(goal) or character}",
            f"Theme/Context: {prompt}",
            f"Write exactly {target_count} sticker reactions.",
            "Read the theme as one small happening the character goes through. The reactions are what a person would "
            "send in a chat while living that happening, in time order from before it to after it. The set covers the "
            "whole experience, including its uncomfortable or funny low point, so the feelings differ in kind and are "
            "not several shades of happy.",
            STICKER_ACTING,
            "Fill five fields for each reaction, in plain English words with no emoji or symbol characters. "
            "framing: in one short clause, how much of the character the picture shows and how large it sits in "
            "the frame, from the head and upper body filling most of the picture to the whole character full-length "
            "at about half the frame height; the pack runs across that range, so no two reactions share a distance, "
            "and the character and its white outline always stay completely inside the picture. "
            "feeling: the feeling in one or two words; no two reactions share it. "
            "body: starting with the word Body, what the whole body is doing, as a painter would draw it, in ten to "
            "eighteen words; the body's outline differs so much between reactions that their silhouettes filled in "
            "solid black could be told apart; it may hold one small prop from the theme. "
            "eyes: the shape the eyes take for this feeling, which replaces their normal shape, described as lines "
            "and shapes a painter can copy and never by repeating how the character's eyes normally look. "
            "mark: one small drawn object or shape that shows this feeling, named as a thing.",
            'Return a JSON object with key "reactions" only.',
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
                    "reactions": {
                        "type": "array",
                        "minItems": target_count,
                        "maxItems": target_count,
                        "items": {
                            "type": "object",
                            "properties": {
                                "framing": {"type": "string"},
                                "feeling": {"type": "string"},
                                "body": {"type": "string"},
                                "eyes": {"type": "string"},
                                "mark": {"type": "string"},
                            },
                            "required": ["framing", "feeling", "body", "eyes", "mark"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["reactions"],
                "additionalProperties": False,
            },
            validate_response=lambda response: {
                **response,
                "expressions": _sticker_reaction(response, target_count),
            },
        )
        return list(payload["expressions"])
    except Exception as exc:
        # A canned mood list makes every pack look the same, so a failed
        # writer stops the run instead.
        raise client.generation_error("sticker_expressions", exc) from exc


def build_sticker_prompt_set(goal: GoalRequest, expressions: list[str]) -> dict[str, Any]:
    """One short prompt per expression: the pose leads, then identity, then the sticker look.

    The renderer follows the start of a short prompt and ignores a pose buried
    in a long one, so nothing else is appended to these at render time.
    """
    identity = include_role_description("", goal).rstrip(".")
    style = style_profile_direction(goal, include_motion=False)
    prompt_sets = [
        {
            "label": f"sticker_{index:02d}",
            "expression": expression,
            "prompt": " ".join(
                part
                for part in (
                    f"{expression.rstrip('.')}.",
                    STICKER_FRAMING,
                    f"The character is {identity}." if identity else "",
                    f"{style.rstrip('.')}." if style else "",
                )
                if part
            ),
            "self_contained": True,
        }
        for index, expression in enumerate(expressions, start=1)
    ]
    return {"prompt_sets": prompt_sets, "prompt_count": len(prompt_sets)}


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
            f"Resolved visual style rules: {style_profile_direction(goal)}",
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


def _sprite_theme(goal: GoalRequest) -> str:
    """What the sprite acts out: the happening this run took from its source.

    The scene brief also carries the run's rendering style, which a sprite
    must not inherit, so the theme is the source-to-scene mapping, or the
    headline when the run has none.
    """
    anchors = goal.constraints.get("news_source_anchors")
    mappings = [
        str(anchor.get("visual_mapping") or "").strip()
        for anchor in (anchors if isinstance(anchors, list) else [])
        if isinstance(anchor, dict)
    ]
    news = goal.constraints.get("news_context")
    headline = (
        str(news.get("title") or news.get("keyword") or "").strip()
        if isinstance(news, dict)
        else ""
    )
    theme = "; ".join(part for part in (*mappings, headline) if part)
    return theme or str(goal.prompt or "").strip()


def build_dynamic_sprite_motion_plan(
    client: PromptClient, goal: GoalRequest
) -> dict[str, Any]:
    """Build direct sprite prompts with only renderer-required metadata."""
    fallback = build_dynamic_sprite_motion_fallback(goal)
    reference_context = build_game_sprite_reference_context(goal)
    target_duration_seconds = min(362 / 24, float(goal.duration_seconds))
    user_prompt = "\n".join(
        part
        for part in (
            f"Subject: {goal.constraints.get('character') or ''}",
            f"Theme (the character acts out its one action): {_sprite_theme(goal)}",
            f"Mandatory game-sprite visual contract: {reference_context['visual_contract']}",
            "Reduce the theme to the one everyday action it is about and at most one prop; drop the rest of the scene. "
            "Stage that action so its phases (getting ready, starting, doing it, what happens, settling back) each change "
            "the silhouette.",
            "Return JSON with image_prompt and video_prompt, plus exactly five ordered beats for the same phases.",
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
                    "beats": {
                        "type": "array",
                        "minItems": 5,
                        "maxItems": 5,
                        "items": {
                            "type": "object",
                            "properties": {
                                "purpose": {"type": "string"},
                                "action": {"type": "string"},
                                "body_change": {"type": "string"},
                                "spatial_change": {"type": "string"},
                                "cause": {"type": "string"},
                                "transition": {"type": "string"},
                            },
                            "required": ["purpose", "action"],
                            "additionalProperties": True,
                        },
                    },
                    "animation_kind": {"type": "string"},
                    "fps": {"type": "number"},
                    "negative_prompt": {"type": "string"},
                    "motion_name": {"type": "string"},
                },
                "required": ["image_prompt", "video_prompt"],
                "additionalProperties": True,
            },
        )
        if not isinstance(payload, dict):
            raise ValueError("dynamic sprite prompt response must be an object")
        beats = normalize_dynamic_sprite_beats(
            payload.get("beats"), list(fallback.get("beats") or [])
        )
        phases = ("anticipation", "charge", "release", "impact_or_peak", "recovery")
        if len(beats) != len(phases):
            beats = list(fallback.get("beats") or [])
        else:
            for beat, phase in zip(beats, phases):
                beat["purpose"] = phase
        creative_video_prompt = str(
            payload.get("video_prompt") or fallback["creative_video_prompt"]
        ).strip()
        creative_image_prompt = str(payload.get("image_prompt") or "").strip()
        image_prompt = (
            compile_dynamic_sprite_image_prompt(
                creative_image_prompt, reference_context["visual_contract"]
            )
            if creative_image_prompt
            else str(fallback["image_prompt"])
        )
        animation_kind = str(
            goal.constraints.get("sprite_animation_kind") or "one_shot"
        ).strip()
        if animation_kind not in {"periodic", "one_shot"}:
            animation_kind = "one_shot"
        normalized = {
            **fallback,
            "motion_name": str(
                payload.get("motion_name") or fallback["motion_name"]
            ).strip(),
            "motion_plan_mode": "direct_prompt",
            "image_prompt": image_prompt,
            "creative_video_prompt": creative_video_prompt,
            "beats": beats,
            "video_prompt": compile_dynamic_sprite_video_prompt(
                creative_video_prompt, reference_context["visual_contract"]
            ),
            "negative_prompt": str(goal.constraints.get("negative_prompt") or ""),
            "animation_kind": animation_kind,
            "fps": max(4.0, min(24.0, float(payload.get("fps", fallback["fps"])))),
            "video_duration_seconds": target_duration_seconds,
            "action_reference_pack": reference_context["pack_version"],
            "frame_map": dynamic_sprite_frame_map(beats),
        }
        return client.mark_payload(normalized)
    except Exception as exc:
        fallback["action_reference_pack"] = reference_context["pack_version"]
        fallback["video_duration_seconds"] = target_duration_seconds
        fallback["video_prompt"] = compile_dynamic_sprite_video_prompt(
            str(fallback.get("creative_video_prompt") or goal.prompt),
            reference_context["visual_contract"],
        )
        return client.template_fallback(
            fallback, exc, fallback_reason="json_parse_failed"
        )


SERIES_SCENE_CONTRACT = (
    "Write one fixed scene as a template for a series of pictures that differ only in who is in them. The template "
    "opens with one sentence naming the selected medium with two or three observable surface cues from the style "
    "rules. It then fixes the camera, the one prop and where it sits, and where the character stands or lies and how "
    "large it is drawn, so the same words give the same composition every time. The one place where the character "
    "goes is written as {character}, exactly once, as a bare noun phrase with no possessive; the scene never writes a cast "
    "member's name and never describes the character's body, species, or colour, because the look supplies that. The picture holds "
    "that one character and the scene only, with no writing, numbers, labels, or signs. For every cast member give a "
    "look of at most twelve words that starts with 'the' and names the shape, the colour, and the one feature that "
    "tells it apart; the look is swapped into the slot of the fixed composition, so it describes the body only and "
    "never an action, a prop, or a setting."
)


def build_series_template(
    client: PromptClient, goal: GoalRequest, members: list[dict[str, Any]]
) -> dict[str, Any]:
    """Write one scene template and a short look per member; no canned scene exists."""
    if client.mode == "template":
        raise PromptGenerationError(
            "Series carousel requires an LLM; canned scenes are not supported"
        )
    cast = [
        {
            "name": str(member["name"]),
            "description": str(member.get("role_description") or "")[:300],
            "keywords": str(member.get("keywords") or ""),
        }
        for member in members
    ]
    user_prompt = "\n".join(
        [
            f"Scene idea: {goal.prompt}",
            f"Style: {goal.style}",
            f"Resolved visual style rules: {style_profile_direction(goal)}",
            f"Cast JSON: {json.dumps(cast, ensure_ascii=False)}",
            SERIES_SCENE_CONTRACT,
            "Return a JSON object with keys template and looks. looks has one item per cast member with keys name "
            "(exactly as given) and look.",
        ]
    )
    manager = client.require_manager()

    def validate_response(payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("series writer must return an object.")
        series_prompt_sets(payload.get("template"), payload.get("looks") or [], members)
        return payload

    payload = client.generate_json(
        manager,
        LONG_VIDEO_SYSTEM_PROMPT,
        user_prompt,
        schema_name="series_template",
        schema={
            "type": "object",
            "properties": {
                "template": {"type": "string"},
                "looks": {
                    "type": "array",
                    "minItems": len(members),
                    "maxItems": len(members),
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "look": {"type": "string"},
                        },
                        "required": ["name", "look"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["template", "looks"],
            "additionalProperties": False,
        },
        validate_response=validate_response,
    )
    return client.mark_payload(
        {
            "template": str(payload["template"]).strip(),
            "prompt_sets": series_prompt_sets(payload["template"], payload["looks"], members),
        }
    )
