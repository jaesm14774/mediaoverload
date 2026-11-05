from __future__ import annotations

import random
import re
import secrets
from typing import Any

from agentic.runtime.contracts import GoalRequest
from agentic.video_directing import (
    apply_video_motion_direction,
    apply_video_opening_frame_direction,
)


_ISOLATED_STICKER_TYPES = frozenset({"sticker_pack", "animated_sticker"})


def selected_role_description(goal: GoalRequest) -> str:
    subject_context = goal.constraints.get("subject_context")
    if isinstance(subject_context, dict):
        identities: list[str] = []
        subjects = subject_context.get("subjects")
        if isinstance(subjects, list):
            for subject in subjects:
                if not isinstance(subject, dict):
                    continue
                name = " ".join(str(subject.get("name") or "").split())
                profile = subject.get("profile")
                profile = profile if isinstance(profile, dict) else {}
                description = " ".join(str(profile.get("role_description") or "").split())
                if name or description:
                    identities.append(f"{name}: {description}" if name and description else name or description)
        if identities:
            return "; ".join(identities)

    profile = goal.constraints.get("character_profile")
    profile = profile if isinstance(profile, dict) else {}
    description = " ".join(str(profile.get("role_description") or "").split())
    name = " ".join(str(goal.constraints.get("character") or "").split())
    if name and description:
        return f"{name}: {description}"
    return name or description


def include_role_description(prompt: object, goal: GoalRequest) -> str:
    prompt_text = str(prompt or "").strip()
    role_description = selected_role_description(goal)
    visual_identity = "; ".join(
        _visual_identity_phrase(identity)
        for identity in role_description.split("; ")
        if identity.strip()
    )
    if not visual_identity or visual_identity.casefold() in prompt_text.casefold():
        return prompt_text
    return "; ".join(part for part in (visual_identity, prompt_text) if part)


def _visual_identity_phrase(identity: str) -> str:
    name, separator, description = identity.partition(": ")
    if not separator:
        return identity.strip()
    description = re.sub(
        rf"^{re.escape(name)}\s+(?:is|are)\s+",
        "",
        description.strip(),
        count=1,
        flags=re.IGNORECASE,
    )
    return f"{name}, {description}" if description else name


IMAGE_PROMPT_CONTRACT = (
    "For an image request, describe one finished frame and one visual thesis: a decisive visible state plus the "
    "relationship or result that makes it matter. Treat the requested and selected cast, action, setting, and any "
    "explicit framing as fixed. If two subjects are marked as required, show both in the same frame and make their "
    "physical interaction visible; style must not add characters, props, or story. Then specify only what helps this "
    "frame: a purposeful viewpoint and relative placement or scale; useful depth layers or negative space; a light "
    "source and direction with a visible effect; one primary medium with two or three observable surface or mark cues. "
    "When a resolved style treatment is supplied separately, do not restate its descriptor or influence list. "
    "Use style composition only when the request leaves framing open. Keep secondary details functional. Avoid generic "
    "quality slogans, contradictory style stacks, and pseudo-text. Write one concise natural English scene prompt."
)

IMAGE_PROMPT_QUALITY_CLAUSE = (
    "Preserve the requested cast, action, and framing. Keep the main relationship readable through viewpoint and "
    "relative scale; make light direction and material cues physically coherent. Style must not add characters or props."
)


def include_image_prompt_quality_clause(
    prompt: object,
    goal: GoalRequest,
    *,
    image_generation: bool = False,
) -> str:
    prompt_text = str(prompt or "").strip()
    if goal.media_type in _ISOLATED_STICKER_TYPES or goal.media_type == "game_sprite":
        return prompt_text
    if goal.media_type != "image" and not image_generation:
        return prompt_text
    if IMAGE_PROMPT_QUALITY_CLAUSE.casefold() in prompt_text.casefold():
        return prompt_text
    return "\n".join(part for part in (prompt_text, IMAGE_PROMPT_QUALITY_CLAUSE) if part)


LONG_VIDEO_SYSTEM_PROMPT = """
You write image and video prompts that follow the user's brief and selected style.
Derive the setting, objects, actors, and actions from this run's brief and source material.
Visual style controls medium, surface, color, light, composition, and motion treatment;
it never changes the requested cast, action, setting, or explicit framing, and it never adds a story,
prop, task, or character anatomy. Use style composition only when the brief leaves framing open.
Preserve the selected characters' canonical identities and essential source context while applying that treatment.
Use source material as context and keep every reported actor, location, cause, and
outcome tied to its source. When the user requests a causal story, follow that direction.
For video, make visible physical action and purposeful camera movement the default.
Use the supplied image as the opening instant, then show how the subject and important
props move, what changes, and how the action resolves. For each shot, name its opening
framing, one motivated camera move or intentional lock, what that choice reveals, and the
resulting composition. Character-led scenes show anticipation, action with follow-through,
and a brief reaction; one or two fitting secondary motions can reinforce the action. Vary
the pace so both movement and reaction are easy to read. Describe the selected style through
its medium and surface, palette and light, optional composition guidance, and motion or transition behavior;
keep those visible rules consistent across shots without overriding requested framing. Keep the action consistent with the request
and facts. Describe the opening keyframe as one frozen instant. Describe temporal action and
camera movement in the video prompt. Preserve deliberately still or contemplative briefs.
For image-only requests, follow the requested composition, action, pacing, text, and
visual treatment. Keep time-based progression in video-specific prompts.
""".strip()


ENGLISH_GENERATION_RESPONSE_CONTRACT = """
Language contract:
- Write every creative field in natural, idiomatic English: storyboards, scene descriptions, image prompts, video prompts, actions, camera directions, audio directions, and visual QA prose.
- Do not write Traditional Chinese, Simplified Chinese, Japanese, or mixed-language prose in those creative fields.
- Preserve a source-language proper noun or exact source text only when the schema explicitly requires it as metadata; never copy it into a creative prompt field.
- Translate the meaning into fluent English rather than translating word for word.
Response in English only.
""".strip()


STICKER_SYSTEM_PROMPT = "Follow the user's sticker brief and selected character identity. Derive scene content from the brief; the selected style controls visual treatment only."


DEFAULT_GAME_SPRITE_VISUAL_CONTRACT = (
    "Crisp 2D pixel-art game asset: hard pixel clusters, a limited readable palette, clean silhouette edges, "
    "deliberate pose exaggeration, and nearest-neighbor-friendly shapes; never painterly, photoreal, tabletop, "
    "soft 3D, or cinematic concept art."
)


DYNAMIC_SPRITE_SYSTEM_PROMPT = (
    "Turn the user's character, theme, and visual style into one playable game-skill effect animation. "
    "A quiet, static, or cinematic source brief is visual inspiration, not permission to leave the "
    "character idle: reinterpret it as the setting and mood for a clear ability action. Choreograph "
    "five readable phases in order: anticipation, charge, release, impact or effect peak, and recovery. "
    "The body must visibly brace, lean, recoil, dash, cast, or follow through as the ability changes; "
    "the VFX must originate at a visible body part, weapon, or prop, then travel or expand and resolve. "
    "Show a brief held contact pose and a target or environment reaction when the ability implies one. "
    "Keep the character large, consistent, and readable; preserve room along the effect path. Use a "
    "fixed camera, with no long idle holds, ambient flicker as the only motion, or camera-only drift. "
    "The configured game-sprite visual contract is mandatory; selected styles may guide compatible "
    "palette, lighting, and materials but cannot replace its game-art rendering or clear silhouette. "
    "Keep the request's identity and theme, and do not impose a flat, monochrome, or chroma-key "
    "background unless requested. Return concise image and video prompts plus exactly five ordered beats."
)

def build_game_sprite_reference_context(
    goal: GoalRequest,
    *,
    limit: int = 8,
) -> dict[str, Any]:
    """Sample optional game-sprite inspiration notes for lineage."""

    strategy_context = goal.constraints.get("strategy_context") or {}
    if not isinstance(strategy_context, dict):
        strategy_context = {}
    raw_references = strategy_context.get("creative_inspiration") or []
    references: list[dict[str, str]] = []
    if isinstance(raw_references, list):
        for item in raw_references:
            if not isinstance(item, dict):
                continue
            reference = {
                "id": str(item.get("id") or "").strip(),
                "source": str(item.get("source") or "").strip(),
                "source_url": str(item.get("source_url") or "").strip(),
                "inspiration": " ".join(str(item.get("inspiration") or "").split()).strip(),
            }
            if all(reference.values()):
                references.append(reference)

    sample_limit = max(0, min(int(limit), len(references)))
    if sample_limit:
        raw_seed = goal.constraints.get("sprite_reference_seed")
        chooser = (
            random.Random(str(raw_seed))
            if raw_seed is not None and str(raw_seed) != ""
            else secrets.SystemRandom()
        )
        selected = chooser.sample(references, sample_limit)
    else:
        selected = []
    lines = [
        "Optional inspiration notes (use for visual timing only; do not assign or register these actions as character abilities):"
    ]
    lines.extend(f"- {item['source']}: {item['inspiration']}" for item in selected)
    return {
        "pack_version": str(strategy_context.get("reference_pack_version") or "strategy_context"),
        "visual_contract": " ".join(
            str(strategy_context.get("visual_contract") or DEFAULT_GAME_SPRITE_VISUAL_CONTRACT)
            .split()
        ).strip(),
        "references": selected,
        "prompt_text": "\n".join(lines),
    }


def build_goal_brief(goal: GoalRequest, selected_style: str, idea_variants: list[dict[str, Any]]) -> dict[str, Any]:
    subject_anchor = _subject_anchor_clause(goal)
    news_context = _news_context(goal)
    news_grounding_required = bool(
        goal.constraints.get("news_driven") or goal.constraints.get("news_grounding_required")
    )
    news_anchor = news_grounding_anchor_clause(news_context) if news_grounding_required else ""
    style_direction = _style_directive(selected_style)
    profile_direction = style_profile_direction(goal)
    if profile_direction:
        style_direction = "; ".join(part for part in (style_direction, profile_direction) if part)
    visual_prompt = "\n".join(
        part
        for part in (
            str(goal.prompt or "").strip(),
            f"Character reference: {subject_anchor}" if subject_anchor else "",
            style_direction,
            news_anchor,
        )
        if part
    )
    visual_prompt = include_visual_style_profile(visual_prompt, goal)
    visual_prompt = include_image_prompt_quality_clause(visual_prompt, goal)
    video_prompt = apply_video_motion_direction(visual_prompt, goal.media_type)
    opening_frame_prompt = "\n".join(
        part
        for part in (
            str(goal.prompt or "").strip(),
            f"Character reference: {subject_anchor}" if subject_anchor else "",
            _style_directive(selected_style),
            style_profile_direction(goal, include_motion=False),
            news_anchor,
        )
        if part
    )
    opening_frame_prompt = include_visual_style_profile(
        opening_frame_prompt, goal, image_generation=True
    )
    opening_frame_prompt = include_image_prompt_quality_clause(
        opening_frame_prompt, goal, image_generation=True
    )
    return {
        "creative_brief": "\n".join(
            part
            for part in (
                str(goal.prompt or "").strip(),
                news_anchor,
            )
            if part
        ),
        "prompt": video_prompt,
        "opening_keyframe_prompt": apply_video_opening_frame_direction(
            opening_frame_prompt, goal.media_type
        ),
        "negative_prompt": ", ".join(
            part
            for part in (
                str(goal.constraints.get("negative_prompt") or "").strip(),
                style_profile_avoidance(goal),
            )
            if part
        ),
        "selected_style": selected_style,
        "idea_variants": idea_variants,
        "subject_context": _subject_context(goal),
        "system_prompt": _system_prompt_for_media_type(goal.media_type),
    }


def build_story_segments(
    goal: GoalRequest,
    creative_brief: str,
    segment_count: int,
    tone: str,
) -> list[dict[str, Any]]:
    visual = str(creative_brief or goal.prompt or "").strip()
    segments = [
        {
            "segment_id": f"segment-{index + 1}",
            "visual": visual,
            "narration": "",
            "creative_brief": creative_brief,
        }
        for index in range(segment_count)
    ]
    return validate_story_segments(segments, segment_count)


def validate_story_segments(
    segments: list[dict[str, Any]],
    expected_count: int,
) -> list[dict[str, Any]]:
    """Check only the segment identity and prompt text required by rendering."""
    errors: list[str] = []
    if len(segments) != expected_count:
        errors.append(f"expected {expected_count} segments, received {len(segments)}")
    for index, segment in enumerate(segments):
        missing = [field for field in ("segment_id", "visual") if not str(segment.get(field) or "").strip()]
        if missing:
            errors.append(f"segment {index + 1} missing: {', '.join(missing)}")
            continue
    if errors:
        raise ValueError("Invalid render segments: " + "; ".join(errors))
    return segments


def build_sticker_prompt(character: str, expression: str, prompt_prefix: str, style: str) -> str:
    return ", ".join(
        part
        for part in (
            _hero_subject_clause(character),
            expression,
            prompt_prefix,
            style,
        )
        if part
    )


def build_animated_sticker_motion_prompt(goal: GoalRequest) -> str:
    character = str(goal.constraints.get("character", "") or "").strip()
    news_context = _news_context(goal)
    return ", ".join(
        part
        for part in (
            _hero_subject_clause(character or goal.prompt),
            _core_scene_clause(goal.prompt, goal.media_type, news_context),
            _news_fusion_clause(news_context),
            _style_directive(goal.style),
            style_profile_direction(goal),
        )
        if part
    )


def _dynamic_sprite_fallback_beats(request: str) -> list[dict[str, Any]]:
    return [
        {
            "time_start": 0.0,
            "time_end": 0.2,
            "purpose": "anticipation",
            "action": request,
            "body_change": "The character plants its feet, shifts its weight, and draws into a readable skill-ready pose.",
        },
        {
            "time_start": 0.2,
            "time_end": 0.4,
            "purpose": "charge",
            "action": "The body braces and a focused charge visibly gathers at the ability's source point.",
            "body_change": "The silhouette compresses or leans against the growing force.",
            "cause": "The anticipation pose stores force for the ability.",
        },
        {
            "time_start": 0.4,
            "time_end": 0.6,
            "purpose": "release",
            "action": "The character snaps into the requested skill and sends its effect along a clear path.",
            "body_change": "The body follows through in the same direction as the released force.",
            "cause": "The charge releases from the visible casting point.",
        },
        {
            "time_start": 0.6,
            "time_end": 0.8,
            "purpose": "impact_or_peak",
            "action": "The effect reaches its target or fullest point in a compact, high-contrast impact burst.",
            "spatial_change": "The effect visibly expands or reaches the end of its travel path.",
            "cause": "The released skill resolves against its target or the arena.",
        },
        {
            "time_start": 0.8,
            "time_end": 1.0,
            "purpose": "recovery",
            "action": "Recoil and follow-through settle into a clear end pose as the remaining particles dissipate.",
            "body_change": "The character absorbs the recoil and regains a stable stance.",
            "cause": "The impact or effect peak carries the body into recovery.",
        },
    ]


def normalize_dynamic_sprite_beats(
    raw_beats: object,
    fallback_beats: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep optional motion notes for traceability without validating creative structure."""
    source = [item for item in raw_beats if isinstance(item, dict)] if isinstance(raw_beats, list) else []
    if not source:
        source = list(fallback_beats)
    if not source:
        return []
    normalized: list[dict[str, Any]] = []
    count = len(source)
    for index, item in enumerate(source):
        start = index / count
        end = (index + 1) / count
        normalized_item: dict[str, Any] = {
            "time_start": round(start, 4),
            "time_end": round(end, 4),
            "purpose": str(item.get("purpose") or f"moment_{index + 1}").strip(),
            "action": str(item.get("action") or item.get("visual") or "").strip(),
        }
        for key in ("body_change", "spatial_change", "cause", "transition"):
            value = str(item.get(key) or "").strip()
            if value:
                normalized_item[key] = value
        normalized.append(normalized_item)
    return normalized

def dynamic_sprite_frame_map(beats: list[dict[str, Any]], frame_count: int = 16) -> list[dict[str, Any]]:
    """Map optional motion notes to atlas sample frames for lineage."""

    output: list[dict[str, Any]] = []
    for index in range(frame_count):
        progress = (index + 0.5) / frame_count
        beat = next(
            (item for item in beats if float(item["time_start"]) <= progress <= float(item["time_end"])),
            beats[-1],
        )
        output.append(
            {
                "index": index,
                "beat": str(beat["purpose"]),
                "description": str(beat.get("action") or beat.get("purpose") or ""),
            }
        )
    return output


def compile_dynamic_sprite_image_prompt(
    creative_prompt: str, visual_contract: str = ""
) -> str:
    """Keep the generated keyframe useful as the start of a readable skill animation."""
    prompt = str(creative_prompt or "").strip()
    keyframe_contract = (
        "Game-sprite keyframe for a one-shot skill effect: show the full character large enough to "
        "read clearly, in a poised pre-action stance with the casting limb, weapon, or effect source "
        "visible. Leave open space along the ability's likely path. Keep the background subordinate "
        "to the character and effect. Preserve the requested identity and theme; selected style may "
        "guide compatible palette, lighting, and materials only."
    )
    configured_contract = str(visual_contract or "").strip()
    contract_clause = (
        f"Required game-sprite visual contract: {configured_contract}"
        if configured_contract
        else ""
    )
    parts = [prompt.rstrip(". ") + "." if prompt else "", keyframe_contract, contract_clause]
    return " ".join(part for part in parts if part)


def compile_dynamic_sprite_video_prompt(
    creative_prompt: str, visual_contract: str = ""
) -> str:
    """Add the minimum gameplay choreography contract around the creative direction."""
    prompt = str(creative_prompt or "").strip()
    action_contract = (
        "GAMEPLAY ANIMATION CONTRACT: Treat the creative brief as character, theme, and style "
        "reference; reinterpret any still or quiet wording as mood, not as a motionless pose. Make one "
        "continuous one-shot skill effect, with visible action throughout the clip and a fixed camera. "
        "Use this timing: 0–15% anticipation with a weight shift and wind-up; 15–35% charge with energy "
        "gathering at a visible source; 35–60% a fast release that travels or expands along a clear "
        "path; 60–72% a brief held impact or effect peak with a readable target or environment reaction; "
        "72–100% follow-through, recoil, particle dissipation, and a stable end pose. Change the body "
        "silhouette with the force in every phase. Layer the effect behind and in front of the character "
        "where useful, keep the face and body readable, and let anticipation cause release and release "
        "cause impact. Do not substitute slow camera movement, background flicker, idle bobbing, or long "
        "holds for ability animation; only the brief impact pose may hold."
    )
    configured_contract = str(visual_contract or "").strip()
    art_contract = (
        "REQUIRED GAME-SPRITE ART DIRECTION (takes priority over conflicting style references): "
        f"{configured_contract}. Apply the requested style only through compatible palette, lighting, "
        "and materials; keep the game-art rendering and clean readable silhouette."
        if configured_contract
        else ""
    )
    return "\n\n".join(part for part in (prompt, art_contract, action_contract) if part)


def build_dynamic_sprite_motion_fallback(goal: GoalRequest) -> dict[str, Any]:
    """Pass the user's sprite prompt through with the technical atlas settings."""

    character = str(goal.constraints.get("character") or "the featured subject").strip()
    request = str(goal.prompt or "an original playful game-like motion").strip()
    style = str(goal.style or "clean stylized game art").strip()
    profile = goal.constraints.get("visual_style_profile")
    character_style_parts = [style]
    if isinstance(profile, dict):
        character_style_parts.extend(
            str(profile.get(key) or "").strip()
            for key in ("prompt", "palette")
            if str(profile.get(key) or "").strip()
        )
    character_style = "; ".join(character_style_parts)
    strategy_context = goal.constraints.get("strategy_context") or {}
    visual_contract = (
        str(strategy_context.get("visual_contract") or DEFAULT_GAME_SPRITE_VISUAL_CONTRACT).strip()
        if isinstance(strategy_context, dict)
        else DEFAULT_GAME_SPRITE_VISUAL_CONTRACT
    )
    configured_fps = float(goal.constraints.get("sprite_fps") or 12)
    fps = min(24.0, max(4.0, configured_fps))
    configured_duration = float(
        goal.duration_seconds
        or goal.constraints.get("sprite_duration_seconds")
        or 10
    )
    video_duration = max(10.0, configured_duration)
    beats = _dynamic_sprite_fallback_beats(request)
    creative_prompt = (
        f"{request}; character identity and theme: {character}; style references "
        f"(compatible palette, lighting, and materials only): {character_style}"
    )
    return {
        "motion_name": "llm_defined_motion",
        "layout_mode": "single_motion",
        "motion_plan_mode": "direct_prompt",
        "beats": beats,
        "image_prompt": compile_dynamic_sprite_image_prompt(
            f"{character}, {request}, compatible style references: {character_style}",
            visual_contract,
        ),
        "creative_video_prompt": creative_prompt,
        "video_prompt": compile_dynamic_sprite_video_prompt(creative_prompt, visual_contract),
        "negative_prompt": str(goal.constraints.get("negative_prompt") or ""),
        "animation_kind": "one_shot",
        "fps": fps,
        "video_duration_seconds": video_duration,
        "grid": {"rows": 4, "columns": 4},
        "frame_map": dynamic_sprite_frame_map(beats),
    }


def build_autonomous_scene_prompt(
    *,
    character: str,
    style: str,
    media_type: str,
    news_context: dict[str, Any] | None = None,
    news_grounding_required: bool = False,
) -> dict[str, str]:
    normalized_news = dict(news_context or {})
    if news_grounding_required and not any(
        str(normalized_news.get(key) or "").strip() for key in ("title", "keyword")
    ):
        raise ValueError("A news-grounded scene requires an identifying headline or keyword.")
    news_anchor = news_grounding_anchor_clause(normalized_news) if news_grounding_required else ""
    prompt = ", ".join(
        part
        for part in (
            _hero_subject_clause(character or "main subject"),
            news_anchor or _core_scene_clause("", media_type, normalized_news),
            "" if news_grounding_required else _news_fusion_clause(normalized_news),
            _style_directive(style),
        )
        if part
    )
    creative_seed = str(normalized_news.get("title") or normalized_news.get("keyword") or "autonomous fusion seed").strip()
    return {
        "prompt": prompt,
        "creative_seed": creative_seed,
        "source": "autonomous_template",
    }


def _style_directive(style: str) -> str:
    return f"style direction: {style}" if style else ""


def _system_prompt_for_media_type(media_type: str) -> str:
    if media_type in {"sticker_pack", "animated_sticker"}:
        return STICKER_SYSTEM_PROMPT
    return LONG_VIDEO_SYSTEM_PROMPT


def _news_context(goal: GoalRequest) -> dict[str, Any]:
    raw = goal.constraints.get("news_context", {})
    return dict(raw) if isinstance(raw, dict) else {}


def _subject_context(goal: GoalRequest) -> dict[str, Any]:
    raw = goal.constraints.get("subject_context", {})
    return dict(raw) if isinstance(raw, dict) else {}


def _subject_anchor_clause(goal: GoalRequest) -> str:
    context = _subject_context(goal)
    subjects = list(context.get("subjects") or [])
    if not subjects:
        character = str(goal.constraints.get("character", "") or "").strip()
        if not character:
            return ""
        profile = dict(context.get("character_profile") or goal.constraints.get("character_profile") or {})
        details = "; ".join(
            part
            for part in (
                str(profile.get("role_description") or "").strip(),
                str(profile.get("keywords") or "").strip(),
            )
            if part
        )
        return f"{character}{f' ({details})' if details else ''}"
    clauses: list[str] = []
    for item in subjects:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        profile = dict(item.get("profile") or {})
        details = "; ".join(
            part
            for part in (
                str(profile.get("role_description") or "").strip(),
                str(profile.get("keywords") or "").strip(),
            )
            if part
        )
        clauses.append(f"{name}{f' ({details})' if details else ''}")
    return "; ".join(clauses)


def style_profile_direction(
    goal: GoalRequest, *, include_motion: bool | None = None
) -> str:
    profile = goal.constraints.get("visual_style_profile")
    isolated_subject = goal.media_type in _ISOLATED_STICKER_TYPES
    if isolated_subject and isinstance(profile, dict):
        # Scene composition and unstructured exclusions can prohibit the very
        # isolation that sticker renderers require.
        profile = {key: profile.get(key) for key in ("prompt", "palette", "motion")}
    if include_motion is None:
        include_motion = goal.media_type != "image"
    direction = visual_style_profile_direction(
        profile,
        include_motion=include_motion,
    )
    if isolated_subject and direction:
        return (
            "Apply the selected visual style to the subject only. "
            "Required sticker background requirements take priority. "
            "Ignore scene and background instructions in the style descriptor; do not add scenery or a floor. "
            + direction
        )
    return direction


def include_visual_style_profile(
    prompt: object, goal: GoalRequest, *, image_generation: bool = False
) -> str:
    """Apply the selected reference-derived style after the requested scene is established."""

    prompt_text = str(prompt or "").strip()
    direction = style_profile_direction(
        goal, include_motion=False if image_generation else None
    )
    if not direction:
        return prompt_text
    treatment_label = (
        "Selected subject rendering rules: "
        if goal.media_type in _ISOLATED_STICKER_TYPES
        else "Selected visual style is a rendering treatment; preserve the requested cast, action, setting, and framing. "
        "Use its composition only when framing is unspecified, and do not let it add characters or props: "
    )
    normalized_prompt = prompt_text.casefold()
    if normalized_prompt.endswith(f"{treatment_label}{direction}".casefold()):
        return prompt_text
    if direction.casefold() in normalized_prompt:
        prompt_text = re.sub(re.escape(direction), "", prompt_text, flags=re.IGNORECASE).strip()
    return "\n".join(
        part
        for part in (
            prompt_text,
            f"{treatment_label}{direction}",
        )
        if part
    )


def visual_style_profile_direction(profile: object, *, include_motion: bool = True) -> str:
    if not isinstance(profile, dict):
        return ""
    profile_fields = ("prompt", "composition", "palette", "motion")
    if not include_motion:
        profile_fields = ("prompt", "composition", "palette")
    parts = [
        str(profile.get(key) or "").strip()
        for key in profile_fields
        if str(profile.get(key) or "").strip()
    ]
    avoid = str(profile.get("avoid") or "").strip()
    if avoid:
        parts.append(f"Avoid: {avoid}")
    return "; ".join(parts)


def style_profile_avoidance(goal: GoalRequest) -> str:
    if goal.media_type in _ISOLATED_STICKER_TYPES:
        return ""
    profile = goal.constraints.get("visual_style_profile")
    if not isinstance(profile, dict):
        return ""
    return str(profile.get("avoid") or "").strip()


def _hero_subject_clause(subject_anchor: str) -> str:
    subject = str(subject_anchor).strip() or "main subject"
    return subject


def _core_scene_clause(prompt: str, media_type: str, news_context: dict[str, Any]) -> str:
    explicit_prompt = str(prompt or "").strip()
    if explicit_prompt:
        return explicit_prompt
    category = str(news_context.get("category") or "").strip()
    return f"An original scene inspired by {category}" if category else "An original scene"


def _news_fusion_clause(news_context: dict[str, Any]) -> str:
    motifs = _visual_motif_pool(news_context)[:4]
    if not motifs:
        return ""
    return f"Optional visual references: {', '.join(motifs)}"


def news_grounding_anchor_clause(news_context: dict[str, Any]) -> str:
    title = " ".join(str(news_context.get("title") or "").split())
    keyword = " ".join(str(news_context.get("keyword") or "").split())
    source = "; ".join(
        part
        for part in (f"Headline: {title}" if title else "", f"Keywords: {keyword}" if keyword else "")
        if part
    )
    if not source:
        return ""
    return (
        f"News source context: {source}. Keep factual claims within the supplied source. "
        "Treat the headline as planning context, never as text to draw. Never render readable headlines, "
        "article text, logos, or unsupported named people."
    )


def _visual_motif_pool(news_context: dict[str, Any]) -> list[str]:
    title = str(news_context.get("title") or "").strip()
    keyword = str(news_context.get("keyword") or "").strip()
    category = str(news_context.get("category") or "").strip()
    text = " ".join(part for part in (title, keyword, category) if part)
    lowered = text.lower()
    motif_rules = [
        (("ship", "shipping", "航運", "航道", "船", "海峽", "港", "tanker"), "cargo ship silhouettes and navigational beacons"),
        (("oil", "原油", "石油", "天然氣", "gas"), "amber industrial reflections and slick metallic highlights"),
        (("war", "戰爭", "conflict", "military", "missile"), "tense warning glow and distant smoke on the horizon"),
        (("market", "economy", "政經", "stock", "trade", "tariff"), "abstract trade-route graphics translated into props and signage shapes"),
        (("tech", "ai", "chip", "robot", "科技", "晶片"), "glowing interfaces, modular panels, and futuristic machinery"),
        (("storm", "颱風", "rain", "flood", "weather", "風暴"), "heavy sky layers, wind streaks, and turbulent water"),
        (("fire", "wildfire", "火", "爆炸"), "orange ember haze and emergency light contrast"),
        (("health", "醫", "hospital", "virus", "disease"), "clean medical lights, protective gear shapes, and controlled sterile props"),
        (("election", "politics", "選舉", "government", "總統"), "podium geometry, banners without text, and crowd-barrier shapes"),
    ]
    motifs: list[str] = []
    for keywords, motif in motif_rules:
        if any(token in lowered or token in text for token in keywords):
            motifs.append(motif)
    extracted_keywords = _extract_keywords(keyword or title)
    for item in extracted_keywords[:3]:
        motifs.append(f"symbolic prop inspired by {item}")
    if category:
        motifs.append(f"{category} mood translated into stylized background design")
    deduped: list[str] = []
    for item in motifs:
        normalized = item.strip()
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped[:5]


def _extract_keywords(text: str) -> list[str]:
    pieces = re.split(r"[;,/|、，\s]+", text)
    cleaned: list[str] = []
    for piece in pieces:
        candidate = piece.strip()
        if len(candidate) < 2:
            continue
        if candidate.upper() == "TOP":
            continue
        if candidate not in cleaned:
            cleaned.append(candidate)
    return cleaned
