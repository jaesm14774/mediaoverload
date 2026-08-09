from __future__ import annotations

import re
from typing import Any

from agentic.storyboard import build_storyboard_segments, load_storyboard, story_state_contract

from agentic.runtime.contracts import GoalRequest
from agentic.minimax_prompting import compose_minimax_h3_prompt, structured_visual_prompt


LONG_VIDEO_SYSTEM_PROMPT = """
You are an expert director and prompt designer for image and video generation.

Non-negotiable rules:
- Keep the same main character identity in every shot.
- If news context is provided, use it only as inspiration for visual motifs, props, tension, or atmosphere.
- Do not recreate a headline literally or stage a newsroom/documentary frame unless explicitly requested.
- The named character must remain the hero; news elements are supporting scene ingredients.
- Extract 2-4 concrete news-inspired elements and blend them into one coherent scene.
- Prefer tangible visuals over abstract summaries: props, architecture, lighting, weather, symbols, motion.
- Prefer substantial actions that can sustain a full clip, not tiny repetitive motions.
- Each shot must visibly progress from the previous one.
- Describe prompts in this order: subject continuity, scene, primary action, environment, camera movement, lighting/style, audio, and end state.
- Put the camera instruction beside the action it controls; use concrete camera language such as follow, push in, pan, tilt, or pull out.
- For image-to-video, treat the first frame as authoritative and describe how it starts moving and evolves instead of redrawing it.
- For text-to-video, establish the identity inside the first moving action instead of opening on a posed portrait.
- When a clip has multiple beats, write timestamped shot progression in chronological order and include the cause/effect handoff.
- Make the result generation-ready for diffusion and image-to-video models.
""".strip()


STICKER_SYSTEM_PROMPT = """
You design high-performing messaging stickers.

Non-negotiable rules:
- Clear silhouette and readable emotion at thumbnail size.
- Exaggerated facial expression and body language.
- Clean background, strong outline, simple shape language.
- One emotion per sticker, visually obvious in under a second.
- If news context exists, reduce it to tiny symbolic accents instead of literal reporting.
""".strip()


def build_goal_brief(goal: GoalRequest, selected_style: str, idea_variants: list[dict[str, Any]]) -> dict[str, Any]:
    character = str(goal.constraints.get("character", "") or "").strip()
    subject_anchor = character or "main subject"
    news_context = _news_context(goal)
    action_directive = _action_directive(goal.media_type, goal.duration_seconds)
    continuity_directive = _continuity_directive(goal.media_type)
    visual_prompt = structured_visual_prompt(
        subject=_hero_subject_clause(subject_anchor),
        scene=_core_scene_clause(goal.prompt, goal.media_type, news_context),
        action=action_directive,
        environment=f"{_news_fusion_clause(news_context)}; {continuity_directive}",
        camera=_camera_beat(0, 2) if goal.media_type in {"long_video", "native_h3_story", "text2video", "text2img2video", "image_to_video"} else "clear focal composition",
        style=_style_directive(selected_style),
        quality=_quality_clause(goal.media_type),
    )
    negative_prompt = ", ".join(
        [
            "ugly",
            "blurry",
            "low quality",
            "bad anatomy",
            "deformed",
            "duplicate subject",
            "identity drift",
            "inconsistent costume",
            "weak composition",
            "headline text",
            "news ticker",
            "literal newspaper layout",
            "watermark",
            "text",
            "minimal motion",
            "static pose",
        ]
    )
    return {
        "creative_brief": f"{goal.prompt} translated into an executable {goal.media_type} workflow with strict subject continuity",
        "prompt": visual_prompt,
        "negative_prompt": negative_prompt,
        "selected_style": selected_style,
        "idea_variants": idea_variants,
        "system_prompt": _system_prompt_for_media_type(goal.media_type),
    }


def build_story_segments(
    goal: GoalRequest,
    creative_brief: str,
    segment_count: int,
    tone: str,
) -> list[dict[str, Any]]:
    storyboard_path = str(goal.constraints.get("storyboard_path", "") or "").strip()
    if storyboard_path:
        storyboard = load_storyboard(storyboard_path)
        return build_storyboard_segments(
            storyboard,
            segment_count=segment_count,
            tone=tone,
            style=goal.style,
            creative_brief=creative_brief,
        )
    character = str(goal.constraints.get("character", "") or "").strip()
    subject_anchor = character or goal.prompt
    news_context = _news_context(goal)
    motif_pool = _visual_motif_pool(news_context)
    segments: list[dict[str, Any]] = []
    for index in range(segment_count):
        stage = _story_stage(index, segment_count)
        camera = _camera_beat(index, segment_count)
        motion = _motion_beat(index, segment_count)
        environment = _environment_beat(goal.prompt, index, segment_count, motif_pool)
        motif_clause = _segment_motif_clause(motif_pool, index)
        visual = structured_visual_prompt(
            subject=_hero_subject_clause(subject_anchor),
            scene=f"story stage: {stage}; tone: {tone}",
            action=motion,
            environment=f"{environment}; {motif_clause}" if motif_clause else environment,
            camera=camera,
            style=_style_directive(goal.style),
            quality=_quality_clause(goal.media_type),
        )
        narration = (
            f"{subject_anchor} {motion.lower()}, pushing the story into the {stage.lower()} beat with clear visual progression."
        )
        segments.append(
            {
                "segment_id": f"segment-{index + 1}",
                "visual": visual,
                "narration": narration,
                "stage": stage,
                "camera": camera,
                "creative_brief": creative_brief,
            }
        )
    return segments


def build_segment_prompt(goal: GoalRequest, segment: dict[str, Any], prior_frame: str | None = None) -> dict[str, Any]:
    character = str(goal.constraints.get("character", "") or "").strip()
    subject_anchor = character or "same main subject"
    news_context = _news_context(goal)
    prompt = structured_visual_prompt(
        subject=_hero_subject_clause(subject_anchor),
        scene=str(segment.get("visual", "")),
        action=str(segment.get("action") or "substantial action with visible start-to-end motion"),
        environment=(
            f"{_news_fusion_clause(news_context)}; preserve facial features, costume, proportions, palette, and iconic character read"
        ),
        camera=str(segment.get("camera") or "coherent scene geography with camera continuity"),
        style=str(goal.style or "stylized cinematic animation"),
        quality="clear motion path, strong silhouette, spatial depth, no documentary text overlays",
    )
    outputs = {
        "segment_id": segment["segment_id"],
        "prompt": prompt,
        "narration": str(segment.get("narration", "")),
    }
    if prior_frame:
        outputs["prior_frame_path"] = prior_frame
    return outputs


def build_minimax_h3_prompt(goal: GoalRequest, segment: dict[str, Any], prior_frame: str | None = None) -> dict[str, Any]:
    """Build an H3-ready audiovisual prompt while preserving Kirby continuity."""
    base = build_segment_prompt(goal, segment, prior_frame=prior_frame)
    character = str(goal.constraints.get("character", "") or "").strip() or "the main character"
    audio_direction = str(
        goal.constraints.get(
            "h3_audio_direction",
            "native stereo audio: playful foot taps, soft environmental ambience, one readable impact accent, "
            "light melodic motif, no subtitles or text overlays",
        )
    )
    story_contract = story_state_contract(segment)
    duration = max(1, int(goal.duration_seconds or 5))
    shot = {
        "time": f"0-{duration}s",
        "title": str(segment.get("narrative_goal") or segment.get("segment_id") or "primary story beat"),
        "action": str(segment.get("visual") or base["prompt"]),
        "camera": str(segment.get("camera") or "camera follows the primary action with a readable change in framing"),
        "state_change": str(segment.get("end_state") or "the primary action reaches a visible next state"),
        "cause": str(segment.get("cause") or "the protagonist acts on the immediate objective"),
        "effect": str(segment.get("effect") or segment.get("next_hook") or "the next story beat becomes possible"),
    }
    prompt = compose_minimax_h3_prompt(
        duration_seconds=duration,
        character=character,
        style=goal.style,
        base_prompt="",
        story_spine=segment.get("story_spine") if isinstance(segment.get("story_spine"), dict) else {},
        shots=[shot],
        audio=audio_direction,
        render_mode="image_to_video" if prior_frame else "text_to_video",
        prior_frame=bool(prior_frame),
    )
    prompt = "\n".join(
        [
            prompt,
            f"Story progression contract: {story_contract}" if story_contract else "",
            "Character lock: preserve the same protagonist from the supplied identity anchor.",
            "Motion direction: advance from the declared start state to the declared end state with one continuous readable primary event.",
            f"Audio direction: {audio_direction}",
        ]
    )
    return {
        **base,
        "prompt": prompt,
        "audio_direction": audio_direction,
        "prompt_format": "minimax_h3_context_ir_local",
    }


def build_sticker_prompt(character: str, expression: str, prompt_prefix: str, style: str) -> str:
    return ", ".join(
        part
        for part in (
            _hero_subject_clause(character),
            expression,
            _core_scene_clause(prompt_prefix, "sticker_pack", {}),
            prompt_prefix,
            style,
            "single character sticker, centered composition, transparent-friendly clean background",
            "bold outline, readable silhouette, exaggerated body language, polished 2D sticker finish, symbolic accents only",
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
            "simple loopable full-body motion, expressive bounce, clear silhouette preservation",
            "avoid camera drift, avoid identity drift, preserve sticker readability",
        )
        if part
    )


def build_autonomous_scene_prompt(
    *,
    character: str,
    style: str,
    media_type: str,
    news_context: dict[str, Any] | None = None,
) -> dict[str, str]:
    normalized_news = dict(news_context or {})
    duration_hint = 15 if media_type == "native_h3_story" else (8 if media_type in {"text2video", "text2img2video", "image_to_video"} else 16)
    prompt = ", ".join(
        part
        for part in (
            _hero_subject_clause(character or "main subject"),
            _core_scene_clause("", media_type, normalized_news),
            _news_fusion_clause(normalized_news),
            _action_directive(media_type, duration_hint),
            _style_directive(style),
            _quality_clause(media_type),
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


def _action_directive(media_type: str, duration_seconds: int) -> str:
    if media_type in {"long_video", "native_h3_story", "text2video", "text2img2video", "animated_sticker", "image_to_video"}:
        return f"meaningful action sequence that can sustain {duration_seconds} seconds"
    return "clear action and visual intent"


def _continuity_directive(media_type: str) -> str:
    if media_type in {"long_video", "native_h3_story", "text2video", "text2img2video", "image_to_video", "animated_sticker"}:
        return "strict continuity of subject identity, pose logic, and scene progression"
    return "consistent subject identity and composition"


def _system_prompt_for_media_type(media_type: str) -> str:
    if media_type in {"sticker_pack", "animated_sticker"}:
        return STICKER_SYSTEM_PROMPT
    return LONG_VIDEO_SYSTEM_PROMPT


def _story_stage(index: int, segment_count: int) -> str:
    if index == 0:
        return "opening"
    if index == segment_count - 1:
        return "conclusion"
    return "development"


def _camera_beat(index: int, segment_count: int) -> str:
    if index == 0:
        return "medium-wide establishing shot with clear subject entrance"
    if index == segment_count - 1:
        return "hero closing shot with payoff framing"
    return "progressive action shot with camera angle change and depth"


def _motion_beat(index: int, segment_count: int) -> str:
    beats = [
        "entering the scene with decisive movement and immediate visual intent",
        "interacting with the environment while advancing the action",
        "shifting position and escalating the scene energy with a stronger pose change",
        "resolving the action with a strong final gesture and clear payoff",
    ]
    if segment_count <= 1:
        return beats[0]
    normalized = round((index / max(1, segment_count - 1)) * (len(beats) - 1))
    return beats[normalized]


def _environment_beat(prompt: str, index: int, segment_count: int, motif_pool: list[str]) -> str:
    if index == 0:
        lead = motif_pool[0] if motif_pool else "environment cue"
        return f"environment established from: {prompt or 'the creative brief'}, anchored by {lead}"
    if index == segment_count - 1:
        tail = motif_pool[-1] if motif_pool else "a transformed backdrop"
        return f"environment shows payoff, aftermath, or destination with {tail}"
    mid = motif_pool[min(index, len(motif_pool) - 1)] if motif_pool else "new depth cues"
    return f"environment evolves with new depth cues, props, or spatial progression featuring {mid}"


def _news_context(goal: GoalRequest) -> dict[str, Any]:
    raw = goal.constraints.get("news_context", {})
    return dict(raw) if isinstance(raw, dict) else {}


def _hero_subject_clause(subject_anchor: str) -> str:
    subject = str(subject_anchor).strip() or "main subject"
    if subject.lower() == "kirby":
        return "Kirby as the unmistakable hero, perfectly round soft pink body, large blue eyes, red boots"
    return f"{subject} as the unmistakable hero and focal subject"


def _core_scene_clause(prompt: str, media_type: str, news_context: dict[str, Any]) -> str:
    explicit_prompt = str(prompt or "").strip()
    if explicit_prompt:
        return explicit_prompt
    category = str(news_context.get("category") or "").strip()
    if media_type in {"long_video", "native_h3_story", "text2video", "text2img2video", "image_to_video"}:
        if category:
            return f"playful cinematic scene inspired by {category} news energy rather than literal reporting"
        return "playful cinematic scene with a clear story beat rather than a static pose"
    if media_type in {"sticker_pack", "animated_sticker"}:
        return "reaction-driven scene with one clear emotion and tiny symbolic props"
    if category:
        return f"single polished illustration inspired by {category} news energy rather than literal reporting"
    return "single polished illustration with a strong focal action"


def _news_fusion_clause(news_context: dict[str, Any]) -> str:
    motifs = _visual_motif_pool(news_context)[:4]
    if not motifs:
        return "original scenario built around clear action, readable staging, and stylized world details"
    return f"news-inspired visual motifs only, not literal reportage: {', '.join(motifs)}"


def _segment_motif_clause(motif_pool: list[str], index: int) -> str:
    if not motif_pool:
        return ""
    primary = motif_pool[index % len(motif_pool)]
    secondary = motif_pool[(index + 1) % len(motif_pool)] if len(motif_pool) > 1 else ""
    if secondary and secondary != primary:
        return f"motif focus: {primary}, supported by {secondary}"
    return f"motif focus: {primary}"


def _quality_clause(media_type: str) -> str:
    if media_type in {"long_video", "native_h3_story", "text2video", "text2img2video", "image_to_video"}:
        return "cinematic lighting, strong silhouette, spatial depth, clear motion path, no documentary text overlays"
    if media_type in {"sticker_pack", "animated_sticker"}:
        return "simple high-contrast silhouette, clean read at thumbnail size, no clutter"
    return "strong focal hierarchy, polished anime rendering, layered environment detail, no documentary text overlays"


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
