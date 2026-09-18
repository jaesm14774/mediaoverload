from __future__ import annotations

import random
import re
import secrets
from typing import Any

from agentic.storyboard import build_storyboard_segments, load_storyboard, story_state_contract

from agentic.runtime.contracts import GoalRequest
from agentic.minimax_prompting import compose_minimax_h3_prompt, short_action_contract, structured_visual_prompt


LONG_VIDEO_SYSTEM_PROMPT = """
You are an expert director and prompt designer for image and video generation.

Non-negotiable rules:
- Keep the same main character identity in every shot.
- Treat news context according to the workflow contract. When the workflow says the
  output is news-grounded, the selected headline must become a concrete visual
  object, action, obstacle, or consequence in the protagonist's causal story; it
  must not be reduced to atmosphere or generic motifs. When the workflow does not
  request news grounding, use the news only as optional visual inspiration.
- Do not recreate a headline literally or stage a newsroom/documentary frame unless explicitly requested.
- The named character must remain the hero, while the news-derived element gives the hero a concrete objective, obstacle, or consequence.
- For news-grounded workflows, preserve one recognizable news anchor across the hook, a later causal beat, and the payoff.
- Default to one named protagonist and one readable causal mechanism, but let the selected visual profile build a layered setting, configured interaction pair, supporting physical forms, and background activity when they reinforce that mechanism. Never invent unrequested characters or duplicate the protagonist.
- Do not introduce speech bubbles, signs, screens, interfaces, readable symbols, pseudo-text, or scribbles; when the source involves communication, translate it into an unmarked physical object or visible action unless literal text is explicitly required.
- Prefer tangible visuals over abstract summaries: props, architecture, lighting, weather, symbols, motion.
- Prefer substantial actions that can sustain a full clip, not tiny repetitive motions.
- Each shot must visibly progress from the previous one.
- Describe prompts in this order: subject continuity, scene, primary action, environment, camera movement, lighting/style, audio, and end state.
- Put the camera instruction beside the action it controls; use concrete camera language such as follow, push in, pan, tilt, or pull out.
- For image-to-video, treat the first frame as authoritative and describe how it starts moving and evolves instead of redrawing it.
- For text-to-video, establish the identity inside the first moving action instead of opening on a posed portrait.
- When a clip has multiple beats, write timestamped shot progression in chronological order and include the cause/effect handoff.
- For cute short-form work, prefer one dominant tactile prop or environmental force, one readable reaction, and one visible reversal; a small character against an oversized object or setting is a useful scale joke when it remains legible.
- Physical comedy should have anticipation, contact or impact, reaction, and a settled result. Use squash-and-stretch, wobble, recoil, or elastic snap only when the visible action causes it.
- If the clip is intended to loop, let the final settled composition echo the opening direction, prop, or pose without hiding the completed payoff.
- Make the result generation-ready for diffusion and image-to-video models.
""".strip()


ENGLISH_GENERATION_RESPONSE_CONTRACT = """
Language contract:
- Write every creative field in natural, idiomatic English: storyboards, scene descriptions, image prompts, video prompts, actions, camera directions, audio directions, and visual QA prose.
- Do not write Traditional Chinese, Simplified Chinese, Japanese, or mixed-language prose in those creative fields.
- Preserve a source-language proper noun or exact source text only when the schema explicitly requires it as metadata; never copy it into a creative prompt field.
- Translate the meaning into fluent English rather than translating word for word.
Response in English only.
""".strip()


IMAGE_PROMPT_CONTRACT = """
Image prompt contract:
- Build one visual thesis: one focal subject or configured interaction, one dominant mechanism, and one readable relationship or emotional beat.
- Describe the decisive visible state, not a list of disconnected objects; translate abstract mood into a pose, prop interaction, deformation, trail, or lighting contrast.
- Preserve the order Subject -> Scene -> Action or visual state -> Environment -> Composition and camera -> Style and lighting -> Quality.
- Give the viewer a clear focal path using intentional negative space or layered depth; state relative scale, containment, attachment, or layer order when those relationships carry the joke.
- Choose one medium and surface language, then name a few observable material cues; do not stack contradictory style labels or vague quality adjectives.
- Use small secondary details only after the main read works. Treat text, charts, panels, and multi-subject layouts as hard geometry constraints when explicitly requested; otherwise avoid them.
- End with precise constraints and a short avoid list: no duplicate subject, accidental extra characters, clutter, watermark, logo, interface, or pseudo-text.
""".strip()


STICKER_SYSTEM_PROMPT = """
You design high-performing messaging stickers.

Non-negotiable rules:
- Clear silhouette and readable emotion at thumbnail size.
- Exaggerated facial expression and body language.
- Clean background, strong outline, simple shape language.
- One emotion per sticker, visually obvious in under a second.
- Prefer a locked or nearly locked camera and a clean, uncluttered field so the emotion survives thumbnail viewing.
- For animated stickers, use one anticipation -> impact -> settle cycle, with elastic deformation caused by the action and an ending pose that can return to the opening pose cleanly.
- If news context exists, reduce it to tiny symbolic accents instead of literal reporting.
""".strip()


DYNAMIC_SPRITE_BACKGROUND_PALETTE = {
    "cyan": "#00e5ff",
    "green": "#18e06f",
    "blue": "#2f6bff",
    "yellow": "#ffe51f",
    "violet": "#7a38ff",
    "orange": "#ff9b2f",
}
DYNAMIC_SPRITE_BACKGROUND_ALIASES = {
    "aqua": "cyan",
    "teal": "cyan",
    "emerald": "green",
    "lime": "green",
    "cobalt": "blue",
    "azure": "blue",
    "gold": "yellow",
    "amber": "orange",
    "purple": "violet",
}
DYNAMIC_SPRITE_DEFAULT_BACKGROUND = "random"


def resolve_dynamic_sprite_background(value: object = None) -> str:
    """Resolve one simple non-red chroma background for a sprite run."""

    raw = str(value or "").strip().casefold()
    if raw in DYNAMIC_SPRITE_BACKGROUND_PALETTE:
        return DYNAMIC_SPRITE_BACKGROUND_PALETTE[raw]
    if raw in DYNAMIC_SPRITE_BACKGROUND_PALETTE.values():
        return raw
    for alias, canonical in DYNAMIC_SPRITE_BACKGROUND_ALIASES.items():
        if alias in raw:
            return DYNAMIC_SPRITE_BACKGROUND_PALETTE[canonical]
    return secrets.choice(tuple(DYNAMIC_SPRITE_BACKGROUND_PALETTE.values()))


def dynamic_sprite_source_contract(background_color: str) -> str:
    return (
        f"isolated game-sprite source, one subject only, flat saturated chroma-key background {background_color}, "
        "no red or magenta background, no floor, tabletop, ground plane, horizon, environment, cast shadow, "
        "contact shadow, text, logo, watermark, panel, split screen, duplicate subject"
    )


def dynamic_sprite_video_contract(background_color: str) -> str:
    return (
        f"Keep the camera locked on the isolated subject against the flat saturated chroma-key background {background_color}; "
        "do not introduce a red or magenta background, floor, tabletop, ground plane, horizon, environment, cast shadow, "
        "or contact shadow. Keep the character recognizable and anatomically consistent while allowing any readable "
        "continuous physical action, including changes in velocity, jumps, strikes, casts, celebrations, dodges, launches, "
        "catches, or an invented action. Never morph the character into a prop, spring, symbol, or different creature, and "
        "never let a prop replace the character. Keep the complete subject inside the central safe area on every beat: "
        "no clipping, leaving frame, extreme scale change, enclosure, or prop occlusion; any prop stays separate and "
        "secondary. Each beat must cause the next beat and the final beat must resolve from the preceding motion."
    )


DYNAMIC_SPRITE_SYSTEM_PROMPT = """
You are a director for automatically generated game motion assets.

Non-negotiable rules:
- Never choose from or assume a fixed action vocabulary. Invent the motion concept from the user's request; jump, strike, cast, celebrate, dodge, dash, and completely new physical actions are examples, not a whitelist.
- Generate a detailed choreography graph with four to eight causally connected beats. Every beat must begin from the previous beat's visible end state; never reset to an unrelated pose between beats.
- Give the arc a readable opening, development, escalation or turn, peak/payoff, and resolution. Add one surprising but physically legible turn when the request allows it.
- When resolved subject context is provided, its selected character and role profile are authoritative; a different character name appearing in the creative request must not replace the resolved subject.
- Keep the subject recognizable and anatomically consistent throughout while allowing any readable body movement, change of velocity, contact, airborne travel, attack, casting, celebration, recoil, or invented physical behavior. Never turn the subject into a prop, spring, symbol, or different creature.
- Keep the complete subject inside a central safe area in every sampled frame; do not clip, eject, enclose, or heavily occlude it, and keep any prop separate and secondary.
- Prefer zero to two optional props. Props may be touched, launched, caught, or transformed independently, but must remain separate from the subject and never replace it.
- Write every creative field in idiomatic English.
- The still prompt must describe one clean source subject, not a contact sheet or a storyboard.
- The video prompt and beats must describe a temporal arc from the initial state through development, peak, and outcome. Include concrete cause and transition language, not a list of disconnected poses.
- Make the motion readable when reduced to sixteen sampled frames. The sprite camera is always locked; do not spend the motion on scenery, camera pushes, or atmospheric establishing shots.
- Use one simple flat saturated chroma-key background from cyan, green, blue, yellow, violet, or orange. The color may vary per run and should suit the subject, but never use red or magenta and never put the background color on the subject.
- The source must be an isolated cutout: no floor, tabletop, ground plane, horizon, environment, cast shadow, or contact shadow.
- Decide whether the motion should loop from the requested idea. Do not force every idea into a loop.
- Avoid text, logos, watermarks, panels, split screens, duplicated subjects, and unrelated props.
""".strip()

DYNAMIC_SPRITE_VIDEO_CONTRACT = dynamic_sprite_video_contract("#00e5ff")
DYNAMIC_SPRITE_NEGATIVE_CONTRACT = (
    "floor, tabletop, ground plane, horizon, environment, cast shadow, contact shadow, "
    "storyboard, contact sheet, split screen, collage, duplicate subject, extra character, text, logo, watermark"
)


def build_game_sprite_reference_context(
    goal: GoalRequest,
    *,
    limit: int = 8,
) -> dict[str, Any]:
    """Sample the selected game_sprite strategy's optional inspiration notes."""

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
        "These notes are optional inspiration only. Do not copy named characters, signature move names, or plots; "
        "do not assign or register abilities. Adapt the physical logic freely to the current request, and ignore "
        "any note that does not fit.",
    ]
    lines.extend(f"- {item['source']}: {item['inspiration']}" for item in selected)
    return {
        "pack_version": str(strategy_context.get("reference_pack_version") or "strategy_context"),
        "visual_contract": " ".join(str(strategy_context.get("visual_contract") or "").split()).strip(),
        "references": selected,
        "prompt_text": "\n".join(lines),
    }


def build_goal_brief(goal: GoalRequest, selected_style: str, idea_variants: list[dict[str, Any]]) -> dict[str, Any]:
    subject_anchor = _subject_anchor_clause(goal)
    news_context = _news_context(goal)
    action_directive = _action_directive(goal.media_type, goal.duration_seconds)
    continuity_directive = _continuity_directive(goal.media_type)
    style_contract = str(goal.constraints.get("visual_style_contract") or "").strip()
    style_direction = _style_directive(selected_style)
    profile_direction = _style_profile_prompt(goal)
    if profile_direction:
        style_direction = "; ".join(part for part in (style_direction, profile_direction) if part)
    if style_contract:
        style_direction = "; ".join(part for part in (style_direction, style_contract) if part)
    visual_prompt = structured_visual_prompt(
        subject=subject_anchor,
        scene=_core_scene_clause(goal.prompt, goal.media_type, news_context),
        action=action_directive,
        environment=f"{_news_fusion_clause(news_context)}; {_interaction_clause(goal)}; {continuity_directive}",
        camera=_camera_beat(0, 2) if goal.media_type in {"long_video", "native_h3_story", "text2video", "text2img2video", "image_to_video"} else "clear focal composition",
        style=style_direction,
        quality=_quality_clause(goal.media_type),
    )
    opening_scene = str(goal.prompt or "").split(";", 1)[0].strip()
    opening_keyframe_prompt = structured_visual_prompt(
        subject=subject_anchor,
        scene=opening_scene or "one coherent scene at the start of the gag with visible foreground, middle action plane, and background",
        action=(
            "single still opening moment only: show the protagonist already beginning the one physical action "
            "with the dominant mechanism, prop, or environmental force; no aftermath, no before-and-after sequence, "
            "no second pose, no duplicate"
        ),
        environment=(
            f"{_interaction_clause(goal)}; {_style_profile_composition(goal)}; "
            "one dominant causal action; the protagonist appears exactly once"
        ),
        camera="single stable readable composition for the first frame of an image-to-video shot",
        style=style_direction,
        quality="clean silhouette, coherent anatomy, no montage, no storyboard panels, no repeated subject",
    )
    negative_prompt = ", ".join(
        [
            "ugly",
            "blurry",
            "low quality",
            "bad anatomy",
            "deformed",
            "duplicate subject" if not _interaction_required(goal) else "identity swap",
            "identity drift",
            "inconsistent costume",
            "weak composition",
            "headline text",
            "news ticker",
            "literal newspaper layout",
            "speech bubble",
            "readable signs",
            "pseudo-text",
            "scribbles",
            "interface",
            "extra characters" if not _interaction_required(goal) else "unwanted third subject",
            "crowd",
            "watermark",
            "text",
            "minimal motion",
            "static pose",
        ]
    )
    return {
        "creative_brief": f"{goal.prompt} translated into an executable {goal.media_type} workflow with strict subject continuity",
        "prompt": visual_prompt,
        "opening_keyframe_prompt": opening_keyframe_prompt,
        "negative_prompt": negative_prompt,
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
    production_profile: str = "",
) -> list[dict[str, Any]]:
    storyboard_path = str(goal.constraints.get("storyboard_path", "") or "").strip()
    if storyboard_path:
        storyboard = load_storyboard(storyboard_path)
        storyboard_brief = " ".join(
            part
            for part in (
                creative_brief,
                f"Resolved protagonist contract: {_subject_anchor_clause(goal)}.",
            )
            if str(part).strip()
        )
        storyboard_segments = build_storyboard_segments(
            storyboard,
            segment_count=segment_count,
            tone=tone,
            style=goal.style,
            creative_brief=storyboard_brief,
        )
        for segment in storyboard_segments:
            action_beats = segment.get("action_beats")
            if not str(segment.get("action") or "").strip() and isinstance(action_beats, list):
                segment["action"] = " ".join(str(beat).strip() for beat in action_beats if str(beat).strip())
        storyboard_segments = validate_story_segments(storyboard_segments, segment_count)
        if str(production_profile or "").strip().lower() == "text2longvideo":
            segment_duration = max(1.0, float(goal.duration_seconds) / max(1, int(segment_count)))
            for segment in storyboard_segments:
                segment["shots"] = build_timed_shot_plan(
                    segment,
                    duration_seconds=segment_duration,
                    shot_count=4,
                    force_multi_beat=True,
                )
        return storyboard_segments
    subject_anchor = _subject_anchor_clause(goal) or goal.prompt
    story_anchor = (
        f"same story world, dominant prop, obstacle, and objective from the goal: {goal.prompt}"
    )
    news_context = _news_context(goal)
    motif_pool = _visual_motif_pool(news_context)
    planned_states = _fallback_story_states(goal, subject_anchor, segment_count)
    segments: list[dict[str, Any]] = []
    for index in range(segment_count):
        stage = _story_stage(index, segment_count)
        camera = _camera_beat(index, segment_count)
        motion = _motion_beat(index, segment_count)
        environment = f"{story_anchor}; {_environment_beat(goal.prompt, index, segment_count, motif_pool)}"
        motif_clause = _segment_motif_clause(motif_pool, index)
        start_state = planned_states[index]
        end_state = planned_states[index + 1]
        visual = structured_visual_prompt(
            subject=subject_anchor,
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
                "action": motion,
                "start_state": start_state,
                "end_state": end_state,
                "cause": f"{subject_anchor} takes the next physical action because the previous state creates a clear immediate objective",
                "effect": f"The changed state opens the {('next' if index < segment_count - 1 else 'resolved')} story beat",
                "creative_brief": creative_brief,
            }
        )
    if str(production_profile or "").strip().lower() == "text2longvideo":
        segment_duration = max(1.0, float(goal.duration_seconds) / max(1, int(segment_count)))
        for segment in segments:
            segment["shots"] = build_timed_shot_plan(
                segment,
                duration_seconds=segment_duration,
                shot_count=4,
                force_multi_beat=True,
            )
    return validate_story_segments(segments, segment_count)


def _fallback_story_states(
    goal: GoalRequest,
    subject_anchor: str,
    segment_count: int,
) -> list[str]:
    """Create concrete state handoffs when the story model cannot fill the schema."""
    prompt = goal.prompt.casefold()
    seed_storm_story = "glowing seed" in prompt and "storm" in prompt
    if seed_storm_story:
        start = (
            f"{subject_anchor} is in the windy meadow with the glowing seed visibly ahead, "
            "before the first action"
        )
        payoff = (
            f"{subject_anchor} has reached a warm clearing and the protected glowing seed has bloomed "
            "into a bright flower"
        )
        middle_states = [
            f"{subject_anchor} has grabbed the glowing seed as the storm wind tears across the meadow",
            f"{subject_anchor} is shielding the glowing seed behind a rock while rain and wind intensify",
            f"{subject_anchor} has carried the glowing seed through the storm toward the warm clearing",
        ]
    else:
        start = f"{subject_anchor} is present at the opening location before the first action"
        payoff = f"{subject_anchor} has completed the objective and created a concrete visible payoff"
        middle_states = [
            f"{subject_anchor} has reached the central objective and visibly changed its position or condition",
            f"{subject_anchor} has secured the objective while the obstacle visibly intensifies",
            f"{subject_anchor} has redirected the objective through a new physical setback",
        ]
    states = [start]
    for index in range(max(0, segment_count - 1)):
        if index < len(middle_states):
            states.append(middle_states[index])
        else:
            states.append(
                f"{subject_anchor} has advanced the objective through a distinct visible change at story beat {index + 2}"
            )
    states.append(payoff)
    return states


def validate_story_segments(
    segments: list[dict[str, Any]],
    expected_count: int,
) -> list[dict[str, Any]]:
    """Check the small structural contract required by the render pipeline."""
    required_fields = (
        "segment_id",
        "visual",
        "narration",
        "action",
        "camera",
        "start_state",
        "end_state",
        "cause",
        "effect",
    )
    errors: list[str] = []
    if len(segments) != expected_count:
        errors.append(f"expected {expected_count} segments, received {len(segments)}")
    for index, segment in enumerate(segments):
        missing = [field for field in required_fields if not str(segment.get(field) or "").strip()]
        if missing:
            errors.append(f"segment {index + 1} missing: {', '.join(missing)}")
            continue
    if errors:
        raise ValueError("Story segment contract failed: " + "; ".join(errors))
    return segments


def build_timed_shot_plan(
    segment: dict[str, Any],
    *,
    duration_seconds: float = 15.0,
    shot_count: int = 4,
    force_multi_beat: bool = False,
) -> list[dict[str, str]]:
    """Return timed beats for one production story segment.

    Production segments need internal temporal structure. Keep short clips as
    one action so they do not conflict with the shared short-action contract.
    The fallback remains useful when an LLM response omits the optional shot
    array.
    """

    duration = max(1.0, float(duration_seconds))
    requested_shots = max(1, int(shot_count))
    max_shots = (
        max(3, min(4, requested_shots))
        if force_multi_beat
        else 1
        if duration <= 6
        else max(3, min(4, requested_shots))
    )
    raw_shots = segment.get("shots")
    if isinstance(raw_shots, list) and len(raw_shots) >= 3:
        normalized: list[dict[str, str]] = []
        for index, item in enumerate(raw_shots[:max_shots]):
            if not isinstance(item, dict):
                continue
            start = duration * index / max(1, len(raw_shots[:max_shots]))
            end = duration * (index + 1) / max(1, len(raw_shots[:max_shots]))
            normalized.append(
                {
                    "time": str(item.get("time") or f"{start:g}-{end:g}s"),
                    "title": str(item.get("title") or f"beat {index + 1}"),
                    "action": str(item.get("action") or segment.get("action") or "visible physical action"),
                    "camera": str(item.get("camera") or segment.get("camera") or "camera follows the action with spatial continuity"),
                    "state_change": str(item.get("state_change") or item.get("end_state") or segment.get("end_state") or "the visible state advances"),
                    "cause": str(item.get("cause") or segment.get("cause") or "the immediate objective drives the next action"),
                    "effect": str(item.get("effect") or segment.get("effect") or "the next beat becomes possible"),
                }
            )
        if len(normalized) >= 3:
            return normalized

    start_state = str(segment.get("start_state") or "the protagonist is ready to act")
    action = str(segment.get("action") or "advances the objective through a clear physical action")
    visual = str(segment.get("visual") or "the established setting")
    camera = str(segment.get("camera") or "the camera follows the action with a clear change in framing")
    cause = str(segment.get("cause") or "the immediate objective creates pressure")
    end_state = str(segment.get("end_state") or "the protagonist reaches the next visible state")
    effect = str(segment.get("effect") or "the next story beat becomes possible")
    fallback = [
        ("Establish and approach", f"Begin from {start_state}; establish {visual} and move toward the objective.", "Open wide, then track toward the protagonist and the dominant prop.", start_state, "the objective becomes immediately readable", "the protagonist commits to the attempt"),
        ("Attempt and contact", f"The protagonist performs the primary action: {action}.", camera, "the attempt is underway", "contact or the first decisive change becomes visible", "the cause creates a new obstacle"),
        ("Reversal and reaction", f"Because {cause}, the protagonist reacts and changes position or strategy while the environment responds.", "Follow the reversal with a pan or tilt, keeping the cause and reaction in the same spatial field.", "the plan changes under pressure", "the setback or reversal is readable", "the protagonist can reach the payoff"),
        ("Payoff and handoff", f"The protagonist completes the beat and reaches {end_state}.", "Follow the final movement, then pull out to hold the resolved composition.", end_state, "the segment settles on the visible result", effect),
    ]
    normalized = []
    for index, (title, shot_action, shot_camera, state_change, shot_cause, shot_effect) in enumerate(fallback):
        start = duration * index / len(fallback)
        end = duration * (index + 1) / len(fallback)
        normalized.append(
            {
                "time": f"{start:g}-{end:g}s",
                "title": title,
                "action": shot_action,
                "camera": shot_camera,
                "state_change": state_change,
                "cause": shot_cause,
                "effect": shot_effect,
            }
        )
    return normalized[:max_shots]


def build_segment_prompt(goal: GoalRequest, segment: dict[str, Any], prior_frame: str | None = None) -> dict[str, Any]:
    subject_anchor = _subject_anchor_clause(goal) or "same main subject"
    news_context = _news_context(goal)
    prompt = structured_visual_prompt(
        subject=subject_anchor,
        scene=str(segment.get("visual", "")),
        action=str(segment.get("action") or "substantial action with visible start-to-end motion"),
        environment=(
            f"{_news_fusion_clause(news_context)}; {_interaction_clause(goal)}; preserve facial features, costume, proportions, palette, and iconic character read"
        ),
        camera=str(segment.get("camera") or "coherent scene geography with camera continuity"),
        style=str(goal.style or "stylized cinematic animation"),
        quality="clear motion path, strong silhouette, spatial depth, no documentary text overlays",
    )
    outputs = {
        "segment_id": segment["segment_id"],
        "prompt": prompt,
        "narration": str(segment.get("narration", "")),
        "subject_context": _subject_context(goal),
    }
    if prior_frame:
        outputs["prior_frame_path"] = prior_frame
    return outputs


def build_minimax_h3_prompt(
    goal: GoalRequest,
    segment: dict[str, Any],
    prior_frame: str | None = None,
    *,
    official_shot_syntax: bool = False,
) -> dict[str, Any]:
    """Build an H3-ready audiovisual prompt while preserving selected-character continuity."""
    base = build_segment_prompt(goal, segment, prior_frame=prior_frame)
    character = _subject_names(goal) or "the main character"
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
        "action": (
            f"Primary physical action: {segment.get('action') or 'continuous visible movement'}. "
            f"Visual staging: {segment.get('visual') or base['prompt']}"
        ),
        "camera": str(segment.get("camera") or "camera follows the primary action with a readable change in framing"),
        "state_change": str(segment.get("end_state") or "the primary action reaches a visible next state"),
        "cause": str(segment.get("cause") or "the protagonist acts on the immediate objective"),
        "effect": str(segment.get("effect") or segment.get("next_hook") or "the next story beat becomes possible"),
    }
    raw_shots = segment.get("shots")
    shots = (
        [dict(item) for item in raw_shots if isinstance(item, dict)]
        if isinstance(raw_shots, list) and len(raw_shots) >= 3
        else [shot]
    )
    prompt = compose_minimax_h3_prompt(
        duration_seconds=duration,
        character=character,
        style=goal.style,
        base_prompt="",
        story_spine=segment.get("story_spine") if isinstance(segment.get("story_spine"), dict) else {},
        shots=shots,
        audio=audio_direction,
        render_mode="image_to_video" if prior_frame else "text_to_video",
        prior_frame=bool(prior_frame),
        subject_context=_subject_context(goal),
        official_shot_syntax=official_shot_syntax,
    )
    prompt = "\n".join(
        [
            prompt,
            f"Story progression contract: {story_contract}" if story_contract else "",
            (
                "Character lock: preserve both declared subject slots from the supplied identity anchor and keep their interaction readable."
                if _interaction_required(goal)
                else "Character lock: preserve the same protagonist from the supplied identity anchor."
            ),
            "Motion direction: advance from the declared start state to the declared end state with one continuous readable primary event.",
            (
                "Long-segment action contract: begin moving within the first half-second; sustain a visible motion path; "
                "include anticipation, a decisive cause-and-effect change, a readable reaction, and a settled result. "
                "Do not spend the segment standing still, watching, waiting, posing, or only making a slow camera push."
            )
            if goal.media_type == "long_video" and duration >= 7
            else "",
            f"Audio direction: {audio_direction}",
        ]
    )
    return {
        **base,
        "prompt": prompt,
        "audio_direction": audio_direction,
        "prompt_format": "minimax_h3_context_ir_local",
        "subject_context": _subject_context(goal),
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
            "bold outline, readable silhouette, exaggerated body language, one dominant prop or symbolic accent, polished 2D sticker finish",
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
            "simple loopable full-body motion with one anticipation, one impact or peak, and one settle; expressive bounce, squash-and-stretch, wobble, or elastic recoil must be caused by the visible action",
            "the final pose or prop position should echo the opening so the loop seam is gentle while the emotion remains readable",
            "minimal uncluttered background, locked camera, clear silhouette preservation",
            "avoid camera drift, avoid identity drift, avoid multiple actions or stacked effects",
        )
        if part
    )


def _dynamic_sprite_fallback_beats(character: str, request: str) -> list[dict[str, Any]]:
    """Build a structural fallback arc without choosing a fixed action."""

    return [
        {
            "time_start": 0.00,
            "time_end": 0.16,
            "purpose": "opening",
            "action": f"{character} holds the source pose and begins the requested motion: {request}.",
            "body_change": "a small readable preparation changes the balance without changing identity",
            "spatial_change": "the subject remains centered while its intended direction becomes clear",
            "cause": "the preparation creates the force for the next movement",
            "transition": "carry the preparation directly into the developing action",
        },
        {
            "time_start": 0.16,
            "time_end": 0.36,
            "purpose": "development",
            "action": f"{character} follows through on {request} with a visible change of position, velocity, or pose.",
            "body_change": "the silhouette stretches or contracts according to the force of the action",
            "spatial_change": "the motion follows one continuous readable path",
            "cause": "the opening impulse drives this larger movement",
            "transition": "the developing movement gathers into a decisive turn",
        },
        {
            "time_start": 0.36,
            "time_end": 0.62,
            "purpose": "peak",
            "action": f"{character} reaches the clearest and most surprising physical peak of {request} while staying recognizable.",
            "body_change": "the main pose change is held long enough to read in a sampled frame",
            "spatial_change": "the trajectory reaches one visible apex or contact point",
            "cause": "the previous movement supplies the momentum for the peak",
            "transition": "release the peak into a controlled recovery",
        },
        {
            "time_start": 0.62,
            "time_end": 0.84,
            "purpose": "resolution",
            "action": f"{character} resolves the consequence of {request} with a readable reaction and stable outcome.",
            "body_change": "the body absorbs the force and returns toward a confident readable pose",
            "spatial_change": "the subject settles without teleporting or changing scale",
            "cause": "the peak naturally causes the recovery",
            "transition": "echo the opening direction if the motion is periodic; otherwise prepare the final hold",
        },
        {
            "time_start": 0.84,
            "time_end": 1.00,
            "purpose": "settle",
            "action": f"{character} holds the earned final pose after completing {request}.",
            "body_change": "a small after-motion confirms the action is finished",
            "spatial_change": "the silhouette remains fully visible inside the frame",
            "cause": "the resolution has completed the action",
            "transition": "return cleanly to the opening direction only for a periodic loop",
        },
    ]


def normalize_dynamic_sprite_beats(
    raw_beats: object,
    fallback_beats: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize an LLM choreography graph while preserving its open vocabulary."""

    if not isinstance(raw_beats, list) or not 4 <= len(raw_beats) <= 8:
        return list(fallback_beats)
    raw_ranges: list[tuple[float, float]] = []
    for item in raw_beats:
        if not isinstance(item, dict):
            return list(fallback_beats)
        try:
            raw_ranges.append((float(item.get("time_start", 0.0)), float(item.get("time_end", 1.0))))
        except (TypeError, ValueError):
            return list(fallback_beats)
    # JSON-schema validation can clamp a model's second-based values to 1.0,
    # leaving several later beats with identical [1.0, 1.0] ranges. Once the
    # order has collapsed, the original durations are unknowable; use evenly
    # spaced choreography slots instead of hiding every beat at the end.
    order_collapsed = any(
        raw_ranges[index][0] <= raw_ranges[index - 1][0]
        or raw_ranges[index][1] <= raw_ranges[index][0]
        for index in range(1, len(raw_ranges))
    ) or any(end <= start for start, end in raw_ranges)
    if order_collapsed:
        raw_ranges = [
            (index / len(raw_ranges), (index + 1) / len(raw_ranges))
            for index in range(len(raw_ranges))
        ]
    # Models sometimes return seconds (0..5) despite the normalized schema
    # asking for fractions. Scale the entire timeline once, rather than
    # clamping every beat independently and collapsing later beats at 1.0.
    timeline_max = max(max(start, end) for start, end in raw_ranges)
    timeline_scale = timeline_max if timeline_max > 1.0 else 1.0
    normalized: list[dict[str, Any]] = []
    previous_end = 0.0
    for index, (item, raw_range) in enumerate(zip(raw_beats, raw_ranges)):
        try:
            raw_start, raw_end = raw_range
            start = max(previous_end, min(1.0, raw_start / timeline_scale))
            end = max(start + 0.01, min(1.0, raw_end / timeline_scale))
        except (TypeError, ValueError):
            return list(fallback_beats)
        normalized.append(
            {
                "time_start": round(start, 4),
                "time_end": round(end, 4),
                "purpose": str(item.get("purpose") or f"beat_{index + 1:02d}").strip(),
                "action": str(item.get("action") or "continue the same physical action").strip(),
                "body_change": str(item.get("body_change") or "preserve a readable subject silhouette").strip(),
                "spatial_change": str(item.get("spatial_change") or "continue the established path").strip(),
                "cause": str(item.get("cause") or "the preceding beat supplies the force").strip(),
                "transition": str(item.get("transition") or "continue directly into the next beat").strip(),
            }
        )
        previous_end = end
    normalized[-1]["time_end"] = 1.0
    return normalized


def dynamic_sprite_frame_map(beats: list[dict[str, Any]], frame_count: int = 16) -> list[dict[str, Any]]:
    """Project the choreography graph onto sampled atlas frames for QA and lineage."""

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
                "description": (
                    f"{beat['action']} Body change: {beat['body_change']}. "
                    f"Spatial change: {beat['spatial_change']}. "
                    f"Cause: {beat['cause']}. Transition: {beat['transition']}."
                ),
            }
        )
    return output


def compile_dynamic_sprite_video_prompt(
    creative_prompt: str,
    beats: list[dict[str, Any]],
    animation_kind: str,
    background_color: str,
) -> str:
    """Compile an open-ended choreography graph into one temporal H3 prompt."""

    beat_lines = [
        (
            f"Beat {index}: {item['purpose']} ({float(item['time_start']):.2f}-{float(item['time_end']):.2f}). "
            f"Action: {item['action']} Body: {item['body_change']} Spatial path: {item['spatial_change']} "
            f"Cause: {item['cause']} Transition: {item['transition']}"
        )
        for index, item in enumerate(beats, start=1)
    ]
    loop_instruction = (
        "Echo the opening direction and energy for a clean periodic loop."
        if animation_kind == "periodic"
        else "End on the earned stable outcome; do not invent a second action after the resolution."
    )
    return " ".join(
        part
        for part in (
            str(creative_prompt or "").strip(),
            "Execute the following sequential choreography as one continuous shot with no pose reset:",
            " ".join(beat_lines),
            loop_instruction,
            dynamic_sprite_video_contract(background_color),
        )
        if part
    )


def build_dynamic_sprite_motion_fallback(goal: GoalRequest) -> dict[str, Any]:
    """Build a no-LLM emergency plan with the same continuous-action structure."""

    character = str(goal.constraints.get("character") or "the featured subject").strip()
    request = str(goal.prompt or "an original playful game-like motion").strip()
    style = str(goal.style or "clean stylized game art").strip()
    configured_fps = float(goal.constraints.get("sprite_fps") or 12)
    fps = min(24.0, max(4.0, configured_fps))
    configured_duration = float(
        goal.duration_seconds
        or goal.constraints.get("sprite_duration_seconds")
        or 5
    )
    video_duration = min(8.0, max(4.0, configured_duration))
    chroma_color = resolve_dynamic_sprite_background(
        goal.constraints.get("sprite_chroma_color") or DYNAMIC_SPRITE_DEFAULT_BACKGROUND
    )
    chroma_threshold = int(goal.constraints.get("sprite_chroma_threshold") or 52)
    loop = not any(token in request.lower() for token in ("one-shot", "one shot", "non-loop", "non loop"))
    beats = _dynamic_sprite_fallback_beats(character, request)
    creative_prompt = (
        f"Animate the same {character} through an original, surprising but readable physical interpretation of: {request}."
    )
    return {
        "motion_name": "llm_defined_motion",
        "layout_mode": "single_motion",
        "motion_plan_mode": "choreographed_action_graph",
        "creative_twist": "one surprising but physically caused turn that remains readable in sixteen sampled frames",
        "beats": beats,
        "image_prompt": (
            f"one single {character}, {request}, {style}, clean game asset source, full subject visible, "
            f"strong readable silhouette, centered composition, stable three-quarter view, "
            f"{dynamic_sprite_source_contract(chroma_color)}"
        ),
        "creative_video_prompt": creative_prompt,
        "video_prompt": compile_dynamic_sprite_video_prompt(
            creative_prompt,
            beats,
            "periodic" if loop else "one_shot",
            chroma_color,
        ),
        "negative_prompt": f"{DYNAMIC_SPRITE_NEGATIVE_CONTRACT}, clutter, camera drift, identity drift",
        "animation_kind": "periodic" if loop else "one_shot",
        "fps": fps,
        "video_duration_seconds": video_duration,
        "grid": {"rows": 4, "columns": 4},
        "frame_map": dynamic_sprite_frame_map(beats),
        "chroma_color": chroma_color,
        "background_color": chroma_color,
        "chroma_threshold": chroma_threshold,
        "identity_repair_applied": "none",
    }


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
    if media_type in {"image", "text2img2img", "image_refine", "image_upscale", "storyboard"}:
        return (
            "one decisive visual state or pose that makes the central idea immediately readable; any prop interaction "
            "must show a clear physical relationship and visible consequence"
        )
    if media_type in {"long_video", "native_h3_story", "text2video", "text2img2video", "animated_sticker", "image_to_video"}:
        short_contract = short_action_contract(duration_seconds, media_type=media_type)
        if short_contract:
            return short_contract
        if int(duration_seconds or 0) <= 15:
            return (
                "one compact causal mini-story in one to three strong action beats: each beat changes the visible "
                "state, one simple prop or force remains the visual anchor, the protagonist has one readable reaction, "
                "and the final beat delivers one memorable physical payoff; if loopable, echo the opening composition"
            )
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
        return "low tracking shot that follows the subject entering at speed, then pushes in on the objective"
    if index == segment_count - 1:
        return "camera rushes with the decisive move, then pulls out to reveal the concrete payoff"
    return "continuous tracking shot with a clear angle change, parallax, and increasing depth"


def _motion_beat(index: int, segment_count: int) -> str:
    beats = [
        "bursting into motion, crossing the space toward the central objective, and making a decisive reach or grab",
        "interacting with the environment while advancing the objective and overcoming a physical setback",
        "shifting position, colliding with the obstacle, and escalating the scene with a stronger pose change",
        "lunging through the final obstacle, triggering a visible change, and landing a strong physical payoff",
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


def _subject_context(goal: GoalRequest) -> dict[str, Any]:
    raw = goal.constraints.get("subject_context", {})
    return dict(raw) if isinstance(raw, dict) else {}


def _interaction_required(goal: GoalRequest) -> bool:
    return bool(_subject_context(goal).get("interaction_contract", {}).get("required", False))


def _subject_names(goal: GoalRequest) -> str:
    subjects = list(_subject_context(goal).get("subjects") or [])
    names = [str(item.get("name") or "").strip() for item in subjects if isinstance(item, dict)]
    names = [name for name in names if name]
    if not names:
        character = str(goal.constraints.get("character", "") or "").strip()
        return character
    return " and ".join(names)


def _subject_anchor_clause(goal: GoalRequest) -> str:
    context = _subject_context(goal)
    subjects = list(context.get("subjects") or [])
    if not subjects:
        character = str(goal.constraints.get("character", "") or "").strip() or "main subject"
        profile = dict(context.get("character_profile") or goal.constraints.get("character_profile") or {})
        details = "; ".join(
            part
            for part in (
                str(profile.get("role_description") or "").strip(),
                str(profile.get("keywords") or "").strip(),
            )
            if part
        )
        return _hero_subject_clause(f"{character}{f' ({details})' if details else ''}")
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
    if _interaction_required(goal):
        return (
            "two distinct subject slots in one coherent frame: "
            + "; ".join(clauses)
            + "; both identities remain readable and visibly interact"
        )
    return _hero_subject_clause(clauses[0] if clauses else "main subject")


def _interaction_clause(goal: GoalRequest) -> str:
    if not _interaction_required(goal):
        return "single-subject continuity; do not add unrequested subjects"
    contract = dict(_subject_context(goal).get("interaction_contract") or {})
    return (
        "two required subjects share the same frame and a readable action relationship; "
        "preserve each identity, spatial position, and role; do not add an unrequested third subject"
        if contract.get("same_frame", True)
        else "two required subjects maintain stable identity and a visible mutual action relationship"
    )


def _style_profile_composition(goal: GoalRequest) -> str:
    profile = goal.constraints.get("visual_style_profile")
    if not isinstance(profile, dict):
        return "coherent setting with readable foreground, middle action plane, and background"
    return "; ".join(
        str(profile.get(key) or "").strip()
        for key in ("composition", "palette", "motion")
        if str(profile.get(key) or "").strip()
    ) or "coherent setting with readable foreground, middle action plane, and background"


def _style_profile_prompt(goal: GoalRequest) -> str:
    profile = goal.constraints.get("visual_style_profile")
    if not isinstance(profile, dict):
        return ""
    return "; ".join(
        str(profile.get(key) or "").strip()
        for key in ("prompt", "composition", "palette", "motion")
        if str(profile.get(key) or "").strip()
    )


def _hero_subject_clause(subject_anchor: str) -> str:
    subject = str(subject_anchor).strip() or "main subject"
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
    return (
        "strong focal hierarchy, one dominant visual mechanism, intentional negative space, observable material cues, "
        "polished rendering, no documentary text overlays"
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
