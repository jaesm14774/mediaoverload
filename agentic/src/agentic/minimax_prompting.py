from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def clean_prompt_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip().rstrip(".")


def subject_identity_lock(character: str) -> str:
    subject = clean_prompt_text(character) or "the main subject"
    if subject.lower() == "kirby":
        return (
            "Kirby is the single unmistakable protagonist: a small round soft-pink hero, large expressive blue eyes, "
            "tiny arms, red boots, clean simple silhouette, stable proportions and palette"
        )
    return f"{subject} is the single unmistakable protagonist with stable identity, proportions, costume, silhouette, and palette"


def structured_visual_prompt(
    *,
    subject: str,
    scene: str,
    action: str,
    environment: str,
    camera: str,
    style: str,
    quality: str,
) -> str:
    """Use one stable semantic order for image and video-adjacent prompts."""
    fields = (
        ("Subject", subject),
        ("Scene", scene),
        ("Action", action),
        ("Environment", environment),
        ("Camera", camera),
        ("Style and lighting", style),
        ("Quality", quality),
    )
    return "\n".join(f"{label}: {clean_prompt_text(value)}." for label, value in fields if clean_prompt_text(value))


def compose_minimax_h3_prompt(
    *,
    duration_seconds: int,
    character: str,
    style: str,
    base_prompt: str = "",
    story_spine: Mapping[str, Any] | None = None,
    setting: str = "",
    visual_language: str = "",
    shots: Sequence[Mapping[str, Any]] = (),
    audio: str = "",
    render_mode: str = "",
    prior_frame: bool = False,
    continuity_rules: Sequence[Any] = (),
) -> str:
    """Compose the local H3 prompt in MiniMax's documented multimodal shape."""
    spine = dict(story_spine or {})
    identity = subject_identity_lock(character)
    mode = clean_prompt_text(render_mode).lower()
    if prior_frame or mode in {"image_to_video", "i2v", "first_last_frame_to_video", "fl2va"}:
        input_relation = (
            "Input relation: the first-frame image is authoritative for the opening appearance, composition, and subject identity; "
            "start moving from that exact image and describe temporal evolution rather than redrawing a new still image"
        )
        if mode in {"first_last_frame_to_video", "fl2va"}:
            input_relation += "; guide the causal motion toward the supplied last-frame state instead of inventing a disconnected ending"
    else:
        input_relation = (
            "Input relation: direct text-to-video; establish the protagonist's identity in the first moving action, "
            "not in a static character sheet"
        )
    prompt_parts = [
        "integrated_multimodal_description:",
        f"Duration: {int(duration_seconds)} seconds ({int(duration_seconds)}-second short film). One continuous short film, not a montage of unrelated clips.",
        f"Character continuity: {identity}.",
        input_relation + ".",
    ]
    for label, key in (
        ("Creative intent", "premise"),
        ("Protagonist objective", "objective"),
        ("Obstacle", "obstacle"),
        ("Stakes", "stakes"),
        ("Climax", "climax"),
        ("Resolution", "resolution"),
    ):
        value = clean_prompt_text(spine.get(key))
        if value:
            prompt_parts.append(f"{label}: {value}.")
    if clean_prompt_text(base_prompt):
        prompt_parts.append(f"Identity and visual anchor: {clean_prompt_text(base_prompt)}.")
    if clean_prompt_text(setting):
        prompt_parts.append(f"World: {clean_prompt_text(setting)}.")
    if clean_prompt_text(visual_language):
        prompt_parts.append(f"Visual language: {clean_prompt_text(visual_language)}.")

    prompt_parts.append("Shot progression:")
    for index, shot in enumerate(shots, start=1):
        time = clean_prompt_text(shot.get("time")) or f"beat {index}"
        title = clean_prompt_text(shot.get("title"))
        action = clean_prompt_text(shot.get("action") or shot.get("visual"))
        camera = clean_prompt_text(shot.get("camera")) or "camera follows the primary action with readable spatial continuity"
        state_change = clean_prompt_text(shot.get("state_change") or shot.get("end_state"))
        cause = clean_prompt_text(shot.get("cause"))
        effect = clean_prompt_text(shot.get("effect"))
        shot_parts = [f"[Shot {index} / SHOT {index} | {time}]"]
        if title:
            shot_parts.append(f"{title}.")
        if cause:
            shot_parts.append(f"Cause: {cause}.")
        shot_parts.append(f"Action: {action or 'the protagonist advances the objective through a visible physical action'}.")
        shot_parts.append(f"Camera: {camera}.")
        if state_change:
            shot_parts.append(f"State change: {state_change}.")
        if effect:
            shot_parts.append(f"Effect and handoff: {effect}.")
        prompt_parts.append(" ".join(shot_parts))

    soundscape = clean_prompt_text(audio) or "natural stereo ambience with clear action-linked sound effects"
    prompt_parts.extend(
        [
            f"overall_soundscape: {soundscape}.",
            "non_diegetic_music: a restrained melodic motif that rises with the obstacle, tightens at the reversal, and resolves with the visible payoff.",
            "Motion contract: visible motion starts in the first half-second; every shot changes composition, action, and mission state; preserve physical cause and effect.",
            "Continuity contract: maintain one protagonist, one readable geography, consistent silhouette and palette, and a visible handoff between adjacent shots.",
            "Visual guardrails: stylized cinematic animation, clear silhouette, readable foreground action, no readable text, logos, subtitles, watermark, duplicate protagonist, frozen pose, or unrelated spectacle.",
        ]
    )
    rules = [clean_prompt_text(item) for item in continuity_rules if clean_prompt_text(item)]
    if rules:
        prompt_parts.append("Additional continuity rules: " + "; ".join(rules) + ".")
    return "\n".join(prompt_parts)
