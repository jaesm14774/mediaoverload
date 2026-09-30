from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

def clean_prompt_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip().rstrip(".")


def subject_reference(
    character: str,
    subject_context: Mapping[str, Any] | None = None,
) -> str:
    context = dict(subject_context or {})
    subjects = [item for item in (context.get("subjects") or []) if isinstance(item, Mapping)]
    if subjects:
        descriptions: list[str] = []
        for item in subjects:
            name = clean_prompt_text(item.get("name"))
            if not name:
                continue
            profile = dict(item.get("profile") or {})
            role_description = clean_prompt_text(profile.get("role_description"))
            keywords = clean_prompt_text(profile.get("keywords"))
            details = "; ".join(
                part
                for part in (
                    role_description,
                    keywords,
                )
                if part
            )
            descriptions.append(f"{name}{f' ({details})' if details else ''}")
        if descriptions:
            return "Character references: " + "; ".join(descriptions) + "."
    subject = clean_prompt_text(character)
    profile = dict(context.get("character_profile") or {})
    if not profile and subjects:
        profile = dict(subjects[0].get("profile") or {})
    role_description = clean_prompt_text(profile.get("role_description"))
    keywords = clean_prompt_text(profile.get("keywords"))
    if role_description:
        supplemental = f" Supplemental visual keywords: {keywords}." if keywords else ""
        return f"Character reference: {subject} — {role_description}.{supplemental}"
    if keywords:
        return f"Character reference: {subject}. Visual keywords: {keywords}."
    return f"Character: {subject}." if subject else ""


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
    subject_context: Mapping[str, Any] | None = None,
    official_shot_syntax: bool = False,
    media_type: str | None = None,
) -> str:
    """Compose the local H3 prompt in MiniMax's documented multimodal shape."""
    spine = dict(story_spine or {})
    context = dict(subject_context or {})
    identity = subject_reference(character, context)
    mode = clean_prompt_text(render_mode).lower()
    if prior_frame or mode in {"image_to_video", "i2v", "first_last_frame_to_video", "fl2va"}:
        input_relation = "Input relation: begin from the supplied first frame and follow the requested motion"
        if mode in {"first_last_frame_to_video", "fl2va"}:
            input_relation = (
                "Input relation: begin from the supplied first frame and make the requested action visibly cause "
                "the supplied last-frame state. Preserve character identity and count, screen direction, setting, "
                "props, drawing style, palette, framing, and camera axis unless the shot progression calls for a "
                "clear change. Make the physical cause, surprise/payoff, and character reaction legible; land on "
                "the supplied last frame and hold the final reveal for about half a second"
            )
    else:
        input_relation = "Input relation: generate from the supplied text brief."
    prompt_parts = [
        "integrated_multimodal_description:",
        f"Duration: {int(duration_seconds)} seconds.",
        identity,
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
        prompt_parts.append(f"Creative brief: {clean_prompt_text(base_prompt)}.")
    if clean_prompt_text(setting):
        prompt_parts.append(f"World: {clean_prompt_text(setting)}.")
    if clean_prompt_text(visual_language):
        prompt_parts.append(f"Visual language: {clean_prompt_text(visual_language)}.")

    if shots and not official_shot_syntax:
        prompt_parts.append("Shot progression:")
    for index, shot in enumerate(shots, start=1):
        time = clean_prompt_text(shot.get("time"))
        title = clean_prompt_text(shot.get("title"))
        action = clean_prompt_text(shot.get("action") or shot.get("visual"))
        camera = clean_prompt_text(shot.get("camera"))
        state_change = clean_prompt_text(shot.get("state_change") or shot.get("end_state"))
        cause = clean_prompt_text(shot.get("cause"))
        effect = clean_prompt_text(shot.get("effect"))
        if official_shot_syntax:
            shot_parts = [f"[Shot {index}]"]
            if index > 1 and time:
                shot_parts.append(f"At {_format_h3_cut_time(time)}, the camera cuts to")
        else:
            shot_parts = [f"[Shot {index}{f' | {time}' if time else ''}]"]
        if title:
            shot_parts.append(f"{title}.")
        if cause:
            shot_parts.append(f"Because {cause}," if official_shot_syntax else f"Cause: {cause}.")
        if action:
            shot_parts.append(f"{action}.")
        if camera:
            shot_parts.append(f"{camera}.")
        if state_change:
            shot_parts.append(
                f"The visible state changes to {state_change}."
                if official_shot_syntax
                else f"State change: {state_change}."
            )
        if effect:
            shot_parts.append(
                f"This causes the next beat: {effect}."
                if official_shot_syntax
                else f"Effect and handoff: {effect}."
            )
        prompt_parts.append(" ".join(shot_parts))

    soundscape = clean_prompt_text(audio)
    if soundscape:
        prompt_parts.append(f"Audio direction: {soundscape}.")
    prompt_parts.append("Follow the supplied creative brief and requested style.")
    rules = [clean_prompt_text(item) for item in continuity_rules if clean_prompt_text(item)]
    if rules:
        prompt_parts.append("Additional continuity rules: " + "; ".join(rules) + ".")
    return "\n".join(prompt_parts)


def _format_h3_cut_time(value: str) -> str:
    """Format a shot range's start as the official H3 cut timestamp."""
    match = re.search(r"(-?\d+(?:\.\d+)?)", str(value or ""))
    seconds = max(0.0, float(match.group(1))) if match else 0.0
    minutes = int(seconds // 60)
    remainder = seconds - (minutes * 60)
    return f"{minutes:02d}:{remainder:06.3f}"
