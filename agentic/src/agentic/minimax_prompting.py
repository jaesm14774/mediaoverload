from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any


def require_h3_prompt(value: Any) -> str:
    """Require non-empty text for the H3 prompt field."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("H3 prompt must be a non-empty scene description string.")
    return value.strip()


def clean_prompt_text(value: Any) -> str:
    if value is None or value == "":
        return ""
    return " ".join(require_h3_prompt(value).split()).rstrip(".")


def format_h3_video_prompt(
    description: str,
    *,
    mode: str,
    duration_seconds: float,
    overall_soundscape: str,
    non_diegetic_music: str = "N/A",
) -> str:
    """Format one visual story using MiniMax's mode-specific prompt contract."""
    scene = require_h3_prompt(description)
    normalized_mode = str(mode or "t2va").strip().casefold().replace("-", "_")
    aliases = {
        "t2v": "t2va",
        "text_to_video": "t2va",
        "i2v": "i2va",
        "image_to_video": "i2va",
        "fl2v": "fl2va",
        "first_last_frame_to_video": "fl2va",
        "l2v": "l2va",
        "last_frame_to_video": "l2va",
        "reference_to_video": "ref2va",
    }
    normalized_mode = aliases.get(normalized_mode, normalized_mode)
    if normalized_mode not in {"t2va", "i2va", "fl2va", "l2va", "ref2va"}:
        raise ValueError(f"Unsupported base H3 prompt mode: {mode}")

    seconds = float(duration_seconds)
    if seconds <= 0:
        raise ValueError("H3 prompt duration must be positive.")
    soundscape = clean_prompt_text(overall_soundscape)
    music = clean_prompt_text(non_diegetic_music) or "N/A"
    if not soundscape:
        raise ValueError("H3 prompt requires an overall soundscape or explicit silence.")
    if not scene.lstrip().startswith("[Shot 1]"):
        scene = f"[Shot 1] {scene}"
    shot_numbers = re.findall(r"\[Shot\s+(\d+)\]", scene, re.IGNORECASE)
    first_shot = shot_numbers[0] if shot_numbers else "1"
    last_shot = shot_numbers[-1] if shot_numbers else first_shot

    alignment = ""
    if normalized_mode == "i2va":
        alignment = (
            "For the target video, at 0.00 seconds into the target video, "
            f"<Picture 1> (from [Shot {first_shot}]) is fully referenced."
        )
    elif normalized_mode == "fl2va":
        alignment = (
            "How the reference pictures align with the target video — "
            f"<Picture 1> (from [Shot {first_shot}]) aligns with the 0.00-second mark of the target video; "
            f"<Picture 2> (from [Shot {last_shot}]) aligns with the {seconds:.2f}-second mark of the target video."
        )
    elif normalized_mode == "l2va":
        alignment = (
            "How the reference pictures align with the target video — "
            f"<Picture 1> (from [Shot {last_shot}]) aligns with the {seconds:.2f}-second mark of the target video."
        )

    sections = [
        f"integrated_multimodal_description: {scene}",
        f"overall_soundscape: {soundscape}",
        f"non_diegetic_music: {music}",
    ]
    return require_h3_prompt("\n\n".join(([alignment] if alignment else []) + sections))


def extract_h3_scene_description(prompt: str) -> str:
    """Return only the integrated visual/audio-timeline section of a base prompt."""
    value = require_h3_prompt(prompt)
    match = re.search(
        r"(?:^|\n)integrated_multimodal_description:\s*(.*?)(?=\n\s*overall_soundscape:|\Z)",
        value,
        re.DOTALL,
    )
    return require_h3_prompt(match.group(1).strip()) if match else value


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
            details = role_description or keywords
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
    overall_soundscape: str = "",
    non_diegetic_music: str = "N/A",
    render_mode: str = "",
    prior_frame: bool = False,
    continuity_rules: Sequence[Any] = (),
    subject_context: Mapping[str, Any] | None = None,
    media_type: str | None = None,
) -> str:
    """Compose a compact H3 scene, then apply the official audiovisual fields."""
    spine = dict(story_spine or {})
    context = dict(subject_context or {})
    identity = subject_reference(character, context)
    opening = [clean_prompt_text(style), identity]
    opening.extend(
        clean_prompt_text(value)
        for value in (setting, visual_language, base_prompt, spine.get("premise"))
        if clean_prompt_text(value)
    )
    objective = clean_prompt_text(spine.get("objective"))
    obstacle = clean_prompt_text(spine.get("obstacle"))
    stakes = clean_prompt_text(spine.get("stakes"))
    story_goal = "; ".join(
        part
        for part in (
            f"The character aims to {objective}" if objective else "",
            f"but {obstacle}" if obstacle else "",
            f"the consequence is {stakes}" if stakes else "",
        )
        if part
    )
    if story_goal:
        opening.append(story_goal)

    prompt_parts: list[str] = []
    for index, shot in enumerate(shots, start=1):
        time = clean_prompt_text(shot.get("time"))
        action = clean_prompt_text(shot.get("action") or shot.get("visual"))
        camera = clean_prompt_text(shot.get("camera"))
        state_change = clean_prompt_text(shot.get("state_change") or shot.get("end_state"))
        cause = clean_prompt_text(shot.get("cause"))
        effect = clean_prompt_text(shot.get("effect"))
        shot_parts = [f"[Shot {index}]"]
        if index == 1:
            shot_parts.extend(part for part in opening if part)
        elif time:
            shot_parts.append(f"At {_format_h3_cut_time(time)}, the camera cuts to")
        if cause:
            shot_parts.append(f"Because {cause},")
        if action:
            shot_parts.append(f"{action}.")
        if camera:
            shot_parts.append(f"{camera}.")
        if state_change:
            shot_parts.append(f"The visible result is {state_change}.")
        if effect:
            shot_parts.append(f"This leads into the next beat: {effect}.")
        prompt_parts.append(" ".join(shot_parts))
    if not prompt_parts:
        prompt_parts.append("[Shot 1] " + " ".join(part for part in opening if part))
    mode = render_mode or ("i2va" if prior_frame else "t2va")
    soundscape = overall_soundscape or "Sound follows the scene's visible physical actions and setting."
    return format_h3_video_prompt(
        " ".join(prompt_parts),
        mode=mode,
        duration_seconds=duration_seconds,
        overall_soundscape=soundscape,
        non_diegetic_music=non_diegetic_music,
    )


def _format_h3_cut_time(value: str) -> str:
    """Format a shot range's start as the official H3 cut timestamp."""
    match = re.search(r"(-?\d+(?:\.\d+)?)", str(value or ""))
    seconds = max(0.0, float(match.group(1))) if match else 0.0
    minutes = int(seconds // 60)
    remainder = seconds - (minutes * 60)
    return f"{minutes:02d}:{remainder:06.3f}"
