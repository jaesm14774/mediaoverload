from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


SHORT_FORM_VIDEO_TYPES = frozenset(
    {
        "text2video",
        "text2img2video",
        "image_to_video",
        "animated_sticker",
    }
)


def clean_prompt_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip().rstrip(".")


def short_action_contract(
    duration_seconds: int | float,
    *,
    media_type: str | None = None,
    subject_count: int = 1,
) -> str:
    """Return the shared, topic-neutral contract for a short causal clip."""
    duration = max(1, int(duration_seconds or 0))
    if duration > 6:
        return ""
    if media_type and media_type not in SHORT_FORM_VIDEO_TYPES:
        return ""
    subject_rule = (
        "keep one protagonist"
        if int(subject_count or 1) <= 1
        else "keep the declared subject slots together with stable individual identities"
    )
    return (
        f"Short-action contract for this {duration}-second clip: use one clear physical action only; {subject_rule}, one dominant physical "
        "mechanism, and one clear objective. Show a readable start state immediately, then anticipation or an "
        "attempt, one decisive cause-and-effect change, a visible reaction or deformation, and a completed end state "
        "with payoff before the end. Tie each camera move to the physical change it reveals. End in a settled visual "
        "state that echoes the opening enough to feel loopable, without hiding the payoff. Do not add a second "
        "plot, unrelated prop, extra protagonist, static pose, abstract action, or montage."
    )


def subject_identity_lock(
    character: str,
    subject_context: Mapping[str, Any] | None = None,
) -> str:
    context = dict(subject_context or {})
    subjects = [item for item in (context.get("subjects") or []) if isinstance(item, Mapping)]
    if context.get("interaction_contract", {}).get("required") and len(subjects) == 2:
        descriptions: list[str] = []
        for item in subjects:
            name = clean_prompt_text(item.get("name")) or "the subject"
            profile = dict(item.get("profile") or {})
            role_description = clean_prompt_text(profile.get("role_description"))
            keywords = clean_prompt_text(profile.get("keywords"))
            details = "; ".join(
                part
                for part in (
                    f"canonical role description: {role_description}" if role_description else "",
                    f"supplemental visual keywords: {keywords}" if keywords else "",
                )
                if part
            )
            descriptions.append(f"{name}{f' ({details})' if details else ''}")
        return (
            "Two required subject slots share one continuous scene: "
            + "; ".join(descriptions)
            + ". The slots may use the same name; preserve each slot separately with its own identity, proportions, costume, silhouette, palette, and readable spatial relationship"
        )
    subject = clean_prompt_text(character) or "the main subject"
    profile = dict(context.get("character_profile") or {})
    if not profile and subjects:
        profile = dict(subjects[0].get("profile") or {})
    role_description = clean_prompt_text(profile.get("role_description"))
    keywords = clean_prompt_text(profile.get("keywords"))
    if role_description:
        supplemental = f" Supplemental visual keywords: {keywords}." if keywords else ""
        return (
            f"Canonical character identity: {subject} — {role_description}."
            f" Preserve this role description's anatomy, silhouette, proportions, costume, and palette exactly;"
            f" do not invent or add conflicting features.{supplemental}"
        )
    if keywords:
        return (
            f"Canonical character identity: {subject}. Supplemental visual keywords: {keywords}."
            " Preserve stable identity, proportions, costume, silhouette, and palette."
        )
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
    subject_context: Mapping[str, Any] | None = None,
) -> str:
    """Compose the local H3 prompt in MiniMax's documented multimodal shape."""
    spine = dict(story_spine or {})
    context = dict(subject_context or {})
    interaction_required = bool(context.get("interaction_contract", {}).get("required", False))
    subject_count = len([item for item in (context.get("subjects") or []) if isinstance(item, Mapping)]) or 1
    identity = subject_identity_lock(character, context)
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
            (
                "Input relation: direct text-to-video; establish both required subject identities in the first moving action, "
                "not in a static character sheet"
                if interaction_required
                else "Input relation: direct text-to-video; establish the protagonist's identity in the first moving action, "
                "not in a static character sheet"
            )
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
            "non_diegetic_music: restrained motif rises with obstacle, tightens at reversal, resolves with payoff.",
            "Motion contract: visible motion starts in the first half-second; every shot changes composition, action, and mission state; preserve physical cause and effect.",
            short_action_contract(duration_seconds, subject_count=subject_count)
            or "Cute gag: one prop causes reaction/payoff; causal deformation; loop the opening.",
            (
                "Continuity gate: keep both required subject slots, their identities, shared world, dominant prop, and news mechanism across shots; "
                "show the interaction causing the payoff; no identity swap, unrequested third subject, new room, device, spectacle, or generic substitute."
                if interaction_required
                else "Continuity gate: keep one protagonist, world, dominant prop, and news mechanism across shots; show that mechanism causing the payoff; no new room, device, character, spectacle, or generic substitute."
            ),
            (
                "Visual guardrails: stylized cinematic animation, clear silhouettes, readable foreground interaction, no readable text, logos, subtitles, watermark, "
                "identity swap, unrequested third subject, frozen pose, or unrelated spectacle."
                if interaction_required
                else "Visual guardrails: stylized cinematic animation, clear silhouette, readable foreground action, no readable text, logos, subtitles, watermark, duplicate protagonist, frozen pose, or unrelated spectacle."
            ),
        ]
    )
    rules = [clean_prompt_text(item) for item in continuity_rules if clean_prompt_text(item)]
    if rules:
        prompt_parts.append("Additional continuity rules: " + "; ".join(rules) + ".")
    return "\n".join(prompt_parts)
