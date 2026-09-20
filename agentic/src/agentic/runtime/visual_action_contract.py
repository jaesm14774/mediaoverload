from __future__ import annotations

from typing import Any


MOTION_MEDIA_TYPES = frozenset(
    {
        "text2video",
        "text2img2video",
        "image_to_video",
        "image_to_video_audio",
        "animated_sticker",
        "long_video",
        "text2longvideo",
        "native_h3_story",
        "native_h3_t2v_story",
        "native_h3_fl2va_story",
        "native_h3_l2va_story",
        "native_h3_ref2va",
        "text2image2native_h3_ref2va",
        "game_sprite",
        # Native H3 callers also pass these canonical render-mode aliases.
        "text_to_video",
        "first_last_to_video",
        "first_last_frame_to_video",
        "last_frame_to_video",
        "reference_to_video",
    }
)

ACTION_BEATS = (
    "hook",
    "mechanism",
    "consequence",
    "reaction",
    "payoff",
)

OPENING_ACTION_LOCK = (
    "Opening action lock: begin with the protagonist already initiating the dominant causal action, "
    "with the action surface, force direction, or attachment point visibly readable. If contact causes "
    "the consequence, show actual contact rather than reaching toward, hovering near, or merely facing "
    "the mechanism."
)


def is_motion_media_type(media_type: str | None) -> bool:
    return str(media_type or "").strip().lower() in MOTION_MEDIA_TYPES


def _subject_rule(subject_count: int) -> str:
    if int(subject_count or 1) <= 1:
        return "one unmistakable protagonist"
    return "the declared subject slots with stable individual identities"


def visual_action_contract(
    duration_seconds: int | float,
    *,
    media_type: str | None = None,
    subject_count: int = 1,
    segment: bool = False,
    loop: bool = False,
) -> str:
    """Return the shared causal-motion contract for every animated route.

    The contract is provider-neutral. Routes compile it differently: a short
    I2V clip compresses the beats, long-video applies them per segment, Native
    H3 maps them to storyboard shots, and game sprites close the loop inside
    the sampled action graph.
    """

    normalized_type = str(media_type or "").strip().lower()
    if normalized_type and not is_motion_media_type(normalized_type):
        return ""
    duration = max(1, int(duration_seconds or 0))
    subjects = _subject_rule(subject_count)

    if normalized_type == "game_sprite":
        ending = (
            "Echo the opening direction and energy for a clean periodic loop."
            if loop
            else "End on the earned stable outcome without starting a second action."
        )
        return (
            f"Game-sprite action contract for this {duration}-second clip: keep {subjects} readable in a fixed asset framing; "
            "choreograph one continuous physical action through anticipation, launch or contact, consequence, readable body or "
            "spatial change, recoil or reaction, and a settled pose. Preserve the same silhouette, palette, and background in "
            f"every sampled frame. {ending} Do not spend the clip on scenery, camera travel, pose resets, or identity-changing morphs."
        )

    if normalized_type in {"long_video", "text2longvideo"} or segment:
        return (
            f"Segment action contract for this {duration}-second story segment: keep {subjects} on one continuous objective; "
            "start with a visible hook or active state, perform one concrete action, show the physical cause and resulting state "
            "change, make the reaction or response readable, and hand off a settled end state to the next segment. Keep one "
            "dominant mechanism and preserve world, identity, spatial direction, and material continuity. Do not fill time with "
            "standing, watching, unrelated props, disconnected spectacle, or an establishing shot without action."
        )

    if duration <= 9:
        return (
            f"Short causal-action contract for this {duration}-second clip: use {subjects}, one dominant physical mechanism, and "
            "one visible objective. The first frame must already contain the hook and action onset. Compress one readable chain "
            "of mechanism, consequence, reaction, and settled payoff; every change must be visibly caused by the preceding action. "
            "If contact drives the mechanism, keep the contact point visible through the consequence. Use foreground, midground, "
            "and background depth when it strengthens the same action. End with a clear result that can echo the opening without "
            "hiding the payoff. Do not add a second plot, unrelated prop, extra protagonist, static pose, or montage."
        )

    return (
        f"Causal-motion contract for this {duration}-second clip: keep {subjects} and one dominant objective legible from the "
        "opening action through the ending. Organize the motion as hook, mechanism, consequence, reaction, and payoff; each beat "
        "must create a visible state change and hand off causally to the next. Use designed spatial depth and concrete material "
        "responses instead of generic camera motion. Preserve identity, geography, and the mechanism across shots; do not add "
        "unrelated subplots, filler poses, or disconnected spectacle."
    )


def default_visual_action_contract(
    duration_seconds: int | float = 6,
    *,
    media_type: str = "text2img2video",
    subject_count: int = 1,
    loop: bool = False,
) -> dict[str, Any]:
    """Return the inspectable contract carried in a goal and run manifest."""

    normalized_type = str(media_type or "").strip().lower()
    if normalized_type and not is_motion_media_type(normalized_type):
        return {}
    route_shape = "sprite_loop" if normalized_type == "game_sprite" else (
        "story_segment"
        if normalized_type in {"long_video", "text2longvideo"}
        else "continuous_clip"
    )
    return {
        "version": 1,
        "media_type": normalized_type,
        "duration_seconds": max(1, int(duration_seconds or 0)),
        "subject_count": max(1, int(subject_count or 1)),
        "beat_order": list(ACTION_BEATS),
        "route_shape": route_shape,
        "loop": bool(loop),
        "requires": [
            "visible_hook",
            "caused_state_change",
            "readable_reaction",
            "settled_payoff",
            "continuity_anchor",
        ],
    }


def enforce_opening_action_lock(prompt: str) -> str:
    """Prevent an I2V opening still from depicting a near-miss setup."""

    text = str(prompt or "").strip()
    if OPENING_ACTION_LOCK.casefold() in text.casefold():
        return text
    if not text:
        return OPENING_ACTION_LOCK
    return f"{text.rstrip()} {OPENING_ACTION_LOCK}"
