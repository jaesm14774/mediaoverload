from __future__ import annotations


VIDEO_MOTION_DIRECTION = (
    "Honor an explicitly still or solemn brief. Otherwise make the clip a lively, charming, readable loop. "
    "The camera makes at most one slow gentle push-in of about 15 percent toward the character, with no pan, orbit, or cut, "
    "and the character and the visible result stay fully inside the frame. "
    "The illustrated look of the frame stays exactly the same, with the same colors, line quality, and painted surfaces; "
    "nothing morphs and no new objects appear. The character is in constant, energetic motion for the whole clip with no "
    "still pause: anticipation with a squat or wind-up, a big action with stretch and squash, then the same action chained "
    "into two or three connected moves with the expression changing from determined to delighted. Props and the "
    "scene's existing elements, such as ripples, rain, leaves, or a swaying umbrella, respond on every beat. The character "
    "is still moving in the last frame. The action stays inside the visible scene: the character bounces, hops, or "
    "moves around within the frame and never travels to a new place that the opening frame does not show. "
    "Do not add a second goal, obstacle, or subplot. "
    "Ground the action in the brief and the character identity."
)

VIDEO_MOTION_DIRECTION_MARKER = "Video action and camera direction:"
VIDEO_OPENING_FRAME_DIRECTION = (
    "When the brief requests a contemplative still composition, preserve it. For a character-led video, depict the split "
    "second before the action begins: the character small and clearly visible in its world, the prop or place "
    "the action involves nearby, and open space around it so nothing crops as it jumps and moves. Build the frame from "
    "objects in the brief and source, and keep every source detail factual."
)
VIDEO_OPENING_FRAME_DIRECTION_MARKER = "Opening-frame action staging:"
VIDEO_SEGMENT_DIRECTION_FIELDS = (
    ("action", "Action"),
    ("cause", "Physical cause"),
    ("effect", "Visible result"),
    ("camera", "Camera"),
)


def requires_video_motion_direction(media_type: str) -> bool:
    normalized = str(media_type or "").strip().casefold()
    return (
        "video" in normalized
        or normalized.startswith("native_h3_")
        or normalized in {"i2v", "t2v", "fl2va", "l2va", "animated_sticker"}
    )


def apply_video_motion_direction(prompt: str, media_type: str) -> str:
    return _prioritize_direction(
        prompt,
        media_type,
        VIDEO_MOTION_DIRECTION_MARKER,
        VIDEO_MOTION_DIRECTION,
    )


def apply_video_opening_frame_direction(prompt: str, media_type: str) -> str:
    return _prioritize_direction(
        prompt,
        media_type,
        VIDEO_OPENING_FRAME_DIRECTION_MARKER,
        VIDEO_OPENING_FRAME_DIRECTION,
    )


def _prioritize_direction(prompt: str, media_type: str, marker: str, direction: str) -> str:
    prompt_text = str(prompt or "").strip()
    if not requires_video_motion_direction(media_type):
        return prompt_text
    clause = f"{marker} {direction}"
    folded_clause = clause.casefold()
    folded_prompt = prompt_text.casefold()
    start = folded_prompt.find(folded_clause)
    while start >= 0:
        prompt_text = f"{prompt_text[:start]} {prompt_text[start + len(clause):]}".strip()
        folded_prompt = prompt_text.casefold()
        start = folded_prompt.find(folded_clause)
    return f"{clause}\n\n{prompt_text}" if prompt_text else clause
