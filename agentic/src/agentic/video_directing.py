from __future__ import annotations


VIDEO_MOTION_DIRECTION = (
    "Honor an explicitly still or solemn brief. Otherwise make the protagonist's readable physical action, "
    "rather than camera drift, the leading event. Let gentle color, light, and painterly surfaces stay gentle "
    "while the character moves with clear intent. For a causal story, keep one visible goal, one obstacle, and one "
    "turn that changes the outcome; each beat changes a visible state and causes the next, and the ending resolves "
    "or reframes the opening. If the brief requests one action, do not invent a second problem or subplot. Establish "
    "a visible goal or obstacle and a loaded pose, "
    "begin the action in the first tenth, then show cause, travel or a decisive pose change, contact or "
    "manipulation, and a visible result. Let that result change the composition and prompt an immediate "
    "reaction consistent with the character and source stakes. For character-led scenes, shape a compact "
    "performance arc: anticipation in the eyes and body, energetic action with follow-through, then a brief "
    "charming reaction such as a surprised blink, delighted glance, or proud bounce when it fits. When a "
    "playful or tender tone fits, let that reaction leave a warm, satisfying afterbeat. Give the "
    "face a readable expression and let the silhouette change. Use a light squash, recoil, wobble, limb gesture, "
    "or accessory lag only when the character and medium support it. Add one or two small secondary motions "
    "tied to the action, such as a rolling prop or nearby leaves responding; these reinforce the protagonist "
    "rather than replace its movement. Vary the rhythm with a quick readable burst, a short breath for the "
    "reaction, and a satisfying payoff; keep visible action or pose progression through most of the shot and "
    "reserve the final tenth for a brief resolved pose. For orb-shaped characters, use a clear full-body path, "
    "pitch, roll, contact, recoil, or turn. For each shot, state the starting framing, one motivated camera move "
    "or intentional lock, its direction and relation to the subject's path, what it reveals, and the ending "
    "framing. A pan or tilt rotates from a fixed position; a dolly, truck, or arc travels through space and creates "
    "parallax; a follow or tracking move keeps pace with the subject; a zoom changes focal length. Keep the "
    "one continuous camera move or a locked shot per shot; state a new shot if the camera needs a second operation. "
    "A pan or tilt changes where the camera points, not subject distance or image scale. A change in subject scale "
    "must come from a matching camera move, zoom, or physical subject travel, and the ending framing must follow "
    "from that move. Keep the character's expressive face and the visible result together in the payoff when both "
    "are story-critical; if the final beat is a reaction close-up, show the result first and preserve enough context "
    "to understand it. Keep the face and every story-critical object within frame with clear edge margins during and "
    "after the move. If a close-up would crop the reaction or result, stop wider or cut to a separate shot. Keep the "
    "character's face, goal, and travel lane easy to read, with the protagonist large "
    "enough to carry the emotion. "
    "Let the camera support the action rather than compete with it. Describe the selected style with stable visible "
    "rules for medium and surface, palette and light, and motion or transitions; keep them consistent across shots. "
    "Use a medium-native reveal only when it strengthens the story turn. Ground goals, props, and consequences in "
    "the brief, source, and character identity."
)

VIDEO_MOTION_DIRECTION_MARKER = "Video action and camera direction:"
VIDEO_OPENING_FRAME_DIRECTION = (
    "When the brief requests a contemplative still composition, preserve it. For a character-led action video, "
    "depict the split second before movement begins with visible anticipation. Aim the protagonist's eyes and a "
    "loaded, angled pose toward a visible goal or obstacle in the same scene; use a small, character-appropriate "
    "cue such as a lean, raised foot, or puffed cheek. Place the goal at a readable distance and reserve a clear "
    "travel lane through the frame. Keep the face, silhouette, and goal easy to see. Use a strong diagonal between "
    "the protagonist and the goal, with foreground and background depth that supports the coming camera move. "
    "Build the frame from objects in the brief and source, and keep every source detail factual."
)
VIDEO_OPENING_FRAME_DIRECTION_MARKER = "Opening-frame action staging:"
VIDEO_SEGMENT_DIRECTION_FIELDS = (
    ("action", "Action"),
    ("cause", "Physical cause"),
    ("effect", "Visible result"),
    ("camera", "Camera"),
)


def format_segment_direction(segment: dict[str, object]) -> str:
    lines = [
        f"{label}: {' '.join(str(segment.get(key) or '').split())}"
        for key, label in VIDEO_SEGMENT_DIRECTION_FIELDS
        if " ".join(str(segment.get(key) or "").split())
    ]
    return "Shot-specific directing notes:\n" + "\n".join(lines) if lines else ""


def apply_segment_video_direction(
    prompt: str,
    segment: dict[str, object],
    media_type: str,
) -> str:
    prompt_text = apply_video_motion_direction(prompt, media_type)
    if not requires_video_motion_direction(media_type):
        return prompt_text
    notes = format_segment_direction(segment)
    if notes and "shot-specific directing notes:" not in prompt_text.casefold():
        prompt_text = f"{prompt_text}\n\n{notes}"
    return prompt_text


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
