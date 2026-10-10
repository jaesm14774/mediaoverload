from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from agentic.minimax_prompting import compose_minimax_h3_prompt

try:
    import yaml
except ImportError:  # pragma: no cover - requirements.txt includes PyYAML
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[3]


class StoryboardError(ValueError):
    """Raised when a reusable story preset cannot produce a valid story plan."""


_NATIVE_STORY_STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "into", "while", "before", "after",
    "then", "than", "for", "its", "their", "they", "them", "only", "must", "will",
    "not", "one", "same", "story", "scene", "shot", "kirby", "protagonist", "original",
    "danger", "problem", "thing", "place", "becomes", "become", "begins", "begin",
}

def _native_story_terms(value: Any) -> set[str]:
    tokens = re.findall(r"[a-z][a-z0-9'-]{2,}|[\u4e00-\u9fff]{2,}", str(value or "").lower())
    return {token for token in tokens if token not in _NATIVE_STORY_STOPWORDS}


def native_surface_variation(creative_brief: str) -> str:
    """Pass the user's creative direction through without content filtering."""
    return re.sub(r"\s+", " ", str(creative_brief or "")).strip()


def resolve_storyboard_path(path_or_name: str | Path) -> Path:
    raw = str(path_or_name).strip()
    if not raw:
        raise StoryboardError("storyboard path or preset name cannot be empty")
    path = Path(raw).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, REPO_ROOT / path]
    if not path.suffix:
        candidates.extend(
            [
                REPO_ROOT / "configs" / "storyboards" / f"{raw}.yaml",
                REPO_ROOT / "configs" / "storyboards" / f"{raw}.yml",
                REPO_ROOT / "configs" / "storyboards" / f"{raw}.json",
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise StoryboardError(f"Storyboard preset not found: {path_or_name}")


def load_storyboard(path_or_name: str | Path) -> dict[str, Any]:
    path = resolve_storyboard_path(path_or_name)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        if yaml is None:
            raise StoryboardError("PyYAML is required to load YAML storyboard presets")
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise StoryboardError(f"Storyboard root must be a mapping: {path}")
    payload["_path"] = str(path)
    return payload


def merge_native_h3_storyboard(
    base_storyboard: dict[str, Any],
    generated_story: dict[str, Any],
) -> dict[str, Any]:
    """Apply a flexible LLM story to the reusable character rules.

    The base preset supplies renderer settings and character identity. Model
    shot count and descriptive fields stay flexible; the application assigns
    valid render timing and fills optional presentation details.
    """
    if not isinstance(base_storyboard, dict):
        raise StoryboardError("Native H3 base storyboard must be a mapping")
    if not isinstance(generated_story, dict):
        raise StoryboardError("Native H3 generated story must be a mapping")
    shots = generated_story.get("native_shots")
    if not isinstance(shots, list):
        shots = generated_story.get("shots") or generated_story.get("beats")
    if not isinstance(shots, list):
        shots = generated_story.get("storyboard")
    if not isinstance(shots, list):
        shots = []

    def first_text(mapping: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = str(mapping.get(key) or "").strip()
            if value:
                return value
        return ""

    duration_seconds = float(
        base_storyboard.get("native_duration_seconds")
        or 15
    )
    generated_description = first_text(
        generated_story,
        "base_prompt",
        "visual_anchor",
        "summary",
        "description",
        "creative_brief",
        "title",
        "name",
    ) or str(base_storyboard.get("base_prompt") or "").strip()
    renderable_shots = [
        dict(raw_shot) if isinstance(raw_shot, dict) else {"action": str(raw_shot).strip()}
        for raw_shot in shots
        if isinstance(raw_shot, (dict, str))
    ]
    base_shots = base_storyboard.get("native_shots")
    base_shots = base_shots if isinstance(base_shots, list) else []
    if not renderable_shots:
        renderable_shots = [{"action": generated_description}]

    boundaries = [duration_seconds * index / len(renderable_shots) for index in range(len(renderable_shots) + 1)]
    render_times = [f"{start:g}-{end:g}s" for start, end in zip(boundaries, boundaries[1:])]

    normalized_shots: list[dict[str, Any]] = []
    for index, raw_shot in enumerate(renderable_shots):
        base_shot = base_shots[index] if index < len(base_shots) and isinstance(base_shots[index], dict) else {}
        action = first_text(
            raw_shot,
            "action",
            "visible_action",
            "subject_action",
            "primary_action",
            "visual_action",
            "visual",
            "description",
            "shot_description",
        )
        if not action:
            action = first_text(
                base_shot,
                "action",
                "visible_action",
                "subject_action",
                "primary_action",
                "visual_action",
                "visual",
                "description",
            )
        if not action:
            action = first_text(
                generated_story,
                "base_prompt",
                "visual_anchor",
                "summary",
                "description",
                "creative_brief",
            ) or str(base_storyboard.get("base_prompt") or "").strip()
        title = first_text(raw_shot, "title", "name", "beat") or first_text(base_shot, "title", "name", "beat")
        camera = first_text(raw_shot, "camera", "camera_direction", "camera_movement", "shot") or first_text(
            base_shot, "camera", "camera_direction", "camera_movement", "shot"
        )
        cause = first_text(raw_shot, "cause", "physical_cause") or first_text(base_shot, "cause", "physical_cause")
        effect = first_text(raw_shot, "effect", "physical_effect") or first_text(base_shot, "effect", "physical_effect")
        state_change = first_text(
            raw_shot,
            "state_change",
            "visible_effect_or_state_change",
            "visible_effect",
            "end_state",
            "effect",
            "result",
        )
        if not state_change:
            state_change = first_text(
                base_shot,
                "state_change",
                "visible_effect_or_state_change",
                "visible_effect",
                "end_state",
                "effect",
                "result",
            )
        normalized = {**raw_shot, "time": render_times[index]}
        for key, value in (("title", title), ("action", action), ("camera", camera), ("state_change", state_change)):
            if value:
                normalized[key] = value
        for key, value in (("cause", cause), ("effect", effect)):
            if value:
                normalized[key] = value
        normalized_shots.append(normalized)

    raw_world = generated_story.get("world")
    generated_world = raw_world if isinstance(raw_world, dict) else {}
    base_world = dict(base_storyboard.get("world") or {})
    world = {
        **base_world,
        **deepcopy(generated_world),
    }
    for key in ("setting", "visual_language"):
        if str(world.get(key) or "").strip().lower().startswith("generated "):
            world[key] = ""
    generated_rules = generated_world.get("continuity_rules")
    if not isinstance(generated_rules, list) or not generated_rules:
        generated_rules = base_world.get("continuity_rules") or []
    world["continuity_rules"] = [str(rule).strip() for rule in generated_rules if str(rule).strip()]

    character = str(base_storyboard.get("character") or "the protagonist").strip()

    def clean_runtime_placeholder(value: Any) -> str:
        text = str(value or "").strip()
        return "" if text.lower().startswith("generated ") else text

    base_spine = {
        key: clean_runtime_placeholder(value)
        for key, value in dict(base_storyboard.get("story_spine") or {}).items()
    }
    generated_spine = generated_story.get("story_spine")
    if isinstance(generated_spine, dict):
        generated_spine = dict(generated_spine)
        if not clean_runtime_placeholder(generated_spine.get("stakes")):
            generated_spine["stakes"] = generated_spine.get("character_scale_stakes") or ""
        if not clean_runtime_placeholder(generated_spine.get("resolution")):
            generated_spine["resolution"] = generated_spine.get("visible_resolution") or ""
    spine = {
        **base_spine,
        **(
            {
                key: clean_runtime_placeholder(value)
                for key, value in generated_spine.items()
                if clean_runtime_placeholder(value)
            }
            if isinstance(generated_spine, dict)
            else {}
        ),
    }
    if normalized_shots:
        last_index = len(normalized_shots) - 1
        stakes_index = min(1, last_index)
        climax_index = min(last_index, max(stakes_index + 1, len(normalized_shots) - 2))
        premise_shot = normalized_shots[0]
        stakes_shot = normalized_shots[stakes_index]
        climax_shot = normalized_shots[climax_index]
        resolution_shot = normalized_shots[-1]
        shot_fallbacks = {
            "premise": first_text(premise_shot, "action", "state_change"),
            "stakes": first_text(stakes_shot, "state_change", "action"),
            "climax": first_text(climax_shot, "action", "state_change"),
            "resolution": first_text(resolution_shot, "state_change", "action"),
        }
        explicit_spine = generated_spine if isinstance(generated_spine, dict) else {}
        for key, fallback in shot_fallbacks.items():
            if not clean_runtime_placeholder(explicit_spine.get(key)) and fallback:
                spine[key] = fallback
    generated_base_prompt = first_text(generated_story, "base_prompt", "visual_anchor")
    base_prompt = generated_base_prompt or str(base_storyboard.get("base_prompt") or "").strip()
    style_description = first_text(generated_story, "style_description", "visual_style")
    if not generated_base_prompt and style_description:
        base_prompt = f"{base_prompt} {style_description}".strip()
    opening_prompt = first_text(
        generated_story,
        "opening_keyframe_prompt",
        "opening_prompt",
        "first_frame_prompt",
    )
    if not opening_prompt:
        opening_prompt = base_prompt
    ending_prompt = first_text(
        generated_story,
        "ending_keyframe_prompt",
        "ending_prompt",
        "last_frame_prompt",
    )
    if not ending_prompt:
        ending_prompt = base_prompt
    negative_prompt = str(base_storyboard.get("negative_prompt") or "").strip()
    native_audio = generated_story.get("native_audio")
    if isinstance(native_audio, dict):
        native_audio = " ".join(
            f"{label}: {str(native_audio.get(key) or '').strip()}"
            for key, label in (("overall_soundscape", "Overall soundscape"), ("non_diegetic_music", "Non-diegetic music"))
            if str(native_audio.get(key) or "").strip()
        )
    native_audio = str(native_audio or "").strip()
    if not native_audio:
        native_audio = "; ".join(
            first_text(shot, "audio_direction", "audio")
            for shot in shots
            if isinstance(shot, dict) and first_text(shot, "audio_direction", "audio")
        )

    gag_card = generated_story.get("gag_card")
    news_trace = generated_story.get("news_trace")
    if isinstance(news_trace, dict):
        news_trace = deepcopy(news_trace)
        if not str(news_trace.get("news_consequence") or "").strip():
            news_trace["news_consequence"] = first_text(
                news_trace,
                "source_consequence",
                "consequence",
                "impact",
            )
    merged = deepcopy(base_storyboard)
    merged.update(
        {
            "name": first_text(generated_story, "name", "title", "story_name") or f"{character} native H3 story",
            "base_prompt": base_prompt,
            "opening_keyframe_prompt": opening_prompt,
            "ending_keyframe_prompt": ending_prompt,
            "negative_prompt": negative_prompt,
            **({"news_trace": deepcopy(news_trace)} if isinstance(news_trace, dict) else {}),
            **({"gag_card": deepcopy(gag_card)} if isinstance(gag_card, dict) else {}),
            "story_spine": spine,
            "world": world,
            "native_audio": native_audio,
            "native_shots": normalized_shots,
            "story_source": "news_llm",
        }
    )
    return merged


def format_native_h3_prompt(
    storyboard: dict[str, Any],
    *,
    style: str = "cinematic 2D anime",
    creative_brief: str = "",
    duration_seconds: int | None = None,
) -> str:
    """Build a native H3 prompt from the user's story and optional shot notes.

    H3 can model a longer continuous clip, so this deliberately avoids repeating
    long-video segment fields. The formatter turns available shot notes into
    one prompt for the automated runner and manifests.
    """
    shots = storyboard.get("native_shots")
    if not isinstance(shots, list):
        shots = []
    world = dict(storyboard.get("world") or {})
    character = str(storyboard.get("character") or "the protagonist").strip()
    duration = int(duration_seconds or storyboard.get("native_duration_seconds") or 15)
    render_mode = str(storyboard.get("render_mode") or "").strip()
    shot_payloads: list[dict[str, Any]] = []
    for index, shot in enumerate(shots, start=1):
        shot_payloads.append(dict(shot) if isinstance(shot, dict) else {"action": str(shot or "")})
    audio = str(storyboard.get("native_audio") or "").strip()
    prompt = compose_minimax_h3_prompt(
        duration_seconds=duration,
        character=character,
        style=style,
        setting=str(world.get("setting") or ""),
        visual_language=str(world.get("visual_language") or ""),
        shots=shot_payloads,
        audio=audio,
        render_mode=render_mode,
        subject_context=dict(storyboard.get("subject_context") or {}),
        media_type=render_mode or "native_h3_story",
    )
    return prompt.strip()


def format_native_h3_keyframe_prompt(
    storyboard: dict[str, Any],
    *,
    style: str = "",
    frame: str = "opening",
) -> str:
    """Describe one selected story beat as one source-grounded still image."""
    normalized_frame = str(frame or "opening").strip().casefold()
    if normalized_frame not in {"opening", "ending"}:
        raise ValueError("Native H3 keyframe frame must be 'opening' or 'ending'.")
    shots = storyboard.get("native_shots")
    if not isinstance(shots, list) or not shots:
        raise StoryboardError("Native H3 keyframe prompts require at least one story shot.")
    shot = shots[0] if normalized_frame == "opening" else shots[-1]
    if not isinstance(shot, dict):
        shot = {"action": str(shot or "")}

    world = dict(storyboard.get("world") or {})
    context = dict(storyboard.get("subject_context") or {})
    raw_news_trace = storyboard.get("news_trace")
    news_trace = dict(raw_news_trace) if isinstance(raw_news_trace, dict) else {}
    news_category = str(news_trace.get("source_category") or "").strip().casefold()
    is_product_news = news_category == "product_or_service"
    is_human_interest = news_category == "human_interest"
    fit_assessment = news_trace.get("fit_assessment")
    fit_assessment = dict(fit_assessment) if isinstance(fit_assessment, dict) else {}
    emotion = str(fit_assessment.get("best_fitting_emotion") or "").strip().casefold()
    if not emotion and isinstance(storyboard.get("gag_card"), dict):
        emotion = "comedy"
    subjects = [
        str(subject.get("name") or "").strip()
        for subject in context.get("subjects", [])
        if isinstance(subject, dict) and str(subject.get("name") or "").strip()
    ]
    character = str(storyboard.get("character") or "").strip()
    if character.casefold() in {"selected protagonist", "the protagonist", "protagonist"} and subjects:
        character = ", ".join(subjects)
    if not character and subjects:
        character = ", ".join(subjects)
    mood_subjects = subjects or [name.strip() for name in character.split(",") if name.strip()]
    kirby_name = next((name for name in mood_subjects if "kirby" in name.casefold()), "")
    waddle_name = next((name for name in mood_subjects if "waddle dee" in name.casefold()), "")

    parts = [f"Draw one storybook still at the exact {normalized_frame} instant: one scene, one readable moment."]
    ending_pose = ""
    if normalized_frame == "ending":
        ending_pose = _format_held_keyframe_reaction(str(shot.get("state_change") or ""))
        if not ending_pose:
            ending_pose = _format_held_keyframe_reaction(str(shot.get("action") or ""))
    source_context_values = [
        news_trace.get(key)
        for key in (
            "source_title",
            "source_fact",
            "source_category",
            "source_concepts",
            "news_mechanism",
            "news_consequence",
        )
    ]
    source_context_text = " ".join(
        " ".join(str(part) for part in value if str(part).strip())
        if isinstance(value, list)
        else str(value or "")
        for value in source_context_values
    ).casefold()
    source_action_sequence = news_trace.get("source_action_sequence")
    repeated_presence_context = " ".join(
        str(item) for item in source_action_sequence
    ).casefold() if isinstance(source_action_sequence, list) else ""
    repeated_presence_terms = (
        "repeat", "return", "again", "revisit", "ongoing", "over time", "long-term", "trust",
        "多次", "反覆", "持續", "長期", "逐漸", "信任", "回訪", "一次又一次",
    )
    frame_text = (
        " ".join(str(shot.get(key) or "") for key in ("action", "physical_cause", "state_change")).casefold()
        if normalized_frame == "opening"
        else ending_pose.casefold()
    )
    visual_translation = " ".join(str(news_trace.get("visual_translation") or "").split()).strip()
    raw_anchors = news_trace.get("visual_anchors")
    anchors = raw_anchors if isinstance(raw_anchors, list) else [raw_anchors] if isinstance(raw_anchors, str) else []
    visual_thesis_text = " ".join([visual_translation, *(str(anchor) for anchor in anchors)]).casefold()
    has_route_thesis = bool(
        re.search(
            r"\bfootprints?\b|\bfootpath\b|\bfootsteps?\b|\b(?:path|route|trail|track)\b|腳印|足跡|路徑|小徑",
            visual_thesis_text,
            re.IGNORECASE,
        )
    )
    has_repeated_presence = any(term in repeated_presence_context for term in repeated_presence_terms)
    has_recurring_footprint_story = is_human_interest and has_route_thesis and has_repeated_presence
    if character:
        parts.append(f"Selected characters: {character}.")
    def append_labeled_detail(label: str, value: Any) -> None:
        text = str(value or "").strip()
        if not text:
            return
        if text.endswith(".."):
            text = text.rstrip(".") + "."
        elif not text.endswith((".", "!", "?", "…")):
            text += "."
        parts.append(f"{label}: {text}")

    visual_identities = []
    for subject in context.get("subjects", []):
        if not isinstance(subject, dict):
            continue
        name = str(subject.get("name") or "").strip()
        profile = subject.get("profile")
        profile = dict(profile) if isinstance(profile, dict) else {}
        appearance = str(
            profile.get("visual_description")
            or profile.get("visual_keywords")
            or profile.get("keywords")
            or ""
        ).strip()
        if name and appearance:
            visual_identities.append(f"{name}: {appearance}")
    if not visual_identities:
        profile = context.get("character_profile")
        profile = dict(profile) if isinstance(profile, dict) else {}
        appearance = str(
            profile.get("visual_description")
            or profile.get("visual_keywords")
            or profile.get("keywords")
            or ""
        ).strip()
        if appearance and character:
            visual_identities.append(f"{character}: {appearance}")
    if visual_identities and normalized_frame == "opening":
        append_labeled_detail("Visual identity", "; ".join(visual_identities))

    if normalized_frame == "opening":
        if has_recurring_footprint_story:
            lead = character.split(",", 1)[0].strip() or "the selected character"
            append_labeled_detail(
                "Main action",
                f"{lead} stands motionless, full body, immediately beside one fully closed wooden doorway. "
                "Show the entire door panel, frame, and threshold distinctly in the same composition.",
            )
            append_labeled_detail(
                "Opening reaction",
                "The door remains firmly shut. The near heel-to-toe prints are visible beside the character's feet, then "
                "disappear behind the adjacent stone wall; no continuation, destination, interior, recipient, or response is shown.",
            )
        else:
            append_labeled_detail("Main action", shot.get("action"))
            append_labeled_detail("Opening reaction", shot.get("state_change"))
    else:
        append_labeled_detail("Held final pose", ending_pose)
        is_chain_news = any(term in source_context_text for term in ("chain", "combine", "串聯", "組合", "結合"))
        has_linked_visual = any(
            term in frame_text for term in ("ring", "loop", "link", "chain", "connect", "joined", "環", "圈")
        )
        if is_chain_news and has_linked_visual:
            parts.append(
                "Main visual focus: the complete linked object is large and unmistakable in the lower-center foreground; "
                "keep every ring separately recognizable and countable, with its own open center and outline; overlap only "
                "where the rings are threaded together, never fuse the chain into one oversized ring, donut, or bagel. "
                "The characters' hands frame the result without covering any link."
            )
        if visual_identities:
            append_labeled_detail("Visual identity", "; ".join(visual_identities))
    if normalized_frame == "ending":
        if has_recurring_footprint_story:
            camera_detail = (
                "Camera: same medium-wide side three-quarter axis as the opening; keep the full character, the same now-ajar "
                "door, and the continuous shoeprint route visible with readable spacing; no close crop."
            )
        else:
            camera_detail = (
                "Camera: three-quarter front two-shot; medium-wide; both full figures with space around them and readable faces; "
                "complete task result large and clear in the lower-center foreground; no close crop."
            )
    else:
        planned_camera = " ".join(str(shot.get("camera") or "").split()).strip()
        if has_recurring_footprint_story:
            camera_detail = (
                "Camera: medium-wide side three-quarter view; show the character's complete body and the entire closed door "
                "including its frame and threshold. Keep only the near heel-to-toe print section in view; a foreground stone "
                "wall hides the route's continuation and destination. Do not show the full lane; no close crop."
            )
        else:
            camera_detail = (
                f"Camera: {planned_camera}; keep the selected characters' faces and the source-linked action readable in this composition."
                if planned_camera
                else "Camera: medium-wide two-shot; keep both characters, faces, hands, and the named action inside frame."
            )
    mood_detail = ""
    if emotion == "comedy":
        if kirby_name and waddle_name:
            mood_detail = (
                f"Freeze on the funny reversal: {kirby_name} is unmistakably caught mid-laugh, with lifted cheeks and a large "
                "open oval mouth whose dark interior is clearly visible and unobstructed; show laughter, not only startled eyes. "
                f"{waddle_name} has bright eyes and rosy cheeks, answering with the named comic lean, bounce, or lifted hand. Preserve "
                "Waddle Dee's familiar simple face without added human eyebrows or a forced laugh mouth. "
                "If cheek or shoulder contact is named, show the shared contact seam visibly flattened. Add no new gag or prop."
            )
        else:
            mood_detail = (
                "Make the named punchline obvious through one surprised face with huge bright eyes and a clearly open laugh, "
                "plus the partner's distinct physical reaction. Avoid angry furrowed brows. If cheek or shoulder contact is named, "
                "show the shared contact seam visibly flattened. Add no new gag or prop."
            )
    elif emotion == "sadness":
        if kirby_name and waddle_name:
            mood_detail = (
                f"Show source-honest sadness: {kirby_name} has lowered eyelids and a small downturned mouth; "
                f"{waddle_name} has downcast eyes and a visibly drooping, close-to-the-friend posture. Preserve "
                "Waddle Dee's familiar simple face without adding human eyebrows or forcing a mouth. Keep tears only if the held pose names them; do not force a smile or cheerful recovery."
            )
        else:
            mood_detail = (
                "Show source-honest sadness through downcast eyes, lowered eyelids, a downturned mouth only where it fits the "
                "character's face design, and slumped shoulders. Keep tears only if the held pose names them; do not force a smile or cheerful recovery."
            )
    elif emotion == "tenderness":
        if kirby_name and waddle_name:
            mood_detail = (
                f"Show tenderness through softened eyes and a small content smile on {kirby_name}; {waddle_name} answers with "
                "soft eyes, rosy cheeks, and a gentle lean against the friend. Preserve Waddle Dee's familiar simple face "
                "without adding human eyebrows or forcing a mouth. Keep the named reciprocal gesture; avoid a broad grin or invented prop."
            )
        elif len(mood_subjects) >= 2:
            mood_detail = (
                "Show tenderness through softened attentive expressions and the specific reciprocal gesture already named in "
                "the held pose. Preserve each character's face design and named body positions; do not invent contact, a prop, or a broad grin."
            )
        else:
            mood_detail = (
                "For this solo character, show tenderness only through softened attentive eyes and the named posture. Preserve "
                "its face design; add no second person, unseen reciprocal gesture, contact, prop, or broad grin."
            )
    elif emotion == "curiosity":
        if kirby_name and waddle_name:
            mood_detail = (
                f"Make curiosity readable at a glance: {kirby_name} has wide bright eyes and lifted cheeks. Draw its tiny nose "
                "above a separate, clearly open dark O-mouth roughly as tall as one blue eye; keep visible space between nose and "
                "mouth so the opening reads as an expression, not a dot or smile line. Leave the whole "
                "mouth unobstructed. Preserve the exact body pose named in Held final pose and keep both eye-lines on its named "
                "story event; do not add any contact or gesture beyond the pose named there. "
                f"{waddle_name} shows curiosity through bright attentive eyes and lifted cheeks while "
                "keeping its familiar simple face; include only a mouth detail named in Held final pose, without human eyebrows "
                "or an invented expression."
            )
        elif waddle_name:
            mood_detail = (
                f"Make curiosity readable on {waddle_name} through bright attentive eyes, rosy cheeks, a head tilt, and a "
                "forward lean with lifted hands. Preserve its familiar simple face without adding human eyebrows or a broad grin; "
                "direct its gaze to the named event."
            )
        elif len(mood_subjects) >= 2:
            mood_detail = (
                f"Make curiosity visible on both named faces: {mood_subjects[0]} has wide sparkling eyes and a clearly open "
                f"O-shaped mouth; {mood_subjects[1]} has bright attentive eyes, lifted cheeks, and a distinct forward lean or "
                "small lifted-hand reaction. Avoid angry furrowed brows; direct both gazes to the named event."
            )
        else:
            mood_detail = (
                "Make curiosity visible with wide sparkling eyes, lifted cheeks, and a clearly open O-shaped mouth where it fits "
                "the character's face design; tilt the head or lean toward the named event. Avoid angry furrowed brows."
            )
    elif emotion in {"awe", "concern", "tension", "solemnity", "relief"}:
        mood_detail = {
            "awe": (
                "Keep the character small against a spacious, source-true environment; show wonder through a lifted gaze and "
                "quiet open posture, with scale and light carrying the feeling. Do not invent spectacle or a comic reaction."
            ),
            "concern": (
                "Use a restrained, attentive expression and grounded posture; let the named distance or supportive closeness "
                "show care. Add no panic, forced smile, tears, or unreported person in danger."
            ),
            "tension": (
                "Hold the character in a still, attentive pose with a steady gaze and slight forward lean; use environmental "
                "shadow or negative space to carry uncertainty. Add no frantic action, violence, or comic surprise."
            ),
            "solemnity": (
                "Use a quiet, composed posture and a restrained gaze; let open space, subdued color, and soft directional light "
                "carry the weight. Do not force tears, a smile, or a dramatic collapse."
            ),
            "relief": (
                "Show a small easing of the shoulders and a softened, attentive gaze; keep relief restrained and do not imply "
                "the situation is resolved beyond what the source reports."
            ),
        }[emotion]
    elif "kirby" in character.casefold() or "bandana waddle dee" in character.casefold():
        mood_detail = "Use readable faces and active posture; keep the story's emotion honest and never force a smile."
    append_labeled_detail("Mood", mood_detail)
    if normalized_frame == "ending":
        mood_part = parts.pop()
        selected_characters_index = next(
            (index for index, part in enumerate(parts) if part.startswith("Selected characters:")),
            0,
        )
        parts.insert(selected_characters_index + 1, mood_part)
    parts.append(camera_detail)
    parts.append(
        "Only the selected cast are characters; keep props inanimate. Preserve each character's supplied appearance "
        "and face design; no unlisted hats or accessories."
    )
    world_visual_language = str(world.get("visual_language") or "").strip()
    normalized_style = style.strip()
    if normalized_style:
        append_labeled_detail("Visual style", normalized_style)
        if world_visual_language and world_visual_language.casefold() != normalized_style.casefold():
            append_labeled_detail("Story-specific art direction", world_visual_language)
    elif world_visual_language:
        append_labeled_detail("Visual style", world_visual_language)
    setting = str(world.get("setting") or "").strip()
    if has_recurring_footprint_story:
        doors = (
            "one close, fully visible closed wooden doorway beside a stone wall that turns out of sight"
            if normalized_frame == "opening"
            else "the same close wooden doorway visibly ajar beside the stone wall"
        )
        setting = f"A miniature Changhua rural lane with a solid, pale stone path and {doors}."
    if len(setting) > 120:
        extra_detail = re.search(r"\s+with\s+", setting, re.IGNORECASE)
        if extra_detail:
            setting = setting[:extra_detail.start()].rstrip(" ,;:")
    append_labeled_detail("Setting", setting)
    integration = " ".join(str(news_trace.get("integration") or "").split()).strip()
    source_limit = " ".join(str(news_trace.get("source_limit") or "").split()).strip()
    if visual_translation:
        append_labeled_detail("Source-to-character visual thesis", visual_translation[:360])
    if integration:
        append_labeled_detail("Character's role in the visual thesis", integration[:280])
    if source_limit:
        append_labeled_detail("Fictional-scene boundary", source_limit[:280])
    if source_context_text.strip():
        parts.append(
            "News visual relationship: make the selected news mechanism part of the main scene the character occupies, "
            "acts within, or visibly responds to; do not pose the character beside a separate news icon. Preserve the "
            "reported trigger, actor, and cause in their actual order. Never turn a language or policy trigger into an "
            "invented numeric code, keypad, or manual button press. If the reported change is automatic, show it acting "
            "on its own and the character responding without causing it. Keep the source mechanism and the character's "
            "readable face in the same focal composition. Use one coherent medium, a story-specific palette and camera, "
            "layered depth, and one article-specific physical relationship; let the profile supply texture while the article "
            "sets palette, light, and composition. The central image should encode this article's cause and reveal rather than "
            "merely label its topic. When a platform or default drives a change across many endpoints, show that source-owned "
            "change spreading across a connected environment; the character witnesses it instead of operating a control."
        )
    if is_human_interest:
        parts.append(
            "Human-interest dignity: for stories about disability, illness, caregiving, poverty, or trauma, do not represent "
            "people receiving care as animals, monsters, props, or cute-character stand-ins. Show only source-supported "
            "actions and outcomes; add no invented client behavior, body condition, visit count, or timeline. Let the "
            "character witness the helper's documented action or represent that helper in a clearly fictional analogy."
        )
        if has_repeated_presence:
            parts.append(
                "This source says progress followed recurring presence: do not suggest one visit or a short wait caused "
                "acceptance. Keep the opening outcome unresolved and suggest sustained effort with an uncounted route motif; "
                "show the selected character once, with no recipient or inferred reply."
            )
    character_names = [*subjects, *(name.strip() for name in character.split(",") if name.strip())]
    normalized_character_names = [name.casefold() for name in character_names if name]
    if has_recurring_footprint_story:
        lead = character.split(",", 1)[0].strip() or "the selected character"
        route_detail = (
            "Opening only: show a short visible section of the already accumulated, uncounted route; let it bend behind a "
            "foreground stone wall and disappear before its full length or destination is visible. Keep the selected door "
            "clearly closed so the later camera move has a real reveal."
            if normalized_frame == "opening"
            else "Final beat only: show the continuous uncounted route and one source-supported doorway visibly ajar with a "
            "clear gap at its edge; keep every other doorway closed and show no recipient."
        )
        parts.append(
            f"Visible metaphor: show {lead} once, full body, beside a curving route of clearly shaped warm-ochre heel-and-toe "
            "prints along a simple pale stone lane. Paint the marks on top of the surface as "
            "flat watercolor pigment, with a distinct sole silhouette and thin dark outline; do not make them holes, pits, "
            f"indentations, or shadows. Keep the route readable with the character's face and both feet inside frame. {route_detail} "
            "The marks symbolize sustained work and are not an exact count of visits."
        )
    source_context_known = bool(source_context_text.strip())
    term_aliases = {
        "conversion": "convert",
        "conversions": "convert",
        "converted": "convert",
        "converting": "convert",
        "converts": "convert",
        "transform": "convert",
        "transformed": "convert",
        "transforming": "convert",
        "transformation": "convert",
        "transformations": "convert",
        "transforms": "convert",
    }

    def canonical_terms(value: str) -> set[str]:
        # Treat hyphenated forms like "route-update" as the same concepts as
        # the source's "route update" wording so article anchors can match.
        tokens = (term for term in re.findall(r"\w+", value.casefold()) if len(term) > 2)
        return {
            term_aliases.get(term, term[:-1] if term.endswith("s") and len(term) > 4 else term)
            for term in tokens
        }

    source_terms = canonical_terms(source_context_text)
    non_news_visual_terms = {
        "background", "color", "colors", "colour", "colours", "composition", "detail", "details",
        "expressive", "foreground", "layered", "light", "lighting", "linework", "paper", "palette",
        "pastel", "silhouette", "texture", "visual", "warm", "watercolor", "watercolour",
    }
    visual_candidates: list[tuple[int, int, int, int, str]] = []
    for index, raw_anchor in enumerate(anchors):
        anchor = " ".join(str(raw_anchor or "").split()).strip()
        if not anchor or len(anchor) > 140:
            continue
        if normalized_frame == "ending":
            for marker in ("transforming into", "transforms into", "transformed into", "morphing into", "morphs into", "morphed into", "morph into", "reshaping into", "reshapes into", "reshaped into", "changing into", "changes into", "changed into", "converting into", "converts into", "converted into", "becomes", "become"):
                marker_index = anchor.casefold().rfind(marker)
                if marker_index >= 0:
                    anchor = anchor[marker_index + len(marker):].strip(" \t,;:.-")
                    break
        if any(name in anchor.casefold() for name in normalized_character_names):
            continue
        terms = canonical_terms(anchor)
        if terms and terms.issubset(non_news_visual_terms):
            continue
        source_score = sum(1 for term in terms if term in source_terms)
        if source_context_known and source_score == 0:
            continue
        frame_score = sum(1 for term in terms if term in frame_text)
        repeated_motif_score = int(
            has_recurring_footprint_story
            and re.search(
                r"\bfootprints?\b|\bfootpath\b|\bfootsteps?\b|\b(?:path|route|trail|track)\b|腳印|足跡|路徑|小徑",
                anchor,
                re.IGNORECASE,
            ) is not None
        )
        visual_candidates.append((repeated_motif_score, source_score, frame_score, -index, anchor))
    if visual_candidates:
        _, _, _, _, anchor = max(visual_candidates)
        is_star_cue = re.search(r"\bstars?\b", anchor, re.IGNORECASE) is not None
        if is_star_cue and not re.search(r"\bstar\s+icon\b", anchor, re.IGNORECASE):
            anchor = re.sub(r"\bstar\b", "star icon", anchor, count=1, flags=re.IGNORECASE)
        anchor_terms = canonical_terms(anchor)
        frame_terms = canonical_terms(frame_text)
        cue_terms = anchor_terms - non_news_visual_terms
        cue_already_present = bool(cue_terms.intersection(frame_terms))
        if cue_already_present:
            cue_detail = (
                f"News visual cue: {anchor} is part of the {'held pose' if normalized_frame == 'ending' else 'main action'}; "
                "render it distinctly from ordinary set dressing with a recognizable silhouette, pattern, or color contrast. "
                "Preserve its described relationship to the character; do not omit it or replace it with a generic lookalike."
            )
            cue_position = next(
                (index for index, part in enumerate(parts) if part.startswith("Visual identity:")),
                next(index for index, part in enumerate(parts) if part.startswith("Camera:")),
            )
            parts.insert(cue_position, cue_detail)
        else:
            if is_product_news:
                parts.append(
                    f"News visual cue: {anchor}, once as the single unbranded analogy integrated into the main cause-and-effect; "
                    "show how the selected character acts on it or responds to its reported change, and keep both faces unobstructed."
                )
            else:
                parts.append(
                    f"News visual cue: {anchor}, once as part of the main environment, action, or visible consequence; "
                    "integrate it with the selected character's action or response instead of placing it as a separate background icon."
                )
    contact_terms = (
        "觸碰", "接觸", "按壓", "擦拭", "擦", "握住", "拍", "戳",
    )
    contact_pattern = re.compile(
        r"\b(?:touch\w*|contact\w*|press\w*|wipe\w*|rub\w*|grasp\w*|grip\w*|tap\w*|nudge\w*|poke\w*|scrub\w*|brush\w*)\b"
        r"|\bagainst\b|"
        + "|".join(re.escape(term) for term in contact_terms),
        re.IGNORECASE,
    )
    tool_pattern = re.compile(
        r"\b(?:tool|spear|brush|spoon|fork|knife|chopstick|stick|wand|pole|bat|pencil|pen|cloth|towel|tissue|utensil|sword|staff)\b",
        re.IGNORECASE,
    )
    active_tool_contact_pattern = re.compile(
        r"\b(?:touch\w*|contact\w*|press\w*|wipe\w*|rub\w*|grasp\w*|grip\w*|tap\w*|nudge\w*|poke\w*|scrub\w*|brush\w*|catch\w*)\b",
        re.IGNORECASE,
    )
    tool_tip_against_pattern = re.compile(
        r"\b(?:tool|spear|brush|spoon|fork|knife|chopstick|stick|wand|pole|bat|pencil|pen|cloth|towel|tissue|utensil|sword|staff)\b"
        r"[^.!?;]{0,24}\b(?:tip|end|butt)\b[^.!?;]{0,28}\bagainst\b",
        re.IGNORECASE,
    )
    has_contact = contact_pattern.search(frame_text) is not None
    clauses = re.split(r"[.!?;]+", frame_text)
    has_named_tool_contact = False
    for clause in clauses:
        tool_matches = list(tool_pattern.finditer(clause))
        if not tool_matches:
            continue
        if tool_tip_against_pattern.search(clause):
            has_named_tool_contact = True
            break
        action_matches = list(active_tool_contact_pattern.finditer(clause))
        if any(
            0 <= action.start() - tool.end() <= 45 or 0 <= tool.start() - action.end() <= 45
            for tool in tool_matches
            for action in action_matches
        ):
            has_named_tool_contact = True
            break
    if has_contact:
        if has_named_tool_contact:
            parts.append("Contact: keep the named tool in hand with its working end on the exact target.")
    appliance_terms = (
        "appliance", "microwave", "oven", "interior", "cavity", "container", "微波爐", "烤箱", "內壁", "容器",
    )
    if any(term in frame_text for term in appliance_terms):
        parts.append("For an appliance/container action, only the named hand and tool enter; keep the character outside.")
    temporal_rule = (
        "Opening cold open: show the selected lead character clearly performing the reported role or responding to the "
        "article-specific mechanism in the same thumbnail-readable composition. Do not let the environment-only cue carry "
        "the frame or delay the character until a later shot. Freeze the clearest source-specific tension before the later payoff."
        if normalized_frame == "opening"
        else "Final beat only; do not replay earlier actions or transformations."
    )
    parts.append(temporal_rule)
    parts.append(
        "Keep characters recognizable. No promotional packaging, extra characters, readable text, pseudo-writing, interface, or logo; "
        "keep the story's single news analogy visible and unbranded."
    )
    return " ".join(parts)


def _format_held_keyframe_reaction(value: str) -> str:
    """Keep an ending frame's outcome while removing narration of earlier states."""
    text = " ".join(str(value or "").split()).strip()
    if not text:
        return ""
    lower_text = text.casefold()
    for marker in ("transitioning from", "transitions from", "changing from", "changes from", "turning from", "turns from"):
        marker_index = lower_text.find(marker)
        if marker_index < 0:
            continue
        lead = text[:marker_index].rstrip(" ,;:-")
        ending = text[marker_index + len(marker):].strip(" ,;:-.")
        to_index = ending.casefold().rfind(" to ")
        if to_index >= 0:
            ending = ending[to_index + len(" to "):].strip(" ,;:-.")
        for temporal_marker in (" after ", " then ", " before "):
            temporal_index = ending.casefold().find(temporal_marker)
            if temporal_index >= 0:
                ending = ending[:temporal_index].strip(" ,;:-.")
        if lead and ending:
            return f"{lead}; final posture: {ending}"
        if ending:
            return f"Final reaction: {ending}"
        return lead
    for marker in (" turns into ", " changes into ", " becomes "):
        marker_index = lower_text.find(marker)
        if marker_index >= 0:
            ending = text[marker_index + len(marker):].strip(" ,;:-.")
            temporal_index = ending.casefold().find(" after ")
            if temporal_index >= 0:
                ending = ending[:temporal_index].strip(" ,;:-.")
            if ending:
                return f"Final reaction: {ending}"
    return text
