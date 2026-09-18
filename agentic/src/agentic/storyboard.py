from __future__ import annotations

import json
import math
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

_DEFAULT_NATIVE_H3_SHOT_TIMES = ("0-4s", "4-10s", "10-15s")
_NATIVE_TIME_RANGE_PATTERN = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*s?\s*-\s*(\d+(?:\.\d+)?)\s*s?\s*$",
    flags=re.IGNORECASE,
)

def native_h3_shot_times(storyboard: dict[str, Any] | None = None) -> tuple[str, ...]:
    """Return the explicit timing contract for a native H3 storyboard."""
    payload = storyboard if isinstance(storyboard, dict) else {}
    declared = payload.get("native_shot_times")
    if not isinstance(declared, list):
        story_generation = payload.get("story_generation") or {}
        declared = story_generation.get("required_shot_times") if isinstance(story_generation, dict) else None
    if not isinstance(declared, list) or not declared:
        return _DEFAULT_NATIVE_H3_SHOT_TIMES
    values = tuple(str(value).strip() for value in declared if str(value).strip())
    return values or _DEFAULT_NATIVE_H3_SHOT_TIMES


def native_h3_duration_from_times(times: tuple[str, ...] | list[str]) -> float:
    """Return the duration implied by the final native H3 beat label."""
    if not times:
        raise StoryboardError("Native H3 timing contract cannot be empty")
    match = _NATIVE_TIME_RANGE_PATTERN.fullmatch(str(times[-1]))
    if not match:
        raise StoryboardError("Native H3 timing contract must use numeric start-end ranges")
    return float(match.group(2))


def validate_native_h3_shot_timing(
    shots: list[dict[str, Any]],
    *,
    duration_seconds: int | float,
) -> tuple[bool, str]:
    """Validate contiguous beat timing without forcing arbitrary inner cuts."""
    previous_end = 0.0
    for index, shot in enumerate(shots, start=1):
        match = _NATIVE_TIME_RANGE_PATTERN.fullmatch(str(shot.get("time") or ""))
        if not match:
            return False, f"shot {index} time must be a numeric start-end range"
        start = float(match.group(1))
        end = float(match.group(2))
        if end <= start:
            return False, f"shot {index} time range must increase"
        if abs(start - previous_end) > 0.05:
            return False, f"shot {index} must start where the previous shot ends"
        previous_end = end
    if abs(previous_end - float(duration_seconds)) > 0.05:
        return False, f"the final shot must end at {duration_seconds:g}s"
    return True, ""


def _native_story_terms(value: Any) -> set[str]:
    tokens = re.findall(r"[a-z][a-z0-9'-]{2,}|[\u4e00-\u9fff]{2,}", str(value or "").lower())
    return {token for token in tokens if token not in _NATIVE_STORY_STOPWORDS}


def ground_native_h3_ending_keyframe_prompt(story: dict[str, Any]) -> str:
    """Keep the ending keyframe visibly tied to the resolved news story."""
    if not isinstance(story, dict):
        return ""
    prompt = str(story.get("ending_keyframe_prompt") or "").strip()
    if not prompt:
        return ""
    world = story.get("world") if isinstance(story.get("world"), dict) else {}
    trace = story.get("news_trace") if isinstance(story.get("news_trace"), dict) else {}
    setting = str(world.get("setting") or "").strip().rstrip(".")
    anchors = [
        str(item).strip().rstrip(".")
        for item in trace.get("visual_anchors", [])
        if str(item).strip()
    ]
    mechanism = str(trace.get("news_mechanism") or "").strip().rstrip(".")
    consequence = str(trace.get("news_consequence") or "").strip().rstrip(".")
    locks: list[str] = []
    if setting:
        locks.append(f"Keep the ending in {setting}.")
    if anchors:
        locks.append("Keep these visible news anchors in the same ending composition: " + "; ".join(anchors) + ".")
    if mechanism:
        locks.append(f"Show the active news mechanism: {mechanism}.")
    if consequence:
        locks.append(f"Show the concrete news consequence: {consequence}.")
    suffix = " ".join(locks)
    if suffix and suffix.casefold() not in prompt.casefold():
        return f"{prompt.rstrip('.')} {suffix}"
    return prompt


def native_surface_variation(creative_brief: str) -> str:
    """Keep autonomous variation bounded so it cannot replace the story spine.

    Native H3 receives a causal storyboard, not an unrestricted scene rewrite.
    LLM-generated briefs can contain a second plot, camera plan, or a different
    visual medium, so only short descriptive labels are allowed through.
    """
    text = re.sub(r"\s+", " ", str(creative_brief or "").strip()).strip(" .")
    if not text:
        return ""
    if len(text.split()) > 14 or len(text) > 110:
        return "a restrained atmospheric variation"
    unsafe_terms = {
        "action", "archive", "camera", "carry", "character", "cinematic", "ending",
        "floating", "kirby", "mission", "protagonist", "release", "run", "scene",
        "shot", "story", "vacuum", "video", "drama", "rendered", "3d", "4d",
    }
    words = {word.lower() for word in re.findall(r"[a-z0-9']+", text)}
    if words & unsafe_terms:
        return "a restrained atmospheric variation"
    return text


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

    The base preset owns identity, safety, and timing.  Creative metadata is
    optional; only the shot list and its visible actions are required to build
    a renderable H3 prompt.  Missing descriptive shot fields receive
    deterministic defaults so a free model is not rejected for presentation
    details that the prompt composer can already infer.
    """
    if not isinstance(base_storyboard, dict):
        raise StoryboardError("Native H3 base storyboard must be a mapping")
    if not isinstance(generated_story, dict):
        raise StoryboardError("Native H3 generated story must be a mapping")
    expected_times = native_h3_shot_times(base_storyboard)
    shots = generated_story.get("native_shots")
    if not isinstance(shots, list):
        shots = generated_story.get("shots") or generated_story.get("beats")
    if not isinstance(shots, list) or len(shots) != len(expected_times):
        raise StoryboardError(f"Generated native H3 story must define exactly {len(expected_times)} native shots")

    def first_text(mapping: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = str(mapping.get(key) or "").strip()
            if value:
                return value
        return ""

    normalized_shots: list[dict[str, Any]] = []
    for index, raw_shot in enumerate(shots):
        if not isinstance(raw_shot, dict):
            raise StoryboardError(f"Generated native H3 shot {index + 1} must be a mapping")
        action = first_text(
            raw_shot,
            "action",
            "primary_action",
            "visual_action",
            "visual",
            "description",
            "shot_description",
        )
        if not action:
            raise StoryboardError(f"Generated native H3 shot {index + 1} must contain an action")
        title = first_text(raw_shot, "title", "name", "beat")
        if not title:
            title = re.split(r"[.;:!?]", action, maxsplit=1)[0].strip()[:160] or f"Beat {index + 1}"
        camera = first_text(raw_shot, "camera", "camera_direction", "camera_movement", "shot")
        if not camera:
            camera = "Follow the primary action with readable spatial continuity."
        state_change = first_text(raw_shot, "state_change", "end_state", "effect", "result")
        if not state_change:
            state_change = action
        normalized_shots.append(
            {
                **raw_shot,
                "time": first_text(raw_shot, "time", "time_range", "timestamp") or expected_times[index],
                "title": title,
                "action": action,
                "camera": camera,
                "state_change": state_change,
            }
        )

    duration_seconds = float(
        base_storyboard.get("native_duration_seconds")
        or native_h3_duration_from_times(expected_times)
    )
    timing_ok, timing_error = validate_native_h3_shot_timing(
        normalized_shots,
        duration_seconds=duration_seconds,
    )
    if not timing_ok:
        raise StoryboardError("Generated native H3 shot timing is invalid: " + timing_error)

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
    subject_context = dict(base_storyboard.get("subject_context") or {})
    subject_items = [item for item in (subject_context.get("subjects") or []) if isinstance(item, dict)]
    interaction_required = bool(
        dict(subject_context.get("interaction_contract") or {}).get("required", False)
    ) and len(subject_items) == 2
    if interaction_required:
        continuity_rules: list[str] = [
            "Exactly the two declared subject slots remain visible when the story requires both; preserve each identity, role, proportions, silhouette, and palette throughout."
        ]
    else:
        continuity_rules = [
            f"Only one {character} appears; preserve the same identity, proportions, silhouette, and palette throughout."
        ]
    for rule in world["continuity_rules"]:
        if interaction_required and "only" in rule.lower() and "protagonist" in rule.lower():
            continue
        if rule not in continuity_rules:
            continuity_rules.append(rule)
    world["continuity_rules"] = continuity_rules

    def clean_runtime_placeholder(value: Any) -> str:
        text = str(value or "").strip()
        return "" if text.lower().startswith("generated ") else text

    base_spine = {
        key: clean_runtime_placeholder(value)
        for key, value in dict(base_storyboard.get("story_spine") or {}).items()
    }
    generated_spine = generated_story.get("story_spine")
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
    spine_defaults = {
        "premise": normalized_shots[0]["action"],
        "objective": "Advance the visible story through a concrete physical payoff.",
        "obstacle": normalized_shots[min(1, len(normalized_shots) - 1)]["action"],
        "stakes": normalized_shots[min(1, len(normalized_shots) - 1)]["state_change"],
        "climax": normalized_shots[-1]["action"],
        "resolution": normalized_shots[-1]["state_change"],
    }
    for key, default in spine_defaults.items():
        if not str(spine.get(key) or "").strip():
            spine[key] = default

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
        opening_prompt = f"{base_prompt}. {normalized_shots[0]['action']}".strip(". ")
    ending_prompt = first_text(
        generated_story,
        "ending_keyframe_prompt",
        "ending_prompt",
        "last_frame_prompt",
    )
    if not ending_prompt:
        ending_prompt = f"{base_prompt}. {normalized_shots[-1]['action']} {normalized_shots[-1]['state_change']}".strip(". ")
    negative_prompt = first_text(generated_story, "negative_prompt", "negative")
    if not negative_prompt:
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
    merged = deepcopy(base_storyboard)
    generated_base_prompt = base_prompt
    if interaction_required:
        subject_names = [
            str(item.get("name") or "").strip()
            for item in subject_items
            if str(item.get("name") or "").strip()
        ]
        generated_base_prompt = (
            f"{generated_base_prompt} Required subject slots share one readable scene: {'; '.join(subject_names)}. "
            "Preserve both identities and show their visible mutual interaction."
        ).strip()
        negative_parts = [
            part.strip()
            for part in negative_prompt.split(",")
            if part.strip() and part.strip().lower() not in {"humans", "extra characters", "duplicate", "duplicate kirby"}
        ]
        negative_parts.extend(["identity swap", "unrequested third subject"])
        negative_prompt = ", ".join(dict.fromkeys(negative_parts))
    merged.update(
        {
            "name": first_text(generated_story, "name", "title", "story_name") or f"{character} native H3 story",
            "base_prompt": generated_base_prompt,
            "opening_keyframe_prompt": opening_prompt,
            "ending_keyframe_prompt": ending_prompt,
            "negative_prompt": negative_prompt,
            **({"news_trace": deepcopy(news_trace)} if isinstance(news_trace, dict) else {}),
            **({"gag_card": deepcopy(gag_card)} if isinstance(gag_card, dict) else {}),
            "story_spine": spine,
            "world": {**world, "continuity_rules": continuity_rules},
            "native_audio": native_audio,
            "native_shots": normalized_shots,
            "native_shot_times": [str(shot["time"]).strip() for shot in normalized_shots],
            "story_source": "news_llm",
        }
    )
    merged["segments"] = _native_shots_to_segments(merged["native_shots"])
    return merged


def _native_shots_to_segments(shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(shots) == 3:
        phases = ("hook", "escalation", "resolution")
    elif len(shots) == 5:
        phases = ("hook", "promise", "escalation", "reversal", "payoff")
    else:
        phases = tuple(["hook", *(["progress"] * max(0, len(shots) - 2)), "payoff"])
    segments: list[dict[str, Any]] = []
    previous_end_state = "The film opens and the problem is not yet understood."
    for index, shot in enumerate(shots):
        time_match = _NATIVE_TIME_RANGE_PATTERN.fullmatch(str(shot["time"]))
        if not time_match:
            raise StoryboardError(
                f"Generated native H3 shot {index + 1} time must use a start-end seconds range"
            )
        start_value = float(time_match.group(1))
        end_value = float(time_match.group(2))
        start_seconds = int(start_value) if start_value.is_integer() else start_value
        end_seconds = int(end_value) if end_value.is_integer() else end_value
        if end_seconds <= start_seconds:
            raise StoryboardError(f"Generated native H3 shot {index + 1} time range must increase")
        title = str(shot["title"]).strip()
        action = str(shot["action"]).strip()
        state_change = str(shot["state_change"]).strip()
        next_hook = (
            f"Continue into the next beat: {str(shots[index + 1]['title']).strip()}."
            if index + 1 < len(shots)
            else "The story resolves here with no new quest."
        )
        segments.append(
            {
                "id": f"native_h3_{index + 1}",
                "time": str(shot["time"]).strip(),
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "duration_seconds": end_seconds - start_seconds,
                "act": index + 1,
                "phase": phases[index],
                "act_goal": f"Make the {phases[index]} beat readable through {title}.",
                "spine_beat": title,
                "narrative_goal": f"Advance the story through {title}.",
                "cause": previous_end_state,
                "action_beats": [action],
                "effect": state_change,
                "must_show": [action, state_change],
                "start_state": previous_end_state,
                "end_state": state_change,
                "next_hook": next_hook,
                "camera": str(shot["camera"]).strip(),
                "environment": "Preserve the same world geography and visual language while the mission state changes.",
                "audio": "Follow the generated native audio direction and make the state change audible.",
            }
        )
        previous_end_state = state_change
    return segments


def build_story_plan(
    storyboard: dict[str, Any],
    *,
    duration_seconds: int | None = None,
    segment_count: int | None = None,
) -> dict[str, Any]:
    raw_segments = storyboard.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise StoryboardError("Storyboard must define a non-empty segments list")
    raw_spine = storyboard.get("story_spine")
    required_spine = ("premise", "objective", "obstacle", "stakes", "climax", "resolution")
    if not isinstance(raw_spine, dict):
        raise StoryboardError("Storyboard must define a story_spine mapping")
    missing_spine = [key for key in required_spine if not str(raw_spine.get(key) or "").strip()]
    if missing_spine:
        raise StoryboardError("Storyboard story_spine missing required fields: " + ", ".join(missing_spine))
    segment_duration = float(storyboard.get("segment_duration_seconds", 5))
    if segment_duration <= 0:
        raise StoryboardError("segment_duration_seconds must be positive")
    if segment_count is None:
        if duration_seconds is None:
            segment_count = int(storyboard.get("default_segment_count", 2))
        else:
            segment_count = max(1, int(math.ceil(float(duration_seconds) / segment_duration)))
    if segment_count < 1 or segment_count > len(raw_segments):
        raise StoryboardError(
            f"Requested {segment_count} segments, but preset provides {len(raw_segments)} explicit story cards"
        )
    selected = [deepcopy(item) for item in raw_segments[:segment_count]]
    seen_ids: set[str] = set()
    for index, segment in enumerate(selected):
        if not isinstance(segment, dict):
            raise StoryboardError(f"Story segment {index + 1} must be a mapping")
        required = (
            "id",
            "act",
            "act_goal",
            "spine_beat",
            "narrative_goal",
            "cause",
            "action_beats",
            "effect",
            "must_show",
            "start_state",
            "end_state",
            "next_hook",
        )
        missing = [key for key in required if not segment.get(key)]
        if missing:
            raise StoryboardError(f"Story segment {index + 1} missing required fields: {', '.join(missing)}")
        segment_id = str(segment["id"])
        if segment_id in seen_ids:
            raise StoryboardError(f"Duplicate story segment id: {segment_id}")
        seen_ids.add(segment_id)
        if not isinstance(segment["action_beats"], list) or not segment["action_beats"]:
            raise StoryboardError(f"Story segment {segment_id} must have at least one action beat")
        if not isinstance(segment["must_show"], list) or not segment["must_show"]:
            raise StoryboardError(f"Story segment {segment_id} must define must_show evidence")
        segment["index"] = index
        segment["segment_number"] = index + 1
        segment["segment_count"] = segment_count
        segment_duration_value = float(segment.get("duration_seconds") or segment_duration)
        if segment_duration_value <= 0:
            raise StoryboardError(f"Story segment {segment_id} duration_seconds must be positive")
        segment["duration_seconds"] = segment_duration_value
    planned_duration = sum(float(segment["duration_seconds"]) for segment in selected)
    return {
        "storyboard_name": str(storyboard.get("name") or "unnamed_storyboard"),
        "storyboard_path": str(storyboard.get("_path") or ""),
        "character": str(storyboard.get("character") or ""),
        "base_prompt": str(storyboard.get("base_prompt") or ""),
        "story_spine": dict(storyboard.get("story_spine") or {}),
        "world": dict(storyboard.get("world") or {}),
        "segment_duration_seconds": segment_duration,
        "segment_count": segment_count,
        "planned_duration_seconds": round(planned_duration, 3),
        "segments": selected,
    }


def format_story_segment_prompt(
    storyboard: dict[str, Any],
    segment: dict[str, Any],
    *,
    base_prompt: str = "",
    previous_end_state: str = "",
    style: str = "",
) -> str:
    def sentence(label: str, value: Any) -> str:
        text = str(value or "").strip().rstrip(".")
        return f"{label}: {text}." if text else ""

    story_spine = dict(storyboard.get("story_spine") or {})
    world = dict(storyboard.get("world") or {})
    continuity_rules = world.get("continuity_rules") or []
    if not isinstance(continuity_rules, list):
        continuity_rules = [str(continuity_rules)]
    action_beats = segment.get("action_beats") or []
    action_text = "; ".join(f"{index + 1}) {str(beat)}" for index, beat in enumerate(action_beats))
    parts = [
        str(base_prompt or storyboard.get("base_prompt") or "").strip(),
        sentence("Core story premise", story_spine.get("premise")),
        sentence("Protagonist objective", story_spine.get("objective")),
        sentence("Global obstacle", story_spine.get("obstacle")),
        sentence("Stakes", story_spine.get("stakes")),
        sentence("Required resolution", story_spine.get("resolution")),
        sentence("World", world.get("setting")),
        sentence("Palette and visual language", world.get("visual_language")),
        sentence("Act goal", segment.get("act_goal")),
        sentence("Story spine beat", segment.get("spine_beat")),
        sentence("Causal link into this segment", segment.get("cause")),
        f"Story phase: {str(segment.get('phase', '')).strip()}; narrative goal: {str(segment.get('narrative_goal', '')).strip().rstrip('.')}." if segment.get("narrative_goal") else "",
        sentence("Start state", segment.get("start_state")),
        sentence("Action beats, in order", action_text),
        sentence("Required visual evidence", "; ".join(str(item).strip().rstrip(".") for item in (segment.get("must_show") or []))),
        sentence("End state", segment.get("end_state")),
        sentence("Causal effect after this segment", segment.get("effect")),
        sentence("Next story hook", segment.get("next_hook")),
        sentence("Camera", segment.get("camera")),
        sentence("Environment progression", segment.get("environment")),
        sentence("Audio direction", segment.get("audio")),
        sentence("Previous segment ended at", previous_end_state) if previous_end_state else "This is the opening segment; establish the world clearly.",
        "Advance the story from the start state to the end state. Do not replay the previous segment's main event or reset to the opening pose.",
        "Every action must serve the protagonist objective and the story spine. Do not introduce a new quest, unrelated prop, disconnected spectacle, or a resolution before the final act.",
        "Use one readable primary event with a clear setup, change, and payoff; keep secondary motion subtle and physically continuous.",
        sentence("Continuity rules", "; ".join(str(item).strip().rstrip(".") for item in continuity_rules)),
        sentence("Style", style),
    ]
    return " ".join(part for part in parts if part.strip())


def format_native_h3_prompt(
    storyboard: dict[str, Any],
    *,
    style: str = "cinematic 2D anime",
    creative_brief: str = "",
    duration_seconds: int | None = None,
) -> str:
    """Build a compact causal multi-shot script for one native H3 clip.

    H3 can model a longer continuous clip, so this deliberately avoids repeating
    the full long-video contract for every five-second segment. The preset owns
    the shot boundaries and state changes; this formatter turns them into one
    prompt that is reusable by the automated runner and visible in manifests.
    """
    shots = storyboard.get("native_shots")
    if not isinstance(shots, list) or not shots:
        raise StoryboardError("Storyboard must define a non-empty native_shots list for native H3 output")
    spine = dict(storyboard.get("story_spine") or {})
    base_prompt = str(storyboard.get("base_prompt") or "").strip()
    world = dict(storyboard.get("world") or {})
    continuity = world.get("continuity_rules") or []
    if not isinstance(continuity, list):
        continuity = [str(continuity)]
    character = str(storyboard.get("character") or "the protagonist").strip()
    duration = int(duration_seconds or storyboard.get("native_duration_seconds") or 15)
    render_mode = str(storyboard.get("render_mode") or "").strip()
    prompt_base = base_prompt
    surface_variation = native_surface_variation(creative_brief)
    if surface_variation:
        prompt_base = (
            f"{prompt_base} Creative variation for this run: {surface_variation}. "
            "Use this only for surface details, lighting, weather, or emotional emphasis; never replace the story objective, causal order, or ending."
        ).strip()
    shot_payloads: list[dict[str, Any]] = []
    for index, shot in enumerate(shots, start=1):
        if not isinstance(shot, dict):
            raise StoryboardError(f"native_shots item {index} must be a mapping")
        required = ("time", "title", "action", "camera", "state_change")
        missing = [key for key in required if not str(shot.get(key) or "").strip()]
        if missing:
            raise StoryboardError(f"native_shots item {index} missing required fields: {', '.join(missing)}")
        shot_payloads.append(dict(shot))
    audio = str(storyboard.get("native_audio") or "").strip()
    prompt = compose_minimax_h3_prompt(
        duration_seconds=duration,
        character=character,
        style=style,
        base_prompt=prompt_base,
        story_spine=spine,
        setting=str(world.get("setting") or ""),
        visual_language=str(world.get("visual_language") or ""),
        shots=shot_payloads,
        audio=audio,
        render_mode=render_mode,
        continuity_rules=continuity,
        subject_context=dict(storyboard.get("subject_context") or {}),
    )
    prompt += "\nAudience story contract: the first beat creates a concrete question, the middle worsens the obstacle or reverses the plan, and the final beat answers the original question with visible payoff evidence."
    gag_card = storyboard.get("gag_card")
    if isinstance(gag_card, dict):
        prompt += (
            "\nSingle visual gag contract: "
            f"Hook already happening: {str(gag_card.get('hook_frame') or '').strip()}. "
            f"Desire: {str(gag_card.get('character_desire') or '').strip()}. "
            f"Prop rule: {str(gag_card.get('prop_rule') or '').strip()}. "
            f"Setback: {str(gag_card.get('setback') or '').strip()}. "
            f"Readable reaction: {str(gag_card.get('expressive_reaction') or '').strip()}. "
            f"Final reversal: {str(gag_card.get('payoff_reversal') or '').strip()}. "
            f"Replay reason: {str(gag_card.get('loop_reason') or '').strip()}."
        )
    news_trace = storyboard.get("news_trace")
    if isinstance(news_trace, dict):
        visual_translation = str(news_trace.get("visual_translation") or "").strip()
        news_mechanism = str(news_trace.get("news_mechanism") or "").strip()
        news_consequence = str(news_trace.get("news_consequence") or "").strip()
        anchor_roles = [
            str(item).strip()
            for item in news_trace.get("anchor_roles", [])
            if str(item).strip()
        ]
        visual_anchors = [
            str(item).strip()
            for item in news_trace.get("visual_anchors", [])
            if str(item).strip()
        ]
        if visual_translation or visual_anchors:
            prompt += (
                "\nNews-grounded visual anchor: "
                f"{visual_translation}. Keep these visible across the causal story: {', '.join(visual_anchors)}."
            ).strip()
        if news_mechanism or news_consequence or anchor_roles:
            prompt += (
                "\nNews mechanism contract: "
                f"active mechanism={news_mechanism}; visible consequence={news_consequence}; "
                f"anchor roles={', '.join(anchor_roles)}. Do not replace the mechanism with a generic floating object."
            ).strip()
    prompt += f"\nEnding frame: show the concrete result of the climax: {str(spine.get('resolution') or '').strip()}."
    prompt += "\nDo not reset to an earlier pose, introduce a new quest, or replace the causal story with unrelated spectacle."
    return prompt.strip()


def build_storyboard_segments(
    storyboard: dict[str, Any],
    *,
    segment_count: int,
    tone: str = "",
    style: str = "",
    creative_brief: str = "",
) -> list[dict[str, Any]]:
    plan = build_story_plan(storyboard, segment_count=segment_count)
    segments: list[dict[str, Any]] = []
    previous_end_state = ""
    for segment in plan["segments"]:
        visual = format_story_segment_prompt(
            storyboard,
            segment,
            base_prompt=creative_brief or plan["base_prompt"],
            previous_end_state=previous_end_state,
            style=style,
        )
        segments.append(
            {
                **segment,
                "story_spine": plan["story_spine"],
                "segment_id": str(segment["id"]),
                "visual": visual,
                "narration": f"{segment['narrative_goal']} The story moves toward {segment['next_hook']}.",
                "tone": tone,
                "creative_brief": creative_brief,
            }
        )
        previous_end_state = str(segment["end_state"])
    return segments


def story_state_contract(segment: dict[str, Any]) -> str:
    return " ".join(
        part
        for part in (
            f"Story phase: {segment.get('phase', '')}",
            f"Act: {segment.get('act', '')}",
            f"Act goal: {segment.get('act_goal', '')}",
            f"Story spine beat: {segment.get('spine_beat', '')}",
            f"Narrative goal: {segment.get('narrative_goal', '')}",
            f"Cause: {segment.get('cause', '')}",
            f"Start state: {segment.get('start_state', '')}",
            f"End state: {segment.get('end_state', '')}",
            f"Effect: {segment.get('effect', '')}",
            f"Must show: {', '.join(str(item) for item in (segment.get('must_show') or []))}",
            f"Next hook: {segment.get('next_hook', '')}",
        )
        if str(part.split(": ", 1)[-1]).strip()
    )
