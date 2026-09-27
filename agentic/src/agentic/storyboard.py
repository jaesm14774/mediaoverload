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
    if not renderable_shots:
        renderable_shots = [{"action": generated_description}]

    boundaries = [duration_seconds * index / len(renderable_shots) for index in range(len(renderable_shots) + 1)]
    render_times = [f"{start:g}-{end:g}s" for start, end in zip(boundaries, boundaries[1:])]

    normalized_shots: list[dict[str, Any]] = []
    for index, raw_shot in enumerate(renderable_shots):
        action = first_text(
            raw_shot,
            "action",
            "subject_action",
            "primary_action",
            "visual_action",
            "visual",
            "description",
            "shot_description",
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
        title = first_text(raw_shot, "title", "name", "beat")
        camera = first_text(raw_shot, "camera", "camera_direction", "camera_movement", "shot")
        state_change = first_text(raw_shot, "state_change", "end_state", "effect", "result")
        normalized = {**raw_shot, "time": render_times[index]}
        for key, value in (("title", title), ("action", action), ("camera", camera), ("state_change", state_change)):
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
        prompt_base = f"{prompt_base} Creative variation for this run: {surface_variation}.".strip()
    shot_payloads: list[dict[str, Any]] = []
    for index, shot in enumerate(shots, start=1):
        shot_payloads.append(dict(shot) if isinstance(shot, dict) else {"action": str(shot or "")})
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
        media_type=render_mode or "native_h3_story",
    )
    news_trace = storyboard.get("news_trace")
    gag_card = storyboard.get("gag_card")
    if isinstance(gag_card, dict):
        ideas = [
            str(gag_card.get(key) or "").strip()
            for key in ("hook_frame", "prop_rule", "expressive_reaction", "loop_reason")
            if str(gag_card.get(key) or "").strip()
        ]
        if ideas:
            prompt += "\nOptional visual ideas: " + "; ".join(ideas) + "."
    if isinstance(news_trace, dict):
        story_arc_intent = str(news_trace.get("story_arc_intent") or "").strip()
        source_fact = str(news_trace.get("source_fact") or "").strip()
        visual_translation = str(news_trace.get("visual_translation") or "").strip()
        news_mechanism = str(news_trace.get("news_mechanism") or "").strip()
        news_consequence = str(news_trace.get("news_consequence") or "").strip()
        source_limit = str(news_trace.get("source_limit") or "").strip()
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
        if story_arc_intent or source_fact or visual_translation or visual_anchors:
            prompt += (
                "\nOptional source context: "
                f"source fact={source_fact}; arc intent={story_arc_intent}; "
                f"visual translation={visual_translation}; visual anchors={', '.join(visual_anchors)}."
            ).strip()
        if news_mechanism or news_consequence or source_limit or anchor_roles:
            prompt += (
                "\nAdditional source context: "
                f"active mechanism={news_mechanism}; visible consequence={news_consequence}; "
                f"source limit={source_limit}; anchor roles={', '.join(anchor_roles)}."
            ).strip()
    if str(spine.get("resolution") or "").strip():
        prompt += f"\nEnding frame idea: {str(spine.get('resolution') or '').strip()}."
    return prompt.strip()
