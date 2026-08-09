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

_NATIVE_MOTION_PATTERN = re.compile(
    r"\b(?:move|moves|moving|reach|reaches|rush|rushes|rushing|pull|pulls|pulled|tear|tears|torn|"
    r"drag|drags|fall|falls|fell|slide|slides|slips|wrap|wraps|break|breaks|crack|cracks|"
    r"burst|bursts|form|forms|whip|whips|blow|blows|scatter|scatters|roll|rolls|crash|crashes|"
    r"splash|splashes|shatter|shatters|emerge|emerges|launch|launches|surge|surges|reverse|reverses|rise|rises|rising|fly|flies|dive|dives|"
    r"leap|leaps|jump|jumps|turn|turns|shift|shifts|erupt|erupts|sweep|sweeps|swirl|swirls|"
    r"drift|drifts|charge|charges|stumble|stumbles|grab|grabs|chase|chases|anchor|anchors|"
    r"release|releases|collaps|tighten|tightens|snap|snaps|spin|spins|run|runs)\b|"
    r"(?:衝|奔|拉|撕|拖|落|滑|裂|爆|湧|逆|升|飛|潛|跳|轉|變|旋|漂|追|抓|錨|釋|崩|斷|跑)",
    flags=re.IGNORECASE,
)

_NATIVE_TURN_PATTERN = re.compile(
    r"\b(?:but|however|yet|fails?|failure|worse|loses?|slips?|snaps?|breaks?|cracks?|"
    r"reverses?|almost|cannot|trapped|threat|surges?|tightens?|collapses?|no longer|"
    r"too late|close call|setback|turning point)\b|(?:但|卻|然而|失敗|更糟|失去|滑落|斷裂|"
    r"破裂|逆轉|幾乎|無法|困住|威脅|加劇|收緊|崩塌|太遲|危機|轉折)",
    flags=re.IGNORECASE,
)

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


def evaluate_native_h3_story_quality(
    story: dict[str, Any],
    *,
    expected_times: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Check whether a native H3 story is audience-readable before rendering.

    This is intentionally deterministic and conservative: it catches a valid
    JSON object that is merely a sequence of attractive poses. It does not claim
    to predict audience retention; the rendered clip still needs media QA.
    """
    errors: list[str] = []
    spine = story.get("story_spine") if isinstance(story, dict) else None
    shots = story.get("native_shots") if isinstance(story, dict) else None
    checks: dict[str, bool] = {}
    shot_times = tuple(expected_times or story.get("native_shot_times") or _DEFAULT_NATIVE_H3_SHOT_TIMES)
    if not isinstance(spine, dict) or not isinstance(shots, list) or len(shots) != len(shot_times):
        return {
            "passed": False,
            "score": 0,
            "checks": {},
            "errors": [f"story spine and exactly {len(shot_times)} shots are required"],
        }

    required_spine = ("premise", "objective", "obstacle", "stakes", "climax", "resolution")
    spine_values = [str(spine.get(key) or "").strip() for key in required_spine]
    checks["concrete_story_spine"] = all(value and not value.lower().startswith("generated ") for value in spine_values)
    if not checks["concrete_story_spine"]:
        errors.append("story spine must contain concrete audience-facing content, not placeholders")
    checks["distinct_spine_roles"] = len(set(value.lower() for value in spine_values)) == len(spine_values)
    if not checks["distinct_spine_roles"]:
        errors.append("premise, objective, obstacle, stakes, climax, and resolution must not collapse into one sentence")

    actions = [str(shot.get("action") or "") for shot in shots if isinstance(shot, dict)]
    cameras = [str(shot.get("camera") or "") for shot in shots if isinstance(shot, dict)]
    states = [str(shot.get("state_change") or "") for shot in shots if isinstance(shot, dict)]
    checks["distinct_mission_states"] = len({state.strip().lower() for state in states}) == len(shot_times)
    if not checks["distinct_mission_states"]:
        errors.append("each shot must leave the protagonist in a different mission state")

    opening_text = f"{actions[0]} {cameras[0]}" if actions and cameras else ""
    checks["hook_visible_motion"] = bool(_NATIVE_MOTION_PATTERN.search(opening_text))
    if not checks["hook_visible_motion"]:
        errors.append("the hook must contain a visible disruption or motion in the opening beat")

    escalation_text = " ".join(actions[1:-1] + states[1:-1]) if len(actions) > 2 and len(states) > 2 else ""
    checks["escalation_or_reversal"] = bool(_NATIVE_TURN_PATTERN.search(escalation_text))
    if not checks["escalation_or_reversal"]:
        errors.append("the middle beat must contain a setback, reversal, deadline, or visibly worsening obstacle")

    payoff_text = " ".join((actions[-1], states[-1], str(spine.get("resolution") or ""))) if actions and states else ""
    resolution_terms = _native_story_terms(spine.get("resolution"))
    payoff_terms = _native_story_terms(payoff_text)
    checks["payoff_evidence"] = bool(resolution_terms & payoff_terms)
    if not checks["payoff_evidence"]:
        errors.append("the final beat must visibly contain evidence of the stated resolution")

    objective_terms = _native_story_terms(spine.get("objective"))
    checks["objective_payoff_link"] = bool(objective_terms & payoff_terms)
    if not checks["objective_payoff_link"]:
        errors.append("the payoff must resolve the original objective instead of starting a new quest")

    total = len(checks)
    score = round(sum(1 for passed in checks.values() if passed) / total * 100) if total else 0
    return {"passed": not errors, "score": score, "checks": checks, "errors": errors}


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


def resolve_native_h3_story(
    base_storyboard: dict[str, Any],
    *,
    character: str,
    style: str,
    duration_seconds: int,
    news_context: dict[str, Any] | None = None,
    creative_brief: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve one news-grounded native H3 story for every entry point.

    News selection, LLM generation, and storyboard validation live here so the
    CLI and scheduler cannot silently drift into different story behavior.
    Every failure propagates to the caller.
    """
    import os

    from agentic.runtime.llm_engine import LLMPromptEngine
    from agentic.tools.context_services import NewsContextService

    resolved_news = dict(news_context or {})
    if resolved_news.get("error"):
        raise RuntimeError(f"Native H3 news context failed: {resolved_news['error']}")
    if not NewsContextService.is_usable_selection(
        str(resolved_news.get("title") or ""),
        str(resolved_news.get("keyword") or ""),
    ):
        selected_news = NewsContextService().get_random_news()
        if selected_news is None:
            raise RuntimeError("Native H3 requires a selectable news item; no news context was available")
        resolved_news = selected_news.to_dict()
    payload = LLMPromptEngine(
        mode=os.environ.get("AGENTIC_LLM_MODE", "llm")
    ).generate_native_h3_storyboard(
        character=character,
        style=style,
        duration_seconds=duration_seconds,
        base_storyboard=base_storyboard,
        news_context=resolved_news,
        creative_brief=creative_brief,
    )
    return merge_native_h3_storyboard(base_storyboard, payload["story"]), payload


def merge_native_h3_storyboard(
    base_storyboard: dict[str, Any],
    generated_story: dict[str, Any],
) -> dict[str, Any]:
    """Apply an LLM-generated story to the reusable character rules.

    The base preset supplies identity and continuity constraints only. Plot,
    setting, props, keyframes, audio, and native shots must come from the
    generated story; missing values are an error rather than a fallback.
    """
    if not isinstance(base_storyboard, dict):
        raise StoryboardError("Native H3 base storyboard must be a mapping")
    if not isinstance(generated_story, dict):
        raise StoryboardError("Native H3 generated story must be a mapping")
    required = (
        "name",
        "base_prompt",
        "opening_keyframe_prompt",
        "ending_keyframe_prompt",
        "negative_prompt",
        "story_spine",
        "world",
        "native_audio",
        "native_shots",
    )
    missing = [key for key in required if not str(generated_story.get(key) or "").strip() and key not in {"story_spine", "world", "native_shots"}]
    if missing:
        raise StoryboardError("Generated native H3 story missing values: " + ", ".join(missing))
    spine = generated_story.get("story_spine")
    world = generated_story.get("world")
    shots = generated_story.get("native_shots")
    expected_times = native_h3_shot_times(base_storyboard)
    if not isinstance(spine, dict) or not isinstance(world, dict) or not isinstance(shots, list) or len(shots) != len(expected_times):
        raise StoryboardError(
            "Generated native H3 story must define story_spine, world, and exactly "
            f"{len(expected_times)} native_shots"
        )
    required_spine = ("premise", "objective", "obstacle", "stakes", "climax", "resolution")
    missing_spine = [key for key in required_spine if not str(spine.get(key) or "").strip()]
    if missing_spine:
        raise StoryboardError("Generated native H3 story_spine missing values: " + ", ".join(missing_spine))
    story_quality = evaluate_native_h3_story_quality(generated_story, expected_times=expected_times)
    if not story_quality["passed"]:
        raise StoryboardError("Generated native H3 story quality is insufficient: " + "; ".join(story_quality["errors"]))
    required_shot = ("time", "title", "action", "camera", "state_change")
    for index, shot in enumerate(shots, start=1):
        if not isinstance(shot, dict):
            raise StoryboardError(f"Generated native H3 shot {index} must be a mapping")
        missing_shot = [key for key in required_shot if not str(shot.get(key) or "").strip()]
        if missing_shot:
            raise StoryboardError(
                f"Generated native H3 shot {index} missing values: " + ", ".join(missing_shot)
            )
    timing_ok, timing_error = validate_native_h3_shot_timing(
        shots,
        duration_seconds=float(
            base_storyboard.get("native_duration_seconds")
            or native_h3_duration_from_times(expected_times)
        ),
    )
    if not timing_ok:
        raise StoryboardError("Generated native H3 shot timing is invalid: " + timing_error)
    base_world = dict(base_storyboard.get("world") or {})
    generated_rules = world.get("continuity_rules") or []
    if not isinstance(generated_rules, list):
        raise StoryboardError("Generated native H3 world.continuity_rules must be a list")
    character = str(base_storyboard.get("character") or "the protagonist").strip()
    continuity_rules: list[str] = [
        f"Only one {character} appears; preserve the same identity, proportions, silhouette, and palette throughout."
    ]
    for rule in generated_rules:
        value = str(rule or "").strip()
        if value and value not in continuity_rules:
            continuity_rules.append(value)
    if not continuity_rules:
        raise StoryboardError("Native H3 story must preserve at least one continuity rule")
    merged = deepcopy(base_storyboard)
    merged.update(
        {
            "name": str(generated_story["name"]).strip(),
            "base_prompt": str(generated_story["base_prompt"]).strip(),
            "opening_keyframe_prompt": str(generated_story["opening_keyframe_prompt"]).strip(),
            "ending_keyframe_prompt": str(generated_story["ending_keyframe_prompt"]).strip(),
            "negative_prompt": str(generated_story["negative_prompt"]).strip(),
            "story_spine": deepcopy(spine),
            "world": {
                **base_world,
                **deepcopy(world),
                "continuity_rules": continuity_rules,
            },
            "native_audio": str(generated_story["native_audio"]).strip(),
            "native_shots": deepcopy(shots),
            "native_shot_times": [str(shot["time"]).strip() for shot in shots],
            "story_quality": deepcopy(story_quality),
            "story_source": "news_llm",
        }
    )
    merged["segments"] = _native_shots_to_segments(merged["native_shots"])
    progression = validate_story_progression(merged["segments"])
    if not progression["passed"]:
        raise StoryboardError("Generated native H3 story progression is invalid: " + "; ".join(progression["errors"]))
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
    progression = validate_story_progression(selected)
    if not progression["passed"]:
        raise StoryboardError("Invalid story progression: " + "; ".join(progression["errors"]))
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
        "progression_check": progression,
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
    )
    prompt += "\nAudience story contract: the first beat creates a concrete question, the middle worsens the obstacle or reverses the plan, and the final beat answers the original question with visible payoff evidence."
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


def validate_story_progression(segments: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    goals = [str(segment.get("narrative_goal", "")).strip().lower() for segment in segments]
    if len(goals) != len(set(goals)):
        errors.append("narrative_goal values must be distinct across segments")
    for index, segment in enumerate(segments):
        for key in ("act", "act_goal", "spine_beat", "cause", "effect", "must_show"):
            if not segment.get(key):
                errors.append(f"segment {index + 1} has no {key} story-spine contract")
        if not str(segment.get("next_hook", "")).strip():
            errors.append(f"segment {index + 1} has no next_hook")
        if index > 0:
            prior_end = str(segments[index - 1].get("end_state", "")).strip().lower()
            current_start = str(segment.get("start_state", "")).strip().lower()
            if not prior_end or not current_start:
                errors.append(f"segment {index + 1} lacks explicit state handoff")
    return {"passed": not errors, "errors": errors, "distinct_goals": len(goals) == len(set(goals))}


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
