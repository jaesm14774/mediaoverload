"""Character Config responsibilities."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import yaml
from agentic.tools.publishing_adapter import FACEBOOK_PROFILE_HANDOFF_PLATFORM
from agentic.app.character_requests import SUBJECT_MODE_SINGLE


SUPPORTED_PUBLISH_PLATFORMS = {
    "twitter",
    "facebook",
    "instagram_graph",
    "youtube",
    FACEBOOK_PROFILE_HANDOFF_PLATFORM,
}


MEDIA_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".mp4",
    ".mov",
    ".avi",
    ".webm",
    ".mkv",
    ".m4v",
}


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}


def _normalize_video_speed_config(value: object) -> dict[str, object]:
    if value is None:
        raw: dict[str, object] = {}
    elif isinstance(value, dict):
        raw = dict(value)
    else:
        raise ValueError("generation.video_speed must be a mapping")
    enabled = bool(raw.get("enabled", False))
    try:
        factor = float(raw.get("factor", 2.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "generation.video_speed.factor must be a number greater than zero"
        ) from exc
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError(
            "generation.video_speed.factor must be a finite number greater than zero"
        )
    return {"enabled": enabled, "factor": factor}


def _normalize_generation_type(
    value: str | None,
) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def load_character_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_global_social_config(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "configs" / "social_media" / "platforms.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(data) if isinstance(data, dict) else {}


def _normalize_platform_configs(
    repo_root: Path, raw_platforms: Any
) -> tuple[dict[str, dict[str, Any]], dict[str, str], list[str]]:
    if not isinstance(raw_platforms, dict):
        return {}, {}, []
    configs: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    skipped: list[str] = []
    for platform_name, platform_config in raw_platforms.items():
        if not isinstance(platform_config, dict) or not bool(
            platform_config.get("enabled", True)
        ):
            continue
        normalized = str(platform_name).lower().strip()
        if normalized == "instagram":
            aliases[normalized] = "instagram_graph"
            normalized = "instagram_graph"
        if normalized not in SUPPORTED_PUBLISH_PLATFORMS:
            skipped.append(str(platform_name))
            continue
        resolved = dict(platform_config)
        if "config_folder_path" in resolved:
            resolved["config_folder_path"] = str(
                _resolve_repo_path(repo_root, str(resolved["config_folder_path"]))
            )
        configs[normalized] = resolved
    return configs, aliases, skipped


def _merge_social_media_config(
    base: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_social_media_config(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _summarize_character_config(
    *,
    path: Path,
    character: dict[str, Any],
    selected_character_name: str,
    character_selection: dict[str, Any],
    generation: dict[str, Any],
    social_media: dict[str, Any],
    strategies: dict[str, Any],
    platform_configs: dict[str, dict[str, Any]],
    effective_longvideo_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    longvideo = dict(strategies.get("text2longvideo", {}) or {})
    longvideo_config = dict(
        effective_longvideo_config
        if effective_longvideo_config is not None
        else longvideo.get("longvideo_config", {}) or {}
    )
    return {
        "config_path": str(path),
        "character_name": str(
            selected_character_name or character.get("name") or path.stem
        ),
        "group_name": str(character.get("group_name") or ""),
        "character_selection": dict(character_selection or {}),
        "configured_subject_mode": str(
            generation.get("subject_mode") or SUBJECT_MODE_SINGLE
        )
        .strip()
        .lower(),
        "subject_mode": str(
            character_selection.get("subject_mode")
            or generation.get("subject_mode")
            or SUBJECT_MODE_SINGLE
        )
        .strip()
        .lower(),
        "subject_mode_weights": dict(generation.get("subject_mode_weights", {}) or {}),
        "output_dir": str(generation.get("output_dir") or ""),
        "generation_type_weights": dict(
            generation.get("generation_type_weights", {}) or {}
        ),
        "style_count": len(
            dict(
                dict(generation.get("creative_profile", {}) or {})
                .get("style", {})
                .get("profiles", {})
                or {}
            )
        ),
        "default_hashtags": [
            str(tag) for tag in (social_media.get("default_hashtags") or []) if tag
        ],
        "enabled_platforms": list(platform_configs.keys()),
        "platform_config_folders": {
            name: str(config.get("config_folder_path", ""))
            for name, config in platform_configs.items()
        },
        "longvideo": {
            "segment_count": int(longvideo_config.get("segment_count", 0) or 0),
            "segment_duration": int(longvideo_config.get("segment_duration", 0) or 0),
            "default_duration_seconds": int(
                longvideo_config.get("default_duration_seconds", 0) or 0
            ),
            "production_profile": str(longvideo_config.get("production_profile") or ""),
            "continuity_mode": str(longvideo_config.get("continuity_mode") or ""),
            "use_tts": bool(longvideo_config.get("use_tts", False)),
            "tts_voice": str(longvideo_config.get("tts_voice", "")),
        },
    }


def _resolve_output_dir(
    repo_root: Path, configured_output_dir: Any, character_name: str
) -> Path:
    base = _resolve_repo_path(
        repo_root, str(configured_output_dir or (repo_root / "output"))
    )
    return Path(base) / character_name.lower()


def _resolve_repo_path(repo_root: Path, raw_path: str) -> Path:
    normalized = raw_path.strip()
    if not normalized:
        return repo_root / "output"
    if normalized.startswith("/app/"):
        return repo_root / normalized.removeprefix("/app/")
    if normalized == "/app":
        return repo_root
    path = Path(normalized)
    if path.is_absolute():
        return path
    return repo_root / path


def _is_media_file(path: str) -> bool:
    return Path(path).suffix.lower() in MEDIA_EXTENSIONS
