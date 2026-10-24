"""Dispatch a selected serial strategy into the existing serial application flow."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from agentic.app.character_config import (
    load_character_config,
    normalize_discord_review_timeout_seconds,
)
from agentic.app.character_requests import CharacterWorkflowRequest


def load_serial_video_settings(
    config_path: Path,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = load_character_config(config_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Character config must contain a YAML mapping: {config_path}")
    character = payload.get("character", {})
    generation = payload.get("generation", {})
    social = payload.get("social_media", {})
    if not isinstance(character, dict) or not isinstance(generation, dict):
        raise ValueError(f"Character config has an invalid character or generation section: {config_path}")
    serial = generation.get("serial_video", {})
    if not isinstance(serial, dict) or not bool(serial.get("enabled", False)):
        raise ValueError(
            f"Serial video is not enabled in {config_path} under generation.serial_video.enabled"
        )
    if not isinstance(social, dict):
        raise ValueError(f"Character config has an invalid social_media section: {config_path}")
    name = str(character.get("name") or config_path.stem).strip()
    return name, dict(generation), dict(serial), dict(social)


def build_serial_video_argv(
    *,
    config_path: Path,
    repo_root: Path,
    episode_index: int,
    prompt: str = "",
    episode_topic: str = "",
    duration_seconds: int | None = None,
    output_dir: str | None = None,
    publish_platforms: Sequence[str] = (),
    publish_mode: str = "",
    dry_run_publish: bool = False,
    comfy_host: str | None = None,
    comfy_port: int | None = None,
    comfy_root: str | None = None,
) -> list[str]:
    character, generation, serial, social = load_serial_video_settings(config_path)
    try:
        duration = int(
            duration_seconds
            if duration_seconds is not None
            else serial.get("default_duration_seconds", 30)
        )
        timeout = normalize_discord_review_timeout_seconds(
            serial.get("discord_review_timeout_seconds")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Serial video duration and review timeout must be integers") from exc
    if duration < 1 or timeout < 1:
        raise ValueError("Serial video duration and Discord review timeout must be positive")

    style = str(serial.get("style") or "cinematic surreal").strip()
    output_value = output_dir or serial.get("output_dir") or generation.get("output_dir") or "output"
    output_path = Path(str(output_value)).expanduser()
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path = output_path.resolve()

    references = serial.get("identity_references", []) or []
    if isinstance(references, str):
        references = [references]
    if not isinstance(references, list):
        raise ValueError("generation.serial_video.identity_references must be a path or list of paths")
    hashtags = serial.get("hashtags") or social.get("default_hashtags") or []
    if isinstance(hashtags, str):
        hashtags = [hashtags]
    if not isinstance(hashtags, list):
        raise ValueError("Serial video hashtags must be a list of strings")

    command = [
        "--media-type", "long_video",
        "--serial-mode",
        "--serial-auto-next",
        "--character", character,
        "--character-config", str(config_path.resolve()),
        "--duration-seconds", str(duration),
        "--style", style,
        "--output-dir", str(output_path),
        "--discord-review-timeout-seconds", str(timeout),
        "--execute",
    ]
    if episode_index == 0 and prompt.strip():
        command.extend(("--goal", prompt.strip()))
    if episode_index == 0 and episode_topic.strip():
        command.extend(("--episode-topic", episode_topic.strip()))
    for raw_path in references:
        path = Path(str(raw_path)).expanduser()
        if not path.is_absolute():
            path = repo_root / path
        command.extend(("--serial-character-reference", str(path.resolve())))
    for platform in publish_platforms:
        command.extend(("--platform", str(platform)))
    for hashtag in hashtags:
        command.extend(("--hashtag", str(hashtag)))
    effective_publish_mode = publish_mode or str(serial.get("publish_mode") or "")
    if effective_publish_mode:
        command.extend(("--publish-mode", effective_publish_mode))
    if dry_run_publish:
        command.append("--dry-run-publish")
    if comfy_host:
        command.extend(("--comfy-host", comfy_host))
    if comfy_port is not None:
        command.extend(("--comfy-port", str(comfy_port)))
    if comfy_root:
        command.extend(("--comfy-root", comfy_root))
    return command


def dispatch_serial_video_request(
    request: CharacterWorkflowRequest,
    *,
    serial_runner: Callable[[Sequence[str]], None] | None = None,
) -> dict[str, Any]:
    if (
        not request.review.publish_after_generate
        or request.review.no_review
        or request.review.stage_probe
    ):
        raise ValueError(
            "serial_video requires Discord review and configured publication; "
            "no-publish, no-review, and stage-probe are unavailable"
        )
    _, _, serial, _ = load_serial_video_settings(request.config_path)
    try:
        episode_count = int(serial.get("episodes_per_run", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError("Serial video episode count must be an integer") from exc
    if episode_count < 1:
        raise ValueError("generation.serial_video.episodes_per_run must be positive")
    if serial_runner is None:
        from agentic.app.main import main as serial_runner
    for episode_index in range(episode_count):
        serial_runner(
            build_serial_video_argv(
                config_path=request.config_path,
                repo_root=request.repo_root,
                episode_index=episode_index,
                prompt=request.generation.prompt,
                duration_seconds=request.generation.duration_seconds,
                output_dir=request.generation.output_dir,
                publish_platforms=request.review.publish_platforms,
                publish_mode=request.review.publish_mode,
                dry_run_publish=request.review.dry_run_publish,
                comfy_host=request.runtime.comfy_host,
                comfy_port=request.runtime.comfy_port,
                comfy_root=request.runtime.comfy_root,
            )
        )
    return {
        "status": "success",
        "generation_type": "serial_video",
        "episodes_started": episode_count,
    }
