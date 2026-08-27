from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agentic.app.character_requests import (
    CharacterGenerationOptions,
    CharacterReviewOptions,
    CharacterRuntimeOptions,
    CharacterWorkflowRequest,
)


_GENERATION_FIELDS = {
    "prompt",
    "temperature",
    "preferred_generation_type",
    "duration_seconds",
    "output_dir",
    "news_driven",
    "news_history_path",
    "routing_history_path",
    "rng",
    "selected_character_name",
    "character_selection",
}
_REVIEW_FIELDS = {
    "dry_run_publish",
    "publish_mode",
    "publish_platforms",
    "facebook_profile_share_url",
    "publish_after_generate",
    "enable_review_loop",
    "review_notes",
    "no_review",
    "stage_probe",
}
_RUNTIME_FIELDS = {"comfy_host", "comfy_port", "comfy_root", "asset_root", "auto_download_assets"}


def _test_resolved_selection(config_path: Path) -> tuple[str, dict[str, Any]] | None:
    """Provide a deterministic selection for payload-only tests using Kirby's random config."""
    if config_path.name.casefold() != "kirby.yaml":
        return None
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    config = config if isinstance(config, dict) else {}
    generation = config.get("generation", {})
    generation = generation if isinstance(generation, dict) else {}
    if str(generation.get("subject_mode") or "single").strip().casefold() != "random":
        return None
    character = config.get("character", {})
    character = character if isinstance(character, dict) else {}
    selected_name = str(character.get("name") or "Kirby").strip()
    group_name = str(character.get("group_name") or "").strip()
    return selected_name, {
        "mode": "config",
        "subject_mode": "single",
        "configured_subject_mode": "random",
        "group_name": group_name,
        "selected_character": selected_name,
        "candidate_count": 1,
        "candidates": [{"name": selected_name, "status": 1, "weight": 1.0}],
        "selected_profile": {},
        "selection_source": "test_fixture_resolved_selection",
    }


def make_character_workflow_request(
    repo_root: Path,
    config_path: str | Path,
    **values: Any,
) -> CharacterWorkflowRequest:
    unknown = set(values) - _GENERATION_FIELDS - _REVIEW_FIELDS - _RUNTIME_FIELDS
    if unknown:
        raise TypeError(f"Unsupported character workflow test options: {sorted(unknown)}")
    generation = {key: values[key] for key in _GENERATION_FIELDS if key in values}
    review = {key: values[key] for key in _REVIEW_FIELDS if key in values}
    runtime = {key: values[key] for key in _RUNTIME_FIELDS if key in values}
    if "selected_character_name" not in generation and "character_selection" not in generation:
        resolved_selection = _test_resolved_selection(Path(config_path))
        if resolved_selection is not None:
            selected_name, selection = resolved_selection
            generation["selected_character_name"] = selected_name
            generation["character_selection"] = selection
    if "publish_platforms" in review:
        review["publish_platforms"] = tuple(review["publish_platforms"] or ())
    return CharacterWorkflowRequest(
        repo_root=Path(repo_root),
        config_path=Path(config_path),
        generation=CharacterGenerationOptions(**generation),
        review=CharacterReviewOptions(**review),
        runtime=CharacterRuntimeOptions(**runtime),
    )
