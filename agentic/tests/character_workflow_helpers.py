from __future__ import annotations

from pathlib import Path
from typing import Any

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
    "publish_after_generate",
    "enable_review_loop",
    "review_notes",
    "no_review",
    "stage_probe",
}
_RUNTIME_FIELDS = {"comfy_host", "comfy_port", "comfy_root", "asset_root", "auto_download_assets"}


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
    if "publish_platforms" in review:
        review["publish_platforms"] = tuple(review["publish_platforms"] or ())
    return CharacterWorkflowRequest(
        repo_root=Path(repo_root),
        config_path=Path(config_path),
        generation=CharacterGenerationOptions(**generation),
        review=CharacterReviewOptions(**review),
        runtime=CharacterRuntimeOptions(**runtime),
    )
