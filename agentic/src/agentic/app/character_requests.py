from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CharacterGenerationOptions:
    prompt: str = ""
    temperature: float = 1.0
    preferred_generation_type: str | None = None
    duration_seconds: int | None = None
    output_dir: str | None = None
    news_driven: bool = False
    news_history_path: str | None = None
    routing_history_path: str | None = None
    rng: random.Random | None = None
    selected_character_name: str | None = None
    character_selection: dict[str, Any] | None = None
    reference_video_source: str | None = None
    reference_video_depth: str | None = None
    reference_video_max_keyframes: int | None = None
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class CharacterReviewOptions:
    dry_run_publish: bool = False
    publish_mode: str = ""
    publish_platforms: tuple[str, ...] = ()
    facebook_profile_share_url: str = ""
    publish_after_generate: bool = True
    enable_review_loop: bool = False
    review_notes: str = ""
    no_review: bool = False
    stage_probe: bool = False

@dataclass(frozen=True, slots=True)
class CharacterRuntimeOptions:
    comfy_host: str | None = None
    comfy_port: int | None = None
    comfy_root: str | None = None
    asset_root: Path | None = None
    auto_download_assets: bool = False


@dataclass(frozen=True, slots=True)
class CharacterWorkflowRequest:
    repo_root: Path
    config_path: Path
    generation: CharacterGenerationOptions = field(default_factory=CharacterGenerationOptions)
    review: CharacterReviewOptions = field(default_factory=CharacterReviewOptions)
    runtime: CharacterRuntimeOptions = field(default_factory=CharacterRuntimeOptions)
