"""One canvas and illustration style per serial series, drawn once and kept."""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

from agentic.app.creative_profiles import resolve_creative_profile
from agentic.app.style_catalog import StyleLibrary
from agentic.runtime.serial_paths import serial_series_artifact_root

LOOK_FILE = "look.json"


def draw_serial_look(
    generation: dict[str, Any],
    series_id: str,
    *,
    style_library: StyleLibrary | None = None,
) -> dict[str, dict[str, Any]]:
    """Draw a series' canvas and style from the model's enabled styles, seeded by series id."""
    creative = dict(generation.get("creative_profile") or {})
    canvas = dict(creative.get("canvas") or {})
    if not canvas:
        raise ValueError("generation.creative_profile.canvas is required for serial video")
    style = dict(creative.get("style") or {})
    return resolve_creative_profile(
        {"canvas": canvas, "style": {"mode": "random", "model": style.get("model")}},
        rng=random.Random(f"serial-look:{series_id}"),
        style_library=style_library,
    )


def serial_series_look(
    generation: dict[str, Any],
    series_id: str,
    *,
    output_root: Path,
    style_library: StyleLibrary | None = None,
) -> dict[str, dict[str, Any]]:
    """Return the look a series was born with, drawing and saving it on first use.

    The saved file, not the config, is the source of truth afterwards: editing the
    enabled styles or the canvas weights never changes a series that already has
    its look, so every later episode keeps the style its references were made in.
    """
    path = serial_series_artifact_root(output_root, series_id) / LOOK_FILE
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    look = draw_serial_look(generation, series_id, style_library=style_library)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".look-", suffix=".tmp", delete=False,
    ) as stream:
        json.dump(look, stream, ensure_ascii=False, indent=2)
        temporary = Path(stream.name)
    os.replace(temporary, path)
    return look
