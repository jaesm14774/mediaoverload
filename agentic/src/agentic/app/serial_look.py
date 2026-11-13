"""One locked canvas and illustration style per serial series."""

from __future__ import annotations

import random
from typing import Any

from agentic.app.creative_profiles import resolve_creative_profile
from agentic.app.style_catalog import StyleLibrary


def serial_series_look(
    generation: dict[str, Any],
    series_id: str,
    *,
    style_library: StyleLibrary | None = None,
) -> dict[str, dict[str, Any]]:
    """Resolve the canvas and style a series keeps for every reference image and episode.

    The draw is seeded by the series id, so any later episode or process lands
    on the same look without storing it, and the style comes only from the
    illustration pool in ``generation.serial_video.style_ids``.
    """
    serial = dict(generation.get("serial_video") or {})
    canvas = dict(dict(generation.get("creative_profile") or {}).get("canvas") or {})
    style_ids = serial.get("style_ids")
    if not canvas or not isinstance(style_ids, list) or not style_ids:
        raise ValueError(
            "generation.creative_profile.canvas and generation.serial_video.style_ids are required for serial video"
        )
    return resolve_creative_profile(
        {
            "canvas": canvas,
            "style": {"mode": "random", "catalog": "krea2", "ids": list(style_ids)},
        },
        rng=random.Random(f"serial-look:{series_id}"),
        style_library=style_library,
    )
