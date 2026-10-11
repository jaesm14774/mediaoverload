"""Three-beat story contracts; image meaning is checked through visual review."""

from __future__ import annotations

import re
from typing import Any

MOTION_COMIC_ROLES = ("setup", "escalation", "payoff")
MOTION_COMIC_DEFAULTS = {
    "duration_seconds": 9,
    "width": 576,
    "height": 1024,
    "fps": 24,
    "image_width": 1024,
    "image_height": 768,
    "steps": 8,
}
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def validate_motion_comic_story(payload: Any) -> dict[str, Any]:
    """Reject incomplete stories before GPU work; this is not a pixel validator."""
    if not isinstance(payload, dict):
        raise ValueError("Motion comic story must be an object")

    def text(
        source: dict, key: str, *, maximum: int | None = None, english: bool = False
    ) -> str:
        value = source.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Motion comic {key} must be non-empty text")
        value = value.strip()
        if maximum is not None and len(value) > maximum:
            raise ValueError(
                f"Motion comic {key} must contain at most {maximum} characters"
            )
        if english and _CJK.search(value):
            raise ValueError(f"Motion comic {key} must be English-only")
        return value

    panels = payload.get("panels")
    if not isinstance(panels, list) or len(panels) != 3:
        raise ValueError("Motion comic requires exactly three story beats")
    normalized = {
        "title": text(payload, "title", maximum=12),
        "visual_style": text(payload, "visual_style", english=True),
        "continuity": text(payload, "continuity", english=True),
        "panels": [],
    }
    for role, panel in zip(MOTION_COMIC_ROLES, panels):
        if not isinstance(panel, dict) or panel.get("role") != role:
            raise ValueError(
                "Motion comic beats must be ordered setup, escalation, payoff"
            )
        normalized["panels"].append(
            {
                "role": role,
                "caption": text(panel, "caption", maximum=24),
                "image_prompt": text(panel, "image_prompt", english=True),
                "visible_state": text(panel, "visible_state", english=True),
                "cause": text(panel, "cause", english=True),
            }
        )
    return normalized
