from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import Any


ALLOWED_ASPECT_RATIOS = frozenset({"16:9", "4:5", "1:1"})
_REQUIRED_STYLE_FIELDS = ("label", "prompt")
_DEFAULT_CREATIVE_PROFILE = {
    "canvas": {
        "mode": "fixed",
        "fixed": "landscape_16_9",
        "profiles": {
            "landscape_16_9": {"aspect_ratio": "16:9", "width": 640, "height": 360},
        },
    },
    "style": {
        "mode": "fixed",
        "fixed": "clean_cel_action",
        "profiles": {
            "clean_cel_action": {
                "label": "clean cel-shaded action",
                "prompt": "polished cel-shaded animation with a designed layered setting and expressive physical action",
                "composition": "readable foreground, middle action plane, and background depth",
                "palette": "controlled saturated color with clear subject separation",
                "motion": "anticipation, contact, recoil, and a settled pose",
                "avoid": "sterile white voids and flat icon composition",
            },
        },
    },
}


def resolve_creative_profile(
    config: dict[str, Any],
    *,
    rng: random.Random | None = None,
) -> dict[str, dict[str, Any]]:
    """Resolve one run-level canvas and visual language from config.

    The selected canvas is the only spatial source of truth for the run. The
    style profile is kept as structured data so prompt builders can use its
    composition/material guidance without turning the config into a single
    giant prompt string.
    """

    if not isinstance(config, dict) or not config:
        config = deepcopy(_DEFAULT_CREATIVE_PROFILE)
    canvas = _resolve_canvas(dict(config.get("canvas") or {}), rng=rng)
    style = _resolve_style(dict(config.get("style") or {}), rng=rng)
    return {"canvas": canvas, "style": style}


def _resolve_canvas(config: dict[str, Any], *, rng: random.Random | None) -> dict[str, Any]:
    profiles = _mapping(config, "profiles", "generation.creative_profile.canvas.profiles")
    profile_id = _select_profile_id(config, profiles, "canvas", rng=rng)
    raw = _mapping(profiles, profile_id, f"canvas profile '{profile_id}'")
    aspect_ratio = str(raw.get("aspect_ratio") or "").strip()
    if aspect_ratio not in ALLOWED_ASPECT_RATIOS:
        raise ValueError(
            f"canvas profile '{profile_id}' must use one of: {', '.join(sorted(ALLOWED_ASPECT_RATIOS))}"
        )
    try:
        width = int(raw["width"])
        height = int(raw["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"canvas profile '{profile_id}' must define integer width and height") from exc
    if width < 256 or height < 256 or width % 8 or height % 8:
        raise ValueError(f"canvas profile '{profile_id}' dimensions must be >=256 and divisible by 8")
    numerator, denominator = (int(part) for part in aspect_ratio.split(":", 1))
    if width * denominator != height * numerator:
        raise ValueError(f"canvas profile '{profile_id}' dimensions do not match aspect ratio {aspect_ratio}")
    return {
        "id": profile_id,
        "aspect_ratio": aspect_ratio,
        "width": width,
        "height": height,
    }


def _resolve_style(config: dict[str, Any], *, rng: random.Random | None) -> dict[str, Any]:
    profiles = _mapping(config, "profiles", "generation.creative_profile.style.profiles")
    profile_id = _select_profile_id(config, profiles, "style", rng=rng)
    raw = _mapping(profiles, profile_id, f"style profile '{profile_id}'")
    missing = [field for field in _REQUIRED_STYLE_FIELDS if not str(raw.get(field) or "").strip()]
    if missing:
        raise ValueError(f"style profile '{profile_id}' is missing: {', '.join(missing)}")
    return {
        "id": profile_id,
        "label": str(raw["label"]).strip(),
        "prompt": str(raw["prompt"]).strip(),
        "composition": str(raw.get("composition") or "").strip(),
        "palette": str(raw.get("palette") or "").strip(),
        "motion": str(raw.get("motion") or "").strip(),
        "avoid": str(raw.get("avoid") or "").strip(),
    }


def _select_profile_id(
    config: dict[str, Any],
    profiles: dict[str, Any],
    kind: str,
    *,
    rng: random.Random | None,
) -> str:
    mode = str(config.get("mode") or "").strip().lower()
    if mode not in {"fixed", "random"}:
        raise ValueError(f"generation.creative_profile.{kind}.mode must be 'fixed' or 'random'")
    if mode == "fixed":
        profile_id = str(config.get("fixed") or "").strip()
        if profile_id not in profiles:
            raise ValueError(f"fixed {kind} profile '{profile_id}' is not defined")
        return profile_id
    weights = config.get("random_weights")
    if not isinstance(weights, dict):
        raise ValueError(f"random {kind} selection requires random_weights")
    candidates: list[tuple[str, float]] = []
    for raw_id, raw_weight in weights.items():
        profile_id = str(raw_id).strip()
        if profile_id not in profiles:
            raise ValueError(f"random {kind} weight references unknown profile '{profile_id}'")
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"random {kind} weight for '{profile_id}' must be a positive number") from exc
        if math.isfinite(weight) and weight > 0:
            candidates.append((profile_id, weight))
    if not candidates:
        raise ValueError(f"random {kind} selection requires at least one positive weight")
    chooser = rng or random
    threshold = chooser.uniform(0, sum(weight for _, weight in candidates))
    cumulative = 0.0
    for profile_id, weight in candidates:
        cumulative += weight
        if threshold <= cumulative:
            return profile_id
    return candidates[-1][0]


def _mapping(value: Any, key: str, label: str) -> dict[str, Any]:
    result = value.get(key) if isinstance(value, dict) else None
    if not isinstance(result, dict) or not result:
        raise ValueError(f"{label} must be a non-empty mapping")
    return {str(name): item for name, item in result.items()}
