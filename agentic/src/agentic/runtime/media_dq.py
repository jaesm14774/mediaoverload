"""Objective output contracts. Creative quality is owned by Discord reviewers."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

from PIL import Image


def expected_subject_count(constraints: dict[str, Any]) -> int | None:
    if "expected_subject_count" in constraints:
        count = constraints["expected_subject_count"]
        if type(count) is not int or count < 0:
            raise ValueError("expected_subject_count must be a non-negative integer")
        return count
    return None


def validate_subject_counts(payload: Any, expected: int, frame_count: int) -> dict[str, Any]:
    counts = payload.get("counts") if isinstance(payload, dict) else None
    valid = (
        isinstance(counts, list) and len(counts) == frame_count
        and all(type(count) is int and count >= 0 for count in counts)
    )
    passed = valid and all(count == expected for count in counts)
    return {
        "passed": bool(passed),
        "status": "pass" if passed else "mismatch" if valid else "unavailable",
        "counts": counts,
        "expected_count": expected,
        "sampled_frame_count": frame_count,
    }


def check_image_contract(path: str, constraints: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    try:
        with Image.open(Path(path)) as image:
            width, height = image.size
            image.verify()
    except (OSError, ValueError) as exc:
        return {"passed": False, "errors": [f"Image cannot be decoded: {exc}"]}
    for key, observed in (("canvas_width", width), ("canvas_height", height)):
        expected = constraints.get(key)
        if expected is not None and observed != int(expected):
            errors.append(f"{key}: expected {expected}, observed {observed}")
    ratio = constraints.get("canvas_aspect_ratio")
    if ratio and not (constraints.get("canvas_width") and constraints.get("canvas_height")):
        expected_ratio = Fraction(str(ratio).replace(":", "/"))
        # The resolved integer canvas may round an ideal aspect ratio by one pixel.
        if abs(width - height * float(expected_ratio)) > 1:
            errors.append(f"aspect ratio: expected {ratio}, observed {width}:{height}")
    return {"passed": not errors, "width": width, "height": height, "errors": errors}
