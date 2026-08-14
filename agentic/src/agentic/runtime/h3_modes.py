"""Canonical MiniMax H3 mode contracts used by planning and E2E execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class H3Mode(str, Enum):
    T2VA = "t2va"
    I2VA = "i2va"
    FL2VA = "fl2va"
    L2VA = "l2va"
    REF2VA = "ref2va"


@dataclass(frozen=True, slots=True)
class H3ModeContract:
    mode: H3Mode
    generation_type: str
    render_mode: str
    workflow_name: str
    requires_first_frame: bool = False
    requires_last_frame: bool = False
    allows_reference_images: bool = False
    allows_reference_videos: bool = False


MODE_CONTRACTS: dict[H3Mode, H3ModeContract] = {
    H3Mode.T2VA: H3ModeContract(
        mode=H3Mode.T2VA,
        generation_type="native_h3_t2v_story",
        render_mode="text_to_video",
        workflow_name="minimax_h3_lowvram_t2v",
    ),
    H3Mode.I2VA: H3ModeContract(
        mode=H3Mode.I2VA,
        generation_type="native_h3_story",
        render_mode="image_to_video",
        workflow_name="minimax_h3_lowvram_i2v",
        requires_first_frame=True,
    ),
    H3Mode.FL2VA: H3ModeContract(
        mode=H3Mode.FL2VA,
        generation_type="native_h3_fl2va_story",
        render_mode="first_last_to_video",
        workflow_name="minimax_h3_lowvram_15s_fl2va_i2v",
        requires_first_frame=True,
        requires_last_frame=True,
    ),
    H3Mode.L2VA: H3ModeContract(
        mode=H3Mode.L2VA,
        generation_type="native_h3_l2va_story",
        render_mode="last_frame_to_video",
        workflow_name="minimax_h3_lowvram_15s_fl2va_i2v",
        requires_last_frame=True,
    ),
    H3Mode.REF2VA: H3ModeContract(
        mode=H3Mode.REF2VA,
        generation_type="native_h3_ref2va",
        render_mode="reference_to_video",
        workflow_name="minimax_h3_ref2va",
        allows_reference_images=True,
        allows_reference_videos=True,
    ),
}

_ALIASES = {
    "t2v": H3Mode.T2VA,
    "t2va": H3Mode.T2VA,
    "native_h3_t2v_story": H3Mode.T2VA,
    "text_to_video": H3Mode.T2VA,
    "i2v": H3Mode.I2VA,
    "i2va": H3Mode.I2VA,
    "native_h3_story": H3Mode.I2VA,
    "image_to_video": H3Mode.I2VA,
    "fl2va": H3Mode.FL2VA,
    "native_h3_fl2va_story": H3Mode.FL2VA,
    "first_last_to_video": H3Mode.FL2VA,
    "l2va": H3Mode.L2VA,
    "native_h3_l2va_story": H3Mode.L2VA,
    "last_frame_to_video": H3Mode.L2VA,
    "ref2va": H3Mode.REF2VA,
    "native_h3_ref2va": H3Mode.REF2VA,
    "text2image2native_h3_ref2va": H3Mode.REF2VA,
    "reference_to_video": H3Mode.REF2VA,
}


def resolve_h3_mode(value: str | H3Mode | None, *, render_mode: str | None = None) -> H3Mode:
    """Resolve a public generation type/render mode into one canonical H3 mode."""

    if isinstance(value, H3Mode):
        return value
    for candidate in (value, render_mode):
        key = str(candidate or "").strip().lower().replace("-", "_")
        if key in _ALIASES:
            return _ALIASES[key]
    raise ValueError(f"Unknown H3 mode: {value or render_mode!r}")


def mode_contract(value: str | H3Mode | None, *, render_mode: str | None = None) -> H3ModeContract:
    return MODE_CONTRACTS[resolve_h3_mode(value, render_mode=render_mode)]


def validate_h3_payload(
    value: str | H3Mode,
    payload: dict[str, Any],
    *,
    render_mode: str | None = None,
) -> None:
    """Reject conditioning inputs that violate the selected H3 mode."""

    contract = mode_contract(value, render_mode=render_mode)
    first_frame = str(payload.get("image_path") or payload.get("input_image_path") or "").strip()
    last_frame = str(payload.get("last_image_path") or payload.get("last_frame_path") or "").strip()
    references = payload.get("reference_manifest") or []
    reference_image_paths = payload.get("reference_image_paths") or []
    reference_video_paths = payload.get("reference_video_paths") or []
    image_refs = [item for item in references if isinstance(item, dict) and item.get("type") == "image"]
    video_refs = [item for item in references if isinstance(item, dict) and item.get("type") == "video"]
    has_image_references = bool(image_refs or reference_image_paths)
    has_video_references = bool(video_refs or reference_video_paths)
    has_references = bool(references or reference_image_paths or reference_video_paths)
    errors: list[str] = []

    if contract.requires_first_frame and not first_frame:
        errors.append(f"{contract.mode.value} requires an opening image")
    if not contract.requires_first_frame and first_frame and contract.mode is H3Mode.T2VA:
        errors.append("t2va must not receive an image conditioning input")
    if contract.requires_last_frame and not last_frame and contract.mode is not H3Mode.REF2VA:
        errors.append(f"{contract.mode.value} requires a landing/last-frame image")
    if contract.mode is H3Mode.I2VA and last_frame:
        errors.append("i2va must not receive a last-frame conditioning input")
    if contract.mode is H3Mode.L2VA and first_frame:
        errors.append("l2va must not receive an opening-frame conditioning input")
    if contract.mode is not H3Mode.REF2VA and has_references:
        errors.append(f"{contract.mode.value} does not accept a reference manifest")
    if contract.mode is H3Mode.REF2VA and not (has_image_references or has_video_references):
        errors.append("ref2va requires at least one image or video reference")
    if contract.mode is H3Mode.REF2VA and payload.get("reference_audio_paths"):
        errors.append("reference audio is intentionally disabled for ref2va")
    if errors:
        raise ValueError("H3 mode contract failed: " + "; ".join(errors))
