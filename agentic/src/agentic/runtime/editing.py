from __future__ import annotations

import math
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
TRANSITIONS = {"fade", "fadeblack", "fadewhite", "wipeleft", "wiperight", "smoothleft", "circleopen"}
MOTIONS = {"none", "slow_zoom_in", "slow_zoom_out", "pan_left", "pan_right", "drift_up", "drift_down"}
EDIT_PROFILES = {
    "baseline_concat",
    "motion_cut_v1",
    "xfade_clean_v1",
    "chapter_dip_v1",
    "editorial_kinetic_v1",
}

MAX_CLIPS = 64
MAX_OUTPUT_DIMENSION = 4096
MAX_FPS = 120.0
MAX_CLIP_DURATION_SECONDS = 300.0
MAX_SOURCE_START_SECONDS = 3600.0
MAX_TARGET_DURATION_SECONDS = 600.0
MAX_TOTAL_DURATION_SECONDS = 600.0
MAX_TRANSITION_DURATION_SECONDS = 30.0
MAX_METADATA_BYTES = 32 * 1024
MAX_RENDER_WORK = 25_000_000_000.0


class EditPlanError(ValueError):
    """Raised when an agent-provided edit plan is unsafe or incomplete."""


@dataclass(frozen=True, slots=True)
class EditClip:
    path: str
    duration_seconds: float | None = None
    source_start_seconds: float = 0.0
    motion: str = "none"
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "duration_seconds": self.duration_seconds,
            "source_start_seconds": self.source_start_seconds,
            "motion": self.motion,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EditClip":
        return cls(
            path=str(payload.get("path") or "").strip(),
            duration_seconds=(
                float(payload["duration_seconds"])
                if payload.get("duration_seconds") is not None
                and payload.get("duration_seconds") != ""
                else None
            ),
            source_start_seconds=float(payload.get("source_start_seconds", 0.0) or 0.0),
            motion=str(payload.get("motion") or "none").strip().lower(),
            label=str(payload.get("label") or "").strip(),
        )


@dataclass(frozen=True, slots=True)
class EditTransition:
    name: str = "fade"
    duration_seconds: float = 0.10

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "duration_seconds": self.duration_seconds}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EditTransition":
        return cls(
            name=str(payload.get("name") or "fade").strip().lower(),
            duration_seconds=float(payload.get("duration_seconds", 0.10) or 0.0),
        )


@dataclass(frozen=True, slots=True)
class EditPlan:
    clips: tuple[EditClip, ...]
    transitions: tuple[EditTransition, ...] = ()
    output_width: int = 576
    output_height: int = 1024
    fps: float = 24.0
    target_duration_seconds: float | None = None
    profile: str = "xfade_clean_v1"
    variant_seed: int = 0
    audio_crossfade: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "EditPlan":
        if not self.clips:
            raise EditPlanError("EditPlan requires at least one clip")
        if len(self.clips) > MAX_CLIPS:
            raise EditPlanError(f"EditPlan cannot contain more than {MAX_CLIPS} clips")
        if self.output_width <= 0 or self.output_height <= 0:
            raise EditPlanError("EditPlan canvas dimensions must be positive")
        if self.output_width > MAX_OUTPUT_DIMENSION or self.output_height > MAX_OUTPUT_DIMENSION:
            raise EditPlanError(f"EditPlan canvas dimensions cannot exceed {MAX_OUTPUT_DIMENSION}")
        if self.output_width % 2 or self.output_height % 2:
            raise EditPlanError("EditPlan canvas dimensions must be even for H.264")
        if not math.isfinite(self.fps) or self.fps <= 0 or self.fps > MAX_FPS:
            raise EditPlanError(f"EditPlan fps must be between 0 and {MAX_FPS}")
        if self.profile not in EDIT_PROFILES:
            raise EditPlanError(f"Unknown edit profile: {self.profile}")
        if (
            self.target_duration_seconds is not None
            and (not math.isfinite(self.target_duration_seconds) or self.target_duration_seconds <= 0)
        ):
            raise EditPlanError("EditPlan target duration must be positive")
        if self.target_duration_seconds is not None and self.target_duration_seconds > MAX_TARGET_DURATION_SECONDS:
            raise EditPlanError(f"EditPlan target duration cannot exceed {MAX_TARGET_DURATION_SECONDS}")
        try:
            metadata_size = len(json.dumps(self.metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        except (TypeError, ValueError) as exc:
            raise EditPlanError("EditPlan metadata must be JSON serializable") from exc
        if metadata_size > MAX_METADATA_BYTES:
            raise EditPlanError(f"EditPlan metadata cannot exceed {MAX_METADATA_BYTES} bytes")
        for clip in self.clips:
            path = Path(clip.path)
            if not clip.path:
                raise EditPlanError("EditPlan clip path cannot be empty")
            if path.suffix.lower() not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
                raise EditPlanError(f"Unsupported edit media type: {clip.path}")
            if (
                clip.duration_seconds is not None
                and (not math.isfinite(clip.duration_seconds) or clip.duration_seconds <= 0)
            ):
                raise EditPlanError(f"Clip duration must be positive: {clip.path}")
            if clip.duration_seconds is not None and clip.duration_seconds > MAX_CLIP_DURATION_SECONDS:
                raise EditPlanError(f"Clip duration cannot exceed {MAX_CLIP_DURATION_SECONDS}: {clip.path}")
            if not math.isfinite(clip.source_start_seconds) or clip.source_start_seconds < 0:
                raise EditPlanError(f"Clip source start cannot be negative: {clip.path}")
            if clip.source_start_seconds > MAX_SOURCE_START_SECONDS:
                raise EditPlanError(f"Clip source start cannot exceed {MAX_SOURCE_START_SECONDS}: {clip.path}")
            if clip.motion not in MOTIONS:
                raise EditPlanError(f"Unknown clip motion: {clip.motion}")
        if len(self.transitions) not in {0, len(self.clips) - 1}:
            raise EditPlanError("EditPlan transitions must contain one entry per clip boundary")
        if self.profile not in {"baseline_concat", "motion_cut_v1"} and len(self.clips) > 1 and not self.transitions:
            raise EditPlanError("Transition profile requires one transition per clip boundary")
        for transition in self.transitions:
            if transition.name not in TRANSITIONS:
                raise EditPlanError(f"Unknown transition: {transition.name}")
            if not math.isfinite(transition.duration_seconds) or transition.duration_seconds <= 0:
                raise EditPlanError("Transition duration must be positive")
            if transition.duration_seconds > MAX_TRANSITION_DURATION_SECONDS:
                raise EditPlanError(f"Transition duration cannot exceed {MAX_TRANSITION_DURATION_SECONDS}")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": 1,
            "clips": [clip.to_dict() for clip in self.clips],
            "transitions": [transition.to_dict() for transition in self.transitions],
            "output_width": self.output_width,
            "output_height": self.output_height,
            "fps": self.fps,
            "target_duration_seconds": self.target_duration_seconds,
            "profile": self.profile,
            "variant_seed": self.variant_seed,
            "audio_crossfade": self.audio_crossfade,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EditPlan":
        if int(payload.get("schema_version", 1)) != 1:
            raise EditPlanError("Unsupported EditPlan schema_version")
        raw_clips = payload.get("clips", [])
        raw_transitions = payload.get("transitions", [])
        raw_metadata = payload.get("metadata") or {}
        raw_audio_crossfade = payload.get("audio_crossfade", True)
        if not isinstance(raw_clips, list) or not isinstance(raw_transitions, list):
            raise EditPlanError("EditPlan clips and transitions must be arrays")
        if not isinstance(raw_metadata, dict):
            raise EditPlanError("EditPlan metadata must be an object")
        if not isinstance(raw_audio_crossfade, bool):
            raise EditPlanError("EditPlan audio_crossfade must be a boolean")
        if any(not isinstance(item, dict) for item in raw_clips + raw_transitions):
            raise EditPlanError("EditPlan clips and transitions must contain objects")
        plan = cls(
            clips=tuple(EditClip.from_dict(item) for item in raw_clips),
            transitions=tuple(EditTransition.from_dict(item) for item in raw_transitions),
            output_width=int(payload.get("output_width", 576)),
            output_height=int(payload.get("output_height", 1024)),
            fps=float(payload.get("fps", 24.0)),
            target_duration_seconds=(
                float(payload["target_duration_seconds"])
                if payload.get("target_duration_seconds") is not None
                and payload.get("target_duration_seconds") != ""
                else None
            ),
            profile=str(payload.get("profile") or "xfade_clean_v1"),
            variant_seed=int(payload.get("variant_seed", 0) or 0),
            audio_crossfade=raw_audio_crossfade,
            metadata=dict(raw_metadata),
        )
        return plan.validate()


def build_edit_plan(
    paths: list[str],
    *,
    profile: str = "xfade_clean_v1",
    output_width: int = 576,
    output_height: int = 1024,
    fps: float = 24.0,
    target_duration_seconds: float | None = None,
    variant_seed: int = 0,
    transition_duration_seconds: float = 0.10,
) -> EditPlan:
    """Build a deterministic plan from ordered generated assets or segments."""

    normalized_paths = [str(Path(path).expanduser().resolve()) for path in paths if str(path).strip()]
    if not normalized_paths:
        raise EditPlanError("At least one input path is required")
    if profile not in EDIT_PROFILES:
        raise EditPlanError(f"Unknown edit profile: {profile}")

    if profile == "baseline_concat":
        transitions: tuple[EditTransition, ...] = ()
        motions = ["none"] * len(normalized_paths)
    elif profile == "motion_cut_v1":
        transitions = ()
        motion_names = ("slow_zoom_in", "pan_left", "slow_zoom_out", "pan_right")
        motions = [
            motion_names[(variant_seed + index) % len(motion_names)]
            if Path(path).suffix.lower() in IMAGE_SUFFIXES
            else "none"
            for index, path in enumerate(normalized_paths)
        ]
    elif profile == "xfade_clean_v1":
        transitions = tuple(EditTransition("fade", transition_duration_seconds) for _ in normalized_paths[1:])
        motions = ["slow_zoom_in" if Path(path).suffix.lower() in IMAGE_SUFFIXES else "none" for path in normalized_paths]
    elif profile == "chapter_dip_v1":
        transitions = tuple(EditTransition("fadeblack", transition_duration_seconds) for _ in normalized_paths[1:])
        motions = ["slow_zoom_in" if Path(path).suffix.lower() in IMAGE_SUFFIXES else "none" for path in normalized_paths]
    else:
        transition_names = ("fade", "wipeleft", "wiperight", "smoothleft", "circleopen")
        motion_names = ("slow_zoom_in", "pan_left", "slow_zoom_out", "pan_right", "drift_up", "drift_down")
        transition_offset = variant_seed % len(transition_names)
        motion_offset = variant_seed % len(motion_names)
        transitions = tuple(
            EditTransition(
                transition_names[(transition_offset + index) % len(transition_names)],
                transition_duration_seconds,
            )
            for index in range(len(normalized_paths) - 1)
        )
        motions = [
            motion_names[(motion_offset + index) % len(motion_names)]
            if Path(path).suffix.lower() in IMAGE_SUFFIXES
            else "none"
            for index, path in enumerate(normalized_paths)
        ]

    return EditPlan(
        clips=tuple(EditClip(path=path, motion=motions[index], label=f"clip-{index + 1:02d}") for index, path in enumerate(normalized_paths)),
        transitions=transitions,
        output_width=int(output_width),
        output_height=int(output_height),
        fps=float(fps),
        target_duration_seconds=target_duration_seconds,
        profile=profile,
        variant_seed=int(variant_seed),
        metadata={"source": "build_edit_plan", "ordered_input_count": len(normalized_paths)},
    ).validate()
