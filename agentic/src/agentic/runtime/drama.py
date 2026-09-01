"""Short-drama planning contracts and the compiler to the edit timeline."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentic.runtime.editing import (
    IMAGE_SUFFIXES,
    MAX_CLIP_DURATION_SECONDS,
    MAX_FPS,
    MAX_METADATA_BYTES,
    MAX_OUTPUT_DIMENSION,
    MAX_SOURCE_START_SECONDS,
    MAX_TOTAL_DURATION_SECONDS,
    MOTIONS,
    TRANSITIONS,
    EditClip,
    EditPlan,
    EditTransition,
)


DRAMA_PLAN_SCHEMA_VERSION = 1
DRAMA_REVIEW_STATUSES = {"pending", "approved", "rejected"}
DRAMA_TRANSITION_CUTS = {"cut", "hard_cut"}
DRAMA_MAX_ID_LENGTH = 160
DRAMA_MAX_TEXT_LENGTH = 8_000
DRAMA_AUDIO_SUFFIXES = {".wav", ".mp3", ".aac", ".m4a", ".flac", ".ogg", ".opus"}


class DramaPlanError(ValueError):
    """Raised when a short-drama plan is incomplete or unsafe to compile."""


def _required_text(value: Any, field_name: str, *, max_length: int = DRAMA_MAX_TEXT_LENGTH) -> str:
    if not isinstance(value, str):
        raise DramaPlanError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise DramaPlanError(f"{field_name} cannot be empty")
    if len(normalized) > max_length:
        raise DramaPlanError(f"{field_name} cannot exceed {max_length} characters")
    return normalized


def _optional_text(value: Any, field_name: str, *, max_length: int = DRAMA_MAX_TEXT_LENGTH) -> str | None:
    if value is None or value == "":
        return None
    return _required_text(value, field_name, max_length=max_length)


def _string_tuple(value: Any, field_name: str, *, required: bool = False) -> tuple[str, ...]:
    if value is None:
        values: list[Any] = []
    elif isinstance(value, list):
        values = value
    elif isinstance(value, tuple):
        values = list(value)
    else:
        raise DramaPlanError(f"{field_name} must be an array of strings")
    result = tuple(_required_text(item, f"{field_name}[]", max_length=DRAMA_MAX_ID_LENGTH) for item in values)
    if len(set(result)) != len(result):
        raise DramaPlanError(f"{field_name} cannot contain duplicates")
    if required and not result:
        raise DramaPlanError(f"{field_name} cannot be empty")
    return result


def _path_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        values: list[Any] = []
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        raise DramaPlanError(f"{field_name} must be an array of paths")
    result = tuple(_required_text(item, f"{field_name}[]", max_length=4_000) for item in values)
    if len(set(result)) != len(result):
        raise DramaPlanError(f"{field_name} cannot contain duplicates")
    for path in result:
        _optional_path(path, f"{field_name}[]")
    return result


def _object_list(value: Any, field_name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise DramaPlanError(f"{field_name} must be an array of objects")
    return value


def _finite_float(value: Any, field_name: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise DramaPlanError(f"{field_name} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DramaPlanError(f"{field_name} must be a number") from exc
    if not math.isfinite(result) or result < minimum:
        raise DramaPlanError(f"{field_name} must be finite and at least {minimum}")
    return result


def _optional_finite_float(value: Any, field_name: str, *, minimum: float = 0.0) -> float | None:
    if value is None or value == "":
        return None
    return _finite_float(value, field_name, minimum=minimum)


def _optional_path(value: Any, field_name: str) -> str | None:
    raw = _optional_text(value, field_name, max_length=4_000)
    if raw is None:
        return None
    suffix = Path(raw).suffix.lower()
    if suffix not in IMAGE_SUFFIXES and suffix not in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}:
        raise DramaPlanError(f"{field_name} must point to an image or video file: {raw}")
    return raw


def _optional_audio_path(value: Any, field_name: str) -> str | None:
    raw = _optional_text(value, field_name, max_length=4_000)
    if raw is None:
        return None
    if Path(raw).suffix.lower() not in DRAMA_AUDIO_SUFFIXES:
        raise DramaPlanError(f"{field_name} must point to an audio file: {raw}")
    return raw


@dataclass(frozen=True, slots=True)
class DialogueCue:
    """Dialogue metadata relative to its scene; audio rendering is a later layer."""

    character_id: str
    text: str
    start_seconds: float = 0.0
    duration_seconds: float | None = None
    audio_path: str | None = None

    def validate(self, scene_duration_seconds: float) -> "DialogueCue":
        _required_text(self.character_id, "dialogue.character_id", max_length=DRAMA_MAX_ID_LENGTH)
        _required_text(self.text, "dialogue.text")
        start = _finite_float(self.start_seconds, "dialogue.start_seconds")
        duration = _optional_finite_float(self.duration_seconds, "dialogue.duration_seconds", minimum=0.001)
        if start >= scene_duration_seconds:
            raise DramaPlanError("dialogue.start_seconds must be inside its scene")
        if duration is not None and start + duration > scene_duration_seconds + 0.001:
            raise DramaPlanError("dialogue cue must finish inside its scene")
        if self.audio_path is not None:
            _optional_audio_path(self.audio_path, "dialogue.audio_path")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "character_id": self.character_id,
            "text": self.text,
            "start_seconds": self.start_seconds,
            "duration_seconds": self.duration_seconds,
            "audio_path": self.audio_path,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DialogueCue":
        return cls(
            character_id=str(payload.get("character_id") or "").strip(),
            text=str(payload.get("text") or "").strip(),
            start_seconds=_finite_float(payload.get("start_seconds", 0.0), "dialogue.start_seconds"),
            duration_seconds=_optional_finite_float(payload.get("duration_seconds"), "dialogue.duration_seconds", minimum=0.001),
            audio_path=_optional_text(payload.get("audio_path"), "dialogue.audio_path", max_length=4_000),
        )


@dataclass(frozen=True, slots=True)
class SfxCue:
    """Sound-effect metadata relative to its scene; mixing is a later layer."""

    name: str
    start_seconds: float
    duration_seconds: float = 0.25
    audio_path: str | None = None
    volume: float = 1.0

    def validate(self, scene_duration_seconds: float) -> "SfxCue":
        _required_text(self.name, "sfx.name", max_length=DRAMA_MAX_ID_LENGTH)
        start = _finite_float(self.start_seconds, "sfx.start_seconds")
        duration = _finite_float(self.duration_seconds, "sfx.duration_seconds", minimum=0.001)
        if start >= scene_duration_seconds:
            raise DramaPlanError("sfx.start_seconds must be inside its scene")
        if start + duration > scene_duration_seconds + 0.001:
            raise DramaPlanError("sfx cue must finish inside its scene")
        volume = _finite_float(self.volume, "sfx.volume")
        if volume > 4.0:
            raise DramaPlanError("sfx.volume cannot exceed 4.0")
        if self.audio_path is not None:
            _optional_audio_path(self.audio_path, "sfx.audio_path")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start_seconds": self.start_seconds,
            "duration_seconds": self.duration_seconds,
            "audio_path": self.audio_path,
            "volume": self.volume,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SfxCue":
        return cls(
            name=str(payload.get("name") or "").strip(),
            start_seconds=_finite_float(payload.get("start_seconds", 0.0), "sfx.start_seconds"),
            duration_seconds=_finite_float(payload.get("duration_seconds", 0.25), "sfx.duration_seconds", minimum=0.001),
            audio_path=_optional_text(payload.get("audio_path"), "sfx.audio_path", max_length=4_000),
            volume=_finite_float(payload.get("volume", 1.0), "sfx.volume"),
        )


@dataclass(frozen=True, slots=True)
class ContinuityContract:
    """Scene-level continuity anchors that must survive generation and editing."""

    character_ids: tuple[str, ...] = ()
    prop_ids: tuple[str, ...] = ()
    location: str = ""
    style_anchor: str = ""
    opening_frame_path: str | None = None
    ending_frame_path: str | None = None

    def validate(self) -> "ContinuityContract":
        _string_tuple(self.character_ids, "continuity.character_ids")
        _string_tuple(self.prop_ids, "continuity.prop_ids")
        if self.location:
            _required_text(self.location, "continuity.location")
        if self.style_anchor:
            _required_text(self.style_anchor, "continuity.style_anchor")
        if self.opening_frame_path is not None:
            _optional_path(self.opening_frame_path, "continuity.opening_frame_path")
        if self.ending_frame_path is not None:
            _optional_path(self.ending_frame_path, "continuity.ending_frame_path")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "character_ids": list(self.character_ids),
            "prop_ids": list(self.prop_ids),
            "location": self.location,
            "style_anchor": self.style_anchor,
            "opening_frame_path": self.opening_frame_path,
            "ending_frame_path": self.ending_frame_path,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ContinuityContract":
        return cls(
            character_ids=_string_tuple(payload.get("character_ids"), "continuity.character_ids"),
            prop_ids=_string_tuple(payload.get("prop_ids"), "continuity.prop_ids"),
            location=str(payload.get("location") or "").strip(),
            style_anchor=str(payload.get("style_anchor") or "").strip(),
            opening_frame_path=_optional_text(payload.get("opening_frame_path"), "continuity.opening_frame_path", max_length=4_000),
            ending_frame_path=_optional_text(payload.get("ending_frame_path"), "continuity.ending_frame_path", max_length=4_000),
        )


@dataclass(frozen=True, slots=True)
class DramaReview:
    status: str = "pending"
    approved_by: str = ""
    notes: str = ""

    def validate(self) -> "DramaReview":
        status = _required_text(self.status, "review.status", max_length=32).lower()
        if status not in DRAMA_REVIEW_STATUSES:
            raise DramaPlanError(f"Unknown review.status: {status}")
        if status == "approved" and not self.approved_by.strip():
            raise DramaPlanError("review.approved_by is required for an approved plan")
        if self.approved_by:
            _required_text(self.approved_by, "review.approved_by", max_length=DRAMA_MAX_ID_LENGTH)
        if self.notes:
            _required_text(self.notes, "review.notes")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "approved_by": self.approved_by, "notes": self.notes}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DramaReview":
        return cls(
            status=str(payload.get("status") or "pending").strip().lower(),
            approved_by=str(payload.get("approved_by") or "").strip(),
            notes=str(payload.get("notes") or "").strip(),
        )


@dataclass(frozen=True, slots=True)
class DramaScene:
    scene_id: str
    beat: str
    duration_seconds: float
    objective: str
    start_state: str
    action_beats: tuple[str, ...]
    end_state: str
    next_hook: str
    cause: str
    effect: str
    character_ids: tuple[str, ...]
    prop_ids: tuple[str, ...] = ()
    location: str = ""
    selected_source_path: str | None = None
    candidate_source_paths: tuple[str, ...] = ()
    source_start_seconds: float = 0.0
    motion: str = "none"
    transition_to_next: str = "cut"
    transition_duration_seconds: float = 0.10
    dialogue: tuple[DialogueCue, ...] = ()
    sfx: tuple[SfxCue, ...] = ()
    continuity: ContinuityContract = field(default_factory=ContinuityContract)

    def validate(self, known_characters: set[str], known_props: set[str], *, require_asset: bool) -> "DramaScene":
        _required_text(self.scene_id, "scene.scene_id", max_length=DRAMA_MAX_ID_LENGTH)
        _required_text(self.beat, "scene.beat", max_length=DRAMA_MAX_ID_LENGTH)
        duration = _finite_float(self.duration_seconds, "scene.duration_seconds", minimum=0.001)
        if duration > MAX_CLIP_DURATION_SECONDS:
            raise DramaPlanError(f"scene.duration_seconds cannot exceed {MAX_CLIP_DURATION_SECONDS}")
        for name, value in (
            ("scene.objective", self.objective),
            ("scene.start_state", self.start_state),
            ("scene.end_state", self.end_state),
            ("scene.next_hook", self.next_hook),
            ("scene.cause", self.cause),
            ("scene.effect", self.effect),
        ):
            _required_text(value, name)
        _string_tuple(self.action_beats, "scene.action_beats", required=True)
        characters = _string_tuple(self.character_ids, "scene.character_ids", required=True)
        props = _string_tuple(self.prop_ids, "scene.prop_ids")
        if not set(characters).issubset(known_characters):
            unknown = sorted(set(characters) - known_characters)[0]
            raise DramaPlanError(f"scene references unknown character: {unknown}")
        if not set(props).issubset(known_props):
            unknown = sorted(set(props) - known_props)[0]
            raise DramaPlanError(f"scene references unknown prop: {unknown}")
        if self.location:
            _required_text(self.location, "scene.location")
        source_path = _optional_path(self.selected_source_path, "scene.selected_source_path")
        if require_asset and source_path is None:
            raise DramaPlanError(f"scene.selected_source_path is required for scene {self.scene_id}")
        _path_tuple(self.candidate_source_paths, "scene.candidate_source_paths")
        source_start = _finite_float(self.source_start_seconds, "scene.source_start_seconds")
        if source_start > MAX_SOURCE_START_SECONDS:
            raise DramaPlanError(f"scene.source_start_seconds cannot exceed {MAX_SOURCE_START_SECONDS}")
        if self.motion not in MOTIONS:
            raise DramaPlanError(f"Unknown scene.motion: {self.motion}")
        transition = self.transition_to_next.strip().lower()
        if transition not in DRAMA_TRANSITION_CUTS and transition not in TRANSITIONS:
            raise DramaPlanError(f"Unknown scene.transition_to_next: {transition}")
        transition_duration = _finite_float(
            self.transition_duration_seconds,
            "scene.transition_duration_seconds",
            minimum=0.0 if transition in DRAMA_TRANSITION_CUTS else 0.001,
        )
        if transition in DRAMA_TRANSITION_CUTS:
            transition_duration = 0.0
        elif transition_duration > 30.0:
            raise DramaPlanError("scene.transition_duration_seconds cannot exceed 30 seconds")
        self.continuity.validate()
        if self.continuity.character_ids and set(self.continuity.character_ids) != set(characters):
            raise DramaPlanError(f"scene {self.scene_id} continuity.character_ids must match scene.character_ids")
        if self.continuity.prop_ids and set(self.continuity.prop_ids) != set(props):
            raise DramaPlanError(f"scene {self.scene_id} continuity.prop_ids must match scene.prop_ids")
        for cue in self.dialogue:
            cue.validate(duration)
            if cue.character_id not in characters:
                raise DramaPlanError(f"dialogue references character not present in scene: {cue.character_id}")
        for cue in self.sfx:
            cue.validate(duration)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "beat": self.beat,
            "duration_seconds": self.duration_seconds,
            "objective": self.objective,
            "start_state": self.start_state,
            "action_beats": list(self.action_beats),
            "end_state": self.end_state,
            "next_hook": self.next_hook,
            "cause": self.cause,
            "effect": self.effect,
            "character_ids": list(self.character_ids),
            "prop_ids": list(self.prop_ids),
            "location": self.location,
            "selected_source_path": self.selected_source_path,
            "candidate_source_paths": list(self.candidate_source_paths),
            "source_start_seconds": self.source_start_seconds,
            "motion": self.motion,
            "transition_to_next": self.transition_to_next,
            "transition_duration_seconds": self.transition_duration_seconds,
            "dialogue": [cue.to_dict() for cue in self.dialogue],
            "sfx": [cue.to_dict() for cue in self.sfx],
            "continuity": self.continuity.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DramaScene":
        raw_continuity = payload.get("continuity") or {}
        if not isinstance(raw_continuity, dict):
            raise DramaPlanError("scene.continuity must be an object")
        dialogue = tuple(DialogueCue.from_dict(item) for item in _object_list(payload.get("dialogue"), "scene.dialogue"))
        sfx = tuple(SfxCue.from_dict(item) for item in _object_list(payload.get("sfx"), "scene.sfx"))
        return cls(
            scene_id=str(payload.get("scene_id") or "").strip(),
            beat=str(payload.get("beat") or "").strip(),
            duration_seconds=_finite_float(payload.get("duration_seconds"), "scene.duration_seconds", minimum=0.001),
            objective=str(payload.get("objective") or "").strip(),
            start_state=str(payload.get("start_state") or "").strip(),
            action_beats=_string_tuple(payload.get("action_beats"), "scene.action_beats"),
            end_state=str(payload.get("end_state") or "").strip(),
            next_hook=str(payload.get("next_hook") or "").strip(),
            cause=str(payload.get("cause") or "").strip(),
            effect=str(payload.get("effect") or "").strip(),
            character_ids=_string_tuple(payload.get("character_ids"), "scene.character_ids"),
            prop_ids=_string_tuple(payload.get("prop_ids"), "scene.prop_ids"),
            location=str(payload.get("location") or "").strip(),
            selected_source_path=_optional_text(payload.get("selected_source_path"), "scene.selected_source_path", max_length=4_000),
            candidate_source_paths=_path_tuple(payload.get("candidate_source_paths"), "scene.candidate_source_paths"),
            source_start_seconds=_finite_float(payload.get("source_start_seconds", 0.0), "scene.source_start_seconds"),
            motion=str(payload.get("motion") or "none").strip().lower(),
            transition_to_next=str(payload.get("transition_to_next") or "cut").strip().lower(),
            transition_duration_seconds=_finite_float(
                payload.get("transition_duration_seconds", 0.10),
                "scene.transition_duration_seconds",
            ),
            dialogue=dialogue,
            sfx=sfx,
            continuity=ContinuityContract.from_dict(raw_continuity),
        )


@dataclass(frozen=True, slots=True)
class DramaPlan:
    plan_id: str
    title: str
    premise: str
    objective: str
    scenes: tuple[DramaScene, ...]
    character_ids: tuple[str, ...]
    prop_ids: tuple[str, ...] = ()
    style: str = ""
    output_width: int = 576
    output_height: int = 1024
    fps: float = 24.0
    target_duration_seconds: float | None = None
    variant_seed: int = 0
    review: DramaReview = field(default_factory=DramaReview)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, *, require_assets: bool = False) -> "DramaPlan":
        _required_text(self.plan_id, "plan_id", max_length=DRAMA_MAX_ID_LENGTH)
        _required_text(self.title, "title", max_length=DRAMA_MAX_ID_LENGTH)
        _required_text(self.premise, "premise")
        _required_text(self.objective, "objective")
        characters = _string_tuple(self.character_ids, "character_ids", required=True)
        props = _string_tuple(self.prop_ids, "prop_ids")
        if not self.scenes:
            raise DramaPlanError("DramaPlan requires at least one scene")
        if len(self.scenes) > 64:
            raise DramaPlanError("DramaPlan cannot contain more than 64 scenes")
        scene_ids = [scene.scene_id for scene in self.scenes]
        if len(set(scene_ids)) != len(scene_ids):
            raise DramaPlanError("DramaPlan scene_id values must be unique")
        for scene in self.scenes:
            scene.validate(set(characters), set(props), require_asset=require_assets)
        total_duration = sum(float(scene.duration_seconds) for scene in self.scenes)
        if total_duration > MAX_TOTAL_DURATION_SECONDS:
            raise DramaPlanError(f"DramaPlan total scene duration cannot exceed {MAX_TOTAL_DURATION_SECONDS}")
        if self.output_width <= 0 or self.output_width > MAX_OUTPUT_DIMENSION or self.output_width % 2:
            raise DramaPlanError("DramaPlan output_width must be positive, even, and within the output limit")
        if self.output_height <= 0 or self.output_height > MAX_OUTPUT_DIMENSION or self.output_height % 2:
            raise DramaPlanError("DramaPlan output_height must be positive, even, and within the output limit")
        fps = _finite_float(self.fps, "fps", minimum=0.001)
        if fps > MAX_FPS:
            raise DramaPlanError(f"DramaPlan fps cannot exceed {MAX_FPS}")
        target = _optional_finite_float(self.target_duration_seconds, "target_duration_seconds", minimum=0.001)
        if target is not None and target > MAX_TOTAL_DURATION_SECONDS:
            raise DramaPlanError(f"DramaPlan target_duration_seconds cannot exceed {MAX_TOTAL_DURATION_SECONDS}")
        self.review.validate()
        if self.style:
            _required_text(self.style, "style")
        try:
            metadata_size = len(json.dumps(self.metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        except (TypeError, ValueError) as exc:
            raise DramaPlanError("DramaPlan metadata must be JSON serializable") from exc
        if metadata_size > MAX_METADATA_BYTES:
            raise DramaPlanError(f"DramaPlan metadata cannot exceed {MAX_METADATA_BYTES} bytes")
        transition_modes = {
            scene.transition_to_next.strip().lower()
            for scene in self.scenes[:-1]
        }
        has_cut = bool(transition_modes & DRAMA_TRANSITION_CUTS)
        has_effect = bool(transition_modes - DRAMA_TRANSITION_CUTS)
        if has_cut and has_effect:
            raise DramaPlanError(
                "P1 cannot compile mixed hard cuts and xfade transitions; use one transition family per timeline"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": DRAMA_PLAN_SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "title": self.title,
            "premise": self.premise,
            "objective": self.objective,
            "scenes": [scene.to_dict() for scene in self.scenes],
            "character_ids": list(self.character_ids),
            "prop_ids": list(self.prop_ids),
            "style": self.style,
            "output_width": self.output_width,
            "output_height": self.output_height,
            "fps": self.fps,
            "target_duration_seconds": self.target_duration_seconds,
            "variant_seed": self.variant_seed,
            "review": self.review.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DramaPlan":
        if not isinstance(payload, dict):
            raise DramaPlanError("DramaPlan must be an object")
        try:
            schema_version = int(payload.get("schema_version", DRAMA_PLAN_SCHEMA_VERSION))
        except (TypeError, ValueError) as exc:
            raise DramaPlanError("DramaPlan schema_version must be an integer") from exc
        if schema_version != DRAMA_PLAN_SCHEMA_VERSION:
            raise DramaPlanError(f"Unsupported DramaPlan schema_version: {schema_version}")
        raw_scenes = _object_list(payload.get("scenes"), "scenes")
        raw_review = payload.get("review") or {}
        raw_metadata = payload.get("metadata") or {}
        if not isinstance(raw_review, dict):
            raise DramaPlanError("review must be an object")
        if not isinstance(raw_metadata, dict):
            raise DramaPlanError("metadata must be an object")
        plan = cls(
            plan_id=str(payload.get("plan_id") or "").strip(),
            title=str(payload.get("title") or "").strip(),
            premise=str(payload.get("premise") or "").strip(),
            objective=str(payload.get("objective") or "").strip(),
            scenes=tuple(DramaScene.from_dict(item) for item in raw_scenes),
            character_ids=_string_tuple(payload.get("character_ids"), "character_ids"),
            prop_ids=_string_tuple(payload.get("prop_ids"), "prop_ids"),
            style=str(payload.get("style") or "").strip(),
            output_width=int(payload.get("output_width", 576)),
            output_height=int(payload.get("output_height", 1024)),
            fps=_finite_float(payload.get("fps", 24.0), "fps", minimum=0.001),
            target_duration_seconds=_optional_finite_float(payload.get("target_duration_seconds"), "target_duration_seconds", minimum=0.001),
            variant_seed=int(payload.get("variant_seed", 0) or 0),
            review=DramaReview.from_dict(raw_review),
            metadata=dict(raw_metadata),
        )
        return plan.validate()


def compile_drama_plan(plan: DramaPlan | dict[str, Any], *, require_assets: bool = True) -> EditPlan:
    """Compile a validated DramaPlan into the existing deterministic EditPlan."""

    drama_plan = DramaPlan.from_dict(plan) if isinstance(plan, dict) else plan
    drama_plan.validate(require_assets=require_assets)
    selected_paths = [scene.selected_source_path for scene in drama_plan.scenes]
    if any(path is None for path in selected_paths):
        raise DramaPlanError("Every scene requires selected_source_path before compilation")

    boundary_scenes = drama_plan.scenes[:-1]
    transitions: tuple[EditTransition, ...]
    transition_modes = [scene.transition_to_next.strip().lower() for scene in boundary_scenes]
    if not transition_modes or all(mode in DRAMA_TRANSITION_CUTS for mode in transition_modes):
        profile = "motion_cut_v1"
        transitions = ()
    else:
        profile = "chapter_dip_v1" if all(mode == "fadeblack" for mode in transition_modes) else "xfade_clean_v1"
        transitions = tuple(
            EditTransition(
                name=mode,
                duration_seconds=scene.transition_duration_seconds,
            )
            for scene, mode in zip(boundary_scenes, transition_modes, strict=True)
        )

    clips = tuple(
        EditClip(
            path=str(Path(scene.selected_source_path).expanduser().resolve()),
            duration_seconds=scene.duration_seconds,
            source_start_seconds=scene.source_start_seconds,
            motion=scene.motion,
            label=f"{scene.scene_id}:{scene.beat}",
        )
        for scene in drama_plan.scenes
    )
    metadata = {
        "source": "drama_plan_compiler_v1",
        "drama_plan_id": drama_plan.plan_id,
        "title": drama_plan.title,
        "review_status": drama_plan.review.status,
        "scene_ids": [scene.scene_id for scene in drama_plan.scenes],
        "beats": [scene.beat for scene in drama_plan.scenes],
        "story_contract": [
            {
                "scene_id": scene.scene_id,
                "objective": scene.objective,
                "start_state": scene.start_state,
                "end_state": scene.end_state,
                "cause": scene.cause,
                "effect": scene.effect,
                "next_hook": scene.next_hook,
            }
            for scene in drama_plan.scenes
        ],
        "dialogue_cue_count": sum(len(scene.dialogue) for scene in drama_plan.scenes),
        "sfx_cue_count": sum(len(scene.sfx) for scene in drama_plan.scenes),
    }
    compiled = EditPlan(
        clips=clips,
        transitions=transitions,
        output_width=drama_plan.output_width,
        output_height=drama_plan.output_height,
        fps=drama_plan.fps,
        target_duration_seconds=drama_plan.target_duration_seconds,
        profile=profile,
        variant_seed=drama_plan.variant_seed,
        audio_crossfade=True,
        metadata=metadata,
    )
    try:
        return compiled.validate()
    except ValueError as exc:
        raise DramaPlanError(f"Compiled DramaPlan is not a valid EditPlan: {exc}") from exc


__all__ = [
    "ContinuityContract",
    "DialogueCue",
    "DramaPlan",
    "DramaPlanError",
    "DramaReview",
    "DramaScene",
    "SfxCue",
    "compile_drama_plan",
]
