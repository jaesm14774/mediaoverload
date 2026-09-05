"""Reference-input helpers for MiniMax H3 Ref2VA workflows.

The local workflow intentionally supports reference images and reference
videos only.  Reference audio is rejected at the boundary so it cannot be
silently threaded into a later ComfyUI request.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping


MAX_REFERENCE_IMAGES = 9
MAX_REFERENCE_VIDEOS = 3
REFERENCE_TYPES = {"image", "video"}
REFERENCE_ROLES = {
    "identity",
    "motion",
    "camera",
    "style",
    "environment",
    "subject",
    "continuation",
}


def _as_path(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return ""
    return str(value).strip()


def _normalise_type(value: Any, path: str) -> str:
    raw = str(value or "").strip().lower()
    suffix = Path(path).suffix.lower()
    if raw in {"audio", "audio_reference", "reference_audio", "sound"}:
        raise ValueError(
            "Reference audio is intentionally disabled for MiniMax H3 in this workflow."
        )
    if raw in {"image", "img", "picture", "photo"}:
        return "image"
    if raw in {"video", "vid", "clip", "reference_video"}:
        return "video"
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        return "image"
    if suffix in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}:
        return "video"
    raise ValueError(
        f"Reference {path!r} must declare type=image or type=video; "
        "audio references are not supported."
    )


def _iter_entries(
    manifest: Any,
    image_paths: Iterable[Any] | None,
    video_paths: Iterable[Any] | None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if manifest is not None:
        if not isinstance(manifest, (list, tuple)):
            raise ValueError("reference_manifest must be a list of reference records.")
        for item in manifest:
            if isinstance(item, (str, Path)):
                entries.append({"path": _as_path(item)})
            elif isinstance(item, Mapping):
                entries.append(dict(item))
            else:
                raise ValueError("Each reference must be a path or an object record.")
    for value in image_paths or []:
        entries.append({"path": _as_path(value), "type": "image"})
    for value in video_paths or []:
        entries.append({"path": _as_path(value), "type": "video"})
    return entries


def normalize_reference_manifest(
    manifest: Any = None,
    *,
    image_paths: Iterable[Any] | None = None,
    video_paths: Iterable[Any] | None = None,
    require_files: bool = False,
    max_images: int = MAX_REFERENCE_IMAGES,
    max_videos: int = MAX_REFERENCE_VIDEOS,
) -> list[dict[str, Any]]:
    """Return deterministic, validated image/video reference records.

    The input accepts either a list of records or separate image/video path
    lists.  Records are retained in input order, while ``tag`` is assigned by
    media type so ComfyUI slot binding remains deterministic.
    """

    entries = _iter_entries(manifest, image_paths, video_paths)
    if not entries:
        raise ValueError(
            "MiniMax H3 Ref2VA requires at least one reference image or video."
        )

    normalised: list[dict[str, Any]] = []
    seen: set[str] = set()
    image_index = 0
    video_index = 0
    for item in entries:
        path = _as_path(item.get("path", item.get("source_path")))
        if not path:
            raise ValueError("Every H3 reference must include a non-empty path.")
        canonical = str(Path(path).expanduser().resolve())
        if canonical in seen:
            raise ValueError(f"Duplicate H3 reference path: {path}")
        seen.add(canonical)

        ref_type = _normalise_type(item.get("type"), path)
        role = str(item.get("role", "subject" if ref_type == "image" else "motion"))
        role = role.strip().lower() or ("subject" if ref_type == "image" else "motion")
        if role not in REFERENCE_ROLES:
            raise ValueError(
                f"Unsupported H3 reference role {role!r}; choose one of "
                f"{sorted(REFERENCE_ROLES)}."
            )

        source = Path(path).expanduser()
        if require_files and not source.is_file():
            raise FileNotFoundError(f"H3 reference file does not exist: {source}")

        if ref_type == "image":
            image_index += 1
            if image_index > max_images:
                raise ValueError(f"At most {max_images} H3 reference images are allowed.")
            tag = str(item.get("tag") or f"reference_image_{image_index}")
        else:
            video_index += 1
            if video_index > max_videos:
                raise ValueError(f"At most {max_videos} H3 reference videos are allowed.")
            tag = str(item.get("tag") or f"reference_video_{video_index}")

        retention = str(
            item.get(
                "retention",
                item.get("preserve", "identity_and_appearance" if role in {"identity", "subject"} else "motion_and_camera"),
            )
        ).strip()
        normalised.append(
            {
                "path": str(source),
                "source_path": str(source),
                "type": ref_type,
                "role": role,
                "tag": tag,
                "prompt_label": f"<Picture {image_index}>" if ref_type == "image" else f"<Video {video_index}>",
                "retention": retention,
                "notes": str(item.get("notes", "")).strip(),
            }
        )

    return normalised


def build_reference_lineage(
    references: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Create lightweight provenance records for run summaries and audits."""

    lineage: list[dict[str, Any]] = []
    for reference in references:
        path = Path(str(reference["path"])).expanduser()
        stat = path.stat() if path.is_file() else None
        lineage.append(
            {
                "path": str(path),
                "type": str(reference["type"]),
                "role": str(reference["role"]),
                "tag": str(reference["tag"]),
                "prompt_label": str(reference.get("prompt_label") or reference["tag"]),
                "retention": str(reference["retention"]),
                "exists": stat is not None,
                "size_bytes": stat.st_size if stat else None,
                "mtime_ns": stat.st_mtime_ns if stat else None,
            }
        )
    return lineage


def format_ref2va_prompt(
    base_prompt: str,
    references: Iterable[Mapping[str, Any]],
    *,
    soundscape: str = "Generate native H3 audio from the scene; do not use reference audio.",
) -> str:
    """Wrap a scene prompt in the current official full-reference contract.

    Ref2VA's full-reference rewrite format is deliberately different from the
    ordinary H3 ``integrated_multimodal_description`` format.  In particular,
    reference images that define reusable subjects are cited as ``<Picture N>``
    sources inside ``<Subject N>`` definitions; they are not incorrectly
    presented as standalone keyframes.  Keep the six section names and the
    reference labels stable so the text encoder can associate them with the
    ordered ComfyUI reference slots.
    """

    refs = list(references)
    subject_lines: list[str] = []
    retention_lines: list[str] = []
    summary_labels: list[str] = []
    has_frame_anchor = False
    for index, ref in enumerate(refs, start=1):
        source_label = str(ref.get("prompt_label") or f"<{('Picture' if ref.get('type') == 'image' else 'Video')} {index}>")
        subject_label = f"<Subject {index}>"
        role = str(ref.get("role") or "subject").strip()
        retention = str(ref.get("retention") or "fully_preserved").strip()
        notes = str(ref.get("notes") or "").strip()
        source_kind = "image" if ref.get("type") == "image" else "video"
        if role == "continuation" and source_kind == "image":
            has_frame_anchor = True
            subject_lines.append(
                f"{source_label} is the lossless first-frame continuity anchor for [Shot 1]."
                + (f" {notes}" if notes else "")
            )
            summary_labels.append(source_label)
            retention_lines.append(
                f"{source_label} ([Shot 1] first frame): fully_preserved - reproduce its opening composition, subject poses, object state, lighting, and spatial landmarks before motion begins."
            )
            continue
        subject_lines.append(
            f"{subject_label} is the {role} content from {source_label}, a referenced {source_kind} asset. "
            f"Use it as a distinct visual subject and retain {retention}."
            + (f" {notes}" if notes else "")
        )
        summary_labels.append(subject_label)
        retention_lines.append(
            f"{subject_label} (appears throughout the target video): fully_preserved - retain its declared {role} characteristics, spatial identity, and visual relationship to the other referenced subjects."
        )

    scene = str(base_prompt).strip()
    summary_subjects = ", ".join(summary_labels) or "the declared references"
    task_prefix = "[keyframe completion + reference generation]" if has_frame_anchor else "[reference generation]"
    summary = (
        f"{task_prefix} The target video uses {summary_subjects} as distinct visual references. "
        "The continuity anchor starts the target timeline; the remaining references guide identity, environment, props, and a coherent causal story rather than a collage of unrelated shots."
        if has_frame_anchor
        else (
            f"{task_prefix} The target video uses {summary_subjects} as distinct visual references. "
            "Their identities and roles guide the same coherent causal story rather than a collage of unrelated shots."
        )
    )
    anchor_instruction = ""
    if has_frame_anchor:
        anchor_instruction = (
            "The target video begins exactly from the declared continuity anchor in [Shot 1]. "
            "Preserve that first-frame composition and current state before introducing the next physical action; do not restart the scene or reinterpret the anchor as a generic reference.\n"
        )
    detailed = (
        "The target video is a polished cinematic animation with stable subject identity, readable scale, and deliberate physical cause and effect. "
        "Each referenced subject must appear only in the role defined above; do not merge, duplicate, or replace the references.\n"
        + anchor_instruction
        + scene
    )
    return "\n".join(
        [
            "subject_definitions:",
            *subject_lines,
            "summary:",
            summary,
            "retention_analysis:",
            *retention_lines,
            "detailed_description:",
            detailed,
            "overall_soundscape:",
            str(soundscape).strip(),
            "non_diegetic_music:",
            "Use only the music direction described in the scene prompt; no reference audio input.",
        ]
    )
