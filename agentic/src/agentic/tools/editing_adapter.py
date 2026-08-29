from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from agentic.runtime.editing import (
    EditClip,
    EditPlan,
    IMAGE_SUFFIXES,
    MAX_RENDER_WORK,
    MAX_TOTAL_DURATION_SECONDS,
)
from agentic.tools.ffmpeg_adapter import FFMPEG_COMMAND_TIMEOUT_SECONDS, FFmpegAdapter


class EditRenderError(RuntimeError):
    """Raised when a deterministic timeline cannot be rendered."""


class OpenCutEditAdapter:
    """Render a small OpenCut-inspired timeline contract with FFmpeg.

    The adapter intentionally keeps the contract independent from FFmpeg filter
    syntax. Agents choose clips, bounded motion, and transitions in ``EditPlan``;
    this class owns normalization, audio continuity, encoding, and evidence.
    """

    def __init__(self, output_root: Path, input_roots: Iterable[Path] = ()) -> None:
        self._ffmpeg = FFmpegAdapter()
        self.output_root = output_root.expanduser().resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        repo_root = Path(__file__).resolve().parents[4]
        configured_roots = [
            Path(raw.strip()).expanduser().resolve()
            for raw in (
                os.environ.get("AGENTIC_ALLOWED_MEDIA_ROOTS", "")
                + ","
                + os.environ.get("AGENTIC_ALLOWED_IMAGE_ROOTS", "")
            ).split(",")
            if raw.strip()
        ]
        self.input_roots = tuple({repo_root, self.output_root, *configured_roots, *(Path(root).expanduser().resolve() for root in input_roots)})
        image_roots = [
            Path(raw.strip()).expanduser().resolve()
            for raw in os.environ.get("AGENTIC_ALLOWED_IMAGE_ROOTS", "").split(",")
            if raw.strip()
        ]
        if not any(self.output_root == root or root in self.output_root.parents for root in image_roots):
            image_roots.append(self.output_root)
            os.environ["AGENTIC_ALLOWED_IMAGE_ROOTS"] = ",".join(str(root) for root in image_roots)

    def render(
        self,
        plan: EditPlan,
        output_path: str,
        *,
        manifest_path: str | None = None,
        contact_sheet_path: str | None = None,
        review_evidence_dir: str | None = None,
    ) -> dict[str, object]:
        plan.validate()
        output = self._resolve_output_path(output_path, "output")
        output.parent.mkdir(parents=True, exist_ok=True)
        source_records = self._source_records(plan)
        source_paths = {Path(str(record["path"])).resolve() for record in source_records}
        manifest_file = self._resolve_output_path(
            manifest_path or str(output.with_suffix(".edit_manifest.json")),
            "manifest",
        )
        contact_file = (
            self._resolve_output_path(contact_sheet_path, "contact sheet")
            if contact_sheet_path
            else None
        )
        evidence_dir = (
            self._resolve_output_path(review_evidence_dir, "review evidence directory")
            if review_evidence_dir
            else None
        )
        if evidence_dir and any(path.parent == evidence_dir for path in source_paths):
            raise EditRenderError("Review evidence directory cannot share a directory with source media")
        artifact_paths = {output, manifest_file}
        if contact_file:
            artifact_paths.add(contact_file)
        self._reject_artifact_collisions(artifact_paths, source_paths)
        durations = [float(record["duration_seconds"]) for record in source_records]
        canonical_plan = replace(
            plan,
            clips=tuple(
                replace(clip, path=str(record["path"]))
                for clip, record in zip(plan.clips, source_records, strict=True)
            ),
        )
        canonical_plan.validate()
        self._validate_transition_durations(canonical_plan, durations)
        self._validate_resource_budget(canonical_plan, durations)

        with tempfile.TemporaryDirectory(prefix="mediaoverload-edit-", dir=str(self.output_root)) as temp_root:
            render_output = Path(temp_root) / output.name
            staged_clips: list[EditClip] = []
            for index, (clip, record) in enumerate(zip(canonical_plan.clips, source_records, strict=True)):
                staged_path = Path(temp_root) / f"input_{index:02d}{Path(clip.path).suffix.lower()}"
                self._stage_input(
                    Path(str(record["path"])),
                    expected_sha256=str(record["sha256"]),
                    destination=staged_path,
                )
                staged_clips.append(replace(clip, path=str(staged_path)))
            render_plan = replace(canonical_plan, clips=tuple(staged_clips))
            can_use_demuxer = render_plan.profile == "baseline_concat" and all(
                Path(clip.path).suffix.lower() not in IMAGE_SUFFIXES
                and clip.source_start_seconds == 0
                and clip.duration_seconds is None
                and clip.motion == "none"
                for clip in render_plan.clips
            )
            if can_use_demuxer:
                self._render_baseline(render_plan, render_output)
            else:
                normalized_paths = []
                for index, (clip, record) in enumerate(zip(render_plan.clips, source_records, strict=True)):
                    normalized_path = Path(temp_root) / f"clip_{index:02d}.mp4"
                    self._normalize_clip(
                        clip,
                        duration=float(record["duration_seconds"]),
                        has_audio=bool(record["has_audio"]),
                        output_path=normalized_path,
                        plan=render_plan,
                    )
                    normalized_paths.append(normalized_path)
                self._render_composition(render_plan, normalized_paths, durations, render_output)

            if render_plan.target_duration_seconds is not None:
                self._fit_target_duration(render_output, float(render_plan.target_duration_seconds))
            os.replace(render_output, output)
        output_probe = self._ffmpeg.probe_media(str(output))
        if contact_file:
            self._ffmpeg.make_contact_sheet(
                video_path=str(output),
                output_path=str(contact_file),
                frame_count=12,
                columns=4,
                scale_width=360,
                duration_seconds=float(output_probe.get("duration") or 0.0),
            )
        evidence_paths = self._build_review_evidence(output, canonical_plan, durations, output_probe, evidence_dir)

        manifest = {
            "schema_version": 1,
            "renderer": "mediaoverload.ffmpeg.opencut_edit",
            "plan": canonical_plan.to_dict(),
            "creative_review": {
                "required": canonical_plan.profile == "editorial_kinetic_v1",
                "status": "unreviewed",
            },
            "sources": source_records,
            "output": {"path": str(output), "sha256": self._sha256(output), "probe": output_probe},
            "render_metrics": {
                "source_duration_seconds": sum(durations),
                "transition_overlap_seconds": sum(
                    transition.duration_seconds for transition in canonical_plan.transitions
                ),
                "rendered_duration_seconds": float(output_probe.get("duration") or 0.0),
            },
            "review_evidence_paths": evidence_paths,
        }
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "video_path": str(output),
            "manifest_path": str(manifest_file),
            "contact_sheet_path": str(contact_file) if contact_file else "",
            "review_evidence_paths": evidence_paths,
            "creative_review_required": canonical_plan.profile == "editorial_kinetic_v1",
            "plan": canonical_plan.to_dict(),
            "probe": output_probe,
            "source_records": source_records,
        }

    def materialize_result(
        self,
        result: dict[str, object],
        *,
        output_path: str,
        manifest_path: str | None = None,
        contact_sheet_path: str | None = None,
        creative_review: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Copy a selected candidate to the caller's requested artifact paths."""

        source_video = Path(str(result.get("video_path") or "")).expanduser().resolve()
        source_manifest = Path(str(result.get("manifest_path") or "")).expanduser().resolve()
        source_contact = Path(str(result.get("contact_sheet_path") or "")).expanduser().resolve()
        if not source_video.is_file() or not source_manifest.is_file():
            raise EditRenderError("Selected edit candidate is missing its video or manifest")
        self._resolve_output_path(str(source_video), "candidate video")
        self._resolve_output_path(str(source_manifest), "candidate manifest")
        output = self._resolve_output_path(output_path, "output")
        manifest_file = self._resolve_output_path(
            manifest_path or str(output.with_suffix(".edit_manifest.json")),
            "manifest",
        )
        manifest_data = json.loads(source_manifest.read_text(encoding="utf-8"))
        if not isinstance(manifest_data, dict):
            raise EditRenderError("Selected edit candidate manifest is not an object")
        source_paths = {
            Path(str(item.get("path"))).expanduser().resolve()
            for item in (manifest_data.get("sources") or [])
            if isinstance(item, dict) and str(item.get("path") or "").strip()
        }
        contact_file = None
        if contact_sheet_path:
            if not source_contact.is_file():
                raise EditRenderError("Selected edit candidate is missing its contact sheet")
            self._resolve_output_path(str(source_contact), "candidate contact sheet")
            contact_file = self._resolve_output_path(contact_sheet_path, "contact sheet")
        artifact_paths = {output, manifest_file}
        if contact_file:
            artifact_paths.add(contact_file)
        self._reject_artifact_collisions(artifact_paths, source_paths)
        if len(artifact_paths) != 2 + int(contact_file is not None):
            raise EditRenderError("Edit materialization artifacts must use distinct paths")
        if manifest_file in {source_video, source_contact} or (contact_file and contact_file in {source_video, source_manifest}):
            raise EditRenderError("Edit materialization artifacts cannot overwrite candidate media")
        output.parent.mkdir(parents=True, exist_ok=True)
        if source_video != output:
            self._copy_atomic(source_video, output)
        final_probe = self._ffmpeg.probe_media(str(output))
        manifest_data["output"] = {
            "path": str(output),
            "sha256": self._sha256(output),
            "probe": final_probe,
        }
        if creative_review is not None:
            manifest_data["creative_review"] = creative_review
        self._write_json_atomic(manifest_file, manifest_data)

        if contact_file:
            if source_contact != contact_file:
                self._copy_atomic(source_contact, contact_file)
        materialized = dict(result)
        materialized.update(
            {
                "video_path": str(output),
                "manifest_path": str(manifest_file),
                "contact_sheet_path": str(contact_file or source_contact),
                "probe": final_probe,
            }
        )
        if creative_review is not None:
            materialized["creative_review"] = creative_review
        return materialized

    def _write_json_atomic(self, path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="mediaoverload-edit-manifest-", dir=str(self.output_root)) as temp_root:
            temporary = Path(temp_root) / path.name
            temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, path)

    def _copy_atomic(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="mediaoverload-edit-copy-", dir=str(self.output_root)) as temp_root:
            temporary = Path(temp_root) / destination.name
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)

    def _build_review_evidence(
        self,
        output: Path,
        plan: EditPlan,
        durations: list[float],
        output_probe: dict[str, object],
        evidence_dir: Path | None,
    ) -> list[str]:
        """Extract overall and join-adjacent frames for visual LLM review."""

        if evidence_dir is None:
            return []
        container_duration = float(output_probe.get("duration") or 0.0)
        video_duration = float(output_probe.get("video_duration") or 0.0)
        duration = min(container_duration, video_duration) if video_duration > 0 else container_duration
        fps = float(output_probe.get("frame_rate") or plan.fps or 24.0)
        if duration <= 0 or fps <= 0:
            return []
        times: list[tuple[str, float]] = [("opening", 0.0)]
        if plan.transitions:
            current_duration = float(durations[0])
            boundaries = []
            for index, transition in enumerate(plan.transitions, start=1):
                boundaries.append((index, max(0.0, current_duration - transition.duration_seconds / 2.0)))
                current_duration += float(durations[index]) - transition.duration_seconds
        else:
            boundaries = []
            current_duration = 0.0
            for index, clip_duration in enumerate(durations[:-1], start=1):
                current_duration += float(clip_duration)
                boundaries.append((index, current_duration))
        for index, boundary in boundaries:
            if boundary >= duration:
                continue
            for label, offset in (("before", -0.12), ("join", 0.0), ("after", 0.12)):
                times.append((f"boundary_{index:02d}_{label}", boundary + offset))
        times.append(("ending", max(0.0, duration - (1.0 / fps))))
        evidence_dir.mkdir(parents=True, exist_ok=True)
        unique: list[tuple[str, float]] = []
        seen: set[float] = set()
        max_timestamp = max(0.0, duration - (1.0 / fps))
        for label, raw_time in times:
            timestamp = min(max(0.0, float(raw_time)), max_timestamp)
            rounded = round(timestamp, 3)
            if rounded in seen:
                continue
            seen.add(rounded)
            unique.append((label, timestamp))
        paths: list[str] = []
        for label, timestamp in unique[:24]:
            frame_path = evidence_dir / f"{label}_{timestamp:.3f}s.jpg"
            self._ffmpeg.extract_frame_at(str(output), str(frame_path), timestamp)
            paths.append(str(frame_path))
        return paths

    def _source_records(self, plan: EditPlan) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for clip in plan.clips:
            path = self._validate_input_path(Path(clip.path))
            if not path.is_file():
                raise FileNotFoundError(f"Edit input does not exist: {path}")
            if path.suffix.lower() in IMAGE_SUFFIXES:
                duration = float(clip.duration_seconds or 3.0)
                has_audio = False
                probe: dict[str, object] = {"path": str(path), "media_type": "image"}
            else:
                probe = self._ffmpeg.probe_media(str(path))
                if not bool(probe.get("has_video")):
                    raise EditRenderError(f"Edit input has no video stream: {path}")
                available = float(probe.get("duration") or 0.0) - clip.source_start_seconds
                duration = float(clip.duration_seconds or available)
                has_audio = bool(probe.get("has_audio"))
                if duration <= 0 or available <= 0 or duration > available + 0.04:
                    raise EditRenderError(
                        f"Clip duration {duration:.3f}s exceeds available source range {available:.3f}s: {path}"
                    )
            records.append(
                {
                    "path": str(path),
                    "sha256": self._sha256(path),
                    "duration_seconds": duration,
                    "source_start_seconds": clip.source_start_seconds,
                    "has_audio": has_audio,
                    "probe": probe,
                }
            )
        return records

    @staticmethod
    def _validate_resource_budget(plan: EditPlan, durations: list[float]) -> None:
        if len(durations) != len(plan.clips) or any(not math.isfinite(duration) or duration <= 0 for duration in durations):
            raise EditRenderError("Edit source durations must be finite and positive")
        total_duration = sum(durations)
        if total_duration > MAX_TOTAL_DURATION_SECONDS:
            raise EditRenderError(
                f"Edit source duration cannot exceed {MAX_TOTAL_DURATION_SECONDS:.0f} seconds"
            )
        render_work = float(plan.output_width) * float(plan.output_height) * float(plan.fps) * total_duration
        if render_work > MAX_RENDER_WORK:
            raise EditRenderError("Edit render work exceeds the configured resource budget")

    def _stage_input(self, source: Path, *, expected_sha256: str, destination: Path) -> None:
        """Snapshot an approved source so FFmpeg cannot follow a later path swap."""

        validated = self._validate_input_path(source)
        if self._sha256(validated) != expected_sha256:
            raise EditRenderError(f"Edit input changed during validation: {validated}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(validated, destination)
        if self._sha256(destination) != expected_sha256:
            destination.unlink(missing_ok=True)
            raise EditRenderError(f"Edit input changed while being staged: {validated}")

    def _validate_input_path(self, raw_path: Path) -> Path:
        path = raw_path.expanduser()
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise EditRenderError(f"Edit input cannot use a symlink path: {path}")
        resolved = path.resolve()
        if not any(resolved == root or root in resolved.parents for root in self.input_roots):
            raise EditRenderError(f"Edit input is outside approved media roots: {resolved}")
        return resolved

    def _resolve_output_path(self, raw_path: str, label: str) -> Path:
        candidate = Path(raw_path).expanduser().resolve()
        if candidate != self.output_root and self.output_root not in candidate.parents:
            raise EditRenderError(f"Edit {label} must stay under the configured output root: {candidate}")
        return candidate

    @staticmethod
    def _reject_artifact_collisions(artifact_paths: set[Path], source_paths: set[Path]) -> None:
        collisions = sorted(path for path in artifact_paths if path in source_paths)
        if collisions:
            raise EditRenderError(f"Edit outputs cannot overwrite source media: {collisions[0]}")

    @staticmethod
    def _validate_transition_durations(plan: EditPlan, durations: list[float]) -> None:
        for index, transition in enumerate(plan.transitions):
            if transition.duration_seconds >= min(durations[index], durations[index + 1]):
                raise EditRenderError(
                    f"Transition {index + 1} ({transition.duration_seconds:.3f}s) must be shorter than adjacent clips"
                )

    def _render_baseline(self, plan: EditPlan, output: Path) -> None:
        inputs = [clip.path for clip in plan.clips]
        if plan.target_duration_seconds is None:
            self._ffmpeg.concat_videos(inputs, str(output), method="demuxer")
            return
        with tempfile.TemporaryDirectory(prefix="mediaoverload-baseline-") as temp_root:
            concatenated = Path(temp_root) / "concat.mp4"
            self._ffmpeg.concat_videos(inputs, str(concatenated), method="demuxer")
            self._ffmpeg.trim_video(str(concatenated), str(output), float(plan.target_duration_seconds))

    def _normalize_clip(
        self,
        clip: EditClip,
        *,
        duration: float,
        has_audio: bool,
        output_path: Path,
        plan: EditPlan,
    ) -> None:
        is_image = Path(clip.path).suffix.lower() in IMAGE_SUFFIXES
        input_args: list[str] = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
        if not is_image and clip.source_start_seconds > 0:
            input_args.extend(["-ss", f"{clip.source_start_seconds:.6f}"])
        if is_image:
            input_args.extend(["-i", clip.path, "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"])
            audio_index = 1
        else:
            input_args.extend(["-i", clip.path])
            if not has_audio:
                input_args.extend(["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"])
                audio_index = 1
            else:
                audio_index = 0

        video_filter = self._video_filter(clip, duration, plan)
        audio_filter = f"[{audio_index}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a]"
        command = [
            *input_args,
            "-filter_complex",
            f"[0:v]{video_filter}[v];{audio_filter}",
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-t",
            f"{duration:.6f}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            f"{plan.fps:.6g}",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-y",
            str(output_path),
        ]
        self._run(command)

    def _video_filter(self, clip: EditClip, duration: float, plan: EditPlan) -> str:
        width = int(plan.output_width)
        height = int(plan.output_height)
        fps = max(1, round(plan.fps))
        frames = max(1, round(duration * fps))
        fit = f"scale={width}:{height}:force_original_aspect_ratio=increase"
        if Path(clip.path).suffix.lower() not in IMAGE_SUFFIXES:
            if clip.motion == "none":
                return f"{fit},crop={width}:{height}:(in_w-out_w)/2:(in_h-out_h)/2,setsar=1,fps={plan.fps:.6g},format=yuv420p"
            overscan = "scale=ceil(iw*1.08/2)*2:ceil(ih*1.08/2)*2"
            denominator = max(1, frames - 1)
            if clip.motion == "slow_zoom_in":
                zoom = f"(1+0.08*n/{denominator})"
                crop = (
                    f"crop=w='trunc(iw/{zoom}/2)*2':h='trunc(ih/{zoom}/2)*2':"
                    "x='(iw-ow)/2':y='(ih-oh)/2'"
                )
            elif clip.motion == "slow_zoom_out":
                zoom = f"(1.08-0.08*n/{denominator})"
                crop = (
                    f"crop=w='trunc(iw/{zoom}/2)*2':h='trunc(ih/{zoom}/2)*2':"
                    "x='(iw-ow)/2':y='(ih-oh)/2'"
                )
            elif clip.motion == "pan_left":
                crop = f"crop={width}:{height}:x='(iw-ow)*n/{denominator}':y='(ih-oh)/2'"
            elif clip.motion == "pan_right":
                crop = f"crop={width}:{height}:x='(iw-ow)*(1-n/{denominator})':y='(ih-oh)/2'"
            elif clip.motion == "drift_up":
                crop = f"crop={width}:{height}:x='(iw-ow)/2':y='(ih-oh)*n/{denominator}'"
            else:
                crop = f"crop={width}:{height}:x='(iw-ow)/2':y='(ih-oh)*(1-n/{denominator})'"
            return f"{fit},{overscan},{crop},scale={width}:{height},setsar=1,fps={plan.fps:.6g},format=yuv420p"

        zoomed = "scale=ceil(iw*1.18/2)*2:ceil(ih*1.18/2)*2"
        denominator = max(1, frames - 1)
        motion = clip.motion
        if motion == "slow_zoom_in":
            zoom = f"min(1+0.10*on/{denominator},1.10)"
            x = "iw/2-(iw/zoom/2)"
            y = "ih/2-(ih/zoom/2)"
        elif motion == "slow_zoom_out":
            zoom = f"max(1.10-0.10*on/{denominator},1.0)"
            x = "iw/2-(iw/zoom/2)"
            y = "ih/2-(ih/zoom/2)"
        elif motion == "pan_left":
            zoom = "1.08"
            x = f"(iw-iw/zoom)*on/{denominator}"
            y = "ih/2-(ih/zoom/2)"
        elif motion == "pan_right":
            zoom = "1.08"
            x = f"(iw-iw/zoom)*(1-on/{denominator})"
            y = "ih/2-(ih/zoom/2)"
        elif motion == "drift_up":
            zoom = "1.08"
            x = "iw/2-(iw/zoom/2)"
            y = f"(ih-ih/zoom)*on/{denominator}"
        elif motion == "drift_down":
            zoom = "1.08"
            x = "iw/2-(iw/zoom/2)"
            y = f"(ih-ih/zoom)*(1-on/{denominator})"
        else:
            zoom = "1.0"
            x = "iw/2-(iw/zoom/2)"
            y = "ih/2-(ih/zoom/2)"
        return (
            f"{fit},{zoomed},zoompan=z='{zoom}':x='{x}':y='{y}':d={frames}:s={width}x{height}:fps={plan.fps:.6g},"
            "setsar=1,format=yuv420p"
        )

    def _render_composition(
        self,
        plan: EditPlan,
        normalized_paths: list[Path],
        durations: list[float],
        output: Path,
    ) -> None:
        if len(normalized_paths) == 1:
            self._run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(normalized_paths[0]),
                    "-c",
                    "copy",
                    "-y",
                    str(output),
                ]
            )
            return

        transitions = list(plan.transitions)
        if not transitions:
            transitions = [type("HardCut", (), {"name": "hard_cut", "duration_seconds": 0.0})() for _ in normalized_paths[1:]]
        if any(transition.name == "hard_cut" for transition in transitions):
            if not all(transition.name == "hard_cut" for transition in transitions):
                raise EditRenderError("hard_cut cannot be mixed with xfade transitions")
            video_filter = f"{''.join(f'[{i}:v][{i}:a]' for i in range(len(normalized_paths)))}concat=n={len(normalized_paths)}:v=1:a=1[v][a]"
        else:
            video_parts: list[str] = []
            audio_parts: list[str] = []
            current_video = "[0:v]"
            current_audio = "[0:a]"
            current_duration = durations[0]
            overlap = 0.0
            for index, transition in enumerate(transitions, start=1):
                duration = transition.duration_seconds
                offset = current_duration - duration
                video_label = f"[v{index}]"
                audio_label = f"[a{index}]"
                video_parts.append(
                    f"{current_video}[{index}:v]xfade=transition={transition.name}:duration={duration:.6f}:offset={offset:.6f}{video_label}"
                )
                if plan.audio_crossfade:
                    audio_parts.append(
                        f"{current_audio}[{index}:a]acrossfade=d={duration:.6f}:c1=tri:c2=tri{audio_label}"
                    )
                current_video = video_label
                current_audio = audio_label
                current_duration += durations[index] - duration
                overlap += duration
            if plan.audio_crossfade:
                audio_filter = ";".join(audio_parts) + f";{current_audio}anull[a]"
            else:
                audio_filter = (
                    f"{''.join(f'[{i}:a]' for i in range(len(normalized_paths)))}"
                    f"concat=n={len(normalized_paths)}:v=0:a=1[a]"
                )
            video_filter = ";".join(video_parts) + f";{current_video}null[v];{audio_filter}"

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            *[item for path in normalized_paths for item in ("-i", str(path))],
            "-filter_complex",
            video_filter,
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            f"{plan.fps:.6g}",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            "-shortest",
            "-y",
            str(output),
        ]
        self._run(command)

    def _trim_output(self, output: Path, duration: float) -> None:
        with tempfile.TemporaryDirectory(prefix="mediaoverload-edit-trim-") as temp_root:
            trimmed = Path(temp_root) / "trimmed.mp4"
            self._ffmpeg.trim_video(str(output), str(trimmed), duration)
            os.replace(trimmed, output)

    def _fit_target_duration(self, output: Path, target_duration: float) -> None:
        current_duration = float(self._ffmpeg.probe_media(str(output)).get("duration") or 0.0)
        if current_duration < target_duration - 0.03:
            pad_duration = target_duration - current_duration + 0.03
            with tempfile.TemporaryDirectory(prefix="mediaoverload-edit-pad-") as temp_root:
                padded = Path(temp_root) / "padded.mp4"
                self._run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-i",
                        str(output),
                        "-vf",
                        f"tpad=stop_mode=clone:stop_duration={pad_duration:.6f}",
                        "-af",
                        f"apad=pad_dur={pad_duration:.6f}",
                        "-t",
                        f"{target_duration:.6f}",
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-c:a",
                        "aac",
                        "-ar",
                        "48000",
                        "-ac",
                        "2",
                        "-movflags",
                        "+faststart",
                        "-y",
                        str(padded),
                    ]
                )
                os.replace(padded, output)
        if float(self._ffmpeg.probe_media(str(output)).get("duration") or 0.0) > target_duration + 0.03:
            self._trim_output(output, target_duration)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _run(command: list[str]) -> None:
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=FFMPEG_COMMAND_TIMEOUT_SECONDS,
            )
        except FileNotFoundError as exc:
            raise EditRenderError("ffmpeg is not installed or not available in PATH") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            raise EditRenderError(f"Edit render failed: {detail}") from exc
        except subprocess.TimeoutExpired as exc:
            raise EditRenderError("Edit render exceeded the ffmpeg command timeout") from exc
        if completed.returncode != 0:
            raise EditRenderError("Edit render returned a non-zero status")
