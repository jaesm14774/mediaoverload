from __future__ import annotations

from pathlib import Path
from typing import Iterable

from agentic.runtime.registry import ToolRegistry
from agentic.runtime.editing import EditPlan
from agentic.tools.editing_adapter import OpenCutEditAdapter
from agentic.tools.ffmpeg_adapter import FFmpegAdapter
from agentic.tools.sprite_adapter import SpriteAdapter
from agentic.tools.tts_adapter import TTSAdapter


class MediaServiceTools:
    def __init__(
        self,
        output_root: Path,
        input_roots: Iterable[Path] = (),
        ffmpeg: FFmpegAdapter | None = None,
    ) -> None:
        self.output_root = output_root
        self.input_roots = tuple(input_roots)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._ffmpeg = ffmpeg
        self._sprite: SpriteAdapter | None = None
        self._editing: OpenCutEditAdapter | None = None
        self._tts: TTSAdapter | None = None

    def extract_last_frame(self, payload: dict[str, object]) -> dict[str, object]:
        service = self._ffmpeg_service()
        output_path = str(payload["output_path"])
        video_path = str(payload["video_path"])
        return {"frame_path": service.extract_last_frame(video_path=video_path, output_path=output_path)}

    def concat_videos(self, payload: dict[str, object]) -> dict[str, object]:
        service = self._ffmpeg_service()
        video_paths = [str(path) for path in payload.get("video_paths", [])]
        output_path = str(payload["output_path"])
        method = str(payload.get("method", "demuxer"))
        return {"video_path": service.concat_videos(video_paths=video_paths, output_path=output_path, method=method)}

    def change_video_speed(self, payload: dict[str, object]) -> dict[str, object]:
        service = self._ffmpeg_service()
        speed = float(payload.get("speed", 1.0))
        return {
            "video_path": service.change_video_speed(
                video_path=str(payload["video_path"]),
                output_path=str(payload["output_path"]),
                speed=speed,
            ),
            "speed": speed,
        }

    def normalize_video_canvas(self, payload: dict[str, object]) -> dict[str, object]:
        service = self._ffmpeg_service()
        width = int(payload["target_width"])
        height = int(payload["target_height"])
        return {
            "video_path": service.normalize_video_canvas(
                video_path=str(payload["video_path"]),
                output_path=str(payload["output_path"]),
                target_width=width,
                target_height=height,
                background=str(payload.get("background", "#15151f")),
            ),
            "width": width,
            "height": height,
        }

    def trim_video(self, payload: dict[str, object]) -> dict[str, object]:
        service = self._ffmpeg_service()
        return {
            "video_path": service.trim_video(
                video_path=str(payload["video_path"]),
                output_path=str(payload["output_path"]),
                duration_seconds=float(payload["duration_seconds"]),
                normalize_audio=bool(payload.get("normalize_audio", False)),
            ),
            "duration_seconds": float(payload["duration_seconds"]),
        }

    def video_to_gif(self, payload: dict[str, object]) -> dict[str, object]:
        service = self._ffmpeg_service()
        return {
            "gif_path": service.video_to_gif(
                video_path=str(payload["video_path"]),
                output_path=str(payload["output_path"]),
                fps=int(payload.get("fps", 12)),
                max_colors=int(payload.get("max_colors", 256)),
                scale_width=int(payload.get("scale_width", 512)),
            )
        }

    def video_qa(self, payload: dict[str, object]) -> dict[str, object]:
        service = self._ffmpeg_service()
        video_path = str(payload["video_path"])
        file_exists = Path(video_path).is_file()
        probe = service.probe_media(video_path) if file_exists else {
            "path": video_path,
            "duration": 0.0,
            "has_video": False,
            "has_audio": False,
            "width": 0,
            "height": 0,
        }
        errors: list[str] = []
        warnings: list[str] = []
        if not file_exists:
            errors.append("video file does not exist")
        target_duration = payload.get("target_duration")
        tolerance = float(payload.get("duration_tolerance", 0.5))
        duration = float(probe.get("duration") or 0.0)
        checks = {
            "file_exists": file_exists,
            "has_video": bool(probe.get("has_video")),
            "dimensions": int(probe.get("width") or 0) > 0 and int(probe.get("height") or 0) > 0,
            "duration": target_duration in {None, ""} or abs(duration - float(target_duration)) <= tolerance,
        }
        expected_width = payload.get("expected_width")
        expected_height = payload.get("expected_height")
        if expected_width not in {None, ""} or expected_height not in {None, ""}:
            checks["expected_dimensions"] = (
                expected_width in {None, ""} or int(probe.get("width") or 0) == int(expected_width)
            ) and (
                expected_height in {None, ""} or int(probe.get("height") or 0) == int(expected_height)
            )
            if not checks["expected_dimensions"]:
                errors.append(
                    f"dimensions {int(probe.get('width') or 0)}x{int(probe.get('height') or 0)} do not match "
                    f"expected {expected_width}x{expected_height}"
                )
        expected_fps = payload.get("expected_fps")
        if expected_fps not in {None, ""}:
            observed_fps = float(probe.get("frame_rate") or 0.0)
            checks["expected_fps"] = abs(observed_fps - float(expected_fps)) <= float(payload.get("fps_tolerance", 0.15))
            if not checks["expected_fps"]:
                errors.append(f"frame rate {observed_fps:.3f} does not match expected {float(expected_fps):.3f}")
        if not bool(probe.get("has_video")):
            errors.append("no video stream")
        if not checks["dimensions"]:
            errors.append("video dimensions are missing")
        if target_duration not in {None, ""} and abs(duration - float(target_duration)) > tolerance:
            errors.append(f"duration {duration:.3f}s is outside target {float(target_duration):.3f}s ± {tolerance:.3f}s")
        require_audio = bool(payload.get("require_audio", False))
        require_stereo = bool(payload.get("require_stereo_audio", False))
        has_audio = bool(probe.get("has_audio"))
        audio_checks = {
            "audio_present": has_audio if require_audio else True,
            "stereo": (
                int(probe.get("channels") or 0) >= 2
                and str(probe.get("channel_layout") or "").lower() in {"stereo", "2.0", "2c", ""
                }
                if require_stereo and has_audio
                else True
            ),
            "duration_alignment": True,
        }
        audio_analysis: dict[str, object] = {}
        audio_limits_requested = any(key in payload for key in ("min_mean_volume_db", "max_peak_db", "max_silence_ratio"))
        should_analyze_audio = (bool(payload.get("analyze_audio", require_audio)) or audio_limits_requested) and has_audio
        if should_analyze_audio:
            try:
                audio_analysis = service.analyze_audio(
                    video_path,
                    silence_threshold_db=float(payload.get("silence_threshold_db", -50.0)),
                    silence_min_seconds=float(payload.get("silence_min_seconds", 0.4)),
                )
                # Loudness and silence are measured, but only explicit numeric
                # requirements may turn these observations into a blocking check.
                for metric, limit_key, compare in (
                    ("mean_volume_db", "min_mean_volume_db", lambda value, limit: value >= limit),
                    ("max_volume_db", "max_peak_db", lambda value, limit: value < limit),
                    ("silence_ratio", "max_silence_ratio", lambda value, limit: value <= limit),
                ):
                    if limit_key not in payload:
                        continue
                    value = audio_analysis.get(metric)
                    valid = value is not None and compare(float(value), float(payload[limit_key]))
                    audio_checks[limit_key] = valid
                    if not valid:
                        errors.append(f"audio {metric}={value} violates {limit_key}={payload[limit_key]}")
            except (OSError, RuntimeError, ValueError) as exc:
                if audio_limits_requested:
                    audio_checks["analysis_available"] = False
                    errors.append(f"required audio measurement failed: {exc}")
                else:
                    warnings.append(f"optional audio measurement unavailable: {exc}")
        elif require_audio and not has_audio:
            errors.append("audio stream is required but missing")
        if audio_limits_requested and not has_audio:
            audio_checks["analysis_available"] = False
            errors.append("explicit audio measurements require an audio stream")
        if require_stereo and has_audio and not audio_checks["stereo"]:
            errors.append("stereo audio is required but the output is not stereo")
        video_duration = float(probe.get("video_duration") or duration)
        audio_duration = float(probe.get("audio_duration") or 0.0)
        if require_audio and audio_duration > 0 and video_duration > 0:
            drift = abs(audio_duration - video_duration)
            audio_checks["duration_alignment"] = drift <= float(payload.get("audio_duration_tolerance", 0.5))
            if not audio_checks["duration_alignment"]:
                errors.append(f"audio/video duration drift is {drift:.3f}s")
        checks.update(audio_checks)
        if bool(payload.get("warn_if_no_audio", False)) and not has_audio:
            warnings.append("no audio stream detected")
        contact_sheet_path = str(payload.get("contact_sheet_path") or "")
        if contact_sheet_path and bool(probe.get("has_video")):
            service.make_contact_sheet(
                video_path=video_path,
                output_path=contact_sheet_path,
                frame_count=int(payload.get("frame_count", 6)),
                columns=int(payload.get("columns", 3)),
                scale_width=int(payload.get("scale_width", 320)),
                duration_seconds=duration,
            )
        return {
            "passed": all(checks.values()),
            "video_path": video_path,
            "file_exists": file_exists,
            "probe": probe,
            "audio_analysis": audio_analysis,
            "checks": checks,
            "duration": duration,
            "target_duration": float(target_duration) if target_duration not in {None, ""} else None,
            "errors": errors,
            "warnings": warnings,
            "contact_sheet_path": contact_sheet_path,
        }

    def merge_audio_video(self, payload: dict[str, object]) -> dict[str, object]:
        service = self._ffmpeg_service()
        return {
            "video_path": service.merge_audio_video(
                video_path=str(payload["video_path"]),
                audio_path=str(payload["audio_path"]),
                output_path=str(payload["output_path"]),
            )
        }

    def concat_audio(self, payload: dict[str, object]) -> dict[str, object]:
        service = self._ffmpeg_service()
        audio_paths = [str(path) for path in payload.get("audio_paths", [])]
        return {
            "audio_path": service.concat_audio(
                audio_paths=audio_paths,
                output_path=str(payload["output_path"]),
            )
        }

    def create_video_from_image(self, payload: dict[str, object]) -> dict[str, object]:
        service = self._ffmpeg_service()
        return {
            "video_path": service.create_video_from_image(
                image_path=str(payload["image_path"]),
                audio_path=str(payload["audio_path"]),
                output_path=str(payload["output_path"]),
                fps=int(payload.get("fps", 30)),
            )
        }

    def video_to_sprite(self, payload: dict[str, object]) -> dict[str, object]:
        return self._sprite_service().video_to_sprite(
            video_path=str(payload["video_path"]),
            output_dir=str(payload["output_dir"]),
            cell_width=int(payload.get("cell_width", 64)),
            cell_height=int(payload.get("cell_height", 64)),
            fps=float(payload.get("fps", 12)),
            loop=bool(payload.get("loop", True)),
            key_color=payload.get("key_color", "magenta"),
            chroma_threshold=int(payload.get("chroma_threshold", 52)),
            frame_map=payload.get("frame_map") if isinstance(payload.get("frame_map"), list) else None,
        )

    def compose_edit(self, payload: dict[str, object]) -> dict[str, object]:
        service = self._editing_service()
        raw_plan = payload.get("edit_plan")
        if not isinstance(raw_plan, dict):
            raise ValueError("media.compose_edit requires an edit_plan object")
        plan = EditPlan.from_dict(raw_plan)
        return service.render(
            plan,
            output_path=str(payload["output_path"]),
            manifest_path=str(payload["manifest_path"]) if payload.get("manifest_path") else None,
            contact_sheet_path=str(payload["contact_sheet_path"]) if payload.get("contact_sheet_path") else None,
            review_evidence_dir=str(payload["review_evidence_dir"]) if payload.get("review_evidence_dir") else None,
        )

    def materialize_edit(self, payload: dict[str, object]) -> dict[str, object]:
        service = self._editing_service()
        raw_result = payload.get("result")
        if not isinstance(raw_result, dict):
            raise ValueError("media.materialize_edit requires a rendered result object")
        return service.materialize_result(
            raw_result,
            output_path=str(payload["output_path"]),
            manifest_path=str(payload["manifest_path"]) if payload.get("manifest_path") else None,
            contact_sheet_path=str(payload["contact_sheet_path"]) if payload.get("contact_sheet_path") else None,
        )

    def generate_tts(self, payload: dict[str, object]) -> dict[str, object]:
        service = self._tts_service()
        output_path = str(payload["output_path"])
        return {
            "audio_path": service.generate_speech_sync(
                text=str(payload["text"]),
                output_path=output_path,
                voice=str(payload.get("voice", "en-US-AriaNeural")),
                rate=str(payload.get("rate", "+0%")),
            )
        }

    def _ffmpeg_service(self) -> FFmpegAdapter:
        if self._ffmpeg is None:
            self._ffmpeg = FFmpegAdapter()
        return self._ffmpeg

    def _sprite_service(self) -> SpriteAdapter:
        if self._sprite is None:
            self._sprite = SpriteAdapter(self._ffmpeg_service())
        return self._sprite

    def _editing_service(self) -> OpenCutEditAdapter:
        if self._editing is None:
            self._editing = OpenCutEditAdapter(output_root=self.output_root, input_roots=self.input_roots)
        return self._editing

    def _tts_service(self) -> TTSAdapter:
        if self._tts is None:
            self._tts = TTSAdapter()
        return self._tts


def register_media_service_tools(
    tool_registry: ToolRegistry,
    output_root: Path,
    input_roots: Iterable[Path] = (),
) -> None:
    tools = MediaServiceTools(output_root=output_root, input_roots=input_roots)
    tool_registry.register("media.extract_last_frame", tools.extract_last_frame, "Extract the last frame from a video")
    tool_registry.register("media.concat_videos", tools.concat_videos, "Concatenate multiple videos")
    tool_registry.register("media.change_video_speed", tools.change_video_speed, "Change video and audio playback speed")
    tool_registry.register("media.normalize_video_canvas", tools.normalize_video_canvas, "Fit video inside an exact canvas while preserving aspect ratio")
    tool_registry.register("media.trim_video", tools.trim_video, "Trim a packaged video to an explicit duration")
    tool_registry.register("media.video_to_gif", tools.video_to_gif, "Convert a video to a GIF")
    tool_registry.register("media.video_to_sprite", tools.video_to_sprite, "Convert a generated motion video to a transparent game sprite atlas")
    tool_registry.register("media.video_qa", tools.video_qa, "Probe duration/streams and create a video contact sheet")
    tool_registry.register("media.merge_audio_video", tools.merge_audio_video, "Merge one audio track into a video")
    tool_registry.register("audio.concat_tracks", tools.concat_audio, "Concatenate multiple audio tracks")
    tool_registry.register("media.create_video_from_image", tools.create_video_from_image, "Create a video from a still image and audio")
    tool_registry.register("media.compose_edit", tools.compose_edit, "Render a deterministic OpenCut-inspired timeline")
    tool_registry.register("media.materialize_edit", tools.materialize_edit, "Materialize the selected edit candidate")
    tool_registry.register("audio.generate_tts_real", tools.generate_tts, "Generate real narration audio")
