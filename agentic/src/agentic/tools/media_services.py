from __future__ import annotations

from pathlib import Path

from agentic.runtime.registry import ToolRegistry
from agentic.tools.ffmpeg_adapter import FFmpegAdapter
from agentic.tools.tts_adapter import TTSAdapter


class MediaServiceTools:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._ffmpeg: FFmpegAdapter | None = None
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
            "loudness": True,
            "silence": True,
            "duration_alignment": True,
        }
        audio_analysis: dict[str, object] = {}
        should_analyze_audio = bool(payload.get("analyze_audio", require_audio)) and has_audio
        if should_analyze_audio:
            try:
                audio_analysis = service.analyze_audio(
                    video_path,
                    silence_threshold_db=float(payload.get("silence_threshold_db", -50.0)),
                    silence_min_seconds=float(payload.get("silence_min_seconds", 0.4)),
                )
                mean_volume = audio_analysis.get("mean_volume_db")
                max_volume = audio_analysis.get("max_volume_db")
                silence_ratio = float(audio_analysis.get("silence_ratio") or 0.0)
                audio_checks["loudness"] = mean_volume is not None and float(mean_volume) >= float(payload.get("min_mean_volume_db", -45.0))
                audio_checks["silence"] = silence_ratio <= float(payload.get("max_silence_ratio", 0.98))
                if max_volume is not None and float(max_volume) >= float(payload.get("max_peak_db", -0.1)):
                    errors.append(f"audio peak {float(max_volume):.2f} dBFS is at or above clipping threshold")
                    audio_checks["loudness"] = False
                if not audio_checks["loudness"]:
                    errors.append("audio mean level is too quiet or could not be measured")
                if not audio_checks["silence"]:
                    errors.append("audio is silent for too much of the rendered duration")
            except (OSError, RuntimeError, ValueError) as exc:
                audio_checks["loudness"] = False
                audio_checks["silence"] = False
                errors.append(f"audio analysis failed: {exc}")
        elif require_audio and not has_audio:
            errors.append("audio stream is required but missing")
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

    def _tts_service(self) -> TTSAdapter:
        if self._tts is None:
            self._tts = TTSAdapter()
        return self._tts


def register_media_service_tools(tool_registry: ToolRegistry, output_root: Path) -> None:
    tools = MediaServiceTools(output_root=output_root)
    tool_registry.register("media.extract_last_frame", tools.extract_last_frame, "Extract the last frame from a video")
    tool_registry.register("media.concat_videos", tools.concat_videos, "Concatenate multiple videos")
    tool_registry.register("media.video_to_gif", tools.video_to_gif, "Convert a video to a GIF")
    tool_registry.register("media.video_qa", tools.video_qa, "Probe duration/streams and create a video contact sheet")
    tool_registry.register("media.merge_audio_video", tools.merge_audio_video, "Merge one audio track into a video")
    tool_registry.register("audio.concat_tracks", tools.concat_audio, "Concatenate multiple audio tracks")
    tool_registry.register("media.create_video_from_image", tools.create_video_from_image, "Create a video from a still image and audio")
    tool_registry.register("audio.generate_tts_real", tools.generate_tts, "Generate real narration audio")
