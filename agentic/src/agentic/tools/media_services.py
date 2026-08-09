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
        probe = service.probe_media(video_path)
        errors: list[str] = []
        warnings: list[str] = []
        target_duration = payload.get("target_duration")
        tolerance = float(payload.get("duration_tolerance", 0.5))
        duration = float(probe.get("duration") or 0.0)
        if not bool(probe.get("has_video")):
            errors.append("no video stream")
        if target_duration not in {None, ""} and abs(duration - float(target_duration)) > tolerance:
            errors.append(f"duration {duration:.3f}s is outside target {float(target_duration):.3f}s ± {tolerance:.3f}s")
        if bool(payload.get("warn_if_no_audio", False)) and not bool(probe.get("has_audio")):
            warnings.append("no audio stream detected")
        contact_sheet_path = str(payload.get("contact_sheet_path") or "")
        if contact_sheet_path and bool(probe.get("has_video")):
            service.make_contact_sheet(
                video_path=video_path,
                output_path=contact_sheet_path,
                frame_count=int(payload.get("frame_count", 6)),
                columns=int(payload.get("columns", 3)),
                scale_width=int(payload.get("scale_width", 320)),
            )
        return {
            "passed": not errors,
            "video_path": video_path,
            "probe": probe,
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
