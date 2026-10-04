from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

from agentic.runtime.motion_graphics import (
    MOTION_GRAPHICS_SAFE_AREA,
    MotionGraphicsPlan,
    render_motion_graphics_frames,
)
from agentic.tools.ffmpeg_adapter import FFmpegAdapter


class MotionGraphicsAdapter:
    """Render declarative Pillow layers and composite them over an unchanged video source."""

    def __init__(self, ffmpeg: FFmpegAdapter | None = None) -> None:
        self.ffmpeg = ffmpeg or FFmpegAdapter()

    def compose(
        self,
        *,
        video_path: str,
        output_path: str,
        plan: MotionGraphicsPlan | dict[str, object],
        work_dir: Path,
    ) -> dict[str, object]:
        source = Path(video_path).expanduser().resolve()
        destination = Path(output_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Motion graphics source video does not exist: {source}")
        if source == destination:
            raise ValueError("Motion graphics output_path must differ from the source video")
        normalized_plan = (
            MotionGraphicsPlan.from_dict(plan.to_dict())
            if isinstance(plan, MotionGraphicsPlan)
            else MotionGraphicsPlan.from_dict(plan)
        )
        source_probe = self.ffmpeg.probe_media(str(source))
        if not bool(source_probe.get("has_video")):
            raise ValueError("Motion graphics source must contain a video stream")
        width = int(source_probe.get("width") or 0)
        height = int(source_probe.get("height") or 0)
        fps = float(source_probe.get("frame_rate") or 0.0)
        duration = float(source_probe.get("video_duration") or source_probe.get("duration") or 0.0)
        if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0 or duration <= 0:
            raise ValueError("Motion graphics source is missing usable video dimensions, frame rate, or duration")
        frame_count = max(1, round(duration * fps))
        if abs(normalized_plan.duration_seconds - duration) > (1 / fps) + 1e-3:
            raise ValueError("Motion graphics plan duration must match the source video within one frame")

        destination.parent.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        frame_pattern = "frame_%06d.png"
        with tempfile.TemporaryDirectory(prefix="motion-graphics-frames-", dir=work_dir) as temp_dir:
            frame_dir = Path(temp_dir)
            render_motion_graphics_frames(
                normalized_plan,
                frame_dir,
                width=width,
                height=height,
                fps=fps,
                frame_count=frame_count,
            )
            self.ffmpeg.overlay_png_sequence(
                video_path=str(source),
                frame_pattern=str(frame_dir / frame_pattern),
                output_path=str(destination),
                fps=fps,
                frame_count=frame_count,
            )

        output_probe = self.ffmpeg.probe_media(str(destination))
        self._verify_contract(source_probe, output_probe, fps)
        plan_path = destination.with_suffix(".motion-graphics-plan.json")
        contact_sheet_path = destination.with_suffix(".contact-sheet.jpg")
        plan_path.write_text(json.dumps(normalized_plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        self.ffmpeg.make_contact_sheet(
            str(destination),
            str(contact_sheet_path),
            frame_count=6,
            columns=3,
            scale_width=480,
            duration_seconds=float(output_probe.get("video_duration") or output_probe.get("duration") or duration),
        )
        return {
            "video_path": str(destination),
            "source_video_path": str(source),
            "probe": output_probe,
            "source_probe": source_probe,
            "contact_sheet_path": str(contact_sheet_path),
            "plan_path": str(plan_path),
            "plan": normalized_plan.to_dict(),
            "frame_count": frame_count,
            "safe_area": {"passed": True, "bounds": dict(MOTION_GRAPHICS_SAFE_AREA)},
        }

    @staticmethod
    def _verify_contract(source: dict[str, object], output: dict[str, object], fps: float) -> None:
        if not bool(output.get("has_video")):
            raise RuntimeError("Motion graphics output has no video stream")
        if (output.get("width"), output.get("height")) != (source.get("width"), source.get("height")):
            raise RuntimeError("Motion graphics output changed the source canvas dimensions")
        if abs(float(output.get("frame_rate") or 0.0) - fps) > 0.15:
            raise RuntimeError("Motion graphics output changed the source frame rate")
        source_duration = float(source.get("video_duration") or source.get("duration") or 0.0)
        output_duration = float(output.get("video_duration") or output.get("duration") or 0.0)
        if abs(output_duration - source_duration) > (1 / fps) + 0.02:
            raise RuntimeError("Motion graphics output changed the source video duration")
        source_has_audio = bool(source.get("has_audio"))
        output_has_audio = bool(output.get("has_audio"))
        if source_has_audio != output_has_audio:
            raise RuntimeError("Motion graphics output changed whether the source has audio")
        if source_has_audio and str(source.get("audio_codec") or "") != str(output.get("audio_codec") or ""):
            raise RuntimeError("Motion graphics output changed the source audio codec")
