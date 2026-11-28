from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from agentic.tools.ffmpeg_adapter import FFmpegAdapter
from agentic.tools.motion_graphics_adapter import MotionGraphicsAdapter

HOLD_SECONDS = 4.0  # the sketch stays fully visible this long, as in the reference clips
FADE_SECONDS = 0.5
PAPER_WIDTH_RATIO = 0.2  # the taped paper's width, as a share of the video's shorter side
MARGIN_RATIO = 0.03
PAPER_ASPECT = 1.18
TILT_DEGREES = -3
TAPE = (232, 214, 168, 170)


def taped_sketch(sketch_path: Path, paper_width: int) -> Image.Image:
    """A kid's drawing on crumpled paper, held by four scotch-tape strips, with a soft shadow."""
    paper = Image.open(sketch_path).convert("RGB")
    crop_width = int(paper.width * 0.84)
    crop_height = min(paper.height, int(crop_width * PAPER_ASPECT))
    left, top = (paper.width - crop_width) // 2, (paper.height - crop_height) // 2
    paper_height = int(paper_width * crop_height / crop_width)
    paper = paper.crop((left, top, left + crop_width, top + crop_height)).resize(
        (paper_width, paper_height), Image.LANCZOS
    )
    pad = max(8, paper_width // 5)
    sheet = Image.new("RGBA", (paper_width + 2 * pad, paper_height + 2 * pad), (0, 0, 0, 0))
    shadow = Image.new("RGBA", sheet.size, (0, 0, 0, 0))
    offset = max(2, paper_width // 30)
    ImageDraw.Draw(shadow).rectangle(
        [pad + offset, pad + offset, pad + paper_width + offset, pad + paper_height + offset], fill=(0, 0, 0, 120)
    )
    sheet = Image.alpha_composite(sheet, shadow.filter(ImageFilter.GaussianBlur(max(2, paper_width // 25))))
    sheet.paste(paper, (pad, pad))
    strip_size = (int(paper_width * 0.34), int(paper_width * 0.11))
    corners = (
        (pad, pad, -38),
        (pad + paper_width, pad, 38),
        (pad, pad + paper_height, 38),
        (pad + paper_width, pad + paper_height, -38),
    )
    for x, y, angle in corners:
        strip = Image.new("RGBA", strip_size, TAPE).rotate(angle, expand=True)
        sheet.alpha_composite(strip, (int(x - strip.width / 2), int(y - strip.height / 2)))
    return sheet.rotate(TILT_DEGREES, expand=True, resample=Image.BICUBIC)


class SketchInsetAdapter:
    """Tape a kid's sketch of the character into the top-left corner of a clip, then let it fade away."""

    def __init__(self, ffmpeg: FFmpegAdapter | None = None) -> None:
        self.ffmpeg = ffmpeg or FFmpegAdapter()

    def compose(self, *, video_path: str, sketch_path: str, output_path: str) -> dict[str, object]:
        source = Path(video_path).expanduser().resolve()
        sketch = Path(sketch_path).expanduser().resolve()
        destination = Path(output_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Sketch inset source video does not exist: {source}")
        if not sketch.is_file():
            raise FileNotFoundError(f"Sketch inset drawing does not exist: {sketch}")
        if source == destination:
            raise ValueError("Sketch inset output_path must differ from the source video")
        source_probe = self.ffmpeg.probe_media(str(source))
        width, height = int(source_probe.get("width") or 0), int(source_probe.get("height") or 0)
        fps = float(source_probe.get("frame_rate") or 0.0)
        duration = float(source_probe.get("video_duration") or source_probe.get("duration") or 0.0)
        if not source_probe.get("has_video") or width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0 or duration <= 0:
            raise ValueError("Sketch inset source is missing usable video dimensions, frame rate, or duration")
        frame_count = max(1, round(duration * fps))
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.ffmpeg.overlay_rgba_frames(
            video_path=str(source),
            frames=self._frames(sketch, width=width, height=height, fps=fps, frame_count=frame_count),
            output_path=str(destination),
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
        )
        output_probe = self.ffmpeg.probe_media(str(destination))
        MotionGraphicsAdapter.verify_contract(source_probe, output_probe, fps)
        return {
            "video_path": str(destination),
            "source_video_path": str(source),
            "probe": output_probe,
            "frame_count": frame_count,
        }

    @staticmethod
    def _frames(sketch: Path, *, width: int, height: int, fps: float, frame_count: int) -> Iterator[bytes]:
        shorter_side = min(width, height)
        inset = taped_sketch(sketch, max(16, int(shorter_side * PAPER_WIDTH_RATIO)))
        margin = int(shorter_side * MARGIN_RATIO)
        blank = bytes(width * height * 4)
        visible = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        visible.alpha_composite(inset, (margin, margin))
        visible_bytes = visible.tobytes()
        fade_frames = max(1, round(FADE_SECONDS * fps))
        for index in range(frame_count):
            elapsed = index / fps
            if elapsed < HOLD_SECONDS:
                yield visible_bytes
            elif index < round(HOLD_SECONDS * fps) + fade_frames:
                strength = 1 - (index - round(HOLD_SECONDS * fps) + 1) / (fade_frames + 1)
                faded = visible.copy()
                faded.putalpha(visible.getchannel("A").point(lambda value, s=strength: int(value * s)))
                yield faded.tobytes()
            else:
                yield blank
