from __future__ import annotations

import math
import re
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont


MotionGraphicsKind = Literal["caption", "burst", "sparkles", "motion_lines"]
MAX_MOTION_GRAPHICS_CUES = 16
MOTION_GRAPHICS_SAFE_AREA = {"left": 0.08, "top": 0.08, "right": 0.92, "bottom": 0.82}
_HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


@dataclass(frozen=True, slots=True)
class MotionGraphicsCue:
    kind: MotionGraphicsKind
    start_seconds: float
    end_seconds: float
    x: float
    y: float
    text: str = ""
    color: str = "#FFFFFF"
    accent: str = "#FF5D8F"
    size: float = 1.0


@dataclass(frozen=True, slots=True)
class MotionGraphicsPlan:
    duration_seconds: float
    cues: list[MotionGraphicsCue]

    @classmethod
    def from_dict(cls, raw: object) -> "MotionGraphicsPlan":
        if not isinstance(raw, dict):
            raise ValueError("motion graphics plan must be an object")
        try:
            duration = float(raw["duration_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("motion graphics duration_seconds must be a number") from exc
        if not math.isfinite(duration) or not 0 < duration <= 120:
            raise ValueError("motion graphics duration_seconds must be greater than zero and at most 120")
        raw_cues = raw.get("cues")
        if not isinstance(raw_cues, list) or not 1 <= len(raw_cues) <= MAX_MOTION_GRAPHICS_CUES:
            raise ValueError(f"motion graphics cues must contain 1 to {MAX_MOTION_GRAPHICS_CUES} items")
        cues: list[MotionGraphicsCue] = []
        supported = {"caption", "burst", "sparkles", "motion_lines"}
        for index, item in enumerate(raw_cues):
            if not isinstance(item, dict):
                raise ValueError(f"motion graphics cue {index} must be an object")
            kind = str(item.get("kind") or "")
            if kind not in supported:
                raise ValueError(f"motion graphics cue {index} has unsupported kind {kind!r}")
            try:
                start = float(item["start_seconds"])
                end = float(item["end_seconds"])
                x = float(item["x"])
                y = float(item["y"])
                size = float(item.get("size", 1.0))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"motion graphics cue {index} has invalid numeric fields") from exc
            numeric = (start, end, x, y, size)
            if not all(math.isfinite(value) for value in numeric):
                raise ValueError(f"motion graphics cue {index} numeric fields must be finite")
            if start < 0 or end <= start or end > duration + 1e-6:
                raise ValueError(f"motion graphics cue {index} must have a valid time range within the source duration")
            safe = MOTION_GRAPHICS_SAFE_AREA
            if not safe["left"] <= x <= safe["right"] or not safe["top"] <= y <= safe["bottom"]:
                raise ValueError(f"motion graphics cue {index} is outside the safe area")
            if not 0.5 <= size <= 1.8:
                raise ValueError(f"motion graphics cue {index} size must be between 0.5 and 1.8")
            text = str(item.get("text") or "").strip()
            if kind == "caption" and (not text or len(text) > 32):
                raise ValueError(f"caption cue {index} text must contain 1 to 32 characters")
            if kind != "caption" and text:
                raise ValueError(f"non-caption cue {index} cannot contain text")
            color = str(item.get("color") or "#FFFFFF")
            accent = str(item.get("accent") or "#FF5D8F")
            if not _HEX_COLOR.fullmatch(color) or not _HEX_COLOR.fullmatch(accent):
                raise ValueError(f"motion graphics cue {index} colors must be #RRGGBB values")
            cues.append(
                MotionGraphicsCue(
                    kind=kind,  # type: ignore[arg-type]
                    start_seconds=start,
                    end_seconds=end,
                    x=x,
                    y=y,
                    text=text,
                    color=color.upper(),
                    accent=accent.upper(),
                    size=size,
                )
            )
        return cls(duration_seconds=duration, cues=cues)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def render_motion_graphics_frames(
    plan: MotionGraphicsPlan,
    output_dir: Path,
    *,
    width: int,
    height: int,
    fps: float,
    frame_count: int,
) -> list[Path]:
    if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
        raise ValueError("motion graphics rendering requires positive canvas, frame rate, and frame count")
    output_dir.mkdir(parents=True, exist_ok=True)
    cue_fonts = [
        _font(max(14, round(height * 0.066 * cue.size)))
        for cue in plan.cues
    ]
    paths: list[Path] = []
    for frame_index in range(frame_count):
        timestamp = frame_index / fps
        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer, "RGBA")
        for cue_index, cue in enumerate(plan.cues):
            if cue.start_seconds <= timestamp < cue.end_seconds:
                progress = (timestamp - cue.start_seconds) / (cue.end_seconds - cue.start_seconds)
                _draw_cue(draw, cue, progress, width, height, cue_fonts[cue_index])
        path = output_dir / f"frame_{frame_index:06d}.png"
        layer.save(path, format="PNG", optimize=True)
        paths.append(path)
    return paths


def motion_graphics_plan_json_schema() -> dict[str, object]:
    cue_schema = {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["caption", "burst", "sparkles", "motion_lines"]},
            "start_seconds": {"type": "number", "minimum": 0},
            "end_seconds": {"type": "number", "exclusiveMinimum": 0},
            "x": {"type": "number", "minimum": 0.08, "maximum": 0.92},
            "y": {"type": "number", "minimum": 0.08, "maximum": 0.82},
            "text": {"type": "string", "maxLength": 32},
            "color": {"type": "string", "pattern": "^#[0-9a-fA-F]{6}$"},
            "accent": {"type": "string", "pattern": "^#[0-9a-fA-F]{6}$"},
            "size": {"type": "number", "minimum": 0.5, "maximum": 1.8},
        },
        "required": ["kind", "start_seconds", "end_seconds", "x", "y", "text", "color", "accent", "size"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "duration_seconds": {"type": "number", "exclusiveMinimum": 0, "maximum": 120},
            "cues": {"type": "array", "minItems": 1, "maxItems": MAX_MOTION_GRAPHICS_CUES, "items": cue_schema},
        },
        "required": ["duration_seconds", "cues"],
        "additionalProperties": False,
    }


def _draw_cue(draw: ImageDraw.ImageDraw, cue: MotionGraphicsCue, progress: float, width: int, height: int, font: ImageFont.FreeTypeFont | ImageFont.ImageFont) -> None:
    center_x = round(cue.x * width)
    center_y = round(cue.y * height)
    pulse = max(0.0, math.sin(math.pi * progress))
    color = _rgba(cue.color, round(238 * max(0.25, pulse)))
    accent = _rgba(cue.accent, round(235 * max(0.25, pulse)))
    scale = cue.size

    if cue.kind == "caption":
        lines = textwrap.wrap(cue.text, width=16, break_long_words=True)[:2]
        text = "\n".join(lines)
        left, top, right, bottom = draw.multiline_textbbox((0, 0), text, font=font, spacing=4, align="center", stroke_width=0)
        text_width = right - left
        text_height = bottom - top
        pad_x = max(12, round(width * 0.025))
        pad_y = max(7, round(height * 0.018))
        box_width = min(round(width * 0.8), text_width + pad_x * 2)
        box_height = text_height + pad_y * 2
        x1 = max(round(width * 0.1), min(width - round(width * 0.1) - box_width, center_x - box_width // 2))
        y1 = max(round(height * 0.1), min(round(height * 0.8) - box_height, center_y - box_height // 2))
        draw.rounded_rectangle((x1, y1, x1 + box_width, y1 + box_height), radius=max(8, round(height * 0.025)), fill=(12, 15, 28, 196), outline=accent, width=max(1, round(height * 0.004)))
        draw.multiline_text((x1 + box_width // 2, y1 + box_height // 2), text, font=font, fill=color, anchor="mm", align="center", spacing=4, stroke_width=max(1, round(height * 0.002)), stroke_fill=(0, 0, 0, 210))
        return

    radius = max(8.0, min(width, height) * 0.18 * scale * (0.65 + 0.35 * pulse))
    if cue.kind == "burst":
        for ray in range(12):
            angle = math.tau * ray / 12 + 0.08 * math.sin(math.tau * progress)
            inner = radius * 0.55
            outer = radius * (1.0 if ray % 2 == 0 else 0.78)
            start = (center_x + math.cos(angle) * inner, center_y + math.sin(angle) * inner)
            end = (center_x + math.cos(angle) * outer, center_y + math.sin(angle) * outer)
            draw.line((start, end), fill=color if ray % 2 == 0 else accent, width=max(2, round(height * 0.009)))
    elif cue.kind == "sparkles":
        orbit = radius * (0.35 + 0.65 * progress)
        for sparkle in range(7):
            angle = math.tau * sparkle / 7 - progress * 1.7
            x = center_x + math.cos(angle) * orbit
            y = center_y + math.sin(angle) * orbit * 0.62
            size = max(3.0, min(width, height) * (0.012 + 0.01 * pulse) * scale)
            _draw_star(draw, x, y, size, color if sparkle % 2 == 0 else accent)
    elif cue.kind == "motion_lines":
        for line in range(5):
            y = center_y + (line - 2) * height * 0.045
            length = radius * (0.7 + 0.3 * math.sin(math.tau * progress + line))
            draw.line((center_x - length, y, center_x + length * 0.35, y), fill=color if line % 2 == 0 else accent, width=max(2, round(height * 0.008)))


def _draw_star(draw: ImageDraw.ImageDraw, x: float, y: float, radius: float, fill: tuple[int, int, int, int]) -> None:
    points = []
    for index in range(8):
        angle = math.tau * index / 8 - math.pi / 2
        distance = radius if index % 2 == 0 else radius * 0.32
        points.append((x + math.cos(angle) * distance, y + math.sin(angle) * distance))
    draw.polygon(points, fill=fill)


def _rgba(color: str, alpha: int) -> tuple[int, int, int, int]:
    return (int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16), max(0, min(255, alpha)))


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path(r"C:\Windows\Fonts\msjh.ttc"),
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                continue
    return ImageFont.load_default()
