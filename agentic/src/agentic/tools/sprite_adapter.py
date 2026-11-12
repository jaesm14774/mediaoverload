"""Deterministic conversion from a generated motion clip to game sprite assets."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageFilter

from agentic.tools.ffmpeg_adapter import FFmpegAdapter


SPRITE_GRID_ROWS = 4
SPRITE_GRID_COLS = 4
SPRITE_FRAME_COUNT = SPRITE_GRID_ROWS * SPRITE_GRID_COLS
SPRITE_DEFAULT_CELL_WIDTH = 128
SPRITE_DEFAULT_CELL_HEIGHT = 128
SPRITE_DEFAULT_FPS = 12
# The master image is rendered on a flat green backdrop. A pixel is backdrop
# when green leads both red and blue by at least this much, which tolerates the
# shade drift video models add without touching Kirby's pinks, reds, or blues.
SPRITE_KEY_DOMINANCE = 40
# A frame whose subject covers more than this multiple of the median area is an
# outlier burst; it is cropped by the shared box rather than enlarging it.
SPRITE_OUTLIER_AREA_RATIO = 1.5
SPRITE_BOX_PADDING = 0.1
# One palette across all frames keeps the sprite stable while it animates.
SPRITE_PALETTE_COLORS = 32
# Every logical sprite pixel is a block of this many cell pixels, so the sprite
# reads as hard pixel clusters instead of a smoothed miniature.
SPRITE_PIXEL_BLOCK = 2
# Previews are enlarged with nearest-neighbor, never smoothed, to about this edge.
SPRITE_PREVIEW_EDGE = 384
# The clip people review and publish: the cut-out sprite looping on a flat
# backdrop, on a square canvas with even sides for the video encoder.
SPRITE_SHOWCASE_EDGE = 720
SPRITE_SHOWCASE_BACKDROP = (247, 240, 228)
SPRITE_SHOWCASE_SECONDS = 5


class SpriteAdapter:
    """Cut the green backdrop out of sampled frames and pack a pixel-art atlas."""

    def __init__(self, ffmpeg: FFmpegAdapter | None = None) -> None:
        self.ffmpeg = ffmpeg or FFmpegAdapter()

    def video_to_sprite(
        self,
        *,
        video_path: str,
        output_dir: str,
        cell_width: int = SPRITE_DEFAULT_CELL_WIDTH,
        cell_height: int = SPRITE_DEFAULT_CELL_HEIGHT,
        fps: float = SPRITE_DEFAULT_FPS,
        loop: bool = True,
        frame_map: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        source = Path(video_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Sprite source video does not exist: {source}")
        probe = self.ffmpeg.probe_media(str(source))
        if not bool(probe.get("has_video")):
            raise ValueError(f"Sprite source does not contain a video stream: {source}")
        duration = float(probe.get("video_duration") or probe.get("duration") or 0.0)
        if duration <= 0:
            raise ValueError(f"Sprite source video has no usable duration: {source}")

        root = Path(output_dir).resolve()
        raw_dir = root / "source_frames"
        raw_dir.mkdir(parents=True, exist_ok=True)
        frame_paths: list[str] = []
        timestamps: list[float] = []
        frame_count = SPRITE_FRAME_COUNT
        # Sample the interior of each interval so the first and final frames
        # do not become duplicate codec boundary frames.
        epsilon = min(1.0 / max(float(probe.get("frame_rate") or 24.0), 1.0), duration / 100.0)
        for index in range(frame_count):
            timestamp = min(
                duration - epsilon,
                duration * ((index + 0.5) / frame_count),
            )
            frame_path = raw_dir / f"source-{index + 1:02d}.png"
            timestamps.append(timestamp)
            frame_paths.append(str(frame_path))
        self.ffmpeg.extract_frames_at(str(source), frame_paths, timestamps)

        result = self.build_from_frames(
            frame_paths,
            output_dir=str(root),
            cell_width=cell_width,
            cell_height=cell_height,
            fps=fps,
            loop=loop,
            frame_map=frame_map,
        )
        result["source_video"] = str(source)
        result["source_probe"] = probe
        manifest_path = Path(str(result["manifest_path"]))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source_video"] = str(source)
        manifest["source_probe"] = probe
        manifest["sampling"] = {"method": "interval_midpoints", "timestamps_seconds": timestamps}
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return result

    def build_from_frames(
        self,
        frame_paths: list[str],
        *,
        output_dir: str,
        cell_width: int = SPRITE_DEFAULT_CELL_WIDTH,
        cell_height: int = SPRITE_DEFAULT_CELL_HEIGHT,
        fps: float = SPRITE_DEFAULT_FPS,
        loop: bool = True,
        frame_map: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        frame_count = len(frame_paths)
        if frame_count != SPRITE_FRAME_COUNT:
            raise ValueError(
                f"game_sprite requires exactly {SPRITE_FRAME_COUNT} frames for a 4x4 atlas; received {frame_count}"
            )
        width = _positive_int(cell_width, "cell_width")
        height = _positive_int(cell_height, "cell_height")
        frame_rate = float(fps)
        if not math.isfinite(frame_rate) or frame_rate <= 0:
            raise ValueError("fps must be a finite number greater than zero")
        root = Path(output_dir).resolve()
        frames_dir = root / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        source_frames: list[Image.Image] = []
        for frame_path in frame_paths:
            source = Path(frame_path).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"Sprite frame does not exist: {source}")
            with Image.open(source) as image:
                source_frames.append(_key_out_green_backdrop(image.convert("RGBA")))
        shared_bbox = _shared_bbox(source_frames)
        if shared_bbox is None:
            raise ValueError("Sprite source frames are fully transparent")
        normalized_frames: list[Image.Image] = []
        frame_records: list[dict[str, Any]] = []
        for index, frame_path in enumerate(frame_paths):
            source = Path(frame_path).resolve()
            normalized = _enlarge(
                _fit_frame(
                    source_frames[index],
                    max(1, width // SPRITE_PIXEL_BLOCK),
                    max(1, height // SPRITE_PIXEL_BLOCK),
                    crop_bbox=shared_bbox,
                ),
                SPRITE_PIXEL_BLOCK,
            )
            if _alpha_bbox(normalized) is None:
                raise ValueError(f"Sprite source frame is fully transparent: {source}")
            frame_copy = normalized.copy()
            normalized_frames.append(frame_copy)
            target = frames_dir / f"frame-{index + 1:02d}.png"
            frame_copy.save(target, format="PNG", optimize=True)
            frame_records.append(
                {
                    "index": index,
                    "row": index // SPRITE_GRID_COLS,
                    "column": index % SPRITE_GRID_COLS,
                    "path": str(target),
                    "sha256": _sha256(target),
                    "bbox": list(frame_copy.getbbox() or ()),
                    "beat": _frame_map_value(frame_map, index, "beat"),
                    "description": _frame_map_value(frame_map, index, "description"),
                }
            )

        normalized_frames = _apply_shared_palette(normalized_frames)
        for record, frame in zip(frame_records, normalized_frames):
            frame.save(record["path"], format="PNG", optimize=True)
            record["sha256"] = _sha256(Path(record["path"]))
        alpha_counts = [_opaque_pixel_count(frame) for frame in normalized_frames]
        atlas = Image.new("RGBA", (width * SPRITE_GRID_COLS, height * SPRITE_GRID_ROWS), (0, 0, 0, 0))
        for index, frame in enumerate(normalized_frames):
            atlas.paste(
                frame,
                ((index % SPRITE_GRID_COLS) * width, (index // SPRITE_GRID_COLS) * height),
                frame,
            )
        atlas_path = root / "atlas.png"
        atlas.save(atlas_path, format="PNG", optimize=True)

        duration_ms = max(1, round(1000.0 / frame_rate))
        preview_scale = max(1, round(SPRITE_PREVIEW_EDGE / max(width, height)))
        preview_frames = [_enlarge(frame, preview_scale) for frame in normalized_frames]
        gif_frames = [_to_transparent_palette(frame) for frame in preview_frames]
        gif_path = root / "animation.gif"
        gif_kwargs: dict[str, Any] = {
            "save_all": True,
            "append_images": gif_frames[1:],
            "duration": duration_ms,
            "disposal": 2,
            "transparency": 255,
        }
        if loop:
            gif_kwargs["loop"] = 0
        gif_frames[0].save(gif_path, format="GIF", **gif_kwargs)

        webp_path = root / "animation.webp"
        preview_frames[0].save(
            webp_path,
            format="WEBP",
            save_all=True,
            append_images=preview_frames[1:],
            duration=duration_ms,
            loop=0 if loop else 1,
            lossless=True,
            method=6,
        )

        showcase_path = self._write_showcase(root, normalized_frames, frame_rate)

        qa = {
            "passed": True,
            "checks": {
                "frame_count": len(normalized_frames) == SPRITE_FRAME_COUNT,
                "cell_dimensions": all(frame.size == (width, height) for frame in normalized_frames),
                "atlas_dimensions": atlas.size == (width * SPRITE_GRID_COLS, height * SPRITE_GRID_ROWS),
                "binary_alpha": all(_has_binary_alpha(frame) for frame in normalized_frames),
                "nonempty_frames": all(count > 0 for count in alpha_counts),
            },
            "opaque_pixel_count": alpha_counts,
        }
        manifest = {
            "schema_version": 1,
            "asset_kind": "game_sprite",
            "grid": {"rows": SPRITE_GRID_ROWS, "columns": SPRITE_GRID_COLS},
            "cell": {"width": width, "height": height},
            "frame_count": SPRITE_FRAME_COUNT,
            "fps": frame_rate,
            "loop": bool(loop),
            "animation_kind": "periodic" if loop else "one_shot",
            "alpha": "binary",
            "background_mode": "green_keyed",
            "palette_colors": SPRITE_PALETTE_COLORS,
            "preview_scale": preview_scale,
            "frames": frame_records,
            "outputs": {
                "atlas": str(atlas_path),
                "gif": str(gif_path),
                "webp": str(webp_path),
                "showcase": str(showcase_path),
                "frames_dir": str(frames_dir),
            },
            "qa": qa,
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "run_dir": str(root),
            "atlas_path": str(atlas_path),
            "gif_path": str(gif_path),
            "webp_path": str(webp_path),
            "showcase_path": str(showcase_path),
            "frame_paths": [str(record["path"]) for record in frame_records],
            "manifest_path": str(manifest_path),
            "saved_files": [
                str(atlas_path),
                str(gif_path),
                str(webp_path),
                str(showcase_path),
                str(manifest_path),
                *[str(record["path"]) for record in frame_records],
            ],
            "qa": qa,
        }

    def _write_showcase(
        self, root: Path, cell_frames: list[Image.Image], frame_rate: float
    ) -> Path:
        """Loop the cut-out sprite on a flat backdrop as an mp4 a feed can play."""
        edge = SPRITE_SHOWCASE_EDGE
        frames: list[Image.Image] = []
        # A whole-number scale keeps every sprite pixel a clean square.
        scale = max(1, (edge * 9 // 10) // max(cell_frames[0].size))
        for frame in cell_frames:
            sprite = _enlarge(frame.convert("RGBA"), scale)
            canvas = Image.new("RGB", (edge, edge), SPRITE_SHOWCASE_BACKDROP)
            canvas.paste(
                sprite,
                ((edge - sprite.width) // 2, (edge - sprite.height) // 2),
                sprite,
            )
            frames.append(canvas)
        loops = max(1, math.ceil(SPRITE_SHOWCASE_SECONDS * frame_rate / len(frames)))
        sequence = frames * loops
        gif_path = root / "showcase.gif"
        sequence[0].save(
            gif_path,
            format="GIF",
            save_all=True,
            append_images=sequence[1:],
            duration=max(1, round(1000.0 / frame_rate)),
            loop=0,
        )
        showcase_path = root / "showcase.mp4"
        self.ffmpeg.gif_to_mp4(str(gif_path), str(showcase_path), fps=frame_rate)
        gif_path.unlink()
        return showcase_path


def _positive_int(value: object, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


def _fit_frame(
    image: Image.Image,
    width: int,
    height: int,
    *,
    crop_bbox: tuple[int, int, int, int] | None = None,
    safe_margin: int = 2,
) -> Image.Image:
    bbox = crop_bbox or _alpha_bbox(image)
    if bbox is None:
        raise ValueError("Cannot fit a fully transparent frame")
    content = image.crop(bbox)
    available_width = max(1, width - safe_margin * 2)
    available_height = max(1, height - safe_margin * 2)
    scale = min(available_width / content.width, available_height / content.height)
    # Premultiplied BOX resampling keeps hard cluster edges and never bleeds the
    # transparent backdrop's color into the silhouette.
    resized = (
        content.convert("RGBa")
        .resize(
            (max(1, round(content.width * scale)), max(1, round(content.height * scale))),
            Image.Resampling.BOX,
        )
        .convert("RGBA")
    )
    result = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    result.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return _binary_alpha(result)


def _shared_bbox(images: list[Image.Image]) -> tuple[int, int, int, int] | None:
    """One crop box for every frame, sized by the ordinary frames.

    A single screen-filling burst must not shrink the character in all the other
    frames, so frames far larger than the median are cropped by the box instead
    of defining it. The box is padded so ordinary effects keep some room.
    """
    boxes = [box for box in (_alpha_bbox(image) for image in images) if box is not None]
    if not boxes:
        return None
    areas = sorted((box[2] - box[0]) * (box[3] - box[1]) for box in boxes)
    ordinary_limit = areas[len(areas) // 2] * SPRITE_OUTLIER_AREA_RATIO
    ordinary = [
        box for box in boxes if (box[2] - box[0]) * (box[3] - box[1]) <= ordinary_limit
    ]
    left = min(box[0] for box in ordinary)
    top = min(box[1] for box in ordinary)
    right = max(box[2] for box in ordinary)
    bottom = max(box[3] for box in ordinary)
    pad_x = round((right - left) * SPRITE_BOX_PADDING)
    pad_y = round((bottom - top) * SPRITE_BOX_PADDING)
    width, height = images[0].size
    return (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(width, right + pad_x),
        min(height, bottom + pad_y),
    )


def _alpha_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    return image.getchannel("A").getbbox()


def _binary_alpha(image: Image.Image) -> Image.Image:
    alpha = image.getchannel("A").point(lambda value: 255 if value >= 128 else 0)
    image.putalpha(alpha)
    return image


def _key_out_green_backdrop(image: Image.Image) -> Image.Image:
    """Make the flat green backdrop transparent and remove its color spill from the edge."""
    red, green, blue, _ = image.split()
    dominance = ImageChops.subtract(green, ImageChops.lighter(red, blue))
    keep = dominance.point(lambda value: 0 if value >= SPRITE_KEY_DOMINANCE else 255)
    # Two pixels of erosion remove the half-blended green halo (olive on orange effects).
    keep = keep.filter(ImageFilter.MinFilter(5))
    despilled_green = ImageChops.darker(green, ImageChops.lighter(red, blue))
    return Image.merge("RGBA", (red, despilled_green, blue, keep))


def _apply_shared_palette(frames: list[Image.Image]) -> list[Image.Image]:
    opaque_pixels = [
        pixel[:3] for frame in frames for pixel in frame.getdata() if pixel[3] == 255
    ]
    sample = Image.new("RGB", (len(opaque_pixels), 1))
    sample.putdata(opaque_pixels)
    palette = sample.quantize(
        colors=SPRITE_PALETTE_COLORS, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
    )
    result: list[Image.Image] = []
    for frame in frames:
        quantized = frame.convert("RGB").quantize(palette=palette, dither=Image.Dither.NONE).convert("RGBA")
        quantized.putalpha(frame.getchannel("A"))
        result.append(quantized)
    return result


def _enlarge(image: Image.Image, scale: int) -> Image.Image:
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def _to_transparent_palette(image: Image.Image) -> Image.Image:
    alpha = image.getchannel("A")
    opaque = Image.new("RGB", image.size, (0, 0, 0))
    opaque.paste(image.convert("RGB"), mask=alpha)
    palette = opaque.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    palette_data = list(palette.getpalette() or [])
    palette_data.extend([0] * (768 - len(palette_data)))
    transparent_index = 255
    palette_data[transparent_index * 3 : transparent_index * 3 + 3] = [0, 0, 0]
    palette.putpalette(palette_data[:768])
    transparent_mask = alpha.point(lambda value: 255 if value == 0 else 0)
    palette.paste(transparent_index, mask=transparent_mask)
    return palette


def _opaque_pixel_count(image: Image.Image) -> int:
    return sum(1 for value in image.getchannel("A").getdata() if value > 0)


def _has_binary_alpha(image: Image.Image) -> bool:
    return set(image.getchannel("A").getdata()).issubset({0, 255})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_map_value(frame_map: list[dict[str, Any]] | None, index: int, key: str) -> str:
    if not isinstance(frame_map, list) or index >= len(frame_map):
        return ""
    item = frame_map[index]
    if not isinstance(item, dict):
        return ""
    return str(item.get(key) or "").strip()
