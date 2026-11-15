"""Deterministic conversion from a generated motion clip to game sprite assets."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

from agentic.tools.ffmpeg_adapter import FFmpegAdapter


SPRITE_GRID_ROWS = 4
SPRITE_GRID_COLS = 4
SPRITE_FRAME_COUNT = SPRITE_GRID_ROWS * SPRITE_GRID_COLS
SPRITE_DEFAULT_CELL_WIDTH = 128
SPRITE_DEFAULT_CELL_HEIGHT = 128
SPRITE_DEFAULT_FPS = 12
# The master image is rendered on a flat green backdrop. A pixel is fully
# backdrop when green leads both red and blue by SPRITE_KEY_FULL, fully subject
# below SPRITE_KEY_NONE, and feathered in between, which tolerates the shade
# drift video models add without touching Kirby's pinks, reds, or blues.
SPRITE_KEY_NONE = 30
SPRITE_KEY_FULL = 60
# Video models invent bokeh, rain, and light spots on the backdrop. The subject
# stays put while those specks drift, so a pixel is the subject's territory when
# it is solid in at least SPRITE_SUBJECT_PERSISTENCE of the frames, widened by
# SPRITE_SUBJECT_REACH source pixels for the motion. Inside that territory,
# pieces closer than SPRITE_SUBJECT_GAP form one cluster and a cluster smaller
# than SPRITE_SUBJECT_MIN_SHARE of the largest is dropped.
SPRITE_SUBJECT_PERSISTENCE = 0.3
SPRITE_SUBJECT_REACH = 12
SPRITE_SUBJECT_GAP = 2
SPRITE_SUBJECT_MIN_SHARE = 0.25
# A frame that is almost entirely solid has no green backdrop to speck; it stays whole.
SPRITE_NO_BACKDROP_SHARE = 0.9
# When the territory keeps less than this share of the frame's own subject, the
# subject is moving and its own cluster is kept whole.
SPRITE_TRAVEL_SHARE = 0.5
# A frame whose subject covers more than this multiple of the median area is an
# outlier burst; it is cropped by the shared box rather than enlarging it.
SPRITE_OUTLIER_AREA_RATIO = 1.5
SPRITE_BOX_PADDING = 0.1
# Previews are enlarged with smooth resampling to about this edge.
SPRITE_PREVIEW_EDGE = 384
# The clip people review and publish: the cut-out sprite looping on a flat
# backdrop, on a square canvas with even sides for the video encoder.
SPRITE_SHOWCASE_EDGE = 720
SPRITE_SHOWCASE_BACKDROP = (247, 240, 228)
SPRITE_SHOWCASE_SECONDS = 5


class SpriteAdapter:
    """Cut the green backdrop out of sampled frames and pack a full-colour atlas."""

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
        source_frames = _isolate_subject(source_frames)
        shared_bbox = _shared_bbox(source_frames)
        if shared_bbox is None:
            raise ValueError("Sprite source frames are fully transparent")
        normalized_frames: list[Image.Image] = []
        frame_records: list[dict[str, Any]] = []
        for index, frame_path in enumerate(frame_paths):
            source = Path(frame_path).resolve()
            normalized = _fit_frame(
                source_frames[index], width, height, crop_bbox=shared_bbox
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
                "transparent_backdrop": all(_corners_transparent(frame) for frame in normalized_frames),
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
            "alpha": "soft",
            "background_mode": "green_keyed",
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
        scale = (edge * 9 // 10) / max(cell_frames[0].size)
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
    resized = _resample(
        content,
        (max(1, round(content.width * scale)), max(1, round(content.height * scale))),
    )
    result = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    result.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return result


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
    return image.getchannel("A").point(lambda value: 255 if value >= 128 else 0).getbbox()


def _corners_transparent(image: Image.Image) -> bool:
    last_x, last_y = image.width - 1, image.height - 1
    return all(
        image.getpixel(corner)[3] == 0
        for corner in ((0, 0), (last_x, 0), (0, last_y), (last_x, last_y))
    )


def _key_out_green_backdrop(image: Image.Image) -> Image.Image:
    """Make the green backdrop transparent, feathered at the edge, and remove colour spill."""
    pixels = np.asarray(image.convert("RGBA"), dtype=np.float32).copy()
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    dominance = green - np.maximum(red, blue)
    backdrop = np.clip(
        (dominance - SPRITE_KEY_NONE) / (SPRITE_KEY_FULL - SPRITE_KEY_NONE), 0.0, 1.0
    )
    pixels[..., 1] = np.minimum(green, np.maximum(red, blue))
    pixels[..., 3] = (1.0 - backdrop) * 255.0
    keyed = Image.fromarray(pixels.round().astype(np.uint8), "RGBA")
    # One pixel of erosion removes the half-blended green halo.
    keyed.putalpha(keyed.getchannel("A").filter(ImageFilter.MinFilter(3)))
    return keyed


def _isolate_subject(frames: list[Image.Image]) -> list[Image.Image]:
    """Drop backdrop specks that drift across the clip while the subject stays."""
    alphas = np.stack([np.asarray(frame.getchannel("A")) for frame in frames])
    solid = alphas >= 128
    territory = ndimage.binary_dilation(
        solid.mean(axis=0) >= SPRITE_SUBJECT_PERSISTENCE, iterations=SPRITE_SUBJECT_REACH
    )
    isolated: list[Image.Image] = []
    for frame, frame_solid, alpha in zip(frames, solid, alphas):
        if frame_solid.mean() >= SPRITE_NO_BACKDROP_SHARE:
            isolated.append(frame)
            continue
        keep = _subject_clusters(frame_solid & territory)
        # A subject that travels is solid in few frames at any one pixel, so
        # the territory shrinks and would erase it; fall back to its own cluster.
        whole = _subject_clusters(frame_solid)
        if keep.sum() < SPRITE_TRAVEL_SHARE * whole.sum():
            keep = whole
        result = frame.copy()
        result.putalpha(Image.fromarray(np.where(keep, alpha, 0).astype(np.uint8)))
        isolated.append(result)
    return isolated


def _subject_clusters(solid: np.ndarray) -> np.ndarray:
    """Pixels of the clusters that are not tiny next to the largest one."""
    clusters, count = ndimage.label(
        ndimage.binary_dilation(solid, iterations=SPRITE_SUBJECT_GAP)
    )
    if count == 0:
        return solid
    sizes = np.asarray(ndimage.sum(solid, clusters, index=range(1, count + 1)))
    big = 1 + np.flatnonzero(sizes >= sizes.max() * SPRITE_SUBJECT_MIN_SHARE)
    return solid & np.isin(clusters, big)


def _resample(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Premultiplied resampling never bleeds the transparent backdrop into the silhouette."""
    return image.convert("RGBa").resize(size, Image.Resampling.LANCZOS).convert("RGBA")


def _enlarge(image: Image.Image, scale: float) -> Image.Image:
    return _resample(image, (round(image.width * scale), round(image.height * scale)))


def _to_transparent_palette(image: Image.Image) -> Image.Image:
    alpha = image.getchannel("A")
    # A hard mask keeps edge colours unblended; a soft one would darken them.
    opaque = Image.new("RGB", image.size, (0, 0, 0))
    opaque.paste(image.convert("RGB"), mask=alpha.point(lambda value: 255 if value >= 128 else 0))
    palette = opaque.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    palette_data = list(palette.getpalette() or [])
    palette_data.extend([0] * (768 - len(palette_data)))
    transparent_index = 255
    palette_data[transparent_index * 3 : transparent_index * 3 + 3] = [0, 0, 0]
    palette.putpalette(palette_data[:768])
    transparent_mask = alpha.point(lambda value: 255 if value < 128 else 0)
    palette.paste(transparent_index, mask=transparent_mask)
    return palette


def _opaque_pixel_count(image: Image.Image) -> int:
    return sum(1 for value in image.getchannel("A").getdata() if value > 0)


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
