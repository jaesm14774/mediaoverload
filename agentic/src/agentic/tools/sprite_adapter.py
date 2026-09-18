"""Deterministic conversion from a generated motion clip to game sprite assets."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

from agentic.tools.ffmpeg_adapter import FFmpegAdapter


SPRITE_GRID_ROWS = 4
SPRITE_GRID_COLS = 4
SPRITE_FRAME_COUNT = SPRITE_GRID_ROWS * SPRITE_GRID_COLS
SPRITE_DEFAULT_CELL_WIDTH = 64
SPRITE_DEFAULT_CELL_HEIGHT = 64
SPRITE_DEFAULT_FPS = 12
SPRITE_DEFAULT_CHROMA_THRESHOLD = 52


class SpriteAdapter:
    """Build a transparent 4x4 sprite atlas without a human curation step."""

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
        key_color: str | Iterable[int] = "magenta",
        chroma_threshold: int = SPRITE_DEFAULT_CHROMA_THRESHOLD,
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
            self.ffmpeg.extract_frame_at(str(source), str(frame_path), timestamp)
            frame_paths.append(str(frame_path))

        result = self.build_from_frames(
            frame_paths,
            output_dir=str(root),
            cell_width=cell_width,
            cell_height=cell_height,
            fps=fps,
            loop=loop,
            key_color=key_color,
            chroma_threshold=chroma_threshold,
            frame_map=frame_map,
        )
        result["source_video"] = str(source)
        result["source_probe"] = probe
        manifest_path = Path(str(result["manifest_path"]))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source_video"] = str(source)
        manifest["source_probe"] = probe
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
        key_color: str | Iterable[int] = "magenta",
        chroma_threshold: int = SPRITE_DEFAULT_CHROMA_THRESHOLD,
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
        threshold = _nonnegative_int(chroma_threshold, "chroma_threshold")
        chroma = _parse_key_color(key_color)

        root = Path(output_dir).resolve()
        frames_dir = root / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        keyed_frames: list[Image.Image] = []
        for frame_path in frame_paths:
            source = Path(frame_path).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"Sprite frame does not exist: {source}")
            with Image.open(source) as image:
                keyed_frames.append(_chroma_key(image.convert("RGBA"), chroma, threshold))
        shared_bbox = _union_bbox(keyed_frames)
        if shared_bbox is None:
            raise ValueError("Sprite frames became fully transparent after chroma removal")
        normalized_frames: list[Image.Image] = []
        frame_records: list[dict[str, Any]] = []
        for index, frame_path in enumerate(frame_paths):
            source = Path(frame_path).resolve()
            normalized = _fit_frame(keyed_frames[index], width, height, crop_bbox=shared_bbox)
            if _alpha_bbox(normalized) is None:
                raise ValueError(f"Sprite frame became fully transparent after chroma removal: {source}")
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

        if len({_sha256(Path(record["path"])) for record in frame_records}) < 2:
            raise ValueError("Sprite source produced no visible frame-to-frame motion")
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
        gif_frames = [_to_transparent_palette(frame) for frame in normalized_frames]
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
        normalized_frames[0].save(
            webp_path,
            format="WEBP",
            save_all=True,
            append_images=normalized_frames[1:],
            duration=duration_ms,
            loop=0 if loop else 1,
            lossless=True,
            method=6,
        )

        qa = {
            "passed": True,
            "checks": {
                "frame_count": len(normalized_frames) == SPRITE_FRAME_COUNT,
                "cell_dimensions": all(frame.size == (width, height) for frame in normalized_frames),
                "atlas_dimensions": atlas.size == (width * SPRITE_GRID_COLS, height * SPRITE_GRID_ROWS),
                "binary_alpha": all(_has_binary_alpha(frame) for frame in normalized_frames),
                "motion_present": len({_sha256(Path(record["path"])) for record in frame_records}) >= 2,
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
            "chroma": {"key_color": list(chroma), "threshold": threshold},
            "frames": frame_records,
            "outputs": {
                "atlas": str(atlas_path),
                "gif": str(gif_path),
                "webp": str(webp_path),
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
            "frame_paths": [str(record["path"]) for record in frame_records],
            "manifest_path": str(manifest_path),
            "saved_files": [
                str(atlas_path),
                str(gif_path),
                str(webp_path),
                str(manifest_path),
                *[str(record["path"]) for record in frame_records],
            ],
            "qa": qa,
        }


def _positive_int(value: object, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


def _nonnegative_int(value: object, name: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _parse_key_color(value: str | Iterable[int]) -> tuple[int, int, int]:
    if isinstance(value, str):
        normalized = value.strip().lower()
        named = {
            "magenta": (255, 0, 255),
            "cyan": (0, 229, 255),
            "green": (24, 224, 111),
            "blue": (47, 107, 255),
            "yellow": (255, 229, 31),
            "violet": (122, 56, 255),
            "orange": (255, 155, 47),
        }
        if normalized in named:
            return named[normalized]
        hex_value = normalized.removeprefix("#")
        if len(hex_value) == 6 and "," not in normalized:
            try:
                return tuple(int(hex_value[index : index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]
            except ValueError as exc:
                raise ValueError("key_color hex value must contain six hexadecimal digits") from exc
        parts = [part.strip() for part in normalized.split(",")]
        if len(parts) == 3:
            try:
                value = [int(part) for part in parts]
            except ValueError as exc:
                raise ValueError("key_color RGB channels must be integers") from exc
        else:
            raise ValueError(
                "key_color must be a supported named color, #RRGGBB, or an RGB sequence"
            )
    values = tuple(int(channel) for channel in value)
    if len(values) != 3 or any(channel < 0 or channel > 255 for channel in values):
        raise ValueError("key_color must contain exactly three channels in the range 0..255")
    return values


def _chroma_key(image: Image.Image, key_color: tuple[int, int, int], threshold: int) -> Image.Image:
    pixels = []
    kr, kg, kb = key_color
    threshold_sq = threshold * threshold
    for red, green, blue, _alpha in image.getdata():
        distance_sq = (red - kr) ** 2 + (green - kg) ** 2 + (blue - kb) ** 2
        pixels.append((red, green, blue, 0 if distance_sq <= threshold_sq else 255))
    image.putdata(pixels)
    return _remove_border_background(image, threshold=max(36, min(96, threshold)))


def _remove_border_background(image: Image.Image, *, threshold: int) -> Image.Image:
    """Remove flat or gently noisy backgrounds that H3 shifts away from the key color."""

    width, height = image.size
    pixels = list(image.getdata())
    visited = bytearray(width * height)
    opaque = [pixel[3] > 0 for pixel in pixels]
    tolerance_sq = threshold * threshold

    def color_distance_sq(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> int:
        return sum((left[channel] - right[channel]) ** 2 for channel in range(3))

    edge_indices = list(range(width))
    edge_indices.extend((height - 1) * width + column for column in range(width))
    edge_indices.extend(row * width for row in range(1, height - 1))
    edge_indices.extend(row * width + width - 1 for row in range(1, height - 1))

    for seed in edge_indices:
        if not opaque[seed] or visited[seed]:
            continue
        seed_color = pixels[seed]
        queue = [seed]
        visited[seed] = 1
        component: list[int] = []
        while queue:
            current = queue.pop()
            component.append(current)
            row, column = divmod(current, width)
            for neighbor in (
                current - 1 if column else -1,
                current + 1 if column + 1 < width else -1,
                current - width if row else -1,
                current + width if row + 1 < height else -1,
            ):
                if neighbor < 0 or visited[neighbor] or not opaque[neighbor]:
                    continue
                if color_distance_sq(pixels[neighbor], seed_color) > tolerance_sq:
                    continue
                visited[neighbor] = 1
                queue.append(neighbor)
        if len(component) >= 16:
            for index in component:
                red, green, blue, _alpha = pixels[index]
                pixels[index] = (red, green, blue, 0)

    image.putdata(pixels)
    return image


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
    resized = content.resize(
        (max(1, round(content.width * scale)), max(1, round(content.height * scale))),
        Image.Resampling.LANCZOS,
    )
    result = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    result.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2), resized)
    return _binary_alpha(result)


def _union_bbox(images: list[Image.Image]) -> tuple[int, int, int, int] | None:
    boxes = [_alpha_bbox(image) for image in images]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes)
    bottom = max(box[3] for box in boxes)
    return left, top, right, bottom


def _alpha_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    return image.getchannel("A").getbbox()


def _binary_alpha(image: Image.Image) -> Image.Image:
    alpha = image.getchannel("A").point(lambda value: 255 if value >= 128 else 0)
    image.putalpha(alpha)
    return image


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
