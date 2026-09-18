"""Package an Image 2 birthday sprite atlas into a validated 8-second GIF."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
AGENTIC_SRC = ROOT / "agentic" / "src"
if str(AGENTIC_SRC) not in sys.path:
    sys.path.insert(0, str(AGENTIC_SRC))

from agentic.tools.sprite_adapter import SpriteAdapter  # noqa: E402


GRID_SIZE = 4
FRAME_COUNT = GRID_SIZE * GRID_SIZE
CELL_SIZE = 128
FPS = 2
FRAME_DURATION_MS = 1000 // FPS
GREETING = "開心生日快樂～～"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-atlas", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path(r"C:\Windows\Fonts\msjhbd.ttc"),
        Path(r"C:\Windows\Fonts\msjh.ttc"),
        Path(r"C:\Windows\Fonts\NotoSansCJKtc-Regular.otf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _normalize_source(source_path: Path, output_dir: Path) -> Image.Image:
    with Image.open(source_path) as source:
        image = source.convert("RGB")
    width = image.width - (image.width % GRID_SIZE)
    height = image.height - (image.height % GRID_SIZE)
    if width <= 0 or height <= 0:
        raise ValueError(f"source atlas is too small: {image.size}")
    image = image.crop((0, 0, width, height))
    normalized = image.resize((CELL_SIZE * GRID_SIZE, CELL_SIZE * GRID_SIZE), Image.Resampling.LANCZOS)
    normalized.save(output_dir / "source_atlas_normalized.png", format="PNG", optimize=True)
    return normalized


def _write_source_frames(normalized: Image.Image, output_dir: Path) -> list[str]:
    source_frames_dir = output_dir / "source_frames"
    source_frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[str] = []
    for index in range(FRAME_COUNT):
        row, column = divmod(index, GRID_SIZE)
        frame = normalized.crop(
            (
                column * CELL_SIZE,
                row * CELL_SIZE,
                (column + 1) * CELL_SIZE,
                (row + 1) * CELL_SIZE,
            )
        )
        path = source_frames_dir / f"source-{index + 1:02d}.png"
        frame.save(path, format="PNG", optimize=True)
        frame_paths.append(str(path))
    return frame_paths


def _draw_background(size: int, frame_index: int) -> Image.Image:
    image = Image.new("RGBA", (size, size), (255, 244, 232, 255))
    pixels = image.load()
    for y in range(size):
        for x in range(size):
            blend = (x + y) / (2 * size)
            pixels[x, y] = (
                int(255 - 12 * blend),
                int(244 + 8 * blend),
                int(232 + 20 * blend),
                255,
            )
    draw = ImageDraw.Draw(image)
    palette = ((255, 176, 196), (255, 215, 92), (125, 211, 252), (174, 226, 152))
    phase = frame_index % len(palette)
    for index in range(12):
        x = 24 + ((index * 83 + frame_index * 5) % (size - 48))
        y = 20 + ((index * 47 + frame_index * 3) % (size - 40))
        radius = 4 + (index % 3)
        color = palette[(index + phase) % len(palette)] + (220,)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    draw.rounded_rectangle((28, 24, size - 28, 118), radius=24, fill=(255, 255, 255, 226), outline=(255, 190, 128, 255), width=4)
    return image


def _compose_showcase_frames(frame_paths: list[str], output_dir: Path) -> list[Image.Image]:
    greeting_font = _load_font(42)
    subtitle_font = _load_font(18)
    frames: list[Image.Image] = []
    for index, frame_path in enumerate(frame_paths):
        canvas = _draw_background(512, index)
        draw = ImageDraw.Draw(canvas)
        greeting_box = draw.textbbox((0, 0), GREETING, font=greeting_font, stroke_width=1)
        greeting_width = greeting_box[2] - greeting_box[0]
        greeting_x = (canvas.width - greeting_width) // 2
        draw.text(
            (greeting_x, 38),
            GREETING,
            font=greeting_font,
            fill=(232, 91, 108, 255),
            stroke_width=2,
            stroke_fill=(255, 255, 255, 255),
        )
        subtitle = "Happy Birthday!"
        subtitle_box = draw.textbbox((0, 0), subtitle, font=subtitle_font)
        draw.text(
            ((canvas.width - (subtitle_box[2] - subtitle_box[0])) // 2, 91),
            subtitle,
            font=subtitle_font,
            fill=(116, 91, 122, 255),
        )
        with Image.open(frame_path) as source:
            character = source.convert("RGBA").resize((300, 300), Image.Resampling.LANCZOS)
        canvas.alpha_composite(character, ((canvas.width - character.width) // 2, 158))
        frames.append(canvas.convert("RGB"))
    return frames


def main() -> None:
    args = parse_args()
    source_atlas = args.source_atlas.resolve()
    output_dir = args.output_dir.resolve()
    if not source_atlas.is_file():
        raise FileNotFoundError(source_atlas)
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized = _normalize_source(source_atlas, output_dir)
    source_frames = _write_source_frames(normalized, output_dir)
    sprite_result = SpriteAdapter().build_from_frames(
        source_frames,
        output_dir=str(output_dir),
        cell_width=CELL_SIZE,
        cell_height=CELL_SIZE,
        fps=FPS,
        loop=True,
        key_color="magenta",
        chroma_threshold=52,
        frame_map=[{"beat": index / FPS, "description": "joyful birthday celebration"} for index in range(FRAME_COUNT)],
    )

    sprite_frame_paths = sprite_result["frame_paths"]
    showcase_frames = _compose_showcase_frames(sprite_frame_paths, output_dir)
    showcase_path = output_dir / "birthday_greeting_8s.gif"
    showcase_frames[0].save(
        showcase_path,
        format="GIF",
        save_all=True,
        append_images=showcase_frames[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        disposal=2,
    )

    high_res_showcase_frames = [
        frame.resize((1024, 1024), Image.Resampling.LANCZOS) for frame in showcase_frames
    ]
    high_res_showcase_path = output_dir / "birthday_greeting_8s_1024.gif"
    high_res_showcase_frames[0].save(
        high_res_showcase_path,
        format="GIF",
        save_all=True,
        append_images=high_res_showcase_frames[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        disposal=2,
    )

    high_res_sprite_result = SpriteAdapter().build_from_frames(
        source_frames,
        output_dir=str(output_dir / "high_res_sprite"),
        cell_width=1024,
        cell_height=1024,
        fps=FPS,
        loop=True,
        key_color="magenta",
        chroma_threshold=52,
        frame_map=[{"beat": index / FPS, "description": "joyful birthday celebration"} for index in range(FRAME_COUNT)],
    )

    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["duration_seconds"] = FRAME_COUNT / FPS
    manifest["fps"] = FPS
    manifest["prompt_lineage"] = {
        "generator": "built-in Image 2",
        "reference_image": "codex-clipboard-df804e06-fd05-4195-ba4c-968ef601de54.png",
        "request": "chibi young man celebrating a happy birthday in a 4x4 game sprite atlas",
    }
    manifest["showcase"] = {
        "gif": str(showcase_path),
        "high_res_gif": str(high_res_showcase_path),
        "greeting": GREETING,
        "duration_seconds": FRAME_COUNT / FPS,
        "frame_duration_ms": FRAME_DURATION_MS,
        "loop": True,
        "upscale": "Lanczos from 512x512 showcase frames",
    }
    manifest["high_res_sprite"] = high_res_sprite_result
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "sprite": sprite_result,
                "showcase_gif": str(showcase_path),
                "high_res_showcase_gif": str(high_res_showcase_path),
                "high_res_sprite_gif": high_res_sprite_result["gif_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
