"""Shared progressive-panel layout and deterministic FFmpeg composition."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps

from agentic.runtime.editing import EditClip, EditPlan, EditTransition
from agentic.runtime.registry import ToolRegistry
from agentic.runtime.story_cards import load_story_card_font, resolve_story_card_font


def wrap_caption(text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines, line = [], ""
    for unit in re.findall(r"[A-Za-z0-9]+|\s+|.", text, flags=re.DOTALL):
        if unit.isspace():
            if line and font.getlength(line + unit) <= width:
                line += unit
            continue
        pieces = (
            list(unit)
            if re.fullmatch(r"[A-Za-z0-9]+", unit) and font.getlength(unit) > width
            else [unit]
        )
        for piece in pieces:
            if line and font.getlength(line + piece) > width:
                lines.append(line.rstrip())
                line = ""
            if not line and piece.isspace():
                continue
            line += piece
    if line:
        lines.append(line.rstrip())
    if not lines:
        lines.append("")
    return lines


def layout_caption(
    text: str,
    font_path: Path,
    preferred_font_size: int,
    panel_width: int,
    panel_height: int,
    padding: int,
) -> tuple[list[str], ImageFont.FreeTypeFont, int, int]:
    """Keep the largest readable font that fits beside a minimum illustration."""
    text_width = max(1, panel_width - 2 * padding)
    minimum_font_size = max(8, preferred_font_size // 2)
    minimum_art_height = max(2 * padding, round(panel_height * 0.28))
    maximum_caption_height = panel_height - minimum_art_height - 2 * padding
    for size in range(preferred_font_size, minimum_font_size - 1, -1):
        font = load_story_card_font(font_path, size)
        line_height = max(1, round(font.size * 1.35))
        lines = wrap_caption(text, font, text_width)
        caption_height = line_height * len(lines) + 2 * padding
        if caption_height <= maximum_caption_height:
            return lines, font, line_height, caption_height
    raise ValueError(
        "Caption cannot fit inside the panel while preserving its illustration area"
    )


def load_recipe(path: Path) -> tuple[dict, list[Path]]:
    story = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(story, dict)
        or not isinstance(story.get("panels"), list)
        or len(story["panels"]) != 3
    ):
        raise ValueError(
            "This experiment requires exactly three illustrated story beats"
        )
    title = story.get("title")
    if not isinstance(title, str) or not 1 <= len(title.strip()) <= 12:
        raise ValueError("Title must contain 1 to 12 characters")
    width, height, fps = (
        int(story.get("width", 576)),
        int(story.get("height", 1024)),
        float(story.get("fps", 24)),
    )
    if not 240 <= width <= 1080 or not width * 1.5 <= height <= min(1920, width * 2):
        raise ValueError(
            "This trial requires a portrait canvas between 240 and 1080 pixels wide"
        )
    paths = []
    for panel in story["panels"]:
        if (
            not isinstance(panel, dict)
            or not isinstance(panel.get("caption"), str)
            or not 1 <= len(panel["caption"].strip()) <= 24
        ):
            raise ValueError("Each caption must contain 1 to 24 characters")
        duration = float(panel["duration_seconds"])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("Panel duration must be positive and finite")
        source = Path(panel["image_path"]).expanduser()
        source = (
            (path.parent / source).resolve(strict=True)
            if not source.is_absolute()
            else source.resolve(strict=True)
        )
        with Image.open(source) as image:
            image.verify()
        paths.append(source)
    durations = [float(panel["duration_seconds"]) for panel in story["panels"]]
    EditPlan(
        clips=tuple(
            EditClip(str(source), duration_seconds=duration)
            for source, duration in zip(paths, durations)
        ),
        output_width=width,
        output_height=height,
        fps=fps,
        profile="motion_cut_v1",
        target_duration_seconds=sum(durations),
    ).validate()
    if any(
        abs(duration * fps - round(duration * fps)) > 1e-5 for duration in durations
    ):
        raise ValueError(
            "Panel reading durations must align with the requested frame rate"
        )
    story = {**story, "width": width, "height": height, "fps": fps}
    return story, paths


def render_pages(story: dict, sources: list[Path], variant: str) -> list[Image.Image]:
    width, height = story["width"], story["height"]
    margin, gap = round(width * 0.08), round(width * 0.025)
    top, bottom, split = (
        round(height * 0.145),
        round(height * 0.79),
        round(height * 0.43),
    )
    font_path = resolve_story_card_font()
    preferred_caption_size = max(12, round(width * 0.047))
    title_font = load_story_card_font(font_path, round(width * 0.076), weight="Bold")
    while title_font.getlength(story["title"]) > width - 2 * margin:
        title_font = load_story_card_font(font_path, title_font.size - 1, weight="Bold")
    illustrations = []
    for source in sources:
        with Image.open(source) as image:
            illustrations.append(ImageOps.exif_transpose(image).convert("RGB"))
    result = []
    for beat in range(3):
        page = Image.new("RGB", (width, height), "#F8F1E7")
        draw = ImageDraw.Draw(page)
        draw.text(
            (margin, round(height * 0.064)),
            story["title"],
            font=title_font,
            fill="#39252F",
        )
        if variant == "slideshow" or beat == 0:
            placements = [(beat, (margin, top, width - margin, bottom))]
        elif beat == 1:
            placements = [
                (0, (margin, top, width - margin, split)),
                (1, (margin, split + gap, width - margin, bottom)),
            ]
        else:
            middle = width // 2
            placements = [
                (0, (margin, top, middle - gap // 2, split)),
                (1, (middle + gap // 2, top, width - margin, split)),
                (2, (margin, split + gap, width - margin, bottom)),
            ]
        for index, box in placements:
            left, upper, right, lower = box
            padding = max(5, round(width * 0.017))
            lines, caption_font, line_height, caption_height = layout_caption(
                story["panels"][index]["caption"],
                font_path,
                preferred_caption_size,
                right - left,
                lower - upper,
                padding,
            )
            art_box = (
                left + padding,
                upper + padding,
                right - padding,
                lower - caption_height - padding,
            )
            art = ImageOps.contain(
                illustrations[index],
                (art_box[2] - art_box[0], art_box[3] - art_box[1]),
                Image.Resampling.LANCZOS,
            )
            draw.rectangle(box, fill="#FFF9F0")
            page.paste(
                art,
                (
                    art_box[0] + (art_box[2] - art_box[0] - art.width) // 2,
                    art_box[1] + (art_box[3] - art_box[1] - art.height) // 2,
                ),
            )
            draw.rectangle(
                box,
                outline="#39252F",
                width=max(1, round(width * (0.007 if index == beat else 0.0035))),
            )
            y = lower - caption_height + padding
            for line in lines:
                draw.text(
                    ((left + right) / 2, y),
                    line,
                    font=caption_font,
                    anchor="mt",
                    fill="#39252F",
                )
                y += line_height
        result.append(page)
    return result


def compose_pages(
    story: dict, prepared: list[Image.Image], directory: Path, tools: ToolRegistry
) -> dict:
    directory.mkdir(parents=True)
    page_paths = []
    for index, page in enumerate(prepared):
        path = directory / f"page-{index + 1}.png"
        page.save(path)
        page_paths.append(path)
    fade = 2 / story["fps"]
    plan = EditPlan(
        clips=tuple(
            EditClip(
                str(path),
                duration_seconds=float(panel["duration_seconds"])
                + (fade if index < 2 else 0),
            )
            for index, (path, panel) in enumerate(zip(page_paths, story["panels"]))
        ),
        transitions=(EditTransition("fade", fade), EditTransition("fade", fade)),
        output_width=story["width"],
        output_height=story["height"],
        fps=story["fps"],
        profile="xfade_clean_v1",
        target_duration_seconds=sum(
            float(panel["duration_seconds"]) for panel in story["panels"]
        ),
    )
    result = tools.call(
        "media.compose_edit",
        {
            "edit_plan": plan.to_dict(),
            "output_path": str(directory / "video.mp4"),
            "contact_sheet_path": str(directory / "contact-sheet.jpg"),
        },
    )
    return result
