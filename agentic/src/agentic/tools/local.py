from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from agentic.runtime.registry import ToolRegistry


class LocalArtifactTools:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)

    def render_storyboard_frames(self, payload: dict[str, object]) -> dict[str, object]:
        prompt = str(payload["prompt"])
        segments = list(payload["segments"])
        style = str(payload.get("style", "cinematic surreal"))
        frame_width = int(payload.get("frame_width", 1280))
        frame_height = int(payload.get("frame_height", 720))
        run_dir = self._resolve_run_dir(prompt)
        frames_dir = run_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        frame_paths: list[str] = []
        for index, segment in enumerate(segments, start=1):
            frame_path = frames_dir / f"frame_{index:02d}.png"
            self._draw_frame(
                frame_path=frame_path,
                title=f"Scene {index}",
                prompt=str(segment.get("visual", "")),
                narration=str(segment.get("narration", "")),
                style=style,
                width=frame_width,
                height=frame_height,
            )
            frame_paths.append(str(frame_path))

        return {
            "run_dir": str(run_dir),
            "frames_dir": str(frames_dir),
            "frame_paths": frame_paths,
            "frame_count": len(frame_paths),
        }

    def package_storyboard(self, payload: dict[str, object]) -> dict[str, object]:
        run_dir = Path(str(payload["run_dir"]))
        frame_paths = [Path(str(path)) for path in payload.get("frame_paths", [])]
        summary = {
            "goal": payload["goal"],
            "style": payload["style"],
            "workflow_name": payload["workflow_name"],
            "frame_count": len(frame_paths),
            "frames": [str(path) for path in frame_paths],
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

        summary_path = run_dir / "storyboard_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        contact_sheet_path = run_dir / "storyboard_contact_sheet.png"
        if frame_paths:
            self._create_contact_sheet(frame_paths, contact_sheet_path)

        notes_path = run_dir / "README.txt"
        notes = [
            f"Goal: {payload['goal']}",
            f"Style: {payload['style']}",
            f"Workflow: {payload['workflow_name']}",
            f"Frames: {len(frame_paths)}",
            "",
            "Generated files:",
            *[str(path) for path in frame_paths],
            str(summary_path),
            str(contact_sheet_path),
        ]
        notes_path.write_text("\n".join(notes), encoding="utf-8")

        return {
            "run_dir": str(run_dir),
            "summary_path": str(summary_path),
            "contact_sheet_path": str(contact_sheet_path),
            "notes_path": str(notes_path),
            "frame_paths": [str(path) for path in frame_paths],
        }

    def _resolve_run_dir(self, prompt: str) -> Path:
        slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")
        slug = slug[:40] or "agentic-run"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_root / f"{timestamp}_{slug}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _draw_frame(
        self,
        frame_path: Path,
        title: str,
        prompt: str,
        narration: str,
        style: str,
        width: int,
        height: int,
    ) -> None:
        image = Image.new("RGB", (width, height), color=(245, 238, 225))
        draw = ImageDraw.Draw(image)
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

        draw.rectangle((40, 40, width - 40, height - 40), outline=(36, 36, 36), width=4)
        draw.rectangle((60, 60, width - 60, 140), fill=(36, 36, 36))
        draw.text((85, 88), title, fill=(245, 238, 225), font=title_font)

        wrapped_prompt = self._wrap_text(f"Visual: {prompt}", 62)
        wrapped_narration = self._wrap_text(f"Narration: {narration}", 62)
        wrapped_style = self._wrap_text(f"Style: {style}", 62)

        y = 190
        for block in (wrapped_prompt, wrapped_narration, wrapped_style):
            draw.multiline_text((90, y), block, fill=(36, 36, 36), font=body_font, spacing=8)
            y += 160

        draw.rectangle((90, height - 150, width - 90, height - 90), fill=(210, 128, 72))
        draw.text((110, height - 128), "Agentic Storyboard Demo", fill=(255, 250, 240), font=body_font)
        image.save(frame_path)

    def _create_contact_sheet(self, frame_paths: list[Path], output_path: Path) -> None:
        images = [Image.open(path).convert("RGB") for path in frame_paths]
        thumb_width = 480
        thumb_height = 270
        cols = 2
        rows = (len(images) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb_width + 60, rows * thumb_height + 60), color=(255, 252, 246))

        for index, image in enumerate(images):
            resized = image.resize((thumb_width, thumb_height))
            x = 20 + (index % cols) * thumb_width
            y = 20 + (index // cols) * thumb_height
            sheet.paste(resized, (x, y))
            image.close()

        sheet.save(output_path)

    @staticmethod
    def _wrap_text(text: str, line_length: int) -> str:
        words = text.split()
        if not words:
            return ""
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if len(candidate) <= line_length:
                current = candidate
                continue
            lines.append(current)
            current = word
        lines.append(current)
        return "\n".join(lines)


def register_local_tools(tool_registry: ToolRegistry, output_root: Path) -> None:
    tools = LocalArtifactTools(output_root=output_root)
    tool_registry.register("local.render_storyboard_frames", tools.render_storyboard_frames, "Render storyboard PNG frames")
    tool_registry.register("local.package_storyboard", tools.package_storyboard, "Build storyboard summary files")
