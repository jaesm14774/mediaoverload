"""Motion-comic execution; shared LLM transport and media compositor stay separate."""

from __future__ import annotations

import json
from pathlib import Path
import secrets

from PIL import Image

from agentic.runtime.contracts import SkillContext, SkillResult
from agentic.runtime.llm_engine import LLMPromptEngine
from agentic.runtime.motion_comics import validate_motion_comic_story
from agentic.runtime.prompting import selected_role_description
from agentic.runtime.registry import SkillRegistry, ToolRegistry
from agentic.skills.shared import build_run_dir
from agentic.tools.motion_comic_adapter import compose_pages, load_recipe, render_pages


class MotionComicSkills:
    def __init__(
        self, tools: ToolRegistry, output_root: Path, prompt_engine: LLMPromptEngine
    ):
        self.tools, self.output_root, self.prompt_engine = (
            tools,
            output_root,
            prompt_engine,
        )

    def write(self, context: SkillContext) -> SkillResult:
        story = self.prompt_engine.build_motion_comic(context.plan.goal)
        directory = build_run_dir(
            self.output_root,
            context.plan.goal.prompt,
            f"motion_comic_{secrets.token_hex(4)}",
        )
        directory.mkdir(parents=True, exist_ok=False)
        path = directory / "story_draft.json"
        path.write_text(
            json.dumps(story, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return SkillResult(
            status="success",
            outputs={
                **story,
                "run_dir": str(directory),
                "story_path": str(path),
                "saved_files": [str(path)],
            },
            logs=[
                "Generated and saved a complete three-beat story with visible-state and cause evidence."
            ],
        )

    def panels(self, context: SkillContext) -> SkillResult:
        written = context.state["motion-comic-write"]
        story = validate_motion_comic_story(written)
        config = context.node.inputs["config"]
        directory = Path(written["run_dir"]) / f"sources_{secrets.token_hex(3)}"
        directory.mkdir()
        seed = context.plan.goal.constraints.get("seed")
        seed = int(seed) if seed is not None else secrets.randbelow(2**31)
        records, paths = [], []
        manifest_path = directory / "panel_generation.json"
        for index, panel in enumerate(story["panels"]):
            payload = {
                "workflow_name": context.node.inputs["workflow_name"],
                "prompt": "\n".join(
                    (
                        panel["image_prompt"],
                        "Visible evidence: " + panel["visible_state"],
                        story["visual_style"],
                        "Resolved subject identity: "
                        + selected_role_description(context.plan.goal),
                        "Show only this panel's current prop state. No text, speech bubbles, borders or watermark.",
                    )
                ),
                "width": int(config["image_width"]),
                "height": int(config["image_height"]),
                "steps": int(config["steps"]),
                "seed": seed + index,
                "image_count": 1,
                "run_dir": str(directory / f"panel-{index + 1}"),
            }
            result = self.tools.call("comfy.workflow.text_to_image", payload)
            images = [
                Path(path)
                for path in result.get("saved_files", [])
                if Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            ]
            if len(images) != 1:
                raise RuntimeError(
                    f"Motion comic beat {index + 1} requires one rendered image"
                )
            with Image.open(images[0]) as image:
                image.verify()
            paths.append(str(images[0].resolve()))
            records.append(
                {
                    "panel": index + 1,
                    "role": panel["role"],
                    "payload": payload,
                    "result": result,
                }
            )
            manifest_path.write_text(
                json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return SkillResult(
            status="success",
            outputs={
                "run_dir": str(directory),
                "image_paths": paths,
                "saved_files": paths,
                "generation_path": str(manifest_path),
                "generation": records,
            },
            metrics={"image_count": len(paths)},
            logs=[
                "Generated one image per causal story beat; story meaning remains subject to visual review."
            ],
        )

    def compose(self, context: SkillContext) -> SkillResult:
        written, generated = (
            context.state["motion-comic-write"],
            context.state["motion-comic-panels"],
        )
        story = validate_motion_comic_story(written)
        config = context.node.inputs["config"]
        fps = float(config["fps"])
        total_frames = round(context.plan.goal.duration_seconds * fps)
        if (
            total_frames < 3
            or abs(total_frames - context.plan.goal.duration_seconds * fps) > 1e-5
        ):
            raise ValueError(
                "Motion comic duration must provide three beats aligned to the FPS"
            )
        counts = [
            total_frames // 3 + (1 if i < total_frames % 3 else 0) for i in range(3)
        ]
        paths = generated["image_paths"]
        if len(paths) != 3:
            raise ValueError(
                "Motion comic composition requires all three rendered images"
            )
        recipe = {
            "title": story["title"],
            "width": int(config["width"]),
            "height": int(config["height"]),
            "fps": fps,
            "panels": [
                {
                    "image_path": path,
                    "caption": panel["caption"],
                    "duration_seconds": frames / fps,
                }
                for panel, path, frames in zip(story["panels"], paths, counts)
            ],
        }
        directory = Path(written["run_dir"]) / f"comic_{secrets.token_hex(3)}"
        directory.mkdir()
        recipe_path = directory / "story.json"
        recipe_path.write_text(
            json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        normalized, sources = load_recipe(recipe_path)
        prepared = render_pages(normalized, sources, "motion_comic")
        result = compose_pages(normalized, prepared, directory / "render", self.tools)
        review = {
            "status": "awaiting_review",
            "criteria": [
                "caption claims match the actual images",
                "visible causal state changes",
                "payoff answers the title",
                "character and prop continuity",
                "readability and pacing",
            ],
        }
        summary_path = directory / "motion_comic_summary.json"
        summary = {
            "goal": context.plan.goal.prompt,
            "story": story,
            "story_path": str(recipe_path),
            "generation": generated["generation"],
            "render": result,
            "perceptual_review": review,
            "prompt_mode": written["prompt_mode"],
            "llm_backend": written.get("llm_backend", {}),
            "llm_model": written.get("llm_model", ""),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return SkillResult(
            status="success",
            outputs={
                **result,
                "final_video_path": result["video_path"],
                "run_dir": str(directory),
                "story_path": str(recipe_path),
                "summary_path": str(summary_path),
                "saved_files": [
                    result["video_path"],
                    str(recipe_path),
                    str(summary_path),
                ],
                "perceptual_review": review,
            },
            logs=[
                "Composed the generated story with the accepted progressive-panel layout; final visual review is pending."
            ],
        )


def register_motion_comic_skills(
    skills: SkillRegistry,
    tools: ToolRegistry,
    output_root: Path,
    *,
    prompt_engine: LLMPromptEngine,
) -> None:
    comic = MotionComicSkills(tools, output_root, prompt_engine)
    skills.register(
        "agent.motion_comic.write",
        comic.write,
        "Generate a three-beat comic story with an LLM",
    )
    skills.register(
        "media.motion_comic.panels",
        comic.panels,
        "Generate illustrations from per-beat visible-state evidence",
    )
    skills.register(
        "media.motion_comic.compose",
        comic.compose,
        "Render a progressive-panel comic video",
    )
