from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from agentic.runtime.contracts import SkillContext, SkillResult
from agentic.runtime.registry import SkillRegistry, ToolRegistry


class ComfyImageSkills:
    def __init__(self, tools: ToolRegistry, output_root: Path) -> None:
        self.tools = tools
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)

    def expand_idea(self, context: SkillContext) -> SkillResult:
        prompt = context.node.inputs["prompt"]
        style = context.node.inputs["style"]
        creative_prompt = f"{prompt}, {style}, highly detailed, cinematic lighting"
        negative_prompt = "ugly, blurry, low quality, bad anatomy, deformed, duplicate, watermark, text"
        return SkillResult(
            status="success",
            outputs={
                "prompt": creative_prompt,
                "negative_prompt": negative_prompt,
            },
            logs=["Prepared the final ComfyUI prompt pair."],
        )

    def ensure_workflow(self, context: SkillContext) -> SkillResult:
        result = self.tools.call(
            "asset.ensure_workflow_ready",
            {
                "workflow_name": context.node.inputs["workflow_name"],
                "auto_download": context.node.inputs.get("auto_download", False),
            },
        )
        return SkillResult(status="success", outputs=result, logs=["Checked ComfyUI workflow manifest."])

    def render_image(self, context: SkillContext) -> SkillResult:
        prompt_bundle = self._resolve_prompt_bundle(context)
        run_dir = self._build_run_dir(context.plan.goal.prompt)
        result = self.tools.call(
            "comfy.render_image",
            {
                "workflow_name": context.node.inputs["workflow_name"],
                "prompt": prompt_bundle["prompt"],
                "negative_prompt": prompt_bundle["negative_prompt"],
                "width": context.node.inputs.get("width", 1024),
                "height": context.node.inputs.get("height", 1024),
                "image_count": int(context.node.inputs.get("image_count", 1)),
                "run_dir": str(run_dir),
            },
        )
        return SkillResult(
            status="success",
            outputs=result,
            metrics={"image_count": len(result["saved_files"])},
            logs=["Rendered a real image through ComfyUI."],
        )

    @staticmethod
    def _resolve_prompt_bundle(context: SkillContext) -> dict[str, str]:
        for dependency in reversed(context.node.depends_on):
            dependency_output = context.state[dependency]
            prompt = dependency_output.get("prompt")
            if isinstance(prompt, str) and prompt:
                return {
                    "prompt": prompt,
                    "negative_prompt": str(dependency_output.get("negative_prompt", "")),
                }
        return {"prompt": context.plan.goal.prompt, "negative_prompt": ""}

    def _build_run_dir(self, prompt: str) -> Path:
        slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")
        slug = slug[:40] or "comfy-image"
        return self.output_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slug}"


def register_comfy_image_skills(skill_registry: SkillRegistry, tool_registry: ToolRegistry, output_root: Path) -> None:
    skills = ComfyImageSkills(tool_registry, output_root)
    skill_registry.register("image.idea.expand", skills.expand_idea, "Prepare a final prompt for ComfyUI")
    skill_registry.register("image.ensure_workflow", skills.ensure_workflow, "Validate ComfyUI image workflow")
    skill_registry.register("image.render", skills.render_image, "Render a real image with ComfyUI")
