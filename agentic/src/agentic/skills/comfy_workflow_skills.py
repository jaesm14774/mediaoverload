from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from agentic.runtime.contracts import SkillContext, SkillResult
from agentic.runtime.registry import SkillRegistry, ToolRegistry


class ComfyWorkflowSkills:
    def __init__(self, tools: ToolRegistry, output_root: Path) -> None:
        self.tools = tools
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)

    def refine_image(self, context: SkillContext) -> SkillResult:
        image_path = context.node.inputs.get("image_path") or self._resolve_first_file(context, ("saved_files",))
        prompt = context.node.inputs.get("prompt") or self._resolve_prompt(context)
        negative_prompt = context.node.inputs.get("negative_prompt") or self._resolve_negative_prompt(context)
        result = self.tools.call(
            "comfy.workflow.image_to_image",
            {
                "run_dir": str(self._build_run_dir(context.plan.goal.prompt, "img2img")),
                "image_path": image_path,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
            },
        )
        return SkillResult(status="success", outputs=result, logs=["Refined image with ComfyUI image-to-image workflow."])

    def upscale_image(self, context: SkillContext) -> SkillResult:
        image_path = context.node.inputs.get("image_path") or self._resolve_first_file(context, ("saved_files",))
        result = self.tools.call(
            "comfy.workflow.image_upscale",
            {
                "run_dir": str(self._build_run_dir(context.plan.goal.prompt, "upscale")),
                "image_path": image_path,
            },
        )
        return SkillResult(status="success", outputs=result, logs=["Upscaled image with ComfyUI workflow."])

    def image_to_video(self, context: SkillContext) -> SkillResult:
        image_path = context.node.inputs.get("image_path") or self._resolve_first_file(context, ("saved_files",))
        prompt = context.node.inputs.get("prompt") or self._resolve_prompt(context)
        result = self.tools.call(
            "comfy.workflow.image_to_video",
            {
                "run_dir": str(self._build_run_dir(context.plan.goal.prompt, "i2v")),
                "image_path": image_path,
                "prompt": prompt,
            },
        )
        return SkillResult(status="success", outputs=result, logs=["Rendered video from still image with ComfyUI."])

    def _build_run_dir(self, prompt: str, suffix: str) -> Path:
        slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")
        slug = slug[:32] or "workflow"
        return self.output_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}_{slug}"

    @staticmethod
    def _resolve_first_file(context: SkillContext, candidate_keys: tuple[str, ...]) -> str:
        for dependency in reversed(context.node.depends_on):
            dependency_output = context.state[dependency]
            for key in candidate_keys:
                value = dependency_output.get(key)
                if isinstance(value, list) and value:
                    return str(value[0])
                if isinstance(value, str):
                    return value
        raise RuntimeError(f"No dependency output found for node '{context.node.node_id}'")

    @staticmethod
    def _resolve_prompt(context: SkillContext) -> str:
        for dependency in reversed(context.node.depends_on):
            dependency_output = context.state[dependency]
            prompt = dependency_output.get("prompt")
            if isinstance(prompt, str) and prompt:
                return prompt
        return context.plan.goal.prompt

    @staticmethod
    def _resolve_negative_prompt(context: SkillContext) -> str:
        for dependency in reversed(context.node.depends_on):
            dependency_output = context.state[dependency]
            negative_prompt = dependency_output.get("negative_prompt")
            if isinstance(negative_prompt, str):
                return negative_prompt
        return ""


def register_comfy_workflow_skills(skill_registry: SkillRegistry, tool_registry: ToolRegistry, output_root: Path) -> None:
    skills = ComfyWorkflowSkills(tool_registry, output_root)
    skill_registry.register("image.refine", skills.refine_image, "Refine an image with a ComfyUI img2img workflow")
    skill_registry.register("image.upscale", skills.upscale_image, "Upscale an image with a ComfyUI workflow")
    skill_registry.register("image.animate", skills.image_to_video, "Animate an image into a video with a ComfyUI workflow")
