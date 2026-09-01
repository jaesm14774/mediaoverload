from __future__ import annotations

from pathlib import Path

from agentic.runtime.contracts import SkillContext, SkillResult
from agentic.runtime.registry import SkillRegistry, ToolRegistry
from agentic.skills.shared import (
    build_run_dir,
    resolve_dependency_negative_prompt,
    resolve_dependency_prompt,
    resolve_dependency_value,
)


class ComfyWorkflowSkills:
    def __init__(self, tools: ToolRegistry, output_root: Path) -> None:
        self.tools = tools
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)

    def refine_image(self, context: SkillContext) -> SkillResult:
        image_path = context.node.inputs.get("image_path") or self._resolve_first_file(context)
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
        image_path = context.node.inputs.get("image_path") or self._resolve_first_file(context)
        try:
            result = self.tools.call(
                "comfy.workflow.image_upscale",
                {
                    "run_dir": str(self._build_run_dir(context.plan.goal.prompt, "upscale")),
                    "image_path": image_path,
                },
            )
            return SkillResult(status="success", outputs=result, logs=["Upscaled image with ComfyUI workflow."])
        except RuntimeError as exc:
            # Upscaling is an enhancement stage for text2img2video. A missing
            # optional custom node must not prevent the already-rendered source
            # image from reaching the video stage; preserve the failure detail.
            return SkillResult(
                status="success",
                outputs={
                    "saved_files": [str(image_path)],
                    "image_path": str(image_path),
                    "upscale_fallback": True,
                    "upscale_error": str(exc),
                },
                logs=[f"Optional upscale unavailable; continuing with source image: {exc}"],
            )

    def image_to_video(self, context: SkillContext) -> SkillResult:
        image_path = context.node.inputs.get("image_path") or self._resolve_first_file(context)
        prompt = context.node.inputs.get("prompt") or self._resolve_prompt(context)
        payload: dict[str, object] = {
            "run_dir": str(self._build_run_dir(context.plan.goal.prompt, "i2v")),
            "workflow_name": str(context.node.inputs.get("workflow_name", "")),
            "image_path": image_path,
            "prompt": prompt,
            "width": context.node.inputs.get("width"),
            "height": context.node.inputs.get("height"),
            "model_profile": str(
                context.node.inputs.get("model_profile")
                or context.plan.goal.constraints.get("native_h3_model_profile")
                or "q4"
            ),
        }
        seed = context.node.inputs.get("seed", context.plan.goal.constraints.get("seed"))
        if seed is not None:
            payload["seed"] = int(seed)
        result = self.tools.call("comfy.workflow.image_to_video", payload)
        return SkillResult(status="success", outputs=result, logs=["Rendered video from still image with ComfyUI."])

    def _build_run_dir(self, prompt: str, suffix: str) -> Path:
        return build_run_dir(self.output_root, prompt, suffix, default_slug="workflow", suffix_first=True)

    _FILE_KEYS: tuple[str, ...] = ("saved_files", "selected_assets", "media_paths", "image_path")

    @classmethod
    def _resolve_first_file(cls, context: SkillContext, candidate_keys: tuple[str, ...] | None = None) -> str:
        keys = candidate_keys if candidate_keys is not None else cls._FILE_KEYS
        value = resolve_dependency_value(context, keys)
        if value is None:
            raise RuntimeError(f"No dependency output found for node '{context.node.node_id}'")
        return value

    @staticmethod
    def _resolve_prompt(context: SkillContext) -> str:
        return resolve_dependency_prompt(context)

    @staticmethod
    def _resolve_negative_prompt(context: SkillContext) -> str:
        return resolve_dependency_negative_prompt(context)


def register_comfy_workflow_skills(skill_registry: SkillRegistry, tool_registry: ToolRegistry, output_root: Path) -> None:
    skills = ComfyWorkflowSkills(tool_registry, output_root)
    skill_registry.register("image.refine", skills.refine_image, "Refine an image with a ComfyUI img2img workflow")
    skill_registry.register("image.upscale", skills.upscale_image, "Upscale an image with a ComfyUI workflow")
    skill_registry.register("image.animate", skills.image_to_video, "Animate an image into a video with a ComfyUI workflow")
