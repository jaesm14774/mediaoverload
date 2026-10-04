from __future__ import annotations

from pathlib import Path

from agentic.runtime.contracts import SkillContext, SkillResult
from agentic.runtime.illustration_style import apply_paper_storybook_art_direction
from agentic.runtime.prompting import include_role_description
from agentic.runtime.registry import SkillRegistry, ToolRegistry
from agentic.skills.shared import asset_check_result, build_run_dir


class ComfyImageSkills:
    def __init__(self, tools: ToolRegistry, output_root: Path) -> None:
        self.tools = tools
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)

    def expand_idea(self, context: SkillContext) -> SkillResult:
        prompt = context.node.inputs["prompt"]
        style = context.node.inputs["style"]
        creative_prompt = "\n".join(
            part for part in (str(prompt).strip(), f"Style: {str(style).strip()}" if str(style).strip() else "") if part
        )
        negative_prompt = ""
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
        return asset_check_result(result, "Checked ComfyUI workflow.")

    def render_image(self, context: SkillContext) -> SkillResult:
        prompt_bundle = self._resolve_prompt_bundle(context)
        run_dir = self._build_run_dir(context.plan.goal.prompt)
        payload = {
            "workflow_name": context.node.inputs["workflow_name"],
            "prompt": prompt_bundle["prompt"],
            "negative_prompt": prompt_bundle["negative_prompt"],
            "width": context.node.inputs.get("width", 1024),
            "height": context.node.inputs.get("height", 1024),
            "image_count": int(context.node.inputs.get("image_count", 1)),
            "run_dir": str(run_dir),
        }
        seed = context.node.inputs.get("seed", context.plan.goal.constraints.get("seed"))
        if seed is not None:
            payload["seed"] = int(seed)
        result = self.tools.call("comfy.render_image", payload)
        return SkillResult(
            status="success",
            outputs=result,
            metrics={"image_count": len(result["saved_files"])},
            logs=["Rendered a real image through ComfyUI."],
        )

    @staticmethod
    def _resolve_prompt_bundle(context: SkillContext) -> dict[str, str]:
        media_type = str(context.plan.goal.media_type or "").strip().lower()
        renders_native_h3_keyframe = media_type in {
            "native_h3_story",
            "native_h3_fl2va_story",
            "native_h3_l2va_story",
        }
        uses_opening_frame = media_type in {
            "text2img2video",
            "image_to_video",
            "image_to_video_audio",
            "native_h3_story",
            "native_h3_fl2va_story",
            "native_h3_l2va_story",
        }
        for dependency in reversed(context.node.depends_on):
            dependency_output = context.state[dependency]
            prompt_key = "opening_keyframe_prompt" if uses_opening_frame else "prompt"
            prompt = dependency_output.get(prompt_key)
            if not isinstance(prompt, str) or not prompt:
                prompt = dependency_output.get("prompt")
            if isinstance(prompt, str) and prompt:
                if not renders_native_h3_keyframe:
                    prompt = include_role_description(prompt, context.plan.goal)
                prompt = apply_paper_storybook_art_direction(
                    prompt,
                    isolated_subject=media_type in {"sticker_pack", "animated_sticker", "game_sprite"},
                )
                return {
                    "prompt": prompt,
                    "negative_prompt": str(dependency_output.get("negative_prompt", "")),
                }
        prompt = str(context.plan.goal.prompt or "")
        if not renders_native_h3_keyframe:
            prompt = include_role_description(prompt, context.plan.goal)
        prompt = apply_paper_storybook_art_direction(
            prompt,
            isolated_subject=media_type in {"sticker_pack", "animated_sticker", "game_sprite"},
        )
        return {"prompt": prompt, "negative_prompt": ""}

    def _build_run_dir(self, prompt: str) -> Path:
        return build_run_dir(self.output_root, prompt, default_slug="comfy-image", max_slug_length=40)


def register_comfy_image_skills(skill_registry: SkillRegistry, tool_registry: ToolRegistry, output_root: Path) -> None:
    skills = ComfyImageSkills(tool_registry, output_root)
    skill_registry.register("image.idea.expand", skills.expand_idea, "Prepare a final prompt for ComfyUI")
    skill_registry.register("image.ensure_workflow", skills.ensure_workflow, "Validate ComfyUI image workflow")
    skill_registry.register("image.render", skills.render_image, "Render a real image with ComfyUI")
