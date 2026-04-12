from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from agentic.runtime.contracts import SkillContext, SkillResult
from agentic.runtime.registry import SkillRegistry, ToolRegistry


def _asset_check_result(result: dict[str, object], success_log: str) -> SkillResult:
    asset_status = result.get("asset_status", [])
    if not isinstance(asset_status, list):
        return SkillResult(status="success", outputs=result, logs=[success_log])

    missing_assets = [
        str(item.get("asset", "")).strip()
        for item in asset_status
        if isinstance(item, dict) and str(item.get("status", "")).lower() != "ready"
    ]
    if not missing_assets:
        return SkillResult(status="success", outputs=result, logs=[success_log])

    workflow_name = str(result.get("workflow_name", "")).strip()
    details = ", ".join(asset for asset in missing_assets if asset) or "unknown assets"
    return SkillResult(
        status="failed",
        outputs=result,
        logs=[f"Workflow assets missing for '{workflow_name}': {details}"],
    )


class LongVideoSkills:
    def __init__(self, tools: ToolRegistry, output_root: Path) -> None:
        self.tools = tools
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)

    def expand_idea(self, context: SkillContext) -> SkillResult:
        prompt = str(context.node.inputs["prompt"])
        style = str(context.node.inputs["style"])
        idea_variants = list(context.node.inputs.get("idea_variants", []))
        selected_variant = idea_variants[0] if idea_variants else {"style": style}
        return SkillResult(
            status="success",
            outputs={
                "creative_brief": f"{prompt} rendered as a {selected_variant.get('style', style)} long-form video narrative",
                "tone": "playful cinematic escalation",
                "idea_variants": idea_variants,
                "prompt": f"{prompt}, {selected_variant.get('style', style)}, cinematic, highly detailed",
                "negative_prompt": "ugly, blurry, low quality, bad anatomy, deformed, duplicate, watermark, text",
            },
            logs=["Expanded goal into a creative brief."],
        )

    def segment_story(self, context: SkillContext) -> SkillResult:
        segment_count = int(context.node.inputs["segment_count"])
        brief = str(context.state["idea-brief"]["creative_brief"])
        segments = [
            {
                "segment_id": f"segment-{index + 1}",
                "visual": f"{brief}; shot {index + 1} of {segment_count}",
                "narration": f"Narration for shot {index + 1} continuing the same story.",
            }
            for index in range(segment_count)
        ]
        return SkillResult(
            status="success",
            outputs={"segments": segments, "segment_count": segment_count},
            metrics={"segment_count": segment_count},
            logs=[f"Planned {segment_count} story segments."],
        )

    def ensure_workflow(self, context: SkillContext) -> SkillResult:
        result = self.tools.call(
            "asset.ensure_workflow_ready",
            {
                "workflow_name": context.node.inputs["workflow_name"],
                "auto_download": context.node.inputs.get("auto_download", False),
            },
        )
        return _asset_check_result(result, "Checked workflow assets.")

    def render_initial_frame(self, context: SkillContext) -> SkillResult:
        segment = self._segment(context)
        run_dir = self._build_run_dir(context.plan.goal.prompt, f"{segment['segment_id']}_image")
        result = self.tools.call(
            "comfy.render_image",
            {
                "workflow_name": context.node.inputs["workflow_name"],
                "run_dir": str(run_dir),
                "prompt": f"{segment['visual']}, {context.plan.goal.style}, cinematic lighting, consistent character design",
                "negative_prompt": str(context.state["idea-brief"].get("negative_prompt", "")),
                "width": context.node.inputs.get("width", 1024),
                "height": context.node.inputs.get("height", 1024),
            },
        )
        return SkillResult(
            status="success",
            outputs=result,
            metrics={"image_count": len(result.get("saved_files", []))},
            logs=[f"Rendered initial keyframe for {segment['segment_id']}."],
        )

    def render_segment_video(self, context: SkillContext) -> SkillResult:
        segment = self._segment(context)
        run_dir = self._build_run_dir(context.plan.goal.prompt, f"{segment['segment_id']}_video")
        image_path = self._optional_dependency_value(context, ("frame_path", "saved_files"))
        if not image_path:
            raise RuntimeError(f"Missing frame input for node '{context.node.node_id}'")

        result = self.tools.call(
            "comfy.render_image_to_video",
            {
                "workflow_name": context.node.inputs["workflow_name"],
                "run_dir": str(run_dir),
                "image_path": image_path,
                "prompt": f"{segment['visual']}, {context.plan.goal.style}, motion continuity, coherent action",
            },
        )
        return SkillResult(
            status="success",
            outputs=result,
            metrics={"video_count": len(result.get("saved_files", []))},
            logs=[f"Rendered {segment['segment_id']} clip."],
        )

    def generate_segment_tts(self, context: SkillContext) -> SkillResult:
        segment = self._segment(context)
        run_dir = self._build_run_dir(context.plan.goal.prompt, f"{segment['segment_id']}_tts")
        audio_dir = run_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        result = self.tools.call(
            "audio.generate_tts_real",
            {
                "text": segment["narration"],
                "output_path": str(audio_dir / f"{segment['segment_id']}.mp3"),
                "voice": context.node.inputs.get("voice", "en-US-AriaNeural"),
                "rate": context.node.inputs.get("rate", "+0%"),
            },
        )
        return SkillResult(status="success", outputs=result, logs=[f"Generated TTS for {segment['segment_id']}."])

    def package_outputs(self, context: SkillContext) -> SkillResult:
        final_node = str(context.node.inputs["final_video_node"])
        preview_node = str(context.node.inputs["preview_node"])
        segment_videos = []
        for dependency in context.node.depends_on:
            if dependency.startswith("segment-video-"):
                saved_files = context.state[dependency].get("saved_files", [])
                if saved_files:
                    segment_videos.append(str(saved_files[0]))

        return SkillResult(
            status="success",
            outputs={
                "video_path": str(context.state[final_node]["video_path"]),
                "preview_gif_path": str(context.state[preview_node]["gif_path"]),
                "segment_videos": segment_videos,
                "segment_count": len(segment_videos),
            },
            metrics={"segment_count": len(segment_videos)},
            logs=["Packaged final long-video artifacts."],
        )

    def _segment(self, context: SkillContext) -> dict[str, object]:
        segment_index = int(context.node.inputs["segment_index"])
        return context.state["script-plan"]["segments"][segment_index]

    def _build_run_dir(self, prompt: str, suffix: str) -> Path:
        slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")
        slug = slug[:32] or "longvideo"
        return self.output_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slug}_{suffix}"

    @staticmethod
    def _optional_dependency_value(context: SkillContext, keys: tuple[str, ...]) -> str | None:
        for dependency in reversed(context.node.depends_on):
            dependency_output = context.state[dependency]
            for key in keys:
                value = dependency_output.get(key)
                if isinstance(value, list) and value:
                    return str(value[0])
                if isinstance(value, str) and value:
                    return value
        return None


def register_longvideo_skills(skill_registry: SkillRegistry, tool_registry: ToolRegistry, output_root: Path) -> None:
    skills = LongVideoSkills(tool_registry, output_root)
    skill_registry.register("idea.expand", skills.expand_idea, "Expand the goal into a creative brief")
    skill_registry.register("script.segment_story", skills.segment_story, "Create long-video story segments")
    skill_registry.register("asset.ensure_workflow", skills.ensure_workflow, "Verify workflow assets")
    skill_registry.register("longvideo.render_initial_frame", skills.render_initial_frame, "Render the opening seed frame")
    skill_registry.register("longvideo.render_segment_video", skills.render_segment_video, "Render one long-video segment")
    skill_registry.register("longvideo.generate_segment_tts", skills.generate_segment_tts, "Generate one narration segment")
    skill_registry.register("longvideo.package_outputs", skills.package_outputs, "Package long-video outputs")
