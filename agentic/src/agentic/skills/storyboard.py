from __future__ import annotations

from agentic.runtime.contracts import SkillContext, SkillResult
from agentic.runtime.registry import SkillRegistry, ToolRegistry


class StoryboardSkills:
    def __init__(self, tools: ToolRegistry) -> None:
        self.tools = tools

    def expand_idea(self, context: SkillContext) -> SkillResult:
        prompt = context.node.inputs["prompt"]
        style = context.node.inputs["style"]
        brief = {
            "creative_brief": f"{prompt} presented as a concise storyboard in {style} style",
            "tone": "clear visual beats",
        }
        return SkillResult(status="success", outputs=brief, logs=["Expanded storyboard goal into a brief."])

    def segment_story(self, context: SkillContext) -> SkillResult:
        segment_count = int(context.node.inputs["segment_count"])
        brief = context.state["idea-brief"]["creative_brief"]
        segments = [
            {
                "segment_id": f"scene-{index + 1}",
                "visual": f"{brief}. Key beat {index + 1} of {segment_count}.",
                "narration": f"Scene {index + 1} advances the story with a clear visual transition.",
            }
            for index in range(segment_count)
        ]
        return SkillResult(
            status="success",
            outputs={"segments": segments, "segment_count": segment_count},
            metrics={"segment_count": segment_count},
            logs=[f"Prepared {segment_count} storyboard scenes."],
        )

    def ensure_workflow(self, context: SkillContext) -> SkillResult:
        result = self.tools.call(
            "asset.ensure_workflow_ready",
            {
                "workflow_name": context.node.inputs["workflow_name"],
                "auto_download": False,
            },
        )
        return SkillResult(status="success", outputs=result, logs=["Validated local storyboard workflow."])

    def render_frames(self, context: SkillContext) -> SkillResult:
        script = context.state["script-plan"]
        result = self.tools.call(
            "local.render_storyboard_frames",
            {
                "prompt": context.plan.goal.prompt,
                "style": context.plan.goal.style,
                "segments": script["segments"],
                "frame_width": context.node.inputs.get("frame_width", 1280),
                "frame_height": context.node.inputs.get("frame_height", 720),
            },
        )
        return SkillResult(
            status="success",
            outputs=result,
            metrics={"frame_count": result["frame_count"]},
            logs=["Rendered storyboard PNG frames."],
        )

    def package_storyboard(self, context: SkillContext) -> SkillResult:
        frames = context.state["storyboard-frames"]
        result = self.tools.call(
            "local.package_storyboard",
            {
                "run_dir": frames["run_dir"],
                "frame_paths": frames["frame_paths"],
                "goal": context.plan.goal.prompt,
                "style": context.plan.goal.style,
                "workflow_name": context.plan.workflow_name,
            },
        )
        return SkillResult(
            status="success",
            outputs=result,
            logs=["Packaged storyboard outputs and metadata."],
        )


def register_storyboard_skills(skill_registry: SkillRegistry, tool_registry: ToolRegistry) -> None:
    skills = StoryboardSkills(tool_registry)
    skill_registry.register("story.idea.expand", skills.expand_idea, "Expand storyboard idea")
    skill_registry.register("story.segment_storyboard", skills.segment_story, "Split storyboard into scenes")
    skill_registry.register("story.ensure_workflow", skills.ensure_workflow, "Validate storyboard workflow")
    skill_registry.register("story.render_frames", skills.render_frames, "Render storyboard frames")
    skill_registry.register("story.package_outputs", skills.package_storyboard, "Package storyboard outputs")
