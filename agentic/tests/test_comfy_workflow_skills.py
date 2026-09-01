from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest, RunState, SkillContext
from agentic.runtime.registry import ToolRegistry
from agentic.skills.comfy_image import ComfyImageSkills
from agentic.skills.comfy_workflow_skills import ComfyWorkflowSkills


class ComfyWorkflowSkillsTests(unittest.TestCase):
    def test_reference_micro_gag_image_render_uses_opening_keyframe_prompt(self) -> None:
        plan = ExecutionPlan(
            goal=GoalRequest(
                prompt="full temporal gag prompt",
                media_type="text2img2video",
                style="anime",
                constraints={"reference_micro_gag_profile": "reference_micro_gag_v1"},
            ),
            workflow_name="text2img2video_v1",
            nodes=[],
        )
        node = ExecutionNode(node_id="render-image", skill_name="image.render", depends_on=["idea-brief"])
        state = RunState(
            goal={},
            metadata={},
            node_outputs={
                "idea-brief": {
                    "prompt": "full temporal gag prompt",
                    "opening_keyframe_prompt": "single opening frame with one hero and one prop",
                    "negative_prompt": "duplicate, text",
                }
            },
        )

        bundle = ComfyImageSkills._resolve_prompt_bundle(SkillContext(plan=plan, node=node, state=state))

        self.assertEqual(bundle["prompt"], "single opening frame with one hero and one prop")
        self.assertEqual(bundle["negative_prompt"], "duplicate, text")

    def test_image_to_video_forwards_fixed_seed(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []
        tools = ToolRegistry()

        def render(payload: dict[str, object]) -> dict[str, object]:
            calls.append(("comfy.workflow.image_to_video", payload))
            return {"saved_files": ["C:/renders/video.mp4"]}

        tools.register("comfy.workflow.image_to_video", render, "test")
        with tempfile.TemporaryDirectory() as temp_dir:
            skills = ComfyWorkflowSkills(tools, Path(temp_dir))
            plan = ExecutionPlan(
                goal=GoalRequest(
                    prompt="animate Kirby",
                    media_type="text2img2video",
                    style="anime",
                    constraints={"seed": 123456},
                ),
                workflow_name="text2img2video_v1",
                nodes=[],
            )
            node = ExecutionNode(
                node_id="animate-video",
                skill_name="image.animate",
                inputs={
                    "workflow_name": "minimax_h3_lowvram_i2v",
                    "image_path": "C:/renders/frame.png",
                    "prompt": "Kirby jumps",
                },
            )

            result = skills.image_to_video(
                SkillContext(plan=plan, node=node, state=RunState(goal={}, metadata={}))
            )

        self.assertEqual(result.status, "success")
        self.assertEqual(calls[0][0], "comfy.workflow.image_to_video")
        self.assertEqual(calls[0][1]["seed"], 123456)

    def test_optional_upscale_failure_keeps_source_image_for_downstream_video(self) -> None:
        tools = ToolRegistry()

        def failing_upscale(_payload: dict[str, object]) -> dict[str, object]:
            raise RuntimeError("missing_node_type: ImageTile+")

        tools.register("comfy.workflow.image_upscale", failing_upscale, "test")
        source = r"C:\renders\source.png"
        with tempfile.TemporaryDirectory() as temp_dir:
            skills = ComfyWorkflowSkills(tools, Path(temp_dir))
            plan = ExecutionPlan(
                goal=GoalRequest(prompt="animate Kirby", media_type="text2img2video", style="anime"),
                workflow_name="text2img2video_v1",
                nodes=[],
            )
            node = ExecutionNode(
                node_id="upscale-image",
                skill_name="image.upscale",
                depends_on=["render-image"],
            )
            state = RunState(
                goal={"prompt": plan.goal.prompt},
                metadata={},
                node_outputs={"render-image": {"saved_files": [source]}},
            )

            result = skills.upscale_image(SkillContext(plan=plan, node=node, state=state))

        self.assertEqual(result.status, "success")
        self.assertEqual(result.outputs["saved_files"], [source])
        self.assertTrue(result.outputs["upscale_fallback"])
        self.assertIn("Optional upscale unavailable", result.logs[0])


if __name__ == "__main__":
    unittest.main()
