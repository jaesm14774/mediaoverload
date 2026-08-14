from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest, RunState, SkillContext
from agentic.runtime.registry import ToolRegistry
from agentic.skills.comfy_workflow_skills import ComfyWorkflowSkills


class ComfyWorkflowSkillsTests(unittest.TestCase):
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
