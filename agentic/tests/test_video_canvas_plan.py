from __future__ import annotations

import unittest
from pathlib import Path

from agentic.app.main import build_runtime
class VideoCanvasPlanTests(unittest.TestCase):
    def test_user_given_i2v_goal_when_plan_is_built_then_canvas_normalization_precedes_video_qa(self) -> None:
        """User Given an I2V goal When its plan is built Then QA consumes the normalized final canvas."""
        repo_root = Path(__file__).resolve().parents[2]
        planner, _runner, _memory = build_runtime(
            repo_root,
            output_root=repo_root / ".tmp-tests" / "visual-action-plan",
            comfy_host="127.0.0.1",
            comfy_port=8188,
        )
        goal = planner.create_goal(
            prompt="Kirby nudges one jelly cube and gets pulled along",
            media_type="text2img2video",
            duration_seconds=6,
            style="tactile storybook watercolor",
            auto_download_assets=False,
            constraints={
                "canvas_width": 640,
                "canvas_height": 360,
                "duration_override_seconds": 6,
                "max_i2v_frames": 132,
                "skip_upscale_for_i2v": True,
            },
        )

        plan = planner.build_plan(goal)
        canvas = next(node for node in plan.nodes if node.node_id == "video-canvas")
        qa = next(node for node in plan.nodes if node.node_id == "video-qa")
        animate = next(node for node in plan.nodes if node.node_id == "animate-video")

        self.assertEqual(canvas.inputs["target_width"], 640)
        self.assertEqual(canvas.inputs["target_height"], 360)
        self.assertEqual(canvas.depends_on, ["animate-video"])
        self.assertEqual(qa.depends_on[0], "video-canvas")
        self.assertEqual(animate.inputs["length"], 132)


if __name__ == "__main__":
    unittest.main()
