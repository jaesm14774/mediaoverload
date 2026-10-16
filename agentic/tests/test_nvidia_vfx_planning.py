from __future__ import annotations

import unittest
from pathlib import Path

from agentic.app.main import build_runtime


class NvidiaVfxPlanningTests(unittest.TestCase):
    def test_user_given_generated_video_when_short_video_plan_is_built_then_qa_and_review_use_nvidia_output(self) -> None:
        """User Given a generated short video When its plan is built Then final QA, preview, and review use NVIDIA VFX output."""
        repo_root = Path(__file__).resolve().parents[2]
        planner, _runner, _memory = build_runtime(
            repo_root,
            output_root=repo_root / ".tmp-tests" / "nvidia-vfx-plan",
            comfy_host="127.0.0.1",
            comfy_port=8188,
        )
        goal = planner.create_goal(
            prompt="A paper boat glides under a warm lantern",
            media_type="text2img2video",
            duration_seconds=6,
            style="painted storybook",
            constraints={
                "canvas_width": 640,
                "canvas_height": 360,
                "duration_override_seconds": 6,
                "max_i2v_frames": 132,
                "skip_upscale_for_i2v": True,
                "enable_review_loop": True,
            },
        )

        plan = planner.build_plan(goal)
        by_id = {node.node_id: node for node in plan.nodes}

        self.assertEqual(by_id["video-nvidia-vfx"].depends_on, ["video-canvas"])
        self.assertEqual(by_id["video-qa"].depends_on[0], "video-nvidia-vfx")
        self.assertEqual(by_id["gif-preview"].depends_on, ["video-nvidia-vfx"])
        self.assertEqual(by_id["video-qa"].inputs["expected_width"], 1280)
        self.assertEqual(by_id["video-qa"].inputs["expected_height"], 720)
        self.assertEqual(by_id["video-qa"].inputs["expected_fps"], 48)
        self.assertEqual(
            by_id["review-video-nvidia-vfx"].depends_on,
            ["review-video-canvas"],
        )
        self.assertIn(
            "review-video-nvidia-vfx",
            by_id["review-final-select"].depends_on,
        )

    def test_user_given_vfx_opt_out_when_short_video_plan_is_built_then_original_canvas_contract_remains(self) -> None:
        """User Given an explicit VFX opt-out When its plan is built Then no NVIDIA node is scheduled and QA keeps source dimensions."""
        repo_root = Path(__file__).resolve().parents[2]
        planner, _runner, _memory = build_runtime(
            repo_root,
            output_root=repo_root / ".tmp-tests" / "nvidia-vfx-opt-out-plan",
            comfy_host="127.0.0.1",
            comfy_port=8188,
        )
        goal = planner.create_goal(
            prompt="A paper boat glides under a warm lantern",
            media_type="text2img2video",
            duration_seconds=6,
            style="painted storybook",
            constraints={
                "canvas_width": 640,
                "canvas_height": 360,
                "duration_override_seconds": 6,
                "max_i2v_frames": 132,
                "skip_upscale_for_i2v": True,
                "enable_nvidia_vfx_postprocess": False,
            },
        )

        plan = planner.build_plan(goal)
        by_id = {node.node_id: node for node in plan.nodes}

        self.assertNotIn("video-nvidia-vfx", by_id)
        self.assertEqual(by_id["video-qa"].depends_on[0], "video-canvas")
        self.assertEqual(by_id["video-qa"].inputs["expected_width"], 640)
        self.assertEqual(by_id["video-qa"].inputs["expected_height"], 360)
        self.assertEqual(by_id["video-qa"].inputs["expected_fps"], 24)

    def test_user_given_other_generated_video_routes_when_plan_is_built_then_final_qa_follows_nvidia_output(self) -> None:
        """User Given native H3, text-to-video, long-video, and motion-comic goals When their plans are built Then final QA follows NVIDIA VFX."""
        repo_root = Path(__file__).resolve().parents[2]
        planner, _runner, _memory = build_runtime(
            repo_root,
            output_root=repo_root / ".tmp-tests" / "nvidia-vfx-route-matrix",
            comfy_host="127.0.0.1",
            comfy_port=8188,
        )
        cases = [
            ("text2video", {}, "video-nvidia-vfx", "animate-video", "video-qa"),
            ("long_video", {}, "video-nvidia-vfx", "video-trim", "longvideo-video-qa"),
            (
                "motion_comic",
                {},
                "motion-comic-nvidia-vfx",
                "motion-comic-compose",
                "motion-comic-qa",
            ),
            (
                "native_h3_t2v_story",
                {
                    "native_h3_storyboard_path": str(
                        repo_root / "configs" / "storyboards" / "native_h3_15s.yaml"
                    )
                },
                "native-h3-nvidia-vfx",
                "native-h3-render",
                "native-h3-qa",
            ),
        ]

        for media_type, constraints, vfx_id, source_id, qa_id in cases:
            with self.subTest(media_type=media_type):
                goal = planner.create_goal(
                    prompt="A paper boat glides under a warm lantern",
                    media_type=media_type,
                    duration_seconds=6,
                    style="painted storybook",
                    constraints=constraints,
                )
                plan = planner.build_plan(goal)
                by_id = {node.node_id: node for node in plan.nodes}

                self.assertEqual(by_id[vfx_id].depends_on, [source_id])
                self.assertIn(vfx_id, by_id[qa_id].depends_on)
                self.assertEqual(by_id[qa_id].inputs["expected_fps"], 48)


if __name__ == "__main__":
    unittest.main()
