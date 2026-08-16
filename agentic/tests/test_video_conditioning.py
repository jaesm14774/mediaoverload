from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic.assets.registry import AssetRegistry
from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest, RunState, SkillContext
from agentic.runtime.video_conditioning import (
    capabilities_from_manifests,
    recipe_candidates,
    sample_recipe_sequence,
)
from agentic.runtime.registry import ToolRegistry
from agentic.skills.longvideo import LongVideoSkills


class VideoConditioningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.registry = AssetRegistry(cls.repo_root / "agentic", asset_root=cls.repo_root)

    def test_workflow_metadata_exposes_provider_neutral_h3_capabilities(self) -> None:
        capabilities = capabilities_from_manifests(self.registry.all_manifests())
        candidates = recipe_candidates(capabilities)

        self.assertIn("anchor_first", candidates)
        self.assertIn("anchor_first_last", candidates)
        self.assertIn("anchor_last", candidates)
        self.assertIn("reference_bundle", candidates)
        self.assertEqual(candidates["anchor_first"][0].workflow_name, "minimax_h3_lowvram_i2v")
        self.assertEqual(candidates["reference_bundle"][0].recipes["reference_bundle"].reference_maximum, 4)

    def test_positive_weight_for_unsupported_recipe_fails_before_render(self) -> None:
        capabilities = capabilities_from_manifests(self.registry.all_manifests())

        with self.assertRaisesRegex(ValueError, "no compatible workflow"):
            sample_recipe_sequence(
                {"anchor_first": 1, "future_control_bundle": 2},
                3,
                recipe_candidates(capabilities),
                seed=7,
            )

    def test_mix_is_reproducible_and_samples_each_segment_independently(self) -> None:
        capabilities = capabilities_from_manifests(self.registry.all_manifests())
        eligible = recipe_candidates(capabilities)
        seed, sequence = sample_recipe_sequence(
            {"anchor_first": 1, "anchor_first_last": 3, "anchor_last": 2, "reference_bundle": 2},
            40,
            eligible,
            seed=20260815,
        )
        _, repeated = sample_recipe_sequence(
            {"anchor_first": 1, "anchor_first_last": 3, "anchor_last": 2, "reference_bundle": 2},
            40,
            eligible,
            seed=20260815,
        )

        self.assertEqual(seed, 20260815)
        self.assertEqual(sequence, repeated)
        self.assertGreaterEqual(len(set(sequence)), 3)

    def test_last_frame_recipe_binds_only_last_anchor_to_h3_renderer(self) -> None:
        captured: dict[str, object] = {}
        tools = ToolRegistry()

        def fake_render(payload: dict[str, object]) -> dict[str, object]:
            captured.update(payload)
            return {"saved_files": ["C:/tmp/segment.mp4"]}

        tools.register("comfy.render_image_to_video", fake_render, "test renderer")
        with tempfile.TemporaryDirectory() as temp_dir:
            skills = LongVideoSkills(tools, Path(temp_dir))
            plan = ExecutionPlan(
                goal=GoalRequest(
                    prompt="Kirby crosses a rainy meadow",
                    style="anime cinematic",
                    constraints={"character": "Kirby"},
                ),
                workflow_name="longvideo_real_v2",
                nodes=[],
            )
            node = ExecutionNode(
                node_id="segment-video-01",
                skill_name="longvideo.render_segment_video",
                inputs={
                    "segment_index": 0,
                    "recipe": "anchor_last",
                    "workflow_name": "minimax_h3_lowvram_15s_fl2va_i2v",
                    "render_tool": "comfy.workflow.image_to_video",
                    "anchor_nodes": {"last": "ending-frame"},
                    "conditioning_plan": {"recipe": "anchor_last", "anchors": {"last": "ending-frame"}},
                    "width": 512,
                    "height": 288,
                    "length": 81,
                    "steps": 16,
                },
            )
            state = RunState(
                goal={"prompt": plan.goal.prompt},
                metadata={},
                node_outputs={
                    "script-plan": {
                        "segments": [
                            {
                                "segment_id": "segment-01",
                                "visual": "Kirby takes one determined step",
                                "action": "walks through the rain",
                                "camera": "slow tracking shot",
                            }
                        ]
                    },
                    "idea-brief": {"negative_prompt": "blurry"},
                    "ending-frame": {"selected_assets": ["C:/tmp/ending.png"]},
                },
            )

            result = skills.render_segment_video(SkillContext(plan=plan, node=node, state=state))

        self.assertEqual(result.status, "success")
        self.assertEqual(captured["h3_mode"], "l2va")
        self.assertFalse(captured["use_first_frame"])
        self.assertTrue(captured["use_last_frame"])
        self.assertNotIn("image_path", captured)
        self.assertEqual(captured["last_image_path"], "C:/tmp/ending.png")
