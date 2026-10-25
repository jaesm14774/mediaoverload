from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic.assets.registry import AssetRegistry
from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest, RunState, SkillContext
from agentic.runtime.video_conditioning import (
    capabilities_from_manifests,
    recipe_candidates,
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
        self.assertIn("t2v", candidates)
        self.assertIn("anchor_first_last", candidates)
        self.assertIn("anchor_last", candidates)
        self.assertIn("reference_bundle", candidates)
        self.assertEqual(candidates["anchor_first"][0].workflow_name, "minimax_h3_i2va")
        self.assertEqual(candidates["reference_bundle"][0].recipes["reference_bundle"].reference_maximum, 4)

    def test_t2v_recipe_has_no_frame_conditioning(self) -> None:
        capabilities = capabilities_from_manifests(self.registry.all_manifests())
        recipe = recipe_candidates(capabilities)["t2v"][0].recipes["t2v"]

        self.assertFalse(recipe.requires_first)
        self.assertFalse(recipe.requires_last)
        self.assertFalse(recipe.requires_references)
        self.assertEqual(recipe.render_tool, "comfy.render_h3")

    def test_last_frame_recipe_binds_only_last_anchor_to_h3_renderer(self) -> None:
        captured: dict[str, object] = {}
        tools = ToolRegistry()

        def fake_render(payload: dict[str, object]) -> dict[str, object]:
            captured.update(payload)
            return {"saved_files": ["C:/tmp/segment.mp4"]}

        tools.register("comfy.render_h3", fake_render, "test renderer")
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
                    "workflow_name": "minimax_h3_fl2va",
                    "render_tool": "comfy.render_h3",
                    "anchor_nodes": {"last": "ending-frame"},
                    "conditioning_plan": {"recipe": "anchor_last", "anchors": {"last": "ending-frame"}},
                    "width": 512,
                    "height": 288,
                    "length": 81,
                    "steps": 16,
                },
                depends_on=["segment-prompt-01"],
            )
            state = RunState(
                goal={"prompt": plan.goal.prompt},
                metadata={},
                node_outputs={
                    "script-plan": {
                        "segments": [
                            {
                                "segment_id": "segment/01:opening",
                                "visual": "Kirby takes one determined step",
                                "action": "walks through the rain",
                                "camera": "slow tracking shot",
                            }
                        ]
                    },
                    "idea-brief": {"negative_prompt": "blurry"},
                    "segment-prompt-01": {
                        "prompt": "Kirby takes one determined step through the rain. The camera tracks alongside him.",
                        "overall_soundscape": "Rain taps the harbor awning and the buoy chain rattles in the gust.",
                        "non_diegetic_music": "N/A",
                    },
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
        self.assertIn("Kirby takes one determined step through the rain", captured["prompt"])
        self.assertIn("overall_soundscape: Rain taps the harbor awning", captured["prompt"])
        self.assertIn("non_diegetic_music: N/A", captured["prompt"])
        self.assertEqual(captured["length"], 81)
        self.assertIn("segment-01-opening_video", str(captured["run_dir"]))

    def test_render_segment_video_consumes_prepared_segment_direction(self) -> None:
        captured: dict[str, object] = {}
        tools = ToolRegistry()

        def fake_render(payload: dict[str, object]) -> dict[str, object]:
            captured.update(payload)
            return {"saved_files": ["C:/tmp/segment.mp4"]}

        tools.register("comfy.render_h3", fake_render, "test renderer")
        with tempfile.TemporaryDirectory() as temp_dir:
            skills = LongVideoSkills(tools, Path(temp_dir))
            plan = ExecutionPlan(
                goal=GoalRequest(
                    prompt="Kirby catches one visible news-derived object",
                    style="anime cinematic",
                    constraints={
                        "character": "Kirby",
                        "news_context": {
                            "title": "A wind warning changes the harbor",
                            "keyword": "wind; harbor",
                        },
                    },
                ),
                workflow_name="longvideo_real_v2",
                nodes=[],
            )
            node = ExecutionNode(
                node_id="segment-video-01",
                skill_name="longvideo.render_segment_video",
                inputs={
                    "segment_index": 0,
                    "recipe": "anchor_first",
                    "workflow_name": "minimax_h3_fl2va",
                    "render_tool": "comfy.render_h3",
                    "anchor_nodes": {"first": "opening-frame"},
                    "width": 512,
                    "height": 288,
                    "length": 81,
                    "steps": 16,
                },
                depends_on=["segment-prompt-01"],
            )
            state = RunState(
                goal={"prompt": plan.goal.prompt},
                metadata={},
                node_outputs={
                    "script-plan": {
                        "segments": [
                            {
                                "segment_id": "segment-1",
                                "visual": "Kirby braces against a gust at the harbor",
                                "action": "grabs a loose warning buoy",
                                "camera": "track with the buoy as it swings",
                                "start_state": "the buoy hangs still",
                                "end_state": "the buoy is secured against the gust",
                                "cause": "the wind pulls the buoy loose",
                                "effect": "Kirby secures the harbor marker",
                            }
                        ]
                    },
                    "idea-brief": {"negative_prompt": "blurry"},
                    "segment-prompt-01": {
                        "prompt": "Keep the buoy visible and show the gust physically pulling it left.",
                        "overall_soundscape": "Wind rushes across the harbor as the buoy chain knocks against the pier.",
                        "non_diegetic_music": "N/A",
                    },
                    "opening-frame": {"selected_assets": ["C:/tmp/opening.png"]},
                },
            )

            result = skills.render_segment_video(SkillContext(plan=plan, node=node, state=state))

        self.assertEqual(result.status, "success")
        self.assertIn("Keep the buoy visible", str(captured["prompt"]))
