from __future__ import annotations

import random
import unittest
from pathlib import Path

from agentic.app.character_workflow import build_goal_payload_from_character_config
from agentic.app.creative_profiles import resolve_creative_profile
from agentic.assets.registry import AssetRegistry
from agentic.runtime.contracts import GoalRequest
from agentic.runtime.planner import TaskPlanner
from character_workflow_helpers import make_character_workflow_request


class CreativeProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.kirby_config = cls.repo_root / "configs" / "characters" / "kirby.yaml"
        cls.asset_registry = AssetRegistry(cls.repo_root / "agentic", asset_root=cls.repo_root)
        cls.planner = TaskPlanner(asset_registry=cls.asset_registry)

    def test_fixed_canvas_and_style_are_resolved_from_config(self) -> None:
        profile = resolve_creative_profile(
            {
                "canvas": {
                    "mode": "fixed",
                    "fixed": "portrait_4_5",
                    "profiles": {
                        "portrait_4_5": {"aspect_ratio": "4:5", "width": 512, "height": 640},
                    },
                },
                "style": {
                    "mode": "fixed",
                    "fixed": "storybook_watercolor",
                    "profiles": {
                        "storybook_watercolor": {
                            "label": "storybook watercolor",
                            "prompt": "soft watercolor paper texture and hand-painted depth",
                        },
                    },
                },
            },
            rng=random.Random(7),
        )

        self.assertEqual(profile["canvas"], {
            "id": "portrait_4_5",
            "aspect_ratio": "4:5",
            "width": 512,
            "height": 640,
        })
        self.assertEqual(profile["style"]["id"], "storybook_watercolor")

    def test_random_profile_is_seeded_and_uses_exact_allowed_ratio(self) -> None:
        config = {
            "canvas": {
                "mode": "random",
                "random_weights": {"landscape_16_9": 3, "portrait_4_5": 2, "square_1_1": 1},
                "profiles": {
                    "landscape_16_9": {"aspect_ratio": "16:9", "width": 640, "height": 360},
                    "portrait_4_5": {"aspect_ratio": "4:5", "width": 512, "height": 640},
                    "square_1_1": {"aspect_ratio": "1:1", "width": 512, "height": 512},
                },
            },
            "style": {
                "mode": "random",
                "random_weights": {"a": 1, "b": 1},
                "profiles": {"a": {"label": "a", "prompt": "a"}, "b": {"label": "b", "prompt": "b"}},
            },
        }

        first = resolve_creative_profile(config, rng=random.Random(11))
        second = resolve_creative_profile(config, rng=random.Random(11))

        self.assertEqual(first, second)
        self.assertIn(first["canvas"]["aspect_ratio"], {"16:9", "4:5", "1:1"})
        self.assertEqual(
            first["canvas"]["width"] / first["canvas"]["height"],
            {"16:9": 16 / 9, "4:5": 4 / 5, "1:1": 1}[first["canvas"]["aspect_ratio"]],
        )

    def test_random_profile_can_produce_more_than_one_canvas_and_style(self) -> None:
        config = {
            "canvas": {
                "mode": "random",
                "random_weights": {"wide": 1, "square": 1},
                "profiles": {
                    "wide": {"aspect_ratio": "16:9", "width": 640, "height": 360},
                    "square": {"aspect_ratio": "1:1", "width": 512, "height": 512},
                },
            },
            "style": {
                "mode": "random",
                "random_weights": {"paint": 1, "graphic": 1},
                "profiles": {
                    "paint": {"label": "paint", "prompt": "paint"},
                    "graphic": {"label": "graphic", "prompt": "graphic"},
                },
            },
        }

        profiles = [resolve_creative_profile(config, rng=random.Random(seed)) for seed in range(12)]

        self.assertGreater(len({profile["canvas"]["id"] for profile in profiles}), 1)
        self.assertGreater(len({profile["style"]["id"] for profile in profiles}), 1)

    def test_character_payload_binds_one_canvas_to_image_and_video_contract(self) -> None:
        payload = build_goal_payload_from_character_config(
            make_character_workflow_request(
                self.repo_root,
                self.kirby_config,
                prompt="Kirby protects a glowing seed from a sudden gust",
                preferred_generation_type="native_h3_story",
                publish_after_generate=False,
                rng=random.Random(2),
            )
        )

        constraints = payload["constraints"]
        canvas = constraints["canvas_profile"]
        self.assertIn(canvas["aspect_ratio"], {"16:9", "4:5", "1:1"})
        self.assertEqual(constraints["canvas_width"], canvas["width"])
        self.assertEqual(constraints["canvas_height"], canvas["height"])
        self.assertEqual(constraints["creative_profile"]["canvas"], canvas)
        self.assertIn(constraints["visual_style_profile"]["id"], {
            "clean_cel_action",
            "painterly_atmosphere",
            "storybook_watercolor",
            "dynamic_ensemble",
            "graphic_color_pop",
            "tactile_tabletop",
        })

    def test_text2img2video_plan_passes_shared_canvas_to_image_and_video(self) -> None:
        goal = GoalRequest(
            prompt="Kirby in a layered neon sky landscape",
            media_type="text2img2video",
            duration_seconds=5,
            style="graphic color pop",
            constraints={
                "canvas_width": 512,
                "canvas_height": 640,
                "duration_override_seconds": 5,
                "video_frame_rate": 24,
            },
        )

        plan = self.planner.build_plan(goal)
        image = next(node for node in plan.nodes if node.node_id == "render-image")
        video = next(node for node in plan.nodes if node.node_id == "animate-video")

        self.assertEqual((image.inputs["width"], image.inputs["height"]), (512, 640))
        self.assertEqual((video.inputs["width"], video.inputs["height"]), (512, 640))

    def test_native_h3_plan_passes_shared_canvas_to_keyframe_and_video(self) -> None:
        goal = GoalRequest(
            prompt="Kirby protects a glowing seed in a storm",
            media_type="native_h3_story",
            duration_seconds=15,
            style="painterly atmospheric adventure",
            constraints={
                "canvas_width": 640,
                "canvas_height": 360,
                "native_h3_storyboard_path": "configs/storyboards/native_h3_15s.yaml",
            },
        )

        plan = self.planner.build_plan(goal)
        opening = next(node for node in plan.nodes if node.node_id == "native-opening-keyframe")
        video = next(node for node in plan.nodes if node.node_id == "native-h3-render")
        canvas = next(node for node in plan.nodes if node.node_id == "native-h3-canvas")
        qa = next(node for node in plan.nodes if node.node_id == "native-h3-qa")

        self.assertEqual((opening.inputs["width"], opening.inputs["height"]), (640, 360))
        self.assertEqual((video.inputs["width"], video.inputs["height"]), (640, 360))
        self.assertEqual(canvas.inputs["target_width"], 640)
        self.assertEqual(canvas.inputs["target_height"], 360)
        self.assertEqual(canvas.depends_on, ["native-h3-render"])
        self.assertEqual(qa.inputs["video_node"], "native-h3-canvas")

    def test_long_video_segments_and_qa_use_shared_canvas(self) -> None:
        goal = GoalRequest(
            prompt="Kirby crosses a layered storm landscape",
            media_type="long_video",
            duration_seconds=10,
            style="painterly atmospheric adventure",
            constraints={
                "canvas_width": 512,
                "canvas_height": 640,
                "segment_count": 2,
            },
        )

        plan = self.planner.build_plan(goal)
        segment = next(node for node in plan.nodes if node.node_id == "segment-video-01")
        qa = next(node for node in plan.nodes if node.node_id == "segment-qa-01")

        self.assertEqual((segment.inputs["width"], segment.inputs["height"]), (512, 640))
        self.assertEqual((qa.inputs["expected_width"], qa.inputs["expected_height"]), (512, 640))


if __name__ == "__main__":
    unittest.main()
