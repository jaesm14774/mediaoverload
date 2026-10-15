from __future__ import annotations

import random
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import yaml

from agentic.app.character_payload import build_goal_payload_from_character_config
from agentic.app.creative_profiles import resolve_creative_profile
from agentic.app.style_catalog import StyleLibrary
from agentic.assets.registry import AssetRegistry
from agentic.runtime.contracts import GoalRequest
from agentic.runtime.planner import TaskPlanner
from agentic.runtime.prompting import (
    build_goal_brief,
    include_visual_style_profile,
    visual_style_profile_direction,
)
from character_workflow_helpers import make_character_workflow_request


class CreativeProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.kirby_config = cls.repo_root / "configs" / "characters" / "kirby.yaml"
        cls.asset_registry = AssetRegistry(cls.repo_root / "agentic", asset_root=cls.repo_root)
        cls.planner = TaskPlanner(asset_registry=cls.asset_registry)

    def test_user_given_source_scenes_when_catalog_styles_are_applied_then_scene_and_full_style_descriptor_are_preserved(self) -> None:
        """User Given source scenes When the catalog styles build briefs Then each original scene and its complete selected style descriptor are preserved."""
        profiles = StyleLibrary().profiles()
        transport_scene = "Magolor watches a train depart after its timetable changes."
        planting_scene = "Magolor tends a seedling in a flowerpot after the soil is replaced."
        for profile_id, profile in profiles.items():
            with self.subTest(profile=profile_id):
                for scene in (transport_scene, planting_scene):
                    goal = GoalRequest(
                        prompt=scene, media_type="text2img2video", style=profile["label"],
                        constraints={"visual_style_profile": profile},
                    )
                    brief = build_goal_brief(goal, profile["label"], [])
                    self.assertIn(scene, brief["prompt"])
                    self.assertIn(profile["prompt"], brief["prompt"])

    def test_user_given_a_selected_group_character_when_style_profiles_are_applied_then_style_stays_generic_and_the_brief_preserves_the_selected_identity(self) -> None:
        """User Given a group-selected character and any configured visual style When the goal payload is built Then style stays identity-independent and the creative brief preserves the selected role profile."""
        original = self.kirby_config.read_text(encoding="utf-8")
        profile_ids = list(StyleLibrary().profiles())[:3]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "profile.yaml"
            for profile_id in profile_ids:
                with self.subTest(profile_id=profile_id):
                    config = yaml.safe_load(original)
                    config["generation"]["subject_mode"] = "single"
                    config["generation"]["creative_profile"]["style"].update(mode="fixed", fixed=profile_id)
                    path.write_text(yaml.safe_dump(config), encoding="utf-8")
                    selection = {
                        "mode": "group", "subject_mode": "single", "selected_character": "MetaKnight",
                        "selected_profile": {"role_description": "a masked knight", "keywords": "mask, sword, cape"},
                    }
                    payload = build_goal_payload_from_character_config(make_character_workflow_request(
                        self.repo_root, path, prompt="Kirby protects one glowing mechanism",
                        preferred_generation_type="native_h3_story", selected_character_name="MetaKnight",
                        character_selection=selection, publish_after_generate=False,
                    ))
                    style = payload["constraints"]["visual_style_profile"]
                    self.assertEqual(style["id"], profile_id)
                    self.assertNotIn("MetaKnight", " ".join(str(value) for value in style.values()))
                    brief = payload["constraints"]["native_h3_creative_brief"]
                    self.assertIn("MetaKnight", brief)
                    self.assertIn("a masked knight", brief)
                    self.assertNotIn("Kirby protects", brief)

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
        self.assertIn(
            constraints["visual_style_profile"]["id"],
            set(StyleLibrary().profiles()),
        )

    def test_user_given_a_selected_style_when_brief_is_built_then_its_visual_rules_are_carried_through(
        self,
    ) -> None:
        """User: Given a selected reference style, When an image or video brief is built, Then its medium, composition, palette, motion, and exclusions reach the generated prompt contract."""
        profile = {
            "label": "bold hand-inked cartoon and comic illustration",
            "prompt": "variable-weight ink outlines and crisp cel shading",
            "composition": "one clear action with a readable silhouette",
            "palette": "saturated coral, cyan, and yellow with controlled contrast",
            "motion": "stable linework with one expressive anticipation and follow-through",
            "avoid": "fake text, logos, interface overlays",
        }

        for media_type in ("image", "text2img2video"):
            with self.subTest(media_type=media_type):
                goal = GoalRequest(
                    prompt="A small hero catches a falling lantern",
                    media_type=media_type,
                    style=profile["label"],
                    constraints={
                        "visual_style_profile": profile,
                        "negative_prompt": "bad anatomy",
                    },
                )

                brief = build_goal_brief(goal, profile["label"], [])

                self.assertIn(profile["prompt"], brief["prompt"])
                self.assertIn(profile["composition"], brief["prompt"])
                self.assertIn(profile["palette"], brief["prompt"])
                if media_type == "image":
                    self.assertNotIn(profile["motion"], brief["prompt"])
                else:
                    self.assertIn(profile["motion"], brief["prompt"])
                self.assertIn(
                    "fake text, logos, interface overlays", brief["negative_prompt"]
                )
                self.assertIn("bad anatomy", brief["negative_prompt"])
                if media_type == "text2img2video":
                    self.assertIn(profile["prompt"], brief["opening_keyframe_prompt"])

    def test_user_selects_a_reference_style_when_renderer_prompt_is_built_then_style_rules_precede_scene_details(
        self,
    ) -> None:
        """User Given a selected reference style and a scene with conflicting background details When the final renderer prompt is built Then the style rules appear before the scene details."""
        pastel = {"label": "pastel illustration", "prompt": "thin warm-gray pencil outlines and pale gouache fills"}
        scene_prompt = "Kirby kneels beside the pot while a leafy garden fills the background."
        prompt_with_existing_direction = "\n".join(
            (
                scene_prompt,
                visual_style_profile_direction(pastel, include_motion=False),
            )
        )
        goal = GoalRequest(
            prompt=scene_prompt,
            media_type="image",
            style=pastel["label"],
            constraints={"visual_style_profile": pastel},
        )

        renderer_prompt = include_visual_style_profile(prompt_with_existing_direction, goal)

        self.assertLess(
            renderer_prompt.index("Selected visual style rules"),
            renderer_prompt.index("Kirby kneels"),
        )
        self.assertEqual(
            renderer_prompt.count(visual_style_profile_direction(pastel, include_motion=False)),
            1,
        )

    def test_user_given_still_image_when_brief_is_built_then_motion_only_style_rules_are_reserved_for_video(
        self,
    ) -> None:
        """User: Given a style profile with motion guidance, When a still or video brief is built, Then only the video brief contains motion-only direction."""
        profile = {
            "label": "bold hand-inked cartoon and comic illustration",
            "prompt": "confident variable-weight ink lines",
            "composition": "one clear physical action",
            "palette": "bright coral and cyan accents",
            "motion": "in video, preserve linework through anticipation and follow-through",
            "avoid": "fake text and interface overlays",
        }

        image_goal = GoalRequest(
            prompt="Kirby catches a falling flower pot.",
            media_type="image",
            style=profile["label"],
            constraints={"visual_style_profile": profile},
        )
        video_goal = GoalRequest(
            prompt="Kirby catches a falling flower pot.",
            media_type="text2img2video",
            style=profile["label"],
            constraints={"visual_style_profile": profile},
        )

        image_brief = build_goal_brief(image_goal, profile["label"], [])
        video_brief = build_goal_brief(video_goal, profile["label"], [])

        self.assertNotIn(profile["motion"], image_brief["prompt"])
        self.assertIn(profile["motion"], video_brief["prompt"])

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
