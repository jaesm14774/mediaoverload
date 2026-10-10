from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agentic.app.character_workflow import build_goal_payload_from_character_config
from character_workflow_helpers import make_character_workflow_request
from agentic.app.main import build_runtime
from agentic.runtime.llm_engine import LLMPromptEngine, PromptGenerationError
from agentic.runtime.model_backends import OpenRouterModelCatalog
from agentic.runtime.contracts import RunState
from agentic.skills.agent_primitives import AgentMediaSkills
from agentic.skills.longvideo import LongVideoSkills
from agentic.storyboard import (
    format_native_h3_keyframe_prompt,
    format_native_h3_prompt,
    load_storyboard,
    merge_native_h3_storyboard,
)
from agentic.tools.context_services import NewsContextService


class NativeH3StoryPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.config_path = cls.repo_root / "configs" / "characters" / "kirby.yaml"

    def test_character_route_builds_native_h3_graph_without_power_shell_entrypoint(self) -> None:
        payload = build_goal_payload_from_character_config(make_character_workflow_request(
            self.repo_root,
            self.config_path,
            prompt="Kirby follows a storm-lit star through the meadow",
            preferred_generation_type="native_h3_story",
            publish_after_generate=False,
        ))
        planner, _runner, _memory = build_runtime(
            self.repo_root,
            output_root=self.repo_root / ".tmp-tests" / "native-h3-plan",
            comfy_host="127.0.0.1",
            comfy_port=8188,
        )
        goal = planner.create_goal(
            prompt=payload["prompt"],
            media_type=payload["media_type"],
            duration_seconds=payload["duration_seconds"],
            style=payload["style"],
            auto_download_assets=False,
            constraints=payload["constraints"],
        )
        plan = planner.build_plan(goal)

        self.assertEqual(goal.media_type, "native_h3_story")
        self.assertEqual(goal.duration_seconds, 15)
        self.assertEqual(plan.workflow_name, "wan2gp_h3_i2va")
        node_ids = [node.node_id for node in plan.nodes]
        expected_node_ids = [
            "native-story-prompt",
            "native-image-asset-check",
            "native-video-asset-check",
            "native-opening-keyframe",
            "native-opening-review",
            "native-keyframe-source-check",
            "native-h3-render",
            "native-h3-speed",
            "native-h3-qa",
            "native-h3-preview",
            "native-h3-package",
        ]
        if "native-h3-canvas" in node_ids:
            expected_node_ids.insert(8, "native-h3-canvas")
        self.assertEqual(
            node_ids,
            expected_node_ids,
        )
        opening = next(node for node in plan.nodes if node.node_id == "native-opening-keyframe")
        opening_review = next(node for node in plan.nodes if node.node_id == "native-opening-review")
        keyframe_source_check = next(node for node in plan.nodes if node.node_id == "native-keyframe-source-check")
        render = next(node for node in plan.nodes if node.node_id == "native-h3-render")
        qa = next(node for node in plan.nodes if node.node_id == "native-h3-qa")
        story_prompt = next(node for node in plan.nodes if node.node_id == "native-story-prompt")
        self.assertEqual(opening.inputs["image_count"], 6)
        self.assertEqual(opening_review.inputs["review_all_candidates"], True)
        self.assertEqual(opening_review.inputs["review_scope"], "first_frame")
        self.assertEqual(opening_review.inputs["review_notes"], "stage: preview")
        self.assertEqual(opening_review.depends_on, ["native-opening-keyframe"])
        self.assertEqual(keyframe_source_check.inputs["opening_node"], "native-opening-review")
        self.assertTrue(keyframe_source_check.inputs["preserve_opening_frame"])
        self.assertFalse(keyframe_source_check.inputs["use_last_frame"])
        self.assertEqual(keyframe_source_check.depends_on, ["native-opening-review"])
        self.assertEqual(keyframe_source_check.inputs["max_regenerations"], 0)
        self.assertEqual(keyframe_source_check.skill_name, "media.image.confirm_keyframe_sources")
        self.assertEqual(keyframe_source_check.stage, "assets")
        self.assertNotIn("identity", keyframe_source_check.tags)
        self.assertIn("native-opening-review", keyframe_source_check.depends_on)
        self.assertIn("native-keyframe-source-check", render.depends_on)
        self.assertFalse(render.inputs["use_last_frame"])
        self.assertEqual(story_prompt.inputs["render_mode"], "image_to_video")
        self.assertEqual(qa.inputs["mode"], "discord_review_evidence_only")
        self.assertEqual(qa.inputs["video_node"], "native-h3-canvas" if "native-h3-canvas" in node_ids else "native-h3-speed")
        self.assertAlmostEqual(qa.inputs["target_duration"], render.inputs["length"] / 24 / 2)
        self.assertEqual(qa.inputs["expected_fps"], 24.0)
        self.assertIsNone(qa.tool_name)
        self.assertEqual(plan.metadata["native_h3"]["keyframe_candidate_count"], 6)
        self.assertFalse(plan.metadata["native_h3"]["use_last_frame"])
        self.assertIn("native_h3", plan.metadata)

    def test_user_gets_reproducible_opening_conditioned_ending_for_fl2va(self) -> None:
        """User Given the same FL2VA story and seed When the planner builds either A/B arm Then only the treatment reuses the opening image for its ending anchor."""
        planner, _runner, _memory = build_runtime(
            self.repo_root,
            output_root=self.repo_root / ".tmp-tests" / "native-h3-paired-anchor-plan",
            comfy_host="127.0.0.1",
            comfy_port=8188,
        )
        common_constraints = {
            "character": "Kirby",
            "native_h3_use_last_frame": True,
            "native_h3_keyframe_candidate_count": 1,
            "native_h3_keyframe_workflow_name": "krea2_turbo",
            "workflow_name": "wan2gp_h3_fl2va",
            "storyboard_path": "configs/storyboards/native_h3_15s.yaml",
            "require_human_review": False,
            "enable_review_loop": False,
            "native_h3_experiment_seed": 730927,
        }

        def make_plan(strategy: str):
            goal = planner.create_goal(
                prompt="Kirby draws one small star that comes alive and surprises him.",
                media_type="native_h3_fl2va_story",
                duration_seconds=15,
                style="hand-drawn paper storybook animation",
                auto_download_assets=False,
                constraints={**common_constraints, "native_h3_frame_pair_strategy": strategy},
            )
            return planner.build_plan(goal)

        treatment = make_plan("opening_conditioned_image_to_image")
        control = make_plan("independent_text_to_image")
        treatment_nodes = {node.node_id: node for node in treatment.nodes}
        control_nodes = {node.node_id: node for node in control.nodes}
        treatment_ending = treatment_nodes["native-ending-keyframe"]
        control_ending = control_nodes["native-ending-keyframe"]

        self.assertTrue(treatment_ending.inputs["use_prior_frame"])
        self.assertEqual(treatment_ending.inputs["identity_refine_workflow_name"], "krea2_turbo_img2img")
        self.assertEqual(treatment_ending.inputs["denoise"], 0.5)
        self.assertEqual(treatment_ending.inputs["seed"], 730928)
        self.assertFalse(control_ending.inputs["use_prior_frame"])
        self.assertEqual(control_ending.inputs["seed"], treatment_ending.inputs["seed"])
        self.assertTrue(
            any(node.node_id == "native-ending-image-refine-asset-check" for node in treatment.nodes)
        )
        self.assertFalse(
            any(node.node_id == "native-ending-image-refine-asset-check" for node in control.nodes)
        )
        for plan in (treatment, control):
            render = next(node for node in plan.nodes if node.node_id == "native-h3-render")
            opening = next(node for node in plan.nodes if node.node_id == "native-opening-keyframe")
            self.assertEqual(opening.inputs["seed"], 730927)
            self.assertEqual(render.inputs["seed"], 730929)
            self.assertTrue(render.inputs["use_last_frame"])

    def test_native_h3_last_frame_mode_adds_review_and_passes_both_frames(self) -> None:
        """User Given FL2VA needs editorial review When the planner creates the ending Then it refines the opening into reviewed candidates before rendering both anchors."""
        payload = build_goal_payload_from_character_config(make_character_workflow_request(
            self.repo_root,
            self.config_path,
            prompt="Kirby protects one glowing seed as a sudden storm reshapes the meadow",
            preferred_generation_type="native_h3_story",
            publish_after_generate=False,
        ))
        payload["constraints"]["native_h3_use_last_frame"] = True
        planner, _runner, _memory = build_runtime(
            self.repo_root,
            output_root=self.repo_root / ".tmp-tests" / "native-h3-last-frame-plan",
            comfy_host="127.0.0.1",
            comfy_port=8188,
        )
        goal = planner.create_goal(
            prompt=payload["prompt"],
            media_type=payload["media_type"],
            duration_seconds=payload["duration_seconds"],
            style=payload["style"],
            auto_download_assets=False,
            constraints=payload["constraints"],
        )
        plan = planner.build_plan(goal)

        node_ids = [node.node_id for node in plan.nodes]
        self.assertIn("native-ending-keyframe", node_ids)
        self.assertIn("native-ending-review", node_ids)
        ending_review = next(node for node in plan.nodes if node.node_id == "native-ending-review")
        ending = next(node for node in plan.nodes if node.node_id == "native-ending-keyframe")
        source_check = next(node for node in plan.nodes if node.node_id == "native-keyframe-source-check")
        render = next(node for node in plan.nodes if node.node_id == "native-h3-render")
        self.assertEqual(ending_review.inputs["review_scope"], "last_frame")
        self.assertEqual(ending.inputs["image_count"], 6)
        self.assertTrue(ending.inputs["use_prior_frame"])
        self.assertEqual(ending.inputs["identity_refine_workflow_name"], "krea2_turbo_img2img")
        self.assertIn(
            "native-ending-image-refine-asset-check",
            node_ids,
        )
        self.assertEqual(ending_review.depends_on, ["native-ending-keyframe"])
        self.assertTrue(source_check.inputs["use_last_frame"])
        self.assertTrue(source_check.inputs["preserve_ending_frame"])
        self.assertEqual(source_check.inputs["ending_node"], "native-ending-review")
        self.assertIn("native-ending-review", source_check.depends_on)
        self.assertTrue(render.inputs["use_last_frame"])
        self.assertTrue(plan.metadata["native_h3"]["use_last_frame"])

    def test_no_review_native_h3_uses_one_opening_candidate_and_skips_review_node(self) -> None:
        payload = build_goal_payload_from_character_config(make_character_workflow_request(
            self.repo_root,
            self.config_path,
            prompt="Kirby protects one glowing seed as a sudden storm reshapes the meadow",
            preferred_generation_type="native_h3_story",
            publish_after_generate=False,
            no_review=True,
        ))
        planner, _runner, _memory = build_runtime(
            self.repo_root,
            output_root=self.repo_root / ".tmp-tests" / "native-h3-no-review-plan",
            comfy_host="127.0.0.1",
            comfy_port=8188,
        )
        goal = planner.create_goal(
            prompt=payload["prompt"],
            media_type=payload["media_type"],
            duration_seconds=payload["duration_seconds"],
            style=payload["style"],
            auto_download_assets=False,
            constraints=payload["constraints"],
        )
        plan = planner.build_plan(goal)
        opening = next(node for node in plan.nodes if node.node_id == "native-opening-keyframe")

        self.assertEqual(opening.inputs["image_count"], 1)
        self.assertNotIn("native-opening-review", [node.node_id for node in plan.nodes])
        self.assertFalse(plan.metadata["native_h3"]["require_human_review"])

    def test_stage_probe_keeps_six_candidates_and_adds_explicit_auto_selection_node(self) -> None:
        payload = build_goal_payload_from_character_config(make_character_workflow_request(
            self.repo_root,
            self.config_path,
            preferred_generation_type="native_h3_story",
            publish_after_generate=False,
            stage_probe=True,
        ))
        planner, _runner, _memory = build_runtime(
            self.repo_root,
            output_root=self.repo_root / ".tmp-tests" / "native-h3-stage-probe-plan",
            comfy_host="127.0.0.1",
            comfy_port=8188,
        )
        goal = planner.create_goal(
            prompt=payload["prompt"],
            media_type=payload["media_type"],
            duration_seconds=payload["duration_seconds"],
            style=payload["style"],
            auto_download_assets=False,
            constraints=payload["constraints"],
        )
        plan = planner.build_plan(goal)

        opening = next(node for node in plan.nodes if node.node_id == "native-opening-keyframe")
        review = next(node for node in plan.nodes if node.node_id == "native-opening-review")
        source_check = next(node for node in plan.nodes if node.node_id == "native-keyframe-source-check")
        self.assertEqual(opening.inputs["image_count"], 6)
        self.assertTrue(review.inputs["auto_select_for_probe"])
        self.assertFalse(review.inputs["require_human_review"])
        self.assertEqual(source_check.inputs["opening_node"], "native-opening-review")
        self.assertFalse(source_check.inputs["preserve_opening_frame"])
        self.assertTrue(plan.metadata["native_h3"]["stage_probe_auto_select"])

    def test_user_given_news_story_when_native_h3_keyframes_are_prepared_then_opening_and_ending_show_only_their_own_story_beat(self) -> None:
        """User Given a selected news story with three visual beats When Native H3 prepares its keyframes Then source facts stay in the story prompt and each drawing prompt shows only its own visible beat."""
        captured: dict[str, object] = {}
        arc_guidance = "Show the same instruction card before and after migration; let the character reaction carry the corrected headline's relief."
        storyboard_fixture = load_storyboard(self.repo_root / "configs/storyboards/native_h3_15s.yaml")
        storyboard_fixture.update(
            {
                "character": "Kirby, Bandana Waddle Dee",
                "subject_context": {
                    "subjects": [
                        {"name": "Kirby", "role": "primary", "profile": {"role_description": "a round pink hero with expressive full-body gestures"}},
                        {"name": "Bandana Waddle Dee", "role": "secondary", "profile": {"role_description": "a warm friend who answers other characters' actions"}},
                    ]
                },
                "world": {
                    "setting": "a warm star meadow",
                    "visual_language": "tactile storybook watercolor with expressive faces",
                },
                "native_shots": [
                    {"action": "Kirby holds one plain blank instruction card below his face while Bandana Waddle Dee leans toward it with curious eyes."},
                    {"action": "Google automatically migrates the saved instructions while the same plain card stays in place and usable."},
                    {"action": "Kirby interrupts a solemn little goodbye with a relieved laugh beside the unchanged card; Bandana Waddle Dee lifts one hand and leans closer."},
                ],
                "news_trace": {
                    "source_category": "product_or_service",
                    "source_title": "Google 11月淘汰Gems，現有設定轉為Skills",
                    "source_fact": "Google says existing Gems will automatically become reusable Skills from November 17, 2026.",
                    "visual_anchors": ["one plain blank instruction card representing saved instructions"],
                    "visual_translation": "the same plain instruction card represents saved instructions before and after the automatic migration",
                    "news_mechanism": "existing saved instructions migrate automatically to Skills; Gems remain usable until the migration date",
                    "news_consequence": "users can keep using their saved instructions after the migration",
                    "source_limit": "The card and character farewell are fictional; the article reports a service migration, not user emotions.",
                },
            }
        )

        class FakeStoryService:
            def resolve(self, _base_storyboard: dict[str, object], **kwargs: object) -> tuple[dict[str, object], dict[str, object]]:
                captured.update(kwargs)
                return dict(storyboard_fixture), {
                    "prompt_mode": "llm",
                    "source": "news",
                    "creative_seed": "news-seed",
                    "news_context": kwargs["news_context"],
                }

        context = SimpleNamespace(
            plan=SimpleNamespace(
                goal=SimpleNamespace(
                    prompt="Autonomous scene generated from the selected news",
                    style="polished 2D anime",
                    constraints={
                        "character": "Kirby, Bandana Waddle Dee",
                        "subject_context": storyboard_fixture["subject_context"],
                        "prompt_source": "news",
                        "native_h3_creative_brief": "compact causal story with one prop and a visible payoff",
                        "native_h3_arc_instruction": arc_guidance,
                        "news_context": {"title": "Google 11月淘汰Gems，現有設定轉為Skills", "keyword": "Google;Gemini;Gems;Skills"},
                    },
                )
            ),
            node=SimpleNamespace(
                inputs={
                    "storyboard_path": "configs/storyboards/native_h3_15s.yaml",
                    "duration_seconds": 15,
                    "style": "polished 2D anime",
                }
            ),
        )
        result = LongVideoSkills(
            tools=SimpleNamespace(),
            output_root=self.repo_root / ".tmp-tests" / "news-only-native-h3",
            story_service=FakeStoryService(),
        ).prepare_native_h3_story(context)

        self.assertEqual(result.status, "success")
        self.assertEqual(captured["creative_brief"], "compact causal story with one prop and a visible payoff")
        self.assertEqual(captured["arc_guidance"], arc_guidance)
        opening_prompt = str(result.outputs["opening_keyframe_prompt"])
        ending_prompt = str(result.outputs["ending_keyframe_prompt"])
        self.assertIn("Kirby", opening_prompt)
        self.assertIn("Bandana Waddle Dee", opening_prompt)
        self.assertIn("warm star meadow", opening_prompt)
        self.assertIn("one plain blank instruction card", opening_prompt)
        self.assertNotIn("gemstone", opening_prompt.casefold())
        self.assertNotIn("Google says existing Gems", str(result.outputs["prompt"]))
        self.assertIn("Google automatically migrates the saved instructions", str(result.outputs["prompt"]))
        self.assertNotIn("Google says existing Gems", opening_prompt)
        self.assertIn("one storybook still", opening_prompt.lower())
        self.assertNotIn("changes into callable form", opening_prompt)
        self.assertIn("Kirby", ending_prompt)
        self.assertIn("Bandana Waddle Dee", ending_prompt)
        self.assertIn("relieved laugh beside the unchanged card", ending_prompt)
        self.assertIn("News visual cue: one plain blank instruction card representing saved instructions is part of the held pose", ending_prompt)
        self.assertNotIn("cheek", ending_prompt.casefold())
        self.assertEqual(ending_prompt.count("News visual cue:"), 1)
        self.assertNotIn("gold Skill star", ending_prompt)

    def test_user_given_h3_template_when_loaded_then_prompt_comes_from_the_current_story(self) -> None:
        """User Given a WanGP H3 template When loaded Then it contains no stale example scene prompt."""
        workflow_path = self.repo_root / "configs" / "workflow" / "wan2gp_h3_fl2va.json"
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        self.assertEqual(workflow["provider"], "wan2gp")
        self.assertNotIn("prompt", workflow)

    def test_human_selected_opening_frame_is_immutable(self) -> None:
        class FakeTools:
            def call(self, tool_name: str, payload: dict[str, object]) -> dict[str, object]:
                raise AssertionError(f"unexpected automatic regeneration: {tool_name}")

        temp_root = self.repo_root / ".tmp-tests" / "immutable-opening"
        temp_root.mkdir(parents=True, exist_ok=True)
        opening_path = temp_root / "selected-asset-2.png"
        ending_path = temp_root / "ending.png"
        opening_path.write_bytes(b"human-selected-opening")
        ending_path.write_bytes(b"generated-ending")
        context = SimpleNamespace(
            state=SimpleNamespace(
                node_outputs={
                    "native-opening-review": {"selected_assets": [str(opening_path)]},
                    "native-ending-keyframe": {"saved_files": [str(ending_path)]},
                    "native-story-prompt": {},
                }
            ),
            node=SimpleNamespace(
                inputs={
                    "opening_node": "native-opening-review",
                    "ending_node": "native-ending-keyframe",
                    "character": "",
                    "preserve_opening_frame": True,
                    "use_last_frame": True,
                    "preserve_ending_frame": True,
                    "max_regenerations": 0,
                },
                depends_on=["native-opening-review", "native-ending-keyframe"],
            ),
            plan=SimpleNamespace(goal=SimpleNamespace(prompt="immutable opening", constraints={})),
        )

        result = AgentMediaSkills(FakeTools(), temp_root).confirm_keyframe_source_files(context)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.outputs["first_frame_path"], str(opening_path))
        self.assertEqual(result.outputs["regenerated_count"], 0)
        self.assertEqual(result.outputs["source_checks"][0]["check"], "file_exists")
        self.assertEqual(result.outputs["identity_check"], "not_applied")

    def test_user_given_creative_direction_when_native_prompt_is_formatted_then_current_h3_brief_and_duration_are_preserved(self) -> None:
        """User Given a scene based on creative direction, When the Native H3 prompt is formatted, Then scene and duration remain without writing instructions."""
        storyboard = load_storyboard(self.repo_root / "configs/storyboards/native_h3_15s.yaml")
        storyboard["native_shots"] = [{"action": "Kirby runs through a blue-hour summer storm.", "camera": "The camera tracks beside Kirby."}]
        prompt = format_native_h3_prompt(
            storyboard,
            creative_brief="a blue-hour summer storm",
            duration_seconds=15,
        )
        self.assertIn("Kirby runs through a blue-hour summer storm", prompt)
        self.assertIn("camera tracks beside Kirby", prompt)
        self.assertIn("Duration: 15 seconds.", prompt)
        self.assertNotIn("contract", prompt.lower())
        self.assertNotIn("Creative variation for this run", prompt)
        self.assertNotIn("Follow the supplied creative brief", prompt)

    def test_user_given_comedy_card_when_native_prompt_is_formatted_then_character_specific_beats_reach_video_prompt(self) -> None:
        """User Given a comedy card and its final shot description, When formatted, Then H3 receives the visible gag while the editorial card stays in the story record."""
        storyboard = load_storyboard(self.repo_root / "configs/storyboards/native_h3_15s.yaml")
        storyboard["gag_card"] = {
            "character_signature": "Kirby inhales and copies the cushion's bounce.",
            "character_desire": "Kirby wants one soft nap",
            "source_prop": "one runaway cushion",
            "prop_rule": "The cushion springs away whenever Kirby lands on it",
            "expectation": "Kirby expects the cushion to stay put for one nap.",
            "physical_escalation": "Each bounce sends Kirby farther across the meadow.",
            "surprising_harmless_reversal": "The cushion reverses its bounce and hugs Kirby.",
            "held_expressive_reaction": "Kirby freezes wide-eyed, then puffs his cheeks.",
        }
        storyboard["native_shots"] = [{
            "action": "The cushion reverses its bounce and hugs Kirby. Kirby freezes wide-eyed, then puffs his cheeks.",
            "camera": "A close view holds on Kirby's face.",
        }]

        prompt = format_native_h3_prompt(storyboard, duration_seconds=15)

        self.assertIn("The cushion reverses its bounce and hugs Kirby", prompt)
        self.assertIn("Kirby freezes wide-eyed, then puffs his cheeks", prompt)
        self.assertIn("close view holds on Kirby's face", prompt)
        self.assertNotIn("Comic staging card", prompt)
        self.assertNotIn("Prop rule:", prompt)
        self.assertEqual(storyboard["gag_card"]["prop_rule"], "The cushion springs away whenever Kirby lands on it")
        self.assertNotIn("contract", prompt.lower())

    def test_user_given_generated_world_without_continuity_when_story_is_merged_then_base_rules_are_preserved(self) -> None:
        """User Given a generated world without continuity rules, When merged, Then the rules remain in the story record while H3 receives the actual scene."""
        base_storyboard = {
            "native_duration_seconds": 15,
            "base_prompt": "Kirby follows a firefly.",
            "world": {
                "setting": "a quiet meadow",
                "continuity_rules": ["Keep the same garden path throughout."],
            },
        }
        generated_story = {
            "world": {"setting": "a moonlit garden"},
            "native_shots": [{"action": "Kirby follows a firefly along the path."}],
        }

        merged = merge_native_h3_storyboard(base_storyboard, generated_story)

        self.assertEqual(merged["world"]["setting"], "a moonlit garden")
        self.assertEqual(
            merged["world"]["continuity_rules"],
            ["Keep the same garden path throughout."],
        )
        prompt = format_native_h3_prompt(merged)
        self.assertNotIn("same garden path throughout", prompt.lower())
        self.assertIn("Kirby follows a firefly along the path", prompt)
        self.assertIn("moonlit garden", prompt)
        self.assertEqual(
            base_storyboard["world"]["continuity_rules"],
            ["Keep the same garden path throughout."],
        )

    def test_native_gag_card_is_preserved_when_story_is_merged(self) -> None:
        base_storyboard = load_storyboard(self.repo_root / "configs/storyboards/native_h3_15s.yaml")
        generated_story = {
            "name": "Kirby and the Runaway Cushion",
            "base_prompt": "One Kirby, polished 2D anime, squash-and-stretch comedy motion.",
            "opening_keyframe_prompt": "Kirby is already dragged sideways by a runaway cushion.",
            "ending_keyframe_prompt": "The cushion rebounds and hugs Kirby in a soft pile.",
            "negative_prompt": "humans, duplicate Kirby, readable text, watermark",
            "news_trace": {
                "contract_version": 2,
                "source_title": "cushion delivery delay",
                "source_concepts": ["cushion"],
                "visual_translation": "The delay becomes a runaway cushion that refuses to arrive calmly.",
                "visual_anchors": ["runaway cushion"],
                "integration": "The cushion's strange movement creates Kirby's one simple physical problem.",
            },
            "gag_card": {
                "character_signature": "Kirby copies the cushion's bounce by inhaling it.",
                "character_desire": "Kirby wants to settle onto the cushion for a nap.",
                "source_prop": "one runaway cushion",
                "prop_rule": "The cushion springs away whenever Kirby lands on it.",
                "expectation": "Kirby expects the cushion to stay put.",
                "physical_escalation": "Each landing sends him farther away.",
                "surprising_harmless_reversal": "The cushion reverses and hugs Kirby.",
                "held_expressive_reaction": "Kirby freezes wide-eyed and puffs his cheeks.",
            },
            "story_spine": {
                "premise": "A runaway cushion keeps escaping Kirby's nap.",
                "objective": "Kirby must catch the cushion and settle onto it.",
                "obstacle": "The cushion springs away whenever he lands.",
                "stakes": "Kirby loses his one chance for a cozy nap.",
                "emotional_arc": "sleepy confidence becomes surprise and delighted relief",
                "climax": "Kirby stops chasing and lets the cushion bounce into him.",
                "resolution": "The cushion hugs Kirby and the nap finally begins.",
            },
            "world": {
                "setting": "a sunny meadow with one oversized pastel cushion",
                "visual_language": "bright candy colors, soft rounded shapes, playful squash-and-stretch",
                "continuity_rules": ["Keep the same oversized cushion visible across all beats."],
            },
            "native_audio": "soft boings, one surprised squeak, then a warm sleepy chime",
            "native_shots": [
                {"time": "0-4s", "title": "Cushion escapes", "action": "The cushion springs away and drags Kirby sideways while he reaches for it.", "camera": "Push in and follow the sideways slide.", "state_change": "Kirby commits to catching the runaway cushion."},
                {"time": "4-10s", "title": "Cushion wins", "action": "Kirby lands on the cushion, but it rebounds and tumbles him through the grass.", "camera": "Track the bounce into a close reaction shot.", "state_change": "Kirby loses his nap and stops chasing the cushion."},
                {"time": "10-15s", "title": "Cushion hugs back", "action": "The cushion rebounds into Kirby's arms and wraps him in a soft hug.", "camera": "Pull out as the tumble settles into a cozy close-up.", "state_change": "Kirby gets the nap he wanted and the cushion is safely with him."},
            ],
        }

        merged = merge_native_h3_storyboard(base_storyboard, generated_story)

        self.assertEqual(merged["gag_card"]["prop_rule"], generated_story["gag_card"]["prop_rule"])
        prompt = format_native_h3_prompt(merged, duration_seconds=15)
        self.assertIn("runaway cushion", prompt)
        self.assertIn("cushion hugs back", prompt.lower())

        pair_base = dict(base_storyboard)
        pair_base["subject_context"] = {
            "subjects": [
                {"role": "primary", "name": "Kirby"},
                {"role": "secondary", "name": "Kirby"},
            ],
            "interaction_contract": {"required": True, "same_frame": True},
        }
        pair_merged = merge_native_h3_storyboard(pair_base, generated_story)
        self.assertEqual(pair_merged["world"]["continuity_rules"], generated_story["world"]["continuity_rules"])
        self.assertEqual(pair_merged["negative_prompt"], "")

    def test_character_route_builds_direct_t2v_story_graph(self) -> None:
        payload = build_goal_payload_from_character_config(make_character_workflow_request(
            self.repo_root,
            self.config_path,
            prompt="Kirby must save one glowing seed before the storm swallows the garden",
            preferred_generation_type="native_h3_t2v_story",
            publish_after_generate=False,
        ))
        planner, _runner, _memory = build_runtime(
            self.repo_root,
            output_root=self.repo_root / ".tmp-tests" / "native-h3-t2v-plan",
            comfy_host="127.0.0.1",
            comfy_port=8188,
        )
        goal = planner.create_goal(
            prompt=payload["prompt"],
            media_type=payload["media_type"],
            duration_seconds=payload["duration_seconds"],
            style=payload["style"],
            auto_download_assets=False,
            constraints=payload["constraints"],
        )
        plan = planner.build_plan(goal)

        self.assertEqual(goal.media_type, "native_h3_t2v_story")
        self.assertEqual(goal.duration_seconds, 15)
        self.assertEqual(plan.workflow_name, "wan2gp_h3_t2va")
        render = next(node for node in plan.nodes if node.node_id == "native-h3-render")
        self.assertEqual(render.tool_name, "wan2gp.render_h3")
        self.assertEqual(plan.metadata["recipe"], "native_h3_t2v_story")
        self.assertEqual(plan.metadata["native_h3"]["length"], 362)
        self.assertEqual(plan.metadata["native_h3"]["backend"], "wan2gp")
        self.assertEqual(plan.metadata["native_h3"]["steps"], 16)
        self.assertNotIn("native-opening-keyframe", [node.node_id for node in plan.nodes])
        self.assertNotIn("native-opening-review", [node.node_id for node in plan.nodes])

    def test_user_given_described_scene_when_native_prompt_is_formatted_then_action_and_camera_are_preserved(self) -> None:
        """User Given a final scene description, When the native prompt is formatted, Then its action and camera remain without appending a separate creative brief."""
        storyboard = load_storyboard(self.repo_root / "configs/storyboards/native_h3_15s.yaml")
        storyboard["native_shots"] = [{
            "action": "Kirby floats through a digital archive, inhales red shards, and transforms them into stars.",
            "camera": "The camera performs an orbital rotation in a 3D void.",
        }]
        prompt = format_native_h3_prompt(
            storyboard,
            creative_brief=(
                "Kirby floats through a digital archive, inhales red shards, and transforms them into stars "
                "while the camera performs an orbital rotation in a 3D void."
            ),
        )
        self.assertIn("digital archive", prompt)
        self.assertIn("red shards", prompt)
        self.assertIn("orbital rotation", prompt)

    def test_native_h3_story_is_generated_from_news_and_replaces_fixed_plot(self) -> None:
        base_storyboard = load_storyboard(self.repo_root / "configs/storyboards/native_h3_15s.yaml")
        generated_story = {
            "name": "Kirby and the Lantern Current",
            "base_prompt": "Kirby is the only protagonist in a moonlit canal city where floating lanterns drift against the tide.",
            "opening_keyframe_prompt": "Opening frame: Kirby stands on a canal bridge as one runaway lantern pulls a ribbon of light into a whirlpool below.",
            "ending_keyframe_prompt": "Ending frame: Kirby rests beside the calm canal while the rescued lanterns form a warm constellation above the water.",
            "negative_prompt": "humans, extra characters, duplicate Kirby, text, watermark, hard cut, identity drift",
            "news_trace": {
                "contract_version": 2,
                "source_title": "city lantern outage paraphrased by provider",
                "source_concepts": ["lantern"],
                "visual_translation": "The outage becomes a runaway city lantern pulled toward a canal whirlpool.",
                "news_mechanism": "the runaway lantern is pulled toward the canal whirlpool",
                "news_consequence": "the lantern rises above the canals after the current reverses",
                "visual_anchors": ["lantern", "whirlpool", "canals"],
                "anchor_roles": ["context", "mechanism", "consequence"],
                "integration": "Kirby protects the lantern while the outage makes the lantern the urgent mission.",
            },
            "story_spine": {
                "premise": "A sudden current is dragging the city's guiding lanterns into a dark canal whirlpool.",
                "objective": "Kirby must redirect the runaway lantern before the city loses its safe path home.",
                "obstacle": "The current is accelerating and the lantern cable is about to snap.",
                "stakes": "Without the lantern, the canal city will be swallowed by darkness.",
                "emotional_arc": "curiosity becomes urgency, courage, and relief",
                "climax": "Kirby anchors the lantern against the current and sends its light back across the canals.",
                "resolution": "The lanterns settle into a warm constellation and the canal becomes safe again.",
            },
            "world": {
                "setting": "a moonlit canal city with narrow bridges and reflective water",
                "visual_language": "indigo night, amber lantern light, soft watercolor anime edges",
                "continuity_rules": [
                    "The same canal bridge and lantern cable remain visible across the story.",
                    "The current always pulls from screen left toward the whirlpool on screen right.",
                ],
            },
            "native_audio": "water rush, cable tension, quick footsteps, one bright lantern chime, then calm night ambience",
            "native_shots": [
                {"time": "0-4s", "title": "Hook - the lantern is pulled away", "action": "A lantern tears loose and drags light toward the whirlpool while Kirby reaches for its cable.", "camera": "Wide bridge reveal pushing into Kirby and the moving lantern.", "state_change": "Kirby understands the lantern is the city's only safe guide and commits to stopping it."},
                {"time": "4-10s", "title": "Escalation - anchor the current", "action": "Kirby slides along the wet bridge and wraps the cable around a post as the current pulls harder.", "camera": "Smooth side track with the cable and whirlpool kept in the same geography.", "state_change": "The cable holds for a moment, but the bridge begins to crack under the force."},
                {"time": "10-15s", "title": "Payoff - light returns", "action": "Kirby releases a burst of warm light through the cable, the current reverses, and the lantern rises above the canals.", "camera": "Follow the light upward, then settle into a calm wide ending frame.", "state_change": "The lanterns form a safe constellation and the original danger is visibly resolved."},
            ],
        }
        engine = LLMPromptEngine(mode="llm")
        with patch.object(engine, "_require_manager", return_value=object()), patch.object(
            LLMPromptEngine, "_chat_json", return_value={"story": generated_story, "creative_seed": "lantern-current", "source": "native_h3_llm"}
        ), patch.object(engine, "backend_info", return_value={"provider": "test"}):
            result = engine.generate_native_h3_storyboard(
                character="Kirby",
                style="polished 2D anime",
                duration_seconds=15,
                base_storyboard=base_storyboard,
                news_context={"title": "city lantern outage", "keyword": "canal safety"},
            )

        merged = merge_native_h3_storyboard(base_storyboard, result["story"])
        prompt = format_native_h3_prompt(merged, style="polished 2D anime", duration_seconds=15)
        self.assertEqual(result["prompt_mode"], "llm")
        self.assertEqual(result["story"]["news_trace"]["source_title"], "city lantern outage")
        self.assertEqual(merged["name"], "Kirby and the Lantern Current")
        self.assertEqual(len(merged["native_shots"]), 3)
        self.assertNotIn("segments", merged)
        self.assertIn("lantern", prompt.lower())
        self.assertNotIn("golden star seed", prompt.lower())
        self.assertNotIn("dark sky rift", prompt.lower())

    def test_user_given_source_context_when_native_prompt_is_formatted_then_only_described_scene_reaches_h3(self) -> None:
        """User Given source context and a final scene, When formatted, Then H3 receives the described action rather than source explanations or role mappings."""
        storyboard = load_storyboard(self.repo_root / "configs/storyboards/native_h3_15s.yaml")
        storyboard["news_trace"] = {
            "visual_translation": "A city blackout closes a canal path.",
            "news_mechanism": "synchronized lights shut down and block the route",
            "news_consequence": "the route reopens when Kirby redirects the final beam",
            "visual_anchors": ["city lights", "blocked route", "warm exit"],
            "anchor_roles": ["context", "mechanism", "consequence"],
            "source_roles": ["Kirby: redirects the beam to reopen the path"],
            "character_mapping": {"Kirby": "the person who clears the path"},
        }
        storyboard["native_shots"] = [{"action": "Kirby redirects the lantern beam and reopens the canal path.", "camera": "The camera follows the beam toward the path."}]
        prompt = format_native_h3_prompt(storyboard, duration_seconds=15)
        self.assertIn("Kirby redirects the lantern beam and reopens the canal path", prompt)
        self.assertIn("camera follows the beam", prompt)
        self.assertNotIn("News grounding context", prompt)
        self.assertNotIn("character role mapping", prompt)
        self.assertNotIn("Preserve who did what to whom", prompt)
        self.assertNotIn("Additional source context", prompt)
        self.assertEqual(storyboard["news_trace"]["news_mechanism"], "synchronized lights shut down and block the route")

    def test_user_given_readable_text_visual_when_native_h3_story_is_generated_then_story_is_not_rejected(self) -> None:
        """User: Given a story mentions a readable document, When Native H3 prepares its render data, Then that creative detail does not block generation."""
        storyboard = load_storyboard(self.repo_root / "configs/storyboards/native_h3_15s.yaml")
        invalid_story = {
            "base_prompt": "Kirby reaches for a glowing document covered in financial symbols.",
            "native_shots": [
                {"time": "0-4s", "action": "Kirby runs toward the loose lantern."},
                {"time": "4-10s", "action": "Kirby catches the lantern cable."},
                {"time": "10-15s", "action": "Kirby anchors the lantern safely."},
            ],
        }
        calls: list[str] = []

        def fake_chat(request):
            calls.append(str(request.user_prompt))
            return {"story": invalid_story, "creative_seed": "lantern-current", "source": "native_h3_llm"}

        engine = LLMPromptEngine(mode="llm")
        with patch.object(engine, "_require_manager", return_value=object()), patch.object(
            LLMPromptEngine, "_chat_json", side_effect=fake_chat
        ), patch.object(engine, "backend_info", return_value={"provider": "test"}):
            result = engine.generate_native_h3_storyboard(
                character="Kirby",
                style="polished 2D anime",
                duration_seconds=15,
                base_storyboard=storyboard,
                news_context={"title": "city lantern outage", "keyword": "canal safety"},
            )

        self.assertEqual(len(calls), 3)
        self.assertIn("document", result["story"]["base_prompt"])

    def test_news_selection_rejects_placeholder_titles(self) -> None:
        self.assertFalse(NewsContextService.is_usable_selection("...", "gold;reserve"))
        self.assertFalse(NewsContextService.is_usable_selection("", "gold;reserve"))
        self.assertTrue(NewsContextService.is_usable_selection("Central bank changes reserve policy", "gold;reserve"))

    def test_user_given_story_missing_optional_spine_fields_when_generated_shots_are_merged_then_the_prompt_gets_grounded_fallbacks(self) -> None:
        """User Given optional spine fields are missing from an otherwise usable generated story When its shots are merged Then visible shot states provide grounded stakes and climax without another generation gate."""
        storyboard = load_storyboard(self.repo_root / "configs/storyboards/native_h3_15s.yaml")
        valid_story = {
            "name": "Kirby and the Lantern Current",
            "base_prompt": "Kirby is the only protagonist in a moonlit canal city with a runaway lantern.",
            "opening_keyframe_prompt": "Opening frame: Kirby reaches for a runaway lantern beside a canal whirlpool.",
            "ending_keyframe_prompt": "Ending frame: Kirby rests beside the calm canal as the rescued lantern rises above the water.",
            "negative_prompt": "humans, extra characters, duplicate Kirby, watermark, hard cuts",
            "news_trace": {
                "contract_version": 2,
                "source_title": "city lantern outage",
                "source_concepts": ["lantern"],
                "visual_translation": "The outage becomes a runaway city lantern pulled toward a canal whirlpool.",
                "news_mechanism": "the runaway lantern pulls its cable toward the canal whirlpool",
                "news_consequence": "the lantern rises above the canals after the current reverses",
                "visual_anchors": ["lantern", "whirlpool", "canals"],
                "anchor_roles": ["context", "mechanism", "consequence"],
                "integration": "Kirby protects the lantern while the outage makes the lantern the urgent mission.",
            },
            "story_spine": {
                "premise": "A current drags a guiding lantern toward a canal whirlpool.",
                "objective": "Kirby must redirect the lantern before the city loses its safe path.",
                "obstacle": "The current accelerates and the cable is about to snap.",
                "stakes": "Without the lantern, the canal city will lose its safe route home.",
                "emotional_arc": "curiosity becomes urgency, courage, and relief",
                "climax": "Kirby anchors the lantern against the current and sends its light back across the canals.",
                "resolution": "The lantern settles into a warm constellation and the canal becomes safe again.",
            },
            "world": {
                "setting": "a moonlit canal city with narrow bridges and reflective water",
                "visual_language": "indigo night, amber lantern light, soft watercolor anime edges",
                "continuity_rules": ["The same canal bridge remains visible across the story."],
            },
            "native_audio": "water rush, cable tension, one bright lantern chime, then calm night ambience",
            "native_shots": [
                {"time": "0-4s", "title": "Hook - the lantern is pulled away", "action": "The lantern tears loose while Kirby reaches for its cable.", "camera": "Wide bridge reveal pushing into Kirby and the moving lantern.", "state_change": "Kirby commits to stopping the runaway lantern."},
                {"time": "4-10s", "title": "Escalation - anchor the current", "action": "Kirby wraps the cable around a post as the current pulls harder.", "camera": "Smooth side track with the cable and whirlpool in the same geography.", "state_change": "The cable holds, but the bridge begins to crack under the force."},
                {"time": "10-15s", "title": "Payoff - light returns", "action": "Kirby releases warm light through the cable and the lantern rises above the canals.", "camera": "Follow the light upward, then settle into a calm wide ending frame.", "state_change": "The lanterns form a safe constellation and the danger is resolved."},
            ],
        }
        missing_stakes = json.loads(json.dumps(valid_story))
        missing_stakes["story_spine"]["stakes"] = ""
        missing_stakes["story_spine"]["climax"] = ""
        class ContractStoryModel:
            last_success_model = "local-contract-model"

            def __init__(self) -> None:
                self.calls = 0

            def chat_completion(self, *, messages: list[dict[str, object]], **_options: object) -> str:
                self.calls += 1
                return json.dumps(
                    {"story": missing_stakes, "creative_seed": "lantern-current", "source": "native_h3_llm"}
                )

        model = ContractStoryModel()
        engine = LLMPromptEngine(
            mode="llm",
            manager=SimpleNamespace(text_model=model, vision_model=model),
        )
        result = engine.generate_native_h3_storyboard(
            character="Kirby",
            style="polished 2D anime",
            duration_seconds=15,
            base_storyboard=storyboard,
            news_context={"title": "city lantern outage", "keyword": "canal safety"},
        )

        self.assertEqual(model.calls, 3)
        self.assertEqual(
            result["story"]["story_spine"]["stakes"],
            valid_story["native_shots"][1]["state_change"],
        )
        self.assertEqual(
            result["story"]["story_spine"]["climax"],
            valid_story["native_shots"][-1]["action"],
        )

    def test_native_h3_creative_brief_is_not_filtered_by_topic_or_digits(self) -> None:
        preserved = LLMPromptEngine._sanitize_native_h3_creative_brief(
            "stock ticker 810.06, chart, report, and glowing symbols"
        )
        self.assertIn("810.06", preserved)
        self.assertIn("chart", preserved)

    def test_user_given_numeric_arc_guidance_when_story_prompt_is_built_then_guidance_is_preserved(self) -> None:
        """User Given a timed source-matched arc When its Native H3 directive is built Then its causal beats reach the story model unchanged."""
        guidance = (
            "Use hook_payoff: open on the ordinary spotted cat (0-4 seconds), reveal a genetic clue, "
            "then resolve with the distinct species discovery; preserve the article's reported facts."
        )

        directive = LLMPromptEngine._format_native_h3_arc_guidance(guidance)

        self.assertIn(guidance, directive)
        self.assertIn("article remains authoritative for facts", directive)

    def test_user_given_news_without_a_custom_brief_when_story_prompt_is_built_then_default_news_direction_is_included(self) -> None:
        """User Given news with no custom creative brief When the story prompt is built Then its default direction keeps the story character-led and emotionally honest."""
        class ContractChatModel:
            last_success_model = "local-contract-model"

            def __init__(self) -> None:
                self.calls: list[list[dict[str, object]]] = []

            def chat_completion(self, *, messages: list[dict[str, object]], **_options: object) -> str:
                self.calls.append(messages)
                return json.dumps(
                    {
                        "story": {
                            "world": {
                                "setting": "A moonlit garden",
                                "continuity_rules": ["Keep the same garden path throughout."],
                            },
                            "native_shots": [
                                {"time": "0-4s", "action": "Kirby starts down the garden path."},
                                {"time": "4-10s", "action": "Kirby follows a firefly around one bend."},
                                {"time": "10-15s", "action": "Kirby returns to the same path as the firefly glows."},
                            ],
                        }
                    }
                )

        model = ContractChatModel()
        engine = LLMPromptEngine(
            mode="llm",
            manager=SimpleNamespace(text_model=model, vision_model=model),
        )
        engine.generate_native_h3_storyboard(
            character="Kirby",
            style="polished 2D anime",
            duration_seconds=15,
            base_storyboard=load_storyboard(self.repo_root / "configs/storyboards/native_h3_15s.yaml"),
            news_context={"title": "A firefly returns to the garden", "keyword": "firefly garden"},
        )

        self.assertEqual(len(model.calls), 3)
        prompt = str(model.calls[1][1]["content"])
        self.assertIn("Creative brief: News scene direction:", prompt)
        self.assertIn("center the selected characters' distinct actions and emotional response", prompt)
        self.assertIn("comedy only when the source mechanism itself creates a clear, harmless physical joke", prompt)
        self.assertIn("otherwise use the strongest source-supported feeling", prompt)
        self.assertIn("Never force cheer into sadness.", prompt)

    def test_user_given_human_interest_news_and_two_characters_when_story_is_generated_then_roles_and_active_welcome_remain_visible(self) -> None:
        """User Given human-interest news and two selected characters When a story prompt is built Then it preserves source roles, bans invented emotional tokens, allows ordinary implied furniture, and ends with an active, mutual welcome."""
        class ContractChatModel:
            last_success_model = "local-contract-model"

            def __init__(self) -> None:
                self.calls: list[list[dict[str, object]]] = []
                self.responses = [
                    {
                        "news_trace": {
                            "source_category": "human_interest",
                            "source_fact": "A student returned to the stadium with former coaches and junior teammates.",
                            "source_roles": {"returning player": "the hometown player", "welcoming person": "a former coach"},
                            "character_mapping": {"Kirby": "the returning player", "Waddle Dee": "a former coach"},
                            "fit_assessment": {"fit": "strong", "best_fitting_emotion": "tenderness"},
                        },
                        "recommended_scene": {"payoff_kind": "tenderness", "character_desire": "to share a proud moment"},
                    },
                    {
                        "story": {
                            "title": "A Homecoming at the Ballpark",
                            "native_shots": [
                                {"action": "Kirby returns to the baseball field as Waddle Dee waits at home plate with open arms."},
                                {"action": "Waddle Dee welcomes Kirby at home plate while Kirby leans forward to meet him."},
                                {"action": "Kirby and Waddle Dee embrace at home plate; Kirby closes his eyes in relief while Waddle Dee's eyes shine with pride."},
                            ],
                        }
                    },
                    {
                        "story": {
                            "title": "A Homecoming at the Ballpark",
                            "native_shots": [
                                {"action": "Kirby returns to the baseball field as Waddle Dee waits at home plate with open arms."},
                                {"action": "Waddle Dee welcomes Kirby at home plate while Kirby leans forward to meet him."},
                                {"action": "Kirby and Waddle Dee embrace at home plate; Kirby closes his eyes in relief while Waddle Dee's eyes shine with pride."},
                            ],
                        }
                    },
                ]

            def chat_completion(self, *, messages: list[dict[str, object]], **_options: object) -> str:
                self.calls.append(messages)
                return json.dumps(self.responses[len(self.calls) - 1])

        model = ContractChatModel()
        engine = LLMPromptEngine(
            mode="llm",
            manager=SimpleNamespace(text_model=model, vision_model=model),
        )
        result = engine.generate_native_h3_storyboard(
            character="Kirby and Waddle Dee",
            subject_context={
                "subjects": [
                    {"role": "primary", "name": "Kirby", "profile": {"role_description": "A round pink hero who inhales."}},
                    {"role": "secondary", "name": "Waddle Dee", "profile": {"role_description": "A loyal friend who cheers for others."}},
                ]
            },
            style="charming storybook animation",
            duration_seconds=15,
            base_storyboard=load_storyboard(self.repo_root / "configs/storyboards/native_h3_15s.yaml"),
            news_context={
                "title": "A former coach and junior teammates welcome a hometown player at the Tigers stadium",
                "keyword": "baseball;stadium;reunion",
                "content": "A young player returned to the stadium with his former coaches and junior teammates.",
            },
        )

        story = result["story"]
        self.assertEqual(story["news_trace"]["source_fact"], "A student returned to the stadium with former coaches and junior teammates.")
        self.assertEqual(story["news_trace"]["fit_assessment"]["fit"], "strong")
        self.assertEqual(story["news_trace"]["character_mapping"]["Kirby"], "the returning player")
        self.assertEqual(story["news_trace"]["character_mapping"]["Waddle Dee"], "a former coach")
        shots = story["native_shots"]
        self.assertGreaterEqual(len(shots), 3)
        self.assertTrue(all("Kirby" in shot["action"] and "Waddle Dee" in shot["action"] for shot in shots))
        self.assertIn("embrace", shots[-1]["action"].lower())
        self.assertIn("relief", shots[-1]["action"].lower())
        self.assertIn("pride", shots[-1]["action"].lower())
        prompts = [str(messages[-1]["content"]).casefold() for messages in model.calls]
        for prompt in prompts:
            self.assertIn("never invent a gift, keepsake, craft token, or unrelated chore", prompt)
        for prompt in prompts[1:]:
            self.assertIn("ordinary furniture and fixtures implied by the reported place and action are allowed", prompt)

    def test_user_given_abstract_product_news_when_story_prompt_is_built_then_curiosity_has_two_active_character_responses(self) -> None:
        """User Given low-stakes abstract migration news and a curiosity payoff When the story prompt is built Then the automatic change stays source-owned and both characters get distinct physical responses after it completes."""

        class ContractChatModel:
            last_success_model = "local-contract-model"

            def __init__(self) -> None:
                self.calls: list[list[dict[str, object]]] = []
                self.responses = [
                    {
                        "news_trace": {
                            "source_category": "product_or_service",
                            "source_fact": "Google says existing Gems migrate to Skills from November 17, 2026; Skills provide similar reusable task instructions.",
                            "source_action_sequence": ["Gems remain usable until November 17", "Gems automatically convert", "Skills provide similar reusable task instructions"],
                            "news_mechanism": "Existing saved instructions migrate to Skills, which provide similar reusable task instructions.",
                            "source_limit": "The instruction-card scene and character reactions are fictional analogies; the article does not report character actions.",
                            "fit_assessment": {
                                "fit": "conditional",
                                "reason": "The abstract migration has no human stakes, and the blank-card analogy alone does not create a visual joke; curious relief can show the transition without implying exact feature parity.",
                                "best_fitting_emotion": "curiosity",
                                "limitation": "The card and character reactions are fictional; the article does not report users' feelings or guarantee complete feature parity.",
                            },
                        },
                        "recommended_scene": {
                            "payoff_kind": "curiosity",
                            "character_desire": "Kirby wants to see how the same saved-instruction analogy carries across the scheduled format change.",
                            "character_signature": "Kirby shows curiosity through large attentive eyes and a full-body lean; Bandana Waddle Dee answers with a loyal head tilt and lifted hand.",
                            "source_prop": "one small plain blank instruction card representing saved Gemini instructions",
                            "prop_rule": "Google's automatic migration moves the same unmarked analogy card between two simple low holders; neither character causes the change, and the blank card does not claim exact feature parity.",
                            "hook": "Kirby and Bandana Waddle Dee notice the same small blank instruction card on a low table.",
                            "escalation": "The card slides by itself from one simple holder into the next while both characters track it with growing curiosity.",
                            "payoff": "The card settles in the new holder; Kirby leans closer with a clear open-mouth look of interest while Bandana Waddle Dee answers with an eager head tilt.",
                            "emotional_shift": "uncertainty becomes curious relief",
                            "held_reaction": "After the card finishes moving on its own, Kirby bends closer with wide curious eyes and a clear open O-mouth; Bandana Waddle Dee tilts their head and lifts one paw to point at the same card.",
                            "emotion": "curiosity",
                            "why_this_fits": "The article reports an abstract automatic migration rather than human drama or a natural joke; one visible card transfer makes the change understandable without making the characters believe the headline or promising exact feature parity.",
                        },
                    },
                    {
                        "story": {
                            "title": "The Card That Travels",
                            "world": {"setting": "a bright artist's studio", "visual_language": "lively watercolor with a warm yellow accent"},
                            "story_spine": {
                                "premise": "Kirby and Bandana Waddle Dee notice one small blank instruction card on a low table.",
                                "stakes": "They wonder how the saved instructions carry across the reported automatic format change.",
                                "climax": "The same card slides by itself from one simple holder into a new one as both characters follow it.",
                                "resolution": "Kirby and Bandana Waddle Dee lean toward the card with clear, curious relief; the card is only an analogy for migration, not a claim of exact feature parity.",
                            },
                            "native_shots": [
                                {"time": "0-5s", "action": "Kirby and Bandana Waddle Dee notice one small plain blank instruction card lying flat on a low table."},
                                {"time": "5-10s", "action": "The same card slides by itself from one simple low holder into another; Kirby and Bandana Waddle Dee follow it with increasingly curious eyes."},
                                {"time": "10-15s", "action": "After the automatic move, Kirby bends closer with wide curious eyes and an open O-mouth; Bandana Waddle Dee tilts their head and points at the card.", "state_change": "Kirby and Bandana Waddle Dee stand side by side with faces visible; Kirby leans toward the card with a clear open O-mouth while Bandana Waddle Dee tilts their head and points to it."},
                            ],
                        }
                    },
                    {
                        "story": {
                            "title": "The Card That Travels",
                            "world": {"setting": "a bright artist's studio", "visual_language": "lively watercolor with a warm yellow accent"},
                            "story_spine": {
                                "premise": "Kirby and Bandana Waddle Dee notice one small blank instruction card on a low table.",
                                "stakes": "They wonder how the saved instructions carry across the reported automatic format change.",
                                "climax": "The same card slides by itself from one simple holder into a new one as both characters follow it.",
                                "resolution": "Kirby and Bandana Waddle Dee lean toward the card with clear, curious relief; the card is only an analogy for migration, not a claim of exact feature parity.",
                            },
                            "native_shots": [
                                {"time": "0-5s", "action": "Kirby and Bandana Waddle Dee notice one small plain blank instruction card lying flat on a low table."},
                                {"time": "5-10s", "action": "The same card slides by itself from one simple low holder into another; Kirby and Bandana Waddle Dee follow it with increasingly curious eyes."},
                                {"time": "10-15s", "action": "After the automatic move, Kirby bends closer with wide curious eyes and an open O-mouth; Bandana Waddle Dee tilts their head and points at the card.", "state_change": "Kirby and Bandana Waddle Dee stand side by side with faces visible; Kirby leans toward the card with a clear open O-mouth while Bandana Waddle Dee tilts their head and points to it."},
                            ],
                        }
                    },
                ]

            def chat_completion(self, *, messages: list[dict[str, object]], **_options: object) -> str:
                self.calls.append(messages)
                return json.dumps(self.responses.pop(0))

        model = ContractChatModel()
        engine = LLMPromptEngine(
            mode="llm",
            manager=SimpleNamespace(text_model=model, vision_model=model),
        )
        article_title = "Google 11月淘汰Gems，現有設定轉為Skills"
        result = engine.generate_native_h3_storyboard(
            character="Kirby",
            subject_context={
                "character_profile": {
                    "role_description": "A round pink hero who inhales and copies abilities.",
                },
                "subjects": [
                    {"name": "Kirby", "role": "primary", "profile": {"visual_description": "round pink body, large blue eyes, red feet"}},
                    {"name": "Bandana Waddle Dee", "role": "secondary", "profile": {"visual_description": "small orange body, cream face, blue bandana, simple oval eyes, rosy cheeks"}},
                ],
            },
            style="charming storybook animation",
            duration_seconds=15,
            base_storyboard=load_storyboard(self.repo_root / "configs/storyboards/native_h3_15s.yaml"),
            news_context={
                "title": article_title,
                "keyword": "Google;Gemini;Gems;Skills",
                "content": "Existing Gems remain usable until November 17, 2026, when they automatically convert into Skills that provide similar reusable instructions. Skills are available to personal users over 18; work and school accounts will receive Skills later.",
            },
        )

        source_fact = "Google says existing Gems migrate to Skills from November 17, 2026; Skills provide similar reusable task instructions."
        self.assertEqual(result["news_editorial_card"]["news_trace"]["fit_assessment"]["fit"], "conditional")
        self.assertEqual(result["news_editorial_card"]["news_trace"]["fit_assessment"]["best_fitting_emotion"], "curiosity")
        self.assertEqual(result["story"]["news_trace"]["source_fact"], source_fact)
        self.assertEqual(result["story"]["news_trace"]["source_title"], article_title)
        self.assertEqual(result["news_editorial_card"]["recommended_scene"]["payoff_kind"], "curiosity")
        self.assertEqual(len(result["story"]["native_shots"]), 3)
        self.assertIn("Bandana Waddle Dee", result["story"]["native_shots"][-1]["action"])
        self.assertIn("open O-mouth", result["story"]["native_shots"][-1]["action"])
        self.assertIn("points at the card", result["story"]["native_shots"][-1]["action"])
        self.assertNotIn("cheeks gently squish together", result["story"]["native_shots"][-1]["action"])
        planned_hold = result["news_editorial_card"]["recommended_scene"]["held_reaction"]
        self.assertNotEqual(result["story"]["native_shots"][-1]["action"], planned_hold)
        self.assertIn("Kirby leans toward the card", result["story"]["native_shots"][-1]["state_change"])
        ending_prompt = format_native_h3_keyframe_prompt(result["story"], frame="ending")
        self.assertIn("points to it", ending_prompt.casefold())
        self.assertNotIn("cheeks gently squish together", ending_prompt.casefold())
        self.assertNotIn("do not add contact, a raised paw", ending_prompt.casefold())
        self.assertIn("same card slides by itself", result["story"]["native_shots"][1]["action"])
        self.assertNotIn("gag_card", result["story"])
        planning_prompt = str(model.calls[0][1]["content"]).casefold()
        story_prompt = str(model.calls[1][1]["content"]).casefold()
        self.assertIn("preserve the reported account audience and transition date", planning_prompt)
        self.assertIn("do not extend a personal-account date to work or school accounts", planning_prompt)
        self.assertIn("never upgrade it to identical", planning_prompt)
        self.assertIn("keep rumors, confirmed announcements, and scheduled changes distinct", planning_prompt)
        self.assertIn("do not treat a misleading or abbreviated headline as a character's knowledge", planning_prompt)
        self.assertIn("the visual gag works without the headline or readable text", planning_prompt)
        self.assertIn("same single analogy object before and after its change", story_prompt)
        self.assertIn("a smile or wide eyes alone is not a punchline", story_prompt)
        self.assertIn("give each selected character a distinct, drawable response suited to the source stakes", story_prompt)
        self.assertIn("let both fictional users contribute distinct voluntary actions", story_prompt)
        self.assertNotIn("their round cheeks gently meet and visibly squish together", story_prompt)
        revision_prompt = str(model.calls[2][1]["content"]).casefold()
        self.assertIn("if no source-connected physical joke survives that silent-frame test", revision_prompt)
        self.assertIn("do not infer complete feature parity", story_prompt)
        self.assertNotIn("frame each shot around the characters, not a product display", story_prompt)

    def test_user_given_news_when_editorial_rewrite_fails_then_renderable_draft_is_preserved(self) -> None:
        """User Given source planning and a draft missing drawable world details When the optional editorial rewrite fails Then the draft keeps its source, setting, and selected style for the keyframe."""
        draft_story = {
            "story": {
                "title": "Kirby's Reusable Star",
                "native_shots": [
                    {"time": "0-4s", "action": "Kirby cups one reusable star."},
                    {"time": "4-10s", "action": "The star helps Kirby twice and passes its effect to a second star."},
                    {"time": "10-15s", "action": "Kirby blinks at the two stars and gives a delighted little bounce."},
                ],
                "gag_card": {
                    "expectation": "The stars solve a tiny task.",
                    "physical_escalation": "The star is used twice.",
                    "surprising_harmless_reversal": "The stars line up neatly.",
                    "held_expressive_reaction": "Kirby smiles.",
                },
            }
        }

        class RewriteUnavailableModel:
            last_success_model = "local-contract-model"

            def __init__(self) -> None:
                self.responses = [
                    json.dumps({
                        "news_trace": {"source_fact": "Existing Gems remain usable until the automatic conversion date; Skills provide similar reusable task instructions."},
                        "recommended_scene": {
                            "payoff_kind": "curiosity",
                            "character_desire": "Kirby wants to keep one tiny task handy.",
                            "source_prop": "one reusable star",
                            "prop_rule": "The same star repeats one helpful action.",
                            "hook": "Kirby finds one star.",
                            "escalation": "Kirby calls it again.",
                            "payoff": "The star remains ready for another use.",
                            "emotional_shift": "curiosity becomes calm delight",
                            "held_reaction": "Kirby smiles with relief.",
                            "emotion": "curiosity",
                            "why_this_fits": "The article describes reusable instructions.",
                        },
                    }),
                    json.dumps(draft_story),
                ]

            def chat_completion(self, *, messages: list[dict[str, object]], **_options: object) -> str:
                if self.responses:
                    return self.responses.pop(0)
                raise RuntimeError("revision provider unavailable")

        model = RewriteUnavailableModel()
        engine = LLMPromptEngine(
            mode="llm",
            manager=SimpleNamespace(text_model=model, vision_model=model),
        )
        result = engine.generate_native_h3_storyboard(
            character="Kirby",
            subject_context={"character_profile": {"role_description": "A round pink hero who inhales and copies abilities."}},
            style="warm storybook",
            duration_seconds=15,
            base_storyboard=load_storyboard(self.repo_root / "configs/storyboards/native_h3_15s.yaml"),
            news_context={"title": "Existing Gems convert into Skills", "keyword": "reusable instructions"},
        )

        self.assertEqual(result["news_story_revision"], "draft_preserved")
        self.assertEqual(
            result["story"]["native_shots"][-1]["action"],
            draft_story["story"]["native_shots"][-1]["action"],
        )
        self.assertTrue(result["story"]["world"]["setting"].strip())
        self.assertTrue(result["story"]["world"]["visual_language"].strip())
        self.assertNotIn("pedestal", result["story"]["world"]["setting"].casefold())
        ending_prompt = format_native_h3_keyframe_prompt(result["story"], style="warm storybook", frame="ending")
        self.assertIn("Setting: a simple storybook setting", ending_prompt)
        self.assertIn("clear space around the source-linked action", ending_prompt)
        self.assertIn("Visual style: warm storybook", ending_prompt)
        self.assertNotIn("gag_card", result["story"])

    def test_user_given_resolved_kirby_role_when_news_story_is_generated_then_final_video_prompt_keeps_identity_and_profile(self) -> None:
        """User Given Kirby's resolved subject profile and selected news When its story is generated Then the final render prompt names Kirby and preserves his visual and role reference."""

        class RoleContractModel:
            last_success_model = "local-contract-model"

            def chat_completion(self, *, messages: list[dict[str, object]], **_options: object) -> str:
                return json.dumps(
                    {
                        "story": {
                            "title": "Kirby and the Saved Instructions",
                            "story_spine": {
                                "premise": "Kirby wants to keep one familiar task ready.",
                                "stakes": "Kirby wants his saved ability to remain useful.",
                                "resolution": "The same blank instruction card represents the saved instructions carried into the Skills format.",
                            },
                            "news_trace": {
                                "source_fact": "Existing saved Gemini instructions automatically convert to Skills, which offer similar reusable task instructions.",
                                "news_mechanism": "Saved custom instructions migrate into a similar reusable format.",
                                "news_consequence": "A user's saved instructions move into Skills, which provide similar reusable task instructions.",
                                "visual_translation": "One plain blank instruction card remains the same object during automatic migration.",
                                "visual_anchors": ["plain blank instruction card representing saved instructions"],
                            },
                            "native_shots": [
                                {
                                    "time": "0-15s",
                                    "action": "The saved instructions migrate automatically while the same plain blank card stays in place; Kirby watches with wide curious eyes.",
                                    "physical_cause": "Google changes the saved instructions automatically; neither character triggers the migration.",
                                    "state_change": "Kirby smiles with curious relief beside the same blank instruction card, now representing the migrated instructions.",
                                }
                            ],
                        }
                    }
                )

        role = "a round pink hero with blue oval eyes and red feet"
        model = RoleContractModel()
        engine = LLMPromptEngine(
            mode="llm",
            manager=SimpleNamespace(text_model=model, vision_model=model),
        )
        story = engine.generate_native_h3_storyboard(
            character="Kirby",
            subject_context={"character_profile": {"role_description": role}},
            style="soft hand-drawn animation",
            duration_seconds=15,
            base_storyboard=load_storyboard(self.repo_root / "configs/storyboards/native_h3_15s.yaml"),
            news_context={"title": "Google Gems migrate to Skills", "content": "Saved instructions stay usable after migration."},
        )["story"]

        render_prompt = format_native_h3_prompt(story, style="soft hand-drawn animation", duration_seconds=15)

        self.assertEqual(story["character"], "Kirby")
        self.assertIn("Character reference: Kirby", render_prompt)
        self.assertIn(role, render_prompt)

    def test_user_given_model_visible_action_fields_when_story_is_merged_then_actions_and_effects_reach_render_prompt(self) -> None:
        """User Given the story model returns visible-action field names When its shots are merged Then the same actions and effects remain available to the render prompt."""
        base_storyboard = load_storyboard(self.repo_root / "configs/storyboards/native_h3_15s.yaml")
        visible_action = "The saved Gemini instructions migrate automatically while the same plain blank instruction card remains on the desk; Kirby watches with wide curious eyes."
        visible_effect = "Kirby smiles with relieved surprise beside the unchanged instruction card, now representing migrated instructions."
        generated_story = {
            "story_spine": {
                "premise": "Saved Gemini instructions migrate automatically while the same blank instruction card remains.",
                "character_scale_stakes": "Kirby wonders how the saved instruction can be reused.",
                "visible_resolution": "Kirby reacts with relieved surprise when the same saved instructions appear in the Skills format.",
            },
            "news_trace": {
                "source_fact": "Existing saved Gemini instructions move into Skills automatically; Skills offer similar reusable task instructions.",
                "source_consequence": "Users can keep calling the converted task instructions.",
                "visual_translation": "The same plain blank instruction card represents saved instructions before and after migration.",
                "visual_anchors": ["plain blank instruction card representing saved instructions"],
                "anchor_roles": {
                    "context": "the existing saved instructions",
                    "mechanism": "the same instruction card remains as the instructions migrate",
                    "consequence": "a user can invoke and reuse a saved instruction",
                },
            },
            "native_shots": [
                {
                    "time": "0-15s",
                    "title": "The instructions migrate, the card stays",
                    "visible_action": visible_action,
                    "physical_cause": "The scheduled system migration updates the instructions without being triggered by Kirby.",
                    "visible_effect_or_state_change": visible_effect,
                    "camera_direction": "Hold a medium-wide view of the desk and Kirby's reaction.",
                }
            ]
        }

        merged = merge_native_h3_storyboard(base_storyboard, generated_story)

        self.assertEqual(merged["native_shots"][0]["action"], visible_action)
        self.assertEqual(merged["native_shots"][0]["state_change"], visible_effect)
        self.assertEqual(
            merged["native_shots"][0]["camera"],
            "Hold a medium-wide view of the desk and Kirby's reaction.",
        )
        self.assertEqual(
            merged["native_shots"][0]["physical_cause"],
            "The scheduled system migration updates the instructions without being triggered by Kirby.",
        )
        self.assertEqual(
            merged["news_trace"]["news_consequence"],
            "Users can keep calling the converted task instructions.",
        )
        self.assertEqual(
            merged["story_spine"]["resolution"],
            "Kirby reacts with relieved surprise when the same saved instructions appear in the Skills format.",
        )
        render_prompt = format_native_h3_prompt(merged, duration_seconds=15)
        self.assertIn(merged["native_shots"][0]["action"], render_prompt)
        self.assertNotIn("\nNews grounding context:", render_prompt)
        self.assertNotIn("\nAdditional source context:", render_prompt)
        self.assertIn("Cause: The scheduled system migration updates the instructions without being triggered by Kirby.", render_prompt)

    def test_user_given_news_gag_when_story_is_formatted_then_physical_cause_and_full_payoff_reach_video_prompt(self) -> None:
        """User Given a character-led news gag, When formatted, Then H3 receives the final shots' cause, setback and payoff without the editorial staging card."""
        base_storyboard = load_storyboard(self.repo_root / "configs/storyboards/native_h3_15s.yaml")
        merged = merge_native_h3_storyboard(
            base_storyboard,
            {
                "title": "The Goodbye That Wasn't",
                "story_spine": {
                    "premise": "Kirby and Bandana Waddle Dee think they must say goodbye to a saved instruction card.",
                    "stakes": "The headline sounds final, but the article says existing Gems are moved to Skills with similar reusable instructions.",
                    "climax": "The same plain card is carried into the new format; Kirby stops his farewell mid-gesture and breaks into relieved laughter while Bandana Waddle Dee perks up beside him.",
                    "resolution": "Kirby laughs with an open mouth beside the unchanged card as Bandana Waddle Dee leans close with one hand raised.",
                },
                "gag_card": {
                    "character_signature": "Kirby expresses every feeling with a clear full-body pose and expressive face.",
                    "character_desire": "Kirby wants to give the saved instructions a proper goodbye because the headline sounds final.",
                    "source_prop": "one plain blank instruction card representing saved Gemini instructions",
                    "prop_rule": "The same card remains available as Google migrates the saved instructions automatically; neither character causes the migration.",
                    "expectation": "Kirby and Bandana Waddle Dee expect the existing saved instructions to disappear after the change.",
                    "physical_escalation": "Kirby begins a solemn little farewell wave beside the card while Bandana Waddle Dee leans in with a sympathetic, drooping pose.",
                    "surprising_harmless_reversal": "The same card is still present in the new format; Kirby cuts the farewell short and pops upright with a relieved laugh while Bandana Waddle Dee lifts one hand in happy surprise.",
                    "held_expressive_reaction": "Kirby laughs with wide bright eyes beside the unchanged blank instruction card; Bandana Waddle Dee leans toward Kirby with an eager head tilt and one hand raised.",
                },
                "world": {
                    "setting": "a bright artist's studio with warm wooden floorboards",
                    "visual_language": "colorful watercolor, readable full-body acting, strong diagonal movement",
                },
                "news_trace": {
                    "source_category": "product_or_service",
                    "source_title": "Google's headline says Gems are phased out, while the article reports automatic migration to Skills",
                    "source_fact": "Existing Gems remain usable until November 17, 2026, when Google automatically converts them to Skills. Skills provide similar reusable task instructions.",
                    "news_mechanism": "Automatic migration of saved Gemini instructions into a similar reusable Skills format.",
                    "news_consequence": "Saved instructions migrate to Skills; the article describes similar reusable task instructions, not complete feature parity.",
                    "visual_translation": "One plain blank instruction card stays the same object while its saved instructions migrate automatically.",
                    "visual_anchors": ["one plain blank instruction card representing saved instructions"],
                    "fit_assessment": {"fit": "conditional", "best_fitting_emotion": "comedy"},
                    "source_limit": "The card and character farewell are fictional analogies; the article reports a product migration, not users' feelings.",
                },
                "native_shots": [
                    {
                        "time": "0-3.75s",
                        "title": "The saved instructions migrate",
                        "action": "The same plain blank instruction card stays on the desk as Google's automatic migration changes the saved instructions.",
                        "physical_cause": "Google performs the migration; neither character triggers it.",
                        "state_change": "The unchanged card remains below both faces while Kirby and Bandana Waddle Dee watch.",
                    },
                    {
                        "time": "3.75-7.5s",
                        "title": "An exaggerated goodbye",
                        "action": "Kirby starts a solemn farewell wave beside the same card while Bandana Waddle Dee leans in sympathetically.",
                        "physical_cause": "The headline's closure wording leads the fictional users to expect a goodbye.",
                        "state_change": "Kirby lowers his gaze in an overdone farewell pose; Bandana Waddle Dee answers with a sympathetic lean.",
                    },
                    {
                        "time": "7.5-11.25s",
                        "title": "The goodbye is interrupted",
                        "action": "Kirby notices the same card carried into the new format, stops the farewell mid-wave, and pops upright with a relieved laugh as Bandana Waddle Dee lifts one hand in surprise.",
                        "physical_cause": "The article's correction reveals that the saved instructions continue under the Skills name.",
                        "state_change": "Kirby laughs with wide bright eyes beside the unchanged card; Bandana Waddle Dee leans closer with one hand raised.",
                    },
                    {
                        "time": "11.25-15s",
                        "title": "Relief",
                        "action": "Kirby laughs with an open mouth beside the unchanged blank card while Bandana Waddle Dee leans close with an eager head tilt and one raised hand.",
                        "physical_cause": "Kirby recognizes the saved instructions in the Skills format and drops the premature farewell.",
                        "state_change": "Kirby and Bandana Waddle Dee hold a clear shared relief pose with the same blank card below both faces.",
                    }
                ],
            },
        )

        prompt = format_native_h3_prompt(merged, duration_seconds=15)

        self.assertIn("Cause: The article's correction reveals that the saved instructions continue under the Skills name.", prompt)
        self.assertIn("Kirby starts a solemn farewell wave", prompt)
        self.assertIn("stops the farewell mid-wave", prompt)
        self.assertIn("hold a clear shared relief pose", prompt)
        self.assertNotIn("Physical escalation:", prompt)
        self.assertNotIn("Surprising reversal:", prompt)

    def test_native_h3_negative_visual_constraints_do_not_erase_creative_brief(self) -> None:
        brief = (
            "Make a compact causal story with one dominant mechanism and a readable payoff. "
            "Do not show readable interfaces or abstract symbols."
        )

        sanitized = LLMPromptEngine._sanitize_native_h3_creative_brief(brief)

        self.assertIn("compact causal story", sanitized)
        self.assertIn("Do not show readable interfaces", sanitized)

    def test_native_h3_story_does_not_fallback_when_llm_is_unavailable(self) -> None:
        engine = LLMPromptEngine(mode="llm", manager=None)
        storyboard = load_storyboard(self.repo_root / "configs/storyboards/native_h3_15s.yaml")
        with patch.object(engine, "_manager_or_none", return_value=None):
            with self.assertRaises(PromptGenerationError):
                engine.generate_native_h3_storyboard(
                    character="Kirby",
                    style="polished 2D anime",
                    duration_seconds=15,
                    base_storyboard=storyboard,
                    news_context={"title": "test", "keyword": "test"},
                )

    def test_native_h3_inspection_is_non_blocking_for_discord_review(self) -> None:
        class FakeTools:
            def call(self, tool_name: str, payload: dict[str, object]) -> dict[str, object]:
                self.tool_name = tool_name
                self.payload = payload
                return {
                    "passed": True,
                    "video_path": str(payload["video_path"]),
                    "file_exists": True,
                    "probe": {
                        "duration": 15.0,
                        "has_video": True,
                        "width": 608,
                        "height": 352,
                        "video_codec": "h264",
                    },
                    "checks": {"file_exists": True, "has_video": True, "dimensions": True, "duration": True},
                    "duration": 15.0,
                    "target_duration": 15.0,
                    "errors": [],
                    "warnings": [],
                    "contact_sheet_path": "",
                }

        state = RunState(
            goal={},
            metadata={},
            node_outputs={
                "native-h3-render": {"saved_files": [str(self.repo_root / ".tmp-tests" / "clip.mp4")], "run_dir": "run"},
            },
        )
        context = SimpleNamespace(
            state=state,
            node=SimpleNamespace(inputs={"target_duration": 15, "duration_tolerance": 0.5}),
            plan=SimpleNamespace(goal=SimpleNamespace(prompt="Kirby story", duration_seconds=15)),
        )

        fake_video = self.repo_root / ".tmp-tests" / "clip.mp4"
        fake_video.parent.mkdir(parents=True, exist_ok=True)
        fake_video.write_bytes(b"test")
        self.addCleanup(lambda: fake_video.unlink(missing_ok=True))
        result = LongVideoSkills(FakeTools(), self.repo_root / ".tmp-tests" / "native-h3-qa").qa_native_h3(context)

        self.assertEqual(result.status, "success")
        self.assertTrue(result.outputs["passed"])
        self.assertNotIn("story_quality", result.outputs)
        self.assertFalse(result.outputs["technical_qa"]["automatic_gate_applied"])
        self.assertTrue(result.outputs["technical_qa"]["checks"]["duration"])

class OpenRouterCatalogTests(unittest.TestCase):
    def test_catalog_filters_zero_price_text_and_vision_models(self) -> None:
        body = {
            "data": [
                {
                    "id": "old/paid-model",
                    "pricing": {"prompt": "0.000001", "completion": "0.000001"},
                    "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
                },
                {
                    "id": "provider/text-free",
                    "created": 2,
                    "context_length": 100000,
                    "pricing": {"prompt": "0", "completion": "0"},
                    "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
                    "supported_parameters": ["response_format"],
                },
                {
                    "id": "provider/vision-free",
                    "created": 3,
                    "context_length": 120000,
                    "pricing": {"prompt": "0", "completion": "0"},
                    "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
                },
                {
                    "id": "provider/content-safety-free",
                    "pricing": {"prompt": "0", "completion": "0"},
                    "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
                },
            ]
        }

        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return json.loads(json.dumps(body))

        OpenRouterModelCatalog._cache.clear()
        with patch("agentic.runtime.model_backends.requests.get", return_value=FakeResponse()):
            text = OpenRouterModelCatalog.candidates("text", limit=5, ttl_seconds=0, force_refresh=True)
            vision = OpenRouterModelCatalog.candidates("vision", limit=5, ttl_seconds=0, force_refresh=True)

        self.assertCountEqual(text, ["provider/text-free", "provider/vision-free"])
        self.assertEqual(vision, ["provider/vision-free"])


class NativeH3KeyframePromptTests(unittest.TestCase):
    def test_user_given_news_story_with_multiple_beats_when_opening_keyframe_is_formatted_then_only_the_first_visual_moment_is_shown(self) -> None:
        """User Given a news story with several shots and two named Kirby characters When its opening keyframe prompt is formatted Then it shows one source-grounded first moment, not a montage of later beats."""
        storyboard = {
            "character": "Kirby, Bandana Waddle Dee",
            "subject_context": {
                "subjects": [
                    {
                        "name": "Kirby",
                        "role": "primary",
                        "profile": {
                            "role_description": "curious pink hero who can inhale and copy abilities",
                            "keywords": "round pink body, blue oval eyes, red feet",
                        },
                    },
                    {
                        "name": "Bandana Waddle Dee",
                        "role": "secondary",
                        "profile": {
                            "role_description": "loyal friend, eager to help",
                            "keywords": "small orange body, blue bandana",
                        },
                    },
                ]
            },
            "world": {
                "setting": "a warm, softly lit star meadow",
                "visual_language": "tactile storybook watercolor with expressive faces",
            },
            "news_trace": {
                "source_category": "product_or_service",
                "source_title": "Google automatically migrates Gems to Skills",
                "source_fact": "Existing saved Gemini instructions automatically become Skills on November 17; Skills offer similar reusable task instructions.",
                "news_mechanism": "Automatic migration of saved instructions into a similar reusable format.",
                "visual_translation": "One plain instruction card remains the same object as its saved instructions migrate.",
                "visual_anchors": ["one plain blank instruction card representing saved Gemini instructions"],
                "source_limit": "The instruction card is a fictional analogy, not a reported physical object.",
            },
            "native_shots": [
                {
                    "time": "0-5s",
                    "action": "Kirby holds one plain blank instruction card at waist height while Bandana Waddle Dee leans toward it with curious eyes.",
                    "physical_cause": "The characters inspect the same saved-instruction card.",
                    "state_change": "Both characters notice the same plain card between them, below their faces.",
                    "camera": "A medium two-shot at their eye level.",
                },
                {
                    "time": "5-10s",
                    "action": "The saved instructions migrate automatically while the blank card stays the same shape.",
                    "physical_cause": "The software migration changes the saved instructions' name, not their continued function.",
                    "state_change": "The same card remains between Kirby and Bandana Waddle Dee after the migration.",
                },
                {
                    "time": "10-15s",
                    "action": "Kirby and Bandana Waddle Dee exchange a curious look beside the same card.",
                    "physical_cause": "They notice that the saved instructions remain available after the change.",
                    "state_change": "Both characters keep their faces visible as they examine the unchanged card.",
                },
            ],
        }

        prompt = format_native_h3_keyframe_prompt(storyboard, style="soft storybook watercolor", frame="opening")

        self.assertIn("Kirby", prompt)
        self.assertIn("Bandana Waddle Dee", prompt)
        self.assertIn("Kirby holds one plain blank instruction card at waist height", prompt)
        self.assertLess(prompt.index("Visual identity"), prompt.index("Main action"))
        self.assertIn("round pink body, blue oval eyes, red feet", prompt)
        self.assertNotIn("can inhale and copy abilities", prompt)
        self.assertIn("one storybook still", prompt.lower())
        self.assertIn("main action", prompt.lower())
        self.assertIn("plain blank instruction card", prompt.lower())
        self.assertNotIn("gemstone", prompt.lower())
        self.assertNotIn("target surface", prompt.lower())
        self.assertNotIn("appliance or container", prompt.lower())
        self.assertIn("Keep characters recognizable", prompt)
        self.assertIn("no promotional packaging", prompt.lower())
        self.assertNotIn("no product display", prompt.lower())
        self.assertNotIn("synchronized loop", prompt)
        self.assertIn("one scene", prompt.lower())

    def test_user_given_news_story_with_multiple_beats_when_ending_keyframe_is_formatted_then_only_the_final_visual_moment_is_shown(self) -> None:
        """User Given a news story with several shots When its ending keyframe prompt is formatted Then it shows the last beat and established cast without replaying earlier actions."""
        storyboard = {
            "character": "Kirby, Bandana Waddle Dee",
            "world": {"setting": "a warm, softly lit star meadow"},
            "news_trace": {
                "source_category": "product_or_service",
                "source_title": "Google automatically migrates Gems to Skills",
                "source_fact": "Saved Gemini instructions are migrated to Skills; Skills provide similar reusable task instructions.",
                "fit_assessment": {"best_fitting_emotion": "curiosity"},
                "visual_translation": "One plain instruction card stays the same as saved instructions migrate.",
                "visual_anchors": "one plain blank instruction card representing saved instructions",
            },
            "native_shots": [
                {"action": "Kirby holds one plain blank instruction card while Bandana Waddle Dee leans in."},
                {"action": "The saved instructions migrate while the card remains the same shape."},
                {
                    "action": "Kirby and Bandana Waddle Dee exchange a curious look beside the same card.",
                    "state_change": "Kirby has wide curious eyes and an open mouth; Bandana Waddle Dee tilts forward beside the unchanged instruction card.",
                },
            ],
        }

        prompt = format_native_h3_keyframe_prompt(storyboard, frame="ending")

        self.assertIn("Held final pose: Kirby has wide curious eyes", prompt)
        self.assertNotIn("wave in a silly synchronized loop", prompt)
        self.assertNotIn("glowing gem", prompt.casefold())
        self.assertNotIn("reusable skill star", prompt.casefold())
        self.assertIn("camera: three-quarter front two-shot; medium-wide", prompt.casefold())
        self.assertIn("both full figures with space around them and readable faces", prompt.casefold())
        self.assertIn("held final pose:", prompt.casefold())
        self.assertIn("one plain blank instruction card representing saved instructions is part of the held pose", prompt)
        self.assertNotIn("Action focus:", prompt)
        self.assertNotIn("faces occupy the upper two-thirds", prompt.casefold())
        self.assertNotIn("rounded corner tab", prompt.casefold())
        self.assertNotIn("cheeks gently meet", prompt.casefold())

    def test_user_given_product_migration_with_a_story_specific_analogy_when_ending_prompt_is_formatted_then_the_llm_pose_and_object_are_preserved(self) -> None:
        """User Given product migration news with a named analogy and two character actions When its ending prompt is formatted Then it preserves those choices without imposing a fixed prop pose or contact gag."""
        storyboard = {
            "character": "Kirby, Bandana Waddle Dee",
            "news_trace": {
                "source_category": "product_or_service",
                "source_title": "Saved instructions migrate to a new format",
                "source_fact": "Existing saved instructions automatically migrate to a similar reusable format.",
                "news_mechanism": "Automatic migration of existing saved instructions.",
                "fit_assessment": {"fit": "conditional", "best_fitting_emotion": "curiosity"},
                "visual_anchors": ["one plain blank instruction card representing saved instructions"],
            },
            "native_shots": [
                {
                    "action": "Kirby holds the plain blank instruction card at chest height while Bandana Waddle Dee points to its small corner tab.",
                    "state_change": "Kirby holds the same blank instruction card at chest height with wide curious eyes and an open O-mouth; Bandana Waddle Dee points to its corner tab and leans in with an attentive head tilt.",
                }
            ],
        }

        prompt = format_native_h3_keyframe_prompt(storyboard, frame="ending")

        self.assertIn("Kirby holds the same blank instruction card at chest height", prompt)
        self.assertIn("Bandana Waddle Dee points to its corner tab", prompt)
        self.assertIn("one plain blank instruction card representing saved instructions is part of the held pose", prompt)
        self.assertNotIn("cheek contact", prompt.casefold())
        self.assertNotIn("rises onto his toes", prompt.casefold())
        self.assertNotIn("small blank instruction card lying flat on a low table", prompt.casefold())

    def test_user_given_non_kirby_comedy_when_ending_prompt_is_formatted_then_cast_identity_is_not_replaced_by_kirby_rules(self) -> None:
        """User Given a comic story with a non-Kirby cast When its ending keyframe prompt is formatted Then it preserves the described gag and adds no Kirby-specific appearance rules."""
        storyboard = {
            "character": "Moss Fox, Tiny Crane",
            "news_trace": {"fit_assessment": {"best_fitting_emotion": "comedy"}},
            "native_shots": [
                {
                    "action": "Moss Fox carefully balances one leaf while Tiny Crane stretches to catch it.",
                    "state_change": "Tiny Crane stands on one leg holding the leaf while Moss Fox laughs beside the crane's wobbling raised wing.",
                }
            ],
        }

        prompt = format_native_h3_keyframe_prompt(storyboard, frame="ending")

        self.assertIn("Selected characters: Moss Fox, Tiny Crane", prompt)
        self.assertIn("Moss Fox laughs beside the crane's wobbling raised wing", prompt)
        self.assertIn("Mood:", prompt)
        self.assertIn("huge bright eyes and a clearly open laugh", prompt)
        self.assertIn("the partner's distinct physical reaction", prompt)
        self.assertNotIn("Kirby pink", prompt)
        self.assertNotIn("Bandana Waddle Dee orange", prompt)
        self.assertNotIn("red feet", prompt)
        self.assertNotIn("diagonal tumble", prompt)

    def test_user_given_joke_contact_in_the_held_pose_when_ending_prompt_is_formatted_then_contact_and_reaction_remain_visible(self) -> None:
        """User Given the final held pose contains the joke's contact When an ending keyframe is formatted Then the still keeps that contact, payload, and reaction without replaying earlier actions."""
        storyboard = {
            "character": "Kirby, Bandana Waddle Dee",
            "news_trace": {"fit_assessment": {"best_fitting_emotion": "comedy"}},
            "native_shots": [
                {
                    "action": "Kirby repeats one wave; their raised hands gently bump together, sending both friends tumbling backward into a diagonal laughing heap.",
                    "physical_cause": "Their raised hands gently bump together during the mirrored wave.",
                    "state_change": "Both characters hold a diagonal heap pose, their raised hands gently touching after the bump, laughing with open mouths.",
                }
            ],
        }

        prompt = format_native_h3_keyframe_prompt(storyboard, frame="ending")

        self.assertIn("Held final pose: Both characters hold a diagonal heap pose, their raised hands gently touching", prompt)
        self.assertIn("gently touching", prompt)
        self.assertIn("Camera: three-quarter front two-shot", prompt)
        self.assertNotIn("Kirby repeats one wave", prompt)

    def test_user_given_news_symbols_when_keyframe_prompt_is_formatted_then_only_the_selected_cast_is_anthropomorphic(self) -> None:
        """User Given a story with an inanimate news symbol When a keyframe prompt is formatted Then the source cue remains an object and only selected characters have faces."""
        storyboard = {
            "character": "Kirby, Bandana Waddle Dee",
            "news_trace": {
                "fit_assessment": {"best_fitting_emotion": "curiosity"},
                "visual_anchors": ["one small blue route-update icon"],
            },
            "native_shots": [
                {
                    "action": "Kirby leans toward the small blue route-update icon while Bandana Waddle Dee points at it.",
                    "state_change": "Kirby and Bandana Waddle Dee hold curious expressions beside the icon.",
                }
            ],
        }

        prompt = format_native_h3_keyframe_prompt(storyboard, frame="ending")

        self.assertIn("Only the selected cast are characters", prompt)
        self.assertIn("keep props inanimate", prompt)
        self.assertIn("News visual cue: one small blue route-update icon is part of the held pose", prompt)
        self.assertIn("preserve its described relationship to the character", prompt.casefold())
        self.assertNotIn("inanimate and secondary", prompt.casefold())
        self.assertIn("three-quarter front", prompt.casefold())
        self.assertIn("keep both eye-lines on its named story event", prompt.casefold())
        self.assertNotIn("Curiosity pose priority: show the characters leaning", prompt)
        self.assertLess(prompt.index("Held final pose:"), prompt.index("Camera:"))
        self.assertNotIn("Contact:", prompt)

    def test_user_given_news_cue_already_in_the_held_pose_when_ending_prompt_is_formatted_then_it_is_not_duplicated(self) -> None:
        """User Given the final pose already contains the news symbol When its ending prompt is formatted Then the cue refers to that same object and does not invite another copy."""
        storyboard = {
            "character": "Kirby, Bandana Waddle Dee",
            "news_trace": {"visual_anchors": ["small blue circular service-update icon"]},
            "native_shots": [
                {
                    "action": "Bandana Waddle Dee nudges a blue marble toward Kirby while the service-update icon remains centered between their hands.",
                    "state_change": "Kirby catches the blue marble in both hands as Bandana Waddle Dee keeps his spear tip against its edge; the service-update icon stays between their hands at eye level.",
                }
            ],
        }

        prompt = format_native_h3_keyframe_prompt(storyboard, frame="ending")

        self.assertIn("News visual cue: small blue circular service-update icon is part of the held pose", prompt)
        self.assertIn("preserve its described relationship to the character", prompt.casefold())
        self.assertNotIn("once in the small background beside the interaction", prompt.casefold())
        self.assertEqual(prompt.count("News visual cue:"), 1)
        self.assertIn("Contact: keep the named tool", prompt)
        self.assertIn("catches the blue marble in both hands", prompt)

    def test_user_given_named_tool_action_when_ending_prompt_is_formatted_then_tool_stays_in_hand_on_exact_cue(self) -> None:
        """User Given a source cue and a tool-led ending action When its keyframe prompt is formatted Then the selected character visibly holds the tool against that same cue."""
        storyboard = {
            "character": "Kirby, Bandana Waddle Dee",
            "news_trace": {"visual_anchors": ["one blue marble"]},
            "native_shots": [
                {
                    "action": "Bandana Waddle Dee taps the blue marble with the rounded tip of his spear while Kirby laughs.",
                    "state_change": "Kirby and Bandana Waddle Dee grin together as Waddle Dee keeps the spear tip against the blue marble.",
                }
            ],
        }

        prompt = format_native_h3_keyframe_prompt(storyboard, frame="ending")

        self.assertIn("Waddle Dee keeps the spear tip against the blue marble", prompt)
        self.assertNotIn("taps the blue marble", prompt)
        self.assertIn("Contact: keep the named tool in hand", prompt)
        self.assertIn("with its working end on the exact target", prompt)

    def test_user_given_unrelated_tool_and_prop_contact_when_ending_prompt_is_formatted_then_the_tool_is_not_redirected_to_that_contact(self) -> None:
        """User Given a spear resting on a shoulder while a separate flagpole touches the news icon When an ending prompt is formatted Then the unrelated prop contact does not redirect the spear toward the icon."""
        storyboard = {
            "character": "Kirby, Bandana Waddle Dee",
            "news_trace": {"visual_anchors": ["one flat gold star icon"]},
            "native_shots": [
                {
                    "action": "Bandana Waddle Dee guides a small flag with his spear and then rests the spear on his shoulder.",
                    "state_change": "Bandana Waddle Dee rests his spear against his shoulder; the flagpole rests against the flat gold star icon on the table while Kirby watches.",
                }
            ],
        }

        prompt = format_native_h3_keyframe_prompt(storyboard, frame="ending")

        self.assertIn("flagpole rests against the flat gold star icon", prompt)
        self.assertNotIn("Keep the named character's tool in that character's hand", prompt)
        self.assertNotIn("keep its named working end in contact", prompt)
        self.assertIn("Camera: three-quarter front two-shot", prompt)

    def test_user_given_transforming_visual_anchor_when_ending_frame_is_formatted_then_only_the_post_conversion_object_is_named(self) -> None:
        """User Given a source cue that describes an object changing form When the ending still is formatted Then it names the final object without asking the image model to replay the earlier transformation."""
        storyboard = {
            "character": "Kirby, Bandana Waddle Dee",
            "news_trace": {
                "visual_anchors": ["a dotted route line transforming into a folded paper route map"],
            },
            "native_shots": [
                {
                    "action": "Kirby and Bandana Waddle Dee laugh together beside the folded paper route map.",
                    "state_change": "Both friends hold a warm, relaxed pose with the folded paper route map resting beside them.",
                }
            ],
        }

        prompt = format_native_h3_keyframe_prompt(storyboard, frame="ending")

        self.assertIn("News visual cue: a folded paper route map is part of the held pose", prompt)
        self.assertNotIn("dotted route line", prompt)
        self.assertNotIn("transforming into", prompt)

    def test_user_given_morphing_visual_anchor_when_ending_frame_is_formatted_then_only_the_final_object_is_named(self) -> None:
        """User Given a visual anchor that says an object is morphing into its final form When the ending keyframe prompt is formatted Then it names only the final form already held in the pose."""
        storyboard = {
            "character": "Kirby, Bandana Waddle Dee",
            "news_trace": {
                "visual_anchors": ["a dotted route line morphing into a folded paper route map"],
            },
            "native_shots": [
                {
                    "action": "Kirby and Bandana Waddle Dee look at the folded paper route map.",
                    "state_change": "Both friends hold a curious pose beside the folded paper route map resting on the table.",
                }
            ],
        }

        prompt = format_native_h3_keyframe_prompt(storyboard, frame="ending")

        self.assertIn("News visual cue: a folded paper route map is part of the held pose", prompt)
        self.assertNotIn("morphing into", prompt.casefold())
        self.assertNotIn("dotted route line", prompt.casefold())

    def test_user_given_contact_that_ended_before_the_held_pose_when_ending_prompt_is_formatted_then_the_old_action_is_not_replayed(self) -> None:
        """User Given an earlier tool interaction whose object is absent from the held pose When the ending keyframe prompt is formatted Then the image prompt shows only the current pose and does not resurrect the finished interaction."""
        storyboard = {
            "character": "Kirby, Bandana Waddle Dee",
            "news_trace": {"visual_anchors": ["one small blue route-update icon"]},
            "native_shots": [
                {
                    "action": "Kirby and Bandana Waddle Dee grin after the catch; Waddle Dee lowers his spear.",
                    "physical_cause": "Waddle Dee's spear catches Kirby's bubble before it pops.",
                    "state_change": "Kirby and Bandana Waddle Dee stand side by side with relaxed hands and bright, satisfied smiles.",
                }
            ],
        }

        prompt = format_native_h3_keyframe_prompt(storyboard, frame="ending")

        self.assertIn("Held final pose: Kirby and Bandana Waddle Dee stand side by side", prompt)
        self.assertNotIn("bubble", prompt.casefold())
        self.assertNotIn("spear tip", prompt.casefold())
        self.assertNotIn("tool visibly in that character's hand", prompt)

    def test_user_given_ending_beat_with_before_and_after_when_single_frame_is_formatted_then_only_the_held_pose_is_sent(self) -> None:
        """User Given an ending beat whose state change describes earlier and later emotions When its keyframe is formatted Then the prompt contains only the held visual pose."""
        storyboard = {
            "character": "Kirby, Bandana Waddle Dee",
            "native_shots": [
                {
                    "action": "Kirby sits up with wide, delighted eyes as Bandana Waddle Dee leans down to help.",
                    "state_change": "Kirby and Bandana Waddle Dee hold a close, relieved embrace; Kirby sits up with wide eyes while Waddle Dee wraps both arms around him.",
                }
            ],
        }

        prompt = format_native_h3_keyframe_prompt(storyboard, frame="ending")

        self.assertIn("Held final pose: Kirby and Bandana Waddle Dee hold a close, relieved embrace", prompt)
        self.assertIn("Kirby sits up with wide eyes", prompt)
        self.assertNotIn("transitions from a surprised tumble", prompt)
        self.assertNotIn("Main action:", prompt)

    def test_user_given_story_details_with_terminal_punctuation_when_keyframe_prompt_is_formatted_then_labels_are_not_duplicated(self) -> None:
        """User Given source, setting, and action details that already end in punctuation When a keyframe prompt is formatted Then each label has one clean sentence ending."""
        storyboard = {
            "world": {"setting": "A quiet meadow.", "visual_language": "Soft watercolor."},
            "news_trace": {
                "source_fact": "A saved action can be reused.",
                "news_mechanism": "The same action is callable again.",
            },
            "native_shots": [
                {
                    "action": "Kirby repeats the same small wave.",
                    "physical_cause": "One saved action is called again.",
                    "state_change": "Waddle Dee's surprised face turns delighted.",
                }
            ],
        }

        prompt = format_native_h3_keyframe_prompt(storyboard, frame="ending")

        self.assertIn("Setting: A quiet meadow.", prompt)
        self.assertIn("Held final pose: Waddle Dee's surprised face turns delighted.", prompt)
        self.assertNotIn("Main action: Kirby repeats the same small wave", prompt)
        self.assertNotIn("meadow..", prompt.casefold())
        self.assertNotIn("wave..", prompt.casefold())

    def test_user_given_comic_news_ending_with_contact_payload_and_cue_when_image_prompt_is_built_then_the_main_visual_reads_first_and_stays_compact(self) -> None:
        """User Given a comic news ending with cheek contact, a completed task, and a small news cue When its image prompt is built Then contact and payoff lead, faces and identity stay readable, and secondary details do not bury the gag."""
        storyboard = {
            "character": "Kirby, Bandana Waddle Dee",
            "subject_context": {
                "subjects": [
                    {"name": "Kirby", "profile": {"visual_description": "round pink body, large blue oval eyes, red feet"}},
                    {"name": "Bandana Waddle Dee", "profile": {"visual_description": "small orange body, blue bandana, short spear"}},
                ]
            },
            "world": {"setting": "a cozy workroom", "visual_language": "expressive watercolor with warm paper texture"},
            "news_trace": {
                "source_title": "A city launches a new transit app",
                "source_fact": "The transit app shows a route update.",
                "source_concepts": ["route update"],
                "fit_assessment": {"best_fitting_emotion": "comedy"},
                "visual_anchors": ["a tiny blue route-update icon"],
            },
            "native_shots": [
                {
                    "action": "Kirby and Bandana Waddle Dee turn toward one another after hanging the picture frame.",
                    "state_change": "Kirby and Bandana Waddle Dee hold a gentle cheek bump, their cheeks visibly squished together. Kirby has bright sparkling eyes and a delighted open-mouth smile; Bandana Waddle Dee has raised brows and a surprised laugh. The small wooden picture frame hangs level on the wall behind them. A tiny blue route-update icon is in the far background.",
                }
            ],
        }

        prompt = format_native_h3_keyframe_prompt(storyboard, style="warm, expressive watercolor", frame="ending")

        self.assertLess(prompt.index("cheek bump"), prompt.index("Camera:"))
        self.assertIn("cheeks visibly squished together", prompt)
        self.assertIn("complete task result large and clear in the lower-center foreground", prompt.casefold())
        self.assertIn("picture frame hangs level", prompt)
        self.assertIn("delighted open-mouth smile", prompt)
        self.assertIn("three-quarter front", prompt.casefold())
        self.assertIn("preserve its described relationship to the character", prompt.casefold())
        self.assertEqual(prompt.count("News visual cue:"), 1)

    def test_user_given_style_words_mixed_with_news_anchors_when_ending_prompt_is_formatted_then_only_the_source_cue_is_used(self) -> None:
        """User Given news anchors mixed with watercolor and palette notes When an ending prompt is formatted Then only the source-linked symbol becomes its news cue and the camera does not invent contact."""
        storyboard = {
            "character": "Kirby, Bandana Waddle Dee",
            "news_trace": {
                "source_title": "A chainable Skills feature arrives",
                "source_fact": "Saved Skills can be chained to complete complex tasks.",
                "source_concepts": ["Skills", "chaining"],
                "fit_assessment": {"best_fitting_emotion": "curiosity"},
                "visual_anchors": [
                    "watercolor pastel colors",
                    "paper texture",
                    "a small gear-shaped Skill icon",
                ],
            },
            "native_shots": [
                {
                    "action": "Kirby and Bandana Waddle Dee look at a small gear-shaped Skill icon between them.",
                    "state_change": "Kirby and Bandana Waddle Dee hold curious expressions as the small gear-shaped Skill icon glows between them.",
                }
            ],
        }

        prompt = format_native_h3_keyframe_prompt(storyboard, frame="ending")
        camera = prompt.split("Camera:", 1)[1].split(".", 1)[0]

        self.assertIn("News visual cue: a small gear-shaped Skill icon is part of the held pose", prompt)
        self.assertNotIn("News visual cue: watercolor", prompt.casefold())
        self.assertNotIn("News visual cue: paper texture", prompt.casefold())
        self.assertNotIn("contact", camera.casefold())

    def test_user_given_news_analogy_is_already_the_held_payload_when_ending_prompt_is_formatted_then_it_is_not_duplicated_in_the_background(self) -> None:
        """User Given a source-linked visual analogy already shown as the held task result When an ending image prompt is formatted Then the cue stays in its described position without adding a duplicate background object."""
        storyboard = {
            "character": "Kirby, Bandana Waddle Dee",
            "subject_context": {
                "subjects": [
                    {"name": "Kirby", "profile": {"visual_description": "round pink body, large blue eyes, red feet"}},
                    {"name": "Bandana Waddle Dee", "profile": {"visual_description": "small orange body, simple face, blue bandana"}},
                ]
            },
            "news_trace": {
                "source_category": "product_or_service",
                "source_fact": "Saved Skills can be chained to complete a complex task.",
                "source_concepts": ["Skills", "chaining"],
                "fit_assessment": {"best_fitting_emotion": "curiosity"},
                "visual_anchors": ["interlocking puzzle blocks representing converted Skills"],
            },
            "native_shots": [
                {
                    "state_change": (
                        "Kirby and Bandana Waddle Dee lean toward interlocking wooden puzzle blocks "
                        "joined in a short chain on the workbench."
                    )
                }
            ],
        }

        prompt = format_native_h3_keyframe_prompt(storyboard, frame="ending")

        self.assertIn("News visual cue: interlocking puzzle blocks representing converted Skills is part of the held pose", prompt)
        self.assertIn("preserve its described relationship to the character", prompt.casefold())
        self.assertNotIn("once in the small background", prompt)
        self.assertNotIn("not used as a prop", prompt.casefold())

    def test_user_given_source_analogy_in_the_character_action_when_ending_prompt_is_formatted_then_the_task_object_stays_visible_and_operable(self) -> None:
        """User Given the news analogy is the object the characters are actively connecting When an ending prompt is formatted Then it stays in the clear foreground and available for that named action."""
        storyboard = {
            "character": "Kirby, Bandana Waddle Dee",
            "subject_context": {
                "subjects": [
                    {"name": "Kirby", "profile": {"visual_description": "round pink body, large blue eyes, red feet"}},
                    {"name": "Bandana Waddle Dee", "profile": {"visual_description": "small orange body, simple face, blue bandana"}},
                ]
            },
            "news_trace": {
                "source_category": "product_or_service",
                "source_fact": "Saved Skills can be chained to complete a complex task.",
                "source_concepts": ["Skills", "chaining"],
                "fit_assessment": {"best_fitting_emotion": "curiosity"},
                "visual_anchors": ["three plain wooden rings linked into one chain, a visual analogy for combined Skills"],
            },
            "native_shots": [
                {
                    "state_change": (
                        "Kirby and Bandana Waddle Dee lean toward the three wooden rings; "
                        "Bandana Waddle Dee threads the final ring through the other two."
                    )
                }
            ],
        }

        prompt = format_native_h3_keyframe_prompt(storyboard, frame="ending")

        self.assertIn("three plain wooden rings linked into one chain, a visual analogy for combined Skills is part of the held pose", prompt)
        self.assertIn("Main visual focus: the complete linked object is large and unmistakable", prompt)
        self.assertIn("keep every ring separately recognizable and countable, with its own open center and outline", prompt)
        self.assertIn("never fuse the chain into one oversized ring, donut, or bagel", prompt)
        self.assertIn("hands frame the result without covering any link", prompt)
        self.assertIn("News visual cue: three plain wooden rings linked into one chain", prompt)
        self.assertIn("preserve its described relationship to the character", prompt.casefold())
        self.assertNotIn("once in the small background", prompt.casefold())
        self.assertIn("complete task result large and clear in the lower-center foreground", prompt.casefold())
        self.assertLess(prompt.index("Held final pose:"), prompt.index("Main visual focus:"))
        self.assertLess(prompt.index("Main visual focus:"), prompt.index("News visual cue:"))
        self.assertLess(prompt.index("News visual cue:"), prompt.index("Visual identity:"))
        self.assertLess(prompt.index("Mood:"), prompt.index("Camera:"))
        self.assertNotIn("not used as a prop", prompt.casefold())

    def test_user_given_kirby_pair_with_distinct_face_designs_when_curiosity_prompt_is_formatted_then_emotion_respects_each_character_and_uses_body_acting(self) -> None:
        """User Given Kirby and Bandana Waddle Dee with their supplied face designs When a curiosity ending prompt is formatted Then Kirby gets a thumbnail-readable open O mouth, Waddle Dee keeps its familiar simple face and shows interest through eyes and posture, and visual identity precedes mood direction."""
        pose = "Kirby and Bandana Waddle Dee face one another beside the event."
        sadness = format_native_h3_keyframe_prompt(
            {
                "character": "Kirby, Bandana Waddle Dee",
                "subject_context": {
                    "subjects": [
                        {"name": "Kirby", "profile": {"visual_description": "round pink body, large blue eyes, red feet"}},
                        {"name": "Bandana Waddle Dee", "profile": {"visual_description": "small orange body, blue bandana, simple face with dark eyes and rosy cheeks"}},
                    ]
                },
                "news_trace": {"fit_assessment": {"best_fitting_emotion": "sadness"}},
                "native_shots": [{"state_change": pose}],
            },
            frame="ending",
        )
        curiosity = format_native_h3_keyframe_prompt(
            {
                "character": "Kirby, Bandana Waddle Dee",
                "subject_context": {
                    "subjects": [
                        {"name": "Kirby", "profile": {"visual_description": "round pink body, large blue eyes, red feet"}},
                        {"name": "Bandana Waddle Dee", "profile": {"visual_description": "small orange body, blue bandana, simple face with dark eyes and rosy cheeks"}},
                    ]
                },
                "news_trace": {"fit_assessment": {"best_fitting_emotion": "curiosity"}},
                "native_shots": [{"state_change": "Bandana Waddle Dee uses both hands to connect the final wooden ring while Kirby watches with curious eyes."}],
            },
            frame="ending",
        )

        self.assertIn("downcast eyes", sadness)
        self.assertIn("Bandana Waddle Dee", sadness)
        self.assertIn("familiar simple face", sadness)
        self.assertIn("do not force a smile", sadness)
        self.assertIn("separate, clearly open dark o-mouth roughly as tall as one blue eye", curiosity.casefold())
        self.assertIn("so the opening reads as an expression, not a dot or smile line", curiosity.casefold())
        self.assertIn("Bandana Waddle Dee shows curiosity through bright attentive eyes and lifted cheeks", curiosity)
        self.assertIn("uses both hands to connect the final wooden ring", curiosity)
        self.assertIn("do not add any contact or gesture beyond the pose named there", curiosity)
        self.assertIn("include only a mouth detail named in Held final pose", curiosity)
        self.assertLess(curiosity.index("Held final pose:"), curiosity.index("Visual identity:"))
        self.assertLess(curiosity.index("Mood:"), curiosity.index("Camera:"))

    def test_user_given_news_analogy_and_character_reaction_when_ending_keyframe_is_formatted_then_limbs_stay_visible_and_props_do_not_gain_marks(self) -> None:
        """User Given a character's curiosity depends on a hand gesture beside an unbranded news analogy When an ending keyframe prompt is formatted Then the frame includes the reaction limbs and result, preserves supplied character design, and keeps the analogy free of pseudo-writing."""
        storyboard = {
            "character": "Kirby, Bandana Waddle Dee",
            "news_trace": {
                "source_fact": "Saved Skills can be chained to complete a task.",
                "source_concepts": ["Skills", "chaining"],
                "fit_assessment": {"best_fitting_emotion": "curiosity"},
                "visual_anchors": ["plain connected wooden blocks representing chained Skills"],
            },
            "native_shots": [
                {
                    "state_change": "Kirby lifts both arms as Bandana Waddle Dee leans toward the connected wooden blocks.",
                }
            ],
        }

        prompt = format_native_h3_keyframe_prompt(storyboard, frame="ending")

        self.assertIn("both full figures with space around them and readable faces", prompt)
        self.assertIn("no close crop", prompt)
        self.assertIn("Preserve each character's supplied appearance and face design", prompt)
        self.assertIn("keep the story's single news analogy visible and unbranded", prompt)


if __name__ == "__main__":
    unittest.main()
