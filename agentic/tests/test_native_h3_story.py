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
from agentic.runtime.prompting import LONG_VIDEO_SYSTEM_PROMPT
from agentic.skills.agent_primitives import AgentMediaSkills
from agentic.skills.longvideo import LongVideoSkills
from agentic.storyboard import (
    evaluate_native_h3_news_grounding,
    evaluate_native_h3_story_quality,
    format_native_h3_prompt,
    ground_native_h3_ending_keyframe_prompt,
    load_storyboard,
    merge_native_h3_storyboard,
    validate_native_h3_shot_timing,
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
        self.assertEqual(plan.workflow_name, "minimax_h3_lowvram_15s_fl2va_i2v")
        self.assertEqual(
            [node.node_id for node in plan.nodes],
            [
                "native-story-prompt",
                "native-image-asset-check",
                "native-video-asset-check",
                "native-opening-keyframe",
                "native-opening-review",
                "native-keyframe-gate",
                "native-h3-render",
                "native-h3-speed",
                "native-h3-qa",
                "native-h3-preview",
                "native-h3-package",
            ],
        )
        opening = next(node for node in plan.nodes if node.node_id == "native-opening-keyframe")
        opening_review = next(node for node in plan.nodes if node.node_id == "native-opening-review")
        keyframe_gate = next(node for node in plan.nodes if node.node_id == "native-keyframe-gate")
        render = next(node for node in plan.nodes if node.node_id == "native-h3-render")
        qa = next(node for node in plan.nodes if node.node_id == "native-h3-qa")
        story_prompt = next(node for node in plan.nodes if node.node_id == "native-story-prompt")
        self.assertEqual(opening.inputs["image_count"], 6)
        self.assertEqual(opening_review.inputs["review_all_candidates"], True)
        self.assertEqual(opening_review.inputs["review_scope"], "first_frame")
        self.assertEqual(opening_review.inputs["review_notes"], "stage: preview")
        self.assertEqual(opening_review.depends_on, ["native-opening-keyframe"])
        self.assertEqual(keyframe_gate.inputs["opening_node"], "native-opening-review")
        self.assertTrue(keyframe_gate.inputs["preserve_opening_frame"])
        self.assertFalse(keyframe_gate.inputs["use_last_frame"])
        self.assertEqual(keyframe_gate.depends_on, ["native-opening-review"])
        self.assertEqual(keyframe_gate.inputs["max_regenerations"], 0)
        self.assertIn("native-opening-review", keyframe_gate.depends_on)
        self.assertIn("native-keyframe-gate", render.depends_on)
        self.assertFalse(render.inputs["use_last_frame"])
        self.assertEqual(story_prompt.inputs["render_mode"], "image_to_video")
        self.assertEqual(qa.inputs["mode"], "technical_and_semantic_qa_before_optional_discord_review")
        self.assertEqual(qa.inputs["video_node"], "native-h3-speed")
        self.assertEqual(qa.inputs["target_duration"], 7.5)
        self.assertEqual(qa.inputs["expected_fps"], 24.0)
        self.assertIsNone(qa.tool_name)
        self.assertEqual(plan.metadata["native_h3"]["keyframe_candidate_count"], 6)
        self.assertFalse(plan.metadata["native_h3"]["use_last_frame"])
        self.assertIn("native_h3", plan.metadata)

    def test_native_h3_last_frame_mode_adds_review_and_passes_both_frames(self) -> None:
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
        gate = next(node for node in plan.nodes if node.node_id == "native-keyframe-gate")
        render = next(node for node in plan.nodes if node.node_id == "native-h3-render")
        self.assertEqual(ending_review.inputs["review_scope"], "last_frame")
        self.assertEqual(ending_review.depends_on, ["native-ending-keyframe"])
        self.assertTrue(gate.inputs["use_last_frame"])
        self.assertTrue(gate.inputs["preserve_ending_frame"])
        self.assertEqual(gate.inputs["ending_node"], "native-ending-review")
        self.assertIn("native-ending-review", gate.depends_on)
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
        gate = next(node for node in plan.nodes if node.node_id == "native-keyframe-gate")
        self.assertEqual(opening.inputs["image_count"], 6)
        self.assertTrue(review.inputs["auto_select_for_probe"])
        self.assertFalse(review.inputs["require_human_review"])
        self.assertEqual(gate.inputs["opening_node"], "native-opening-review")
        self.assertFalse(gate.inputs["preserve_opening_frame"])
        self.assertTrue(plan.metadata["native_h3"]["stage_probe_auto_select"])

    def test_news_only_native_h3_does_not_validate_autonomous_brief_as_user_objective(self) -> None:
        captured: dict[str, object] = {}
        storyboard_fixture = load_storyboard(self.repo_root / "configs/storyboards/kirby_native_15s.yaml")
        storyboard_fixture.update(
            {
                "opening_keyframe_prompt": "Kirby reacts to a visible news-derived disruption in a clear meadow composition.",
                "ending_keyframe_prompt": "Kirby resolves the news-derived disruption and restores the scene.",
                "world": {"setting": "a high-tech semiconductor laboratory"},
                "news_trace": {
                    "visual_anchors": ["lab conveyor", "scanning arch", "sealed bubble"],
                    "news_mechanism": "the scanning arch traps the subject inside the bubble",
                    "news_consequence": "the sealed bubble leaves the lab route blocked",
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
                    "story_quality": {},
                    "news_grounding": {},
                }

        context = SimpleNamespace(
            plan=SimpleNamespace(
                goal=SimpleNamespace(
                    prompt="Autonomous scene generated from the selected news",
                    style="polished 2D anime",
                    constraints={
                        "character": "Kirby",
                        "prompt_source": "news",
                        "native_h3_creative_brief": "cute micro-gag with one prop and a visible payoff",
                        "news_context": {"title": "AI companion robot arrives", "keyword": "AI;robot"},
                    },
                )
            ),
            node=SimpleNamespace(
                inputs={
                    "storyboard_path": "configs/storyboards/kirby_native_15s.yaml",
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
        self.assertEqual(captured["creative_brief"], "cute micro-gag with one prop and a visible payoff")
        ending_prompt = str(result.outputs["ending_keyframe_prompt"])
        self.assertIn("high-tech semiconductor laboratory", ending_prompt)
        self.assertIn("scanning arch traps the subject", ending_prompt)
        self.assertIn("sealed bubble leaves the lab route blocked", ending_prompt)

    def test_native_h3_workflow_manifest_has_runtime_prompt_placeholder(self) -> None:
        workflow_path = self.repo_root / "configs" / "workflow" / "minimax_h3_lowvram_15s_fl2va_i2v.json"
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        prompt = str(workflow["5"]["inputs"]["prompt"])

        self.assertIn("runtime prompt placeholder", prompt.lower())
        self.assertNotIn("golden star seed", prompt.lower())

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

        result = AgentMediaSkills(FakeTools(), temp_root).validate_character_frames(context)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.outputs["first_frame_path"], str(opening_path))
        self.assertEqual(result.outputs["regenerated_count"], 0)
        self.assertEqual(result.outputs["identity_reports"][0]["validation"], "human_selected_immutable")

    def test_native_prompt_includes_creative_variation_and_cleans_title_artifact(self) -> None:
        storyboard = load_storyboard(self.repo_root / "configs/storyboards/kirby_native_15s.yaml")
        prompt = format_native_h3_prompt(
            storyboard,
            creative_brief="a blue-hour summer storm",
            duration_seconds=15,
        )
        self.assertIn("Creative variation for this run", prompt)
        self.assertIn("Hook", prompt)
        self.assertNotIn("??", prompt)
        self.assertIn("15-second", prompt)
        self.assertIn("Cute gag:", prompt)
        self.assertIn("loop the opening", prompt)

    def test_native_prompt_carries_the_single_visual_gag_contract(self) -> None:
        storyboard = load_storyboard(self.repo_root / "configs/storyboards/kirby_native_15s.yaml")
        storyboard["gag_card"] = {
            "hook_frame": "Kirby is already being dragged by a runaway cushion",
            "character_desire": "Kirby wants one soft nap",
            "prop_rule": "The cushion springs away whenever Kirby lands on it",
            "setback": "The cushion slings Kirby into a fluffy tumble",
            "expressive_reaction": "Kirby freezes wide-eyed, then puffs his cheeks",
            "payoff_reversal": "The cushion snaps back and hugs Kirby instead",
            "loop_reason": "The final hug looks like the opening bounce in reverse",
        }

        prompt = format_native_h3_prompt(storyboard, duration_seconds=15)

        self.assertIn("Single visual gag contract", prompt)
        self.assertIn("Prop rule", prompt)
        self.assertIn("Final reversal", prompt)
        self.assertIn("Replay reason", prompt)

    def test_native_gag_card_is_preserved_when_story_is_merged(self) -> None:
        base_storyboard = load_storyboard(self.repo_root / "configs/storyboards/kirby_native_15s.yaml")
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
                "hook_frame": "Kirby is already dragged sideways by a runaway cushion.",
                "character_desire": "Kirby wants to settle onto the cushion for a nap.",
                "prop_rule": "The cushion springs away whenever Kirby lands on it.",
                "setback": "The cushion slings Kirby into a fluffy tumble.",
                "expressive_reaction": "Kirby freezes wide-eyed and puffs his cheeks.",
                "payoff_reversal": "The cushion snaps back and hugs Kirby instead.",
                "loop_reason": "The final hug echoes the opening sideways bounce in reverse.",
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
        self.assertTrue(
            any("Exactly the two declared subject slots" in rule for rule in pair_merged["world"]["continuity_rules"])
        )
        self.assertFalse(any("Only one" in rule for rule in pair_merged["world"]["continuity_rules"]))
        self.assertNotIn("duplicate Kirby", pair_merged["negative_prompt"])
        self.assertIn("unrequested third subject", pair_merged["negative_prompt"])

    def test_native_gag_card_quality_rejects_missing_content(self) -> None:
        story = {
            "story_spine": {
                "premise": "A cushion escapes Kirby.",
                "objective": "Kirby must catch the cushion.",
                "obstacle": "The cushion springs away.",
                "stakes": "Kirby loses his nap.",
                "climax": "Kirby lets the cushion bounce into him.",
                "resolution": "The cushion hugs Kirby.",
            },
            "gag_card": {"hook_frame": "Kirby is dragged by a cushion"},
            "native_shots": [
                {"action": "The cushion springs away and drags Kirby.", "camera": "Follow the slide.", "state_change": "Kirby chases the cushion."},
                {"action": "The cushion bounces Kirby into the grass.", "camera": "Push into Kirby's reaction.", "state_change": "Kirby loses his nap."},
                {"action": "The cushion hugs Kirby.", "camera": "Pull out on the cozy pile.", "state_change": "Kirby gets his nap."},
            ],
        }

        quality = evaluate_native_h3_story_quality(story)

        self.assertFalse(quality["passed"])
        self.assertFalse(quality["checks"]["gag_card_complete"])
        self.assertTrue(any("gag_card" in error for error in quality["errors"]))

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
        self.assertEqual(plan.workflow_name, "minimax_h3_lowvram_t2v")
        render = next(node for node in plan.nodes if node.node_id == "native-h3-render")
        self.assertEqual(render.tool_name, "comfy.workflow.text_to_video")
        self.assertEqual(plan.metadata["recipe"], "native_h3_t2v_story")
        self.assertEqual(plan.metadata["native_h3"]["length"], 362)
        self.assertTrue(plan.metadata["native_h3"]["lowvram_preview"])
        self.assertEqual(plan.metadata["native_h3"]["steps"], 16)
        self.assertNotIn("native-opening-keyframe", [node.node_id for node in plan.nodes])
        self.assertNotIn("native-opening-review", [node.node_id for node in plan.nodes])

    def test_native_h3_timing_allows_adjusted_contiguous_20_second_beats(self) -> None:
        shots = [
            {"time": "0-2.5s"},
            {"time": "2.5-6.25s"},
            {"time": "6.25-11.5s"},
            {"time": "11.5-16.75s"},
            {"time": "16.75-20s"},
        ]
        valid, error = validate_native_h3_shot_timing(shots, duration_seconds=20)
        self.assertTrue(valid, error)

        invalid, error = validate_native_h3_shot_timing(
            [{"time": "0-2s"}, {"time": "2.5-20s"}],
            duration_seconds=20,
        )
        self.assertFalse(invalid)
        self.assertIn("start where the previous shot ends", error)

    def test_native_prompt_rejects_unbounded_scene_rewrite(self) -> None:
        storyboard = load_storyboard(self.repo_root / "configs/storyboards/kirby_native_15s.yaml")
        prompt = format_native_h3_prompt(
            storyboard,
            creative_brief=(
                "Kirby floats through a digital archive, inhales red shards, and transforms them into stars "
                "while the camera performs an orbital rotation in a 3D void."
            ),
        )
        self.assertIn("a restrained atmospheric variation", prompt)
        self.assertNotIn("digital archive", prompt)
        self.assertNotIn("red shards", prompt)

    def test_native_h3_story_is_generated_from_news_and_replaces_fixed_plot(self) -> None:
        base_storyboard = load_storyboard(self.repo_root / "configs/storyboards/kirby_native_15s.yaml")
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
        self.assertEqual(len(merged["segments"]), 3)
        self.assertIn("lantern", prompt.lower())
        self.assertNotIn("golden star seed", prompt.lower())
        self.assertNotIn("dark sky rift", prompt.lower())

    def test_native_h3_news_grounding_rejects_generic_story_without_trace(self) -> None:
        generic_story = {
            "name": "Kirby Seed Storm",
            "story_spine": {
                "premise": "A storm threatens a glowing seed.",
                "objective": "Kirby must protect the seed.",
                "obstacle": "The wind tears the seed away.",
                "stakes": "The meadow will wither.",
                "climax": "Kirby plants the seed.",
                "resolution": "The meadow blooms again.",
            },
            "native_shots": [
                {"action": "A storm tears a seed loose.", "camera": "Follow the seed.", "state_change": "The seed is lost."},
                {"action": "Kirby chases the seed through wind.", "camera": "Track Kirby.", "state_change": "Kirby reaches the seed."},
                {"action": "Kirby plants the seed and the meadow blooms.", "camera": "Pull out wide.", "state_change": "The meadow is restored."},
            ],
        }

        quality = evaluate_native_h3_news_grounding(
            generic_story,
            {"title": "AI companion robot arrives", "keyword": "AI;robot"},
            creative_brief="Kirby protects one glowing seed",
        )

        self.assertFalse(quality["passed"])
        self.assertFalse(quality["checks"]["news_trace_present"])

    def test_native_h3_ending_keyframe_prompt_locks_news_scene_and_payoff(self) -> None:
        story = {
            "ending_keyframe_prompt": "Magolor remains suspended in the foreground.",
            "world": {"setting": "a high-tech semiconductor laboratory"},
            "news_trace": {
                "visual_anchors": ["lab conveyor", "scanning arch", "sealed bubble"],
                "news_mechanism": "the scanning arch traps the subject inside the bubble",
                "news_consequence": "the sealed bubble leaves the lab route blocked",
            },
        }

        prompt = ground_native_h3_ending_keyframe_prompt(story)

        self.assertIn("high-tech semiconductor laboratory", prompt)
        self.assertIn("lab conveyor", prompt)
        self.assertIn("scanning arch", prompt)
        self.assertIn("sealed bubble", prompt)
        self.assertIn("scanning arch traps the subject", prompt)
        self.assertIn("sealed bubble leaves the lab route blocked", prompt)

    def test_native_h3_news_grounding_accepts_small_anchor_wording_variations(self) -> None:
        story = {
            "name": "Kirby and the Typhoon Seed",
            "story_spine": {
                "premise": "Kirby protects a glowing seed from a typhoon vortex.",
                "objective": "Protect the glowing seed.",
            },
            "news_trace": {
                "contract_version": 2,
                "source_title": "Typhoon warning",
                "source_concepts": ["typhoon"],
                "visual_translation": "A typhoon vortex funnel threatens the meadow.",
                "news_mechanism": "the typhoon vortex pulls the glowing seed across the meadow",
                "news_consequence": "the typhoon vortex still pulls the glowing seed across the meadow in the payoff",
                "visual_anchors": ["typhoon vortex funnel", "glowing seed", "meadow"],
                "anchor_roles": ["context", "mechanism", "consequence"],
                "integration": "The typhoon vortex threatens the glowing seed and forces Kirby to protect it.",
            },
            "native_shots": [
                {
                    "action": "A typhoon vortex pulls at the glowing seed across the meadow.",
                    "camera": "Follow the seed.",
                    "state_change": "The seed is displaced.",
                }
                for _ in range(5)
            ],
        }
        quality = evaluate_native_h3_news_grounding(
            story,
            {"title": "Typhoon warning", "keyword": "typhoon", "category": "weather"},
            creative_brief="Kirby protects the glowing seed",
        )

        self.assertTrue(quality["passed"], quality)

    def test_native_h3_news_contract_requires_mechanism_and_consequence(self) -> None:
        story = {
            "name": "Kirby and the Lantern Outage",
            "story_spine": {
                "premise": "A city lantern system begins shutting down in sequence.",
                "objective": "Kirby must reopen the canal path before the last light disappears.",
                "obstacle": "A synchronized blackout closes the path and reverses Kirby's route.",
                "stakes": "The canal becomes impassable when the final lantern goes dark.",
                "climax": "Kirby redirects the last beam and reopens the canal path.",
                "resolution": "Warm lantern light returns and the canal path is open again.",
            },
            "news_trace": {
                "contract_version": 2,
                "source_title": "city lantern outage",
                "source_concepts": ["outage", "lantern"],
                "visual_translation": "The outage becomes a synchronized lantern blackout that blocks a canal path and can be reversed by Kirby.",
                "news_mechanism": "synchronized lantern blackout blocks the canal path",
                "news_consequence": "the canal path reopens into warm light",
                "visual_anchors": ["city lanterns", "synchronized lantern blackout", "canal path reopens"],
                "anchor_roles": ["context", "mechanism", "consequence"],
                "integration": "The outage makes the canal path close, so Kirby must restore the lantern sequence before the route disappears.",
            },
            "native_shots": [
                {"action": "Kirby runs beneath the city lanterns as every light flickers in sequence."},
                {"action": "The synchronized lantern blackout blocks the canal path and forces Kirby to reverse."},
                {"action": "Kirby redirects the last lantern beam and the canal path reopens into warm light."},
            ],
        }
        quality = evaluate_native_h3_news_grounding(
            story,
            {"title": "city lantern outage", "keyword": "outage;lantern", "category": "city"},
        )
        self.assertTrue(quality["passed"], quality)
        self.assertTrue(quality["checks"]["news_mechanism_reaches_story"])
        self.assertTrue(quality["checks"]["news_consequence_reaches_payoff"])

        collapsed = json.loads(json.dumps(story))
        collapsed["news_trace"]["visual_anchors"] = ["glowing orb", "glowing orb", "glowing orb"]
        collapsed["news_trace"]["anchor_roles"] = ["prop", "prop", "prop"]
        collapsed["news_trace"]["news_mechanism"] = "the orb floats"
        collapsed["news_trace"]["news_consequence"] = "the orb glows steadily"
        collapsed["native_shots"] = [
            {"action": "Kirby catches the glowing orb."},
            {"action": "The glowing orb slips and Kirby catches it again."},
            {"action": "The glowing orb glows steadily in Kirby's hands."},
        ]
        rejected = evaluate_native_h3_news_grounding(
            collapsed,
            {"title": "city lantern outage", "keyword": "outage;lantern", "category": "city"},
        )
        self.assertFalse(rejected["passed"])
        self.assertFalse(rejected["checks"]["news_anchor_roles_complete"])
        self.assertFalse(rejected["checks"]["news_anchor_diversity"])
        self.assertFalse(rejected["checks"]["news_anchor_not_default_object_loop"])

    def test_native_h3_render_prompt_carries_news_mechanism_contract(self) -> None:
        storyboard = load_storyboard(self.repo_root / "configs/storyboards/kirby_native_15s.yaml")
        storyboard["news_trace"] = {
            "visual_translation": "A city blackout closes a canal path.",
            "news_mechanism": "synchronized lights shut down and block the route",
            "news_consequence": "the route reopens when Kirby redirects the final beam",
            "visual_anchors": ["city lights", "blocked route", "warm exit"],
            "anchor_roles": ["context", "mechanism", "consequence"],
        }
        prompt = format_native_h3_prompt(storyboard, duration_seconds=15)
        self.assertIn("News mechanism contract", prompt)
        self.assertIn("synchronized lights shut down", prompt)
        self.assertIn("Do not replace the mechanism with a generic floating object", prompt)

    def test_native_h3_news_grounding_keeps_compound_anchor_in_payoff(self) -> None:
        story = {
            "name": "Kirby and the Shadow Agent",
            "story_spine": {
                "premise": "Kirby confronts a rogue autonomous energy entity.",
                "objective": "Kirby must neutralize the aggressive energy source.",
                "resolution": "The monolith is destroyed and a stable glow remains.",
            },
            "news_trace": {
                "contract_version": 2,
                "source_title": "AI agent attack from abroad",
                "source_concepts": ["AI自主攻擊"],
                "visual_translation": "The AI自主攻擊 becomes a featureless black monolith that attacks autonomously.",
                "news_mechanism": "the featureless black monolith attacks Kirby with dark energy",
                "news_consequence": "the stable glow remains after Kirby destroys the featureless black monolith",
                "visual_anchors": ["Featureless black monolith", "dark energy", "stable glow"],
                "anchor_roles": ["context", "mechanism", "consequence"],
                "integration": "The AI自主攻擊 becomes the featureless black monolith that Kirby must neutralize.",
            },
            "native_shots": [
                {"action": "A featureless black monolith lashes out at Kirby."},
                {"action": "The featureless black monolith pulls Kirby toward it."},
                {"action": "Kirby destroys the monolith and a stable glow remains."},
            ],
        }

        quality = evaluate_native_h3_news_grounding(
            story,
            {"title": "AI agent attack from abroad", "keyword": "AI自主攻擊", "category": "news"},
        )

        self.assertTrue(quality["passed"], quality)

    def test_native_h3_safety_contract_rejects_readable_text_without_semantic_repair(self) -> None:
        storyboard = load_storyboard(self.repo_root / "configs/storyboards/kirby_native_15s.yaml")
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
            with self.assertRaisesRegex(PromptGenerationError, "forbidden readable-text visual cues"):
                engine.generate_native_h3_storyboard(
                    character="Kirby",
                    style="polished 2D anime",
                    duration_seconds=15,
                    base_storyboard=storyboard,
                    news_context={"title": "city lantern outage", "keyword": "canal safety"},
                )

        self.assertEqual(len(calls), 1)

    def test_native_h3_visual_cue_filter_allows_unmarked_physical_surfaces(self) -> None:
        allowed = LLMPromptEngine._find_native_h3_forbidden_visual_cues(
            "no text or logos; glowing floor panels, a data-core pedestal, clear screen geography, readable cause-and-effect, and a calm light display"
        )
        self.assertEqual(allowed, [])

        physical_stamp = LLMPromptEngine._find_native_h3_forbidden_visual_cues(
            "a giant unmarked golden stamp slams down, and the tracks are stamped by the physical coin"
        )
        self.assertEqual(physical_stamp, [])

        rejected = LLMPromptEngine._find_native_h3_forbidden_visual_cues(
            "a floor panel displays readable text beside a glowing button"
        )
        self.assertIn("readable text", rejected)
        self.assertIn("text-bearing surface", rejected)

        stamped_text = LLMPromptEngine._find_native_h3_forbidden_visual_cues(
            "a physical stamp presses readable words into the paper"
        )
        self.assertIn("stamp with readable content", stamped_text)

    def test_native_h3_rejects_text_cues_in_story_spine_and_keyframe_prompts(self) -> None:
        story = load_storyboard(self.repo_root / "configs/storyboards/kirby_native_15s.yaml")
        story.update(
            {
                "name": "Kirby and the Lantern Current",
                "base_prompt": "Kirby is the only protagonist in a moonlit canal city with a runaway lantern.",
                "opening_keyframe_prompt": "Kirby reaches for a lantern labeled APPROVED.",
                "ending_keyframe_prompt": "The lantern is stamped APPROVED as Kirby celebrates.",
                "native_audio": "water rush, cable tension, lantern chime, calm night ambience",
                "news_trace": {
                    "contract_version": 2,
                    "source_title": "city lantern outage",
                    "source_concepts": ["lantern"],
                    "visual_translation": "The outage becomes a runaway lantern.",
                    "news_mechanism": "the runaway lantern pulls its cable toward the whirlpool",
                    "news_consequence": "the lantern rises above the canal as warm light spreads",
                    "visual_anchors": ["lantern", "whirlpool", "warm light"],
                    "anchor_roles": ["context", "mechanism", "consequence"],
                    "integration": "Kirby protects the lantern while the outage makes it urgent.",
                },
                "native_shots": [
                    {"time": "0-4s", "title": "Lantern breaks loose", "action": "A runaway lantern yanks its cable toward the whirlpool while Kirby lunges for it.", "camera": "Follow Kirby and the moving lantern across the bridge.", "state_change": "Kirby commits to stopping the lantern before the canal loses its safe guide."},
                    {"time": "4-10s", "title": "Cable starts to fail", "action": "Kirby wraps the cable around a post as the current pulls harder and the bridge cracks.", "camera": "Push in on the tightening cable and Kirby's strained grip.", "state_change": "The cable holds briefly, but the cracking bridge makes the rescue more dangerous."},
                    {"time": "10-15s", "title": "Lantern returns", "action": "Kirby redirects the current and lifts the lantern above the canal as warm light spreads.", "camera": "Pull out from Kirby to the restored lantern constellation.", "state_change": "The lantern is safe, the canal is illuminated, and the original danger is resolved."},
                ],
            }
        )
        story["story_spine"].update(
            {
                "premise": "A current drags the city's guiding lantern toward a canal whirlpool.",
                "objective": "Kirby must redirect the lantern before the city loses its safe path.",
                "obstacle": "The current accelerates and the cable begins to snap.",
                "stakes": "Without the lantern, the canal city will be swallowed by darkness.",
                "emotional_arc": "curiosity becomes urgency, courage, and relief",
                "climax": "Kirby anchors the lantern against the current and redirects its light.",
                "resolution": "The lantern is stamped APPROVED and the canal is safe again.",
            }
        )

        with self.assertRaises(PromptGenerationError) as raised:
            LLMPromptEngine._validate_native_h3_story_payload(
                {"story": story},
                expected_times=tuple(str(shot["time"]) for shot in story["native_shots"]),
                duration_seconds=15,
                news_context={"title": "city lantern outage", "keyword": "lantern"},
            )

        self.assertIn("label", str(raised.exception))

    def test_native_h3_normalizes_timestamp_only_hook_from_opening_prompt(self) -> None:
        payload = {
            "story": {
                "opening_keyframe_prompt": "Kirby clutches a glowing coin as blue wind ribbons whip around him.",
                "gag_card": {"hook_frame": "0s"},
                "native_shots": [
                    {"action": "Kirby leaps while the wind knocks the glowing coin loose."}
                ],
            }
        }

        normalized = LLMPromptEngine._normalize_native_h3_story_payload(payload)

        self.assertEqual(
            normalized["story"]["gag_card"]["hook_frame"],
            payload["story"]["opening_keyframe_prompt"],
        )

        mismatched = {
            "story": {
                "opening_keyframe_prompt": "Kirby clutches a glowing coin as blue wind ribbons whip around him.",
                "gag_card": {"hook_frame": "0s"},
                "native_shots": [{"action": "Kirby watches a red apple roll across a quiet table."}],
            }
        }

        still_invalid = LLMPromptEngine._normalize_native_h3_story_payload(mismatched)

        self.assertEqual(still_invalid["story"]["gag_card"]["hook_frame"], "0s")

    def test_news_selection_rejects_placeholder_titles(self) -> None:
        self.assertFalse(NewsContextService.is_usable_selection("...", "gold;reserve"))
        self.assertFalse(NewsContextService.is_usable_selection("", "gold;reserve"))
        self.assertTrue(NewsContextService.is_usable_selection("Central bank changes reserve policy", "gold;reserve"))

    def test_native_h3_optional_semantic_fields_do_not_trigger_repair(self) -> None:
        storyboard = load_storyboard(self.repo_root / "configs/storyboards/kirby_native_15s.yaml")
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
        calls: list[tuple[str, str]] = []

        def fake_chat(request):
            calls.append((str(request.schema_name), str(request.user_prompt)))
            return {"story": missing_stakes, "creative_seed": "lantern-current", "source": "native_h3_llm"}

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

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "native_h3_storyboard")
        self.assertEqual(
            result["story"]["story_spine"]["stakes"],
            valid_story["native_shots"][1]["state_change"],
        )
        self.assertEqual(
            result["story"]["story_spine"]["climax"],
            valid_story["native_shots"][-1]["action"],
        )

    def test_native_h3_free_model_minimal_story_is_normalized_with_advisory_scores(self) -> None:
        storyboard = load_storyboard(self.repo_root / "configs/storyboards/kirby_native_15s.yaml")
        minimal_story = {
            "story_name": "Kirby and the Loose Lantern",
            "style_description": "Soft pastel animation with readable squash-and-stretch motion.",
            "native_shots": [
                {"time_range": "0-4s", "primary_action": "Kirby chases a loose blue lantern."},
                {"time_range": "4-10s", "primary_action": "Kirby catches the lantern cable as it pulls harder."},
                {"time_range": "10-15s", "primary_action": "Kirby anchors the lantern beside the bridge."},
            ],
        }
        requests = []

        def fake_chat(request):
            requests.append(request)
            return {"story": minimal_story, "source": "native_h3_llm"}

        engine = LLMPromptEngine(mode="llm")
        with patch.object(engine, "_require_manager", return_value=object()), patch.object(
            LLMPromptEngine, "_chat_json", side_effect=fake_chat
        ), patch.object(engine, "backend_info", return_value={"provider": "test"}):
            result = engine.generate_native_h3_storyboard(
                character="Kirby",
                style="polished 2D anime",
                duration_seconds=15,
                base_storyboard=storyboard,
                news_context={"title": "unrelated source", "keyword": "unrelated"},
                creative_brief="cute single gag",
            )

        self.assertEqual(len(requests), 1)
        self.assertFalse(requests[0].use_response_format)
        self.assertEqual(result["story"]["name"], "Kirby and the Loose Lantern")
        self.assertIn("Soft pastel animation", result["story"]["base_prompt"])
        self.assertNotIn("news_trace", result["story"])
        self.assertTrue(all(shot["title"] for shot in result["story"]["native_shots"]))
        self.assertTrue(all(shot["camera"] for shot in result["story"]["native_shots"]))
        self.assertTrue(all(shot["state_change"] for shot in result["story"]["native_shots"]))
        self.assertFalse(result["news_grounding"]["passed"])

    def test_native_h3_risky_creative_brief_is_sanitized(self) -> None:
        sanitized = LLMPromptEngine._sanitize_native_h3_creative_brief(
            "stock ticker 810.06, chart, report, and glowing symbols"
        )
        self.assertIn("abstract atmosphere", sanitized)
        self.assertNotIn("810.06", sanitized)

    def test_native_h3_negative_visual_constraints_do_not_erase_creative_brief(self) -> None:
        brief = (
            "Make a cute micro-gag with one dominant mechanism and a readable payoff. "
            "Do not show readable interfaces or abstract symbols."
        )

        sanitized = LLMPromptEngine._sanitize_native_h3_creative_brief(brief)

        self.assertIn("cute micro-gag", sanitized)
        self.assertIn("Do not show readable interfaces", sanitized)

    def test_shared_video_system_prompt_does_not_downgrade_news_grounding(self) -> None:
        self.assertIn("news-grounded", LONG_VIDEO_SYSTEM_PROMPT)
        self.assertIn("causal story", LONG_VIDEO_SYSTEM_PROMPT)
        self.assertNotIn("use it only as inspiration for visual motifs", LONG_VIDEO_SYSTEM_PROMPT)

    def test_native_h3_story_does_not_fallback_when_llm_is_unavailable(self) -> None:
        engine = LLMPromptEngine(mode="llm", manager=None)
        storyboard = load_storyboard(self.repo_root / "configs/storyboards/kirby_native_15s.yaml")
        with patch.object(engine, "_manager_or_none", return_value=None):
            with self.assertRaises(PromptGenerationError):
                engine.generate_native_h3_storyboard(
                    character="Kirby",
                    style="polished 2D anime",
                    duration_seconds=15,
                    base_storyboard=storyboard,
                    news_context={"title": "test", "keyword": "test"},
                )

    def test_native_h3_qa_reads_story_quality_from_run_state(self) -> None:
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
                "native-story-prompt": {"story_quality": {"passed": False, "score": 42}},
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
        self.assertEqual(result.outputs["story_quality"]["score"], 42)
        self.assertFalse(result.outputs["technical_qa"]["bypassed"])
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


if __name__ == "__main__":
    unittest.main()
