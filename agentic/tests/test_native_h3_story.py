from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agentic.app.character_workflow import build_goal_payload_from_character_config
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
    load_storyboard,
    merge_native_h3_storyboard,
    repair_native_h3_news_trace_integration,
    validate_native_h3_shot_timing,
)
from agentic.tools.context_services import NewsContextService


class NativeH3StoryPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.config_path = cls.repo_root / "configs" / "characters" / "kirby.yaml"

    def test_character_route_builds_native_h3_graph_without_power_shell_entrypoint(self) -> None:
        payload = build_goal_payload_from_character_config(
            self.repo_root,
            self.config_path,
            prompt="Kirby follows a storm-lit star through the meadow",
            preferred_generation_type="native_h3_story",
            publish_after_generate=False,
        )
        planner, _runner, _memory = build_runtime(
            self.repo_root / "agentic",
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
        self.assertEqual(opening.inputs["image_count"], 6)
        self.assertEqual(opening_review.inputs["review_all_candidates"], True)
        self.assertEqual(opening_review.inputs["review_scope"], "first_frame")
        self.assertEqual(opening_review.depends_on, ["native-opening-keyframe"])
        self.assertEqual(keyframe_gate.inputs["opening_node"], "native-opening-review")
        self.assertTrue(keyframe_gate.inputs["preserve_opening_frame"])
        self.assertFalse(keyframe_gate.inputs["use_last_frame"])
        self.assertEqual(keyframe_gate.depends_on, ["native-opening-review"])
        self.assertEqual(keyframe_gate.inputs["max_regenerations"], 0)
        self.assertIn("native-opening-review", keyframe_gate.depends_on)
        self.assertIn("native-keyframe-gate", render.depends_on)
        self.assertFalse(render.inputs["use_last_frame"])
        self.assertEqual(qa.inputs["mode"], "technical_and_semantic_qa_before_optional_discord_review")
        self.assertIsNone(qa.tool_name)
        self.assertEqual(plan.metadata["native_h3"]["keyframe_candidate_count"], 6)
        self.assertFalse(plan.metadata["native_h3"]["use_last_frame"])
        self.assertIn("native_h3", plan.metadata)

    def test_native_h3_last_frame_mode_adds_review_and_passes_both_frames(self) -> None:
        payload = build_goal_payload_from_character_config(
            self.repo_root,
            self.config_path,
            prompt="Kirby protects one glowing seed as a sudden storm reshapes the meadow",
            preferred_generation_type="native_h3_story",
            publish_after_generate=False,
        )
        payload["constraints"]["native_h3_use_last_frame"] = True
        planner, _runner, _memory = build_runtime(
            self.repo_root / "agentic",
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
        payload = build_goal_payload_from_character_config(
            self.repo_root,
            self.config_path,
            prompt="Kirby protects one glowing seed as a sudden storm reshapes the meadow",
            preferred_generation_type="native_h3_story",
            publish_after_generate=False,
            no_review=True,
        )
        planner, _runner, _memory = build_runtime(
            self.repo_root / "agentic",
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

    def test_news_only_native_h3_does_not_validate_autonomous_brief_as_user_objective(self) -> None:
        captured: dict[str, object] = {}
        storyboard_fixture = load_storyboard(self.repo_root / "configs/storyboards/kirby_native_15s.yaml")
        storyboard_fixture.update(
            {
                "opening_keyframe_prompt": "Kirby reacts to a visible news-derived disruption in a clear meadow composition.",
                "ending_keyframe_prompt": "Kirby resolves the news-derived disruption and restores the scene.",
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
                        "news_context": {"title": "AI companion robot arrives", "keyword": "AI;robot"},
                    },
                )
            ),
            node=SimpleNamespace(
                inputs={
                    "storyboard_path": "configs/storyboards/kirby_native_15s_5beat.yaml",
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
        self.assertEqual(captured["creative_brief"], "")

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

    def test_character_route_builds_direct_t2v_story_graph(self) -> None:
        payload = build_goal_payload_from_character_config(
            self.repo_root,
            self.config_path,
            prompt="Kirby must save one glowing seed before the storm swallows the garden",
            preferred_generation_type="native_h3_t2v_story",
            publish_after_generate=False,
        )
        planner, _runner, _memory = build_runtime(
            self.repo_root / "agentic",
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
        self.assertEqual(plan.workflow_name, "minimax_h3_lowvram_i2v")
        render = next(node for node in plan.nodes if node.node_id == "native-h3-render")
        self.assertEqual(render.tool_name, "comfy.workflow.image_to_video")
        self.assertEqual(plan.metadata["recipe"], "native_h3_story")
        self.assertEqual(plan.metadata["native_h3"]["length"], 362)
        self.assertFalse(plan.metadata["native_h3"]["lowvram_preview"])
        self.assertEqual(plan.metadata["native_h3"]["steps"], 16)
        self.assertIn("native-opening-keyframe", [node.node_id for node in plan.nodes])
        self.assertIn("native-opening-review", [node.node_id for node in plan.nodes])

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
                "source_title": "city lantern outage paraphrased by provider",
                "source_concepts": ["lantern"],
                "visual_translation": "The outage becomes a runaway city lantern pulled toward a canal whirlpool.",
                "visual_anchors": ["lantern", "whirlpool"],
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

    def test_native_h3_news_grounding_accepts_small_anchor_wording_variations(self) -> None:
        story = {
            "name": "Kirby and the Typhoon Seed",
            "story_spine": {
                "premise": "Kirby protects a glowing seed from a typhoon vortex.",
                "objective": "Protect the glowing seed.",
            },
            "news_trace": {
                "source_title": "Typhoon warning",
                "source_concepts": ["typhoon"],
                "visual_translation": "A typhoon vortex funnel threatens the meadow.",
                "visual_anchors": ["typhoon vortex funnel", "glowing seed"],
                "integration": "The typhoon vortex threatens the glowing seed and forces Kirby to protect it.",
            },
            "native_shots": [
                {
                    "action": "A typhoon vortex pulls at the glowing seed.",
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

    def test_native_h3_news_grounding_keeps_compound_anchor_in_payoff(self) -> None:
        story = {
            "name": "Kirby and the Shadow Agent",
            "story_spine": {
                "premise": "Kirby confronts a rogue autonomous energy entity.",
                "objective": "Kirby must neutralize the aggressive energy source.",
                "resolution": "The monolith is destroyed and a stable orb remains.",
            },
            "news_trace": {
                "source_title": "AI agent attack from abroad",
                "source_concepts": ["AI自主攻擊"],
                "visual_translation": "The AI自主攻擊 becomes a featureless black monolith that attacks autonomously.",
                "visual_anchors": ["Featureless black monolith"],
                "integration": "The AI自主攻擊 becomes the featureless black monolith that Kirby must neutralize.",
            },
            "native_shots": [
                {"action": "A featureless black monolith lashes out at Kirby."},
                {"action": "The featureless black monolith pulls Kirby toward it."},
                {"action": "Kirby destroys the monolith and a stable orb remains."},
            ],
        }

        quality = evaluate_native_h3_news_grounding(
            story,
            {"title": "AI agent attack from abroad", "keyword": "AI自主攻擊", "category": "news"},
        )

        self.assertTrue(quality["passed"], quality)

    def test_native_h3_trace_repair_bridges_translated_news_anchor(self) -> None:
        story = {
            "name": "Kirby vs. The Shadow Agent",
            "story_spine": {
                "objective": "Neutralize the aggressive energy source.",
                "resolution": "The dark entity is destroyed and a stable orb remains.",
            },
            "news_trace": {
                "source_title": "AI agent attack from abroad",
                "source_concepts": ["AI自主攻擊"],
                "visual_translation": "The AI agent is represented by a featureless black monolith that acts autonomously.",
                "visual_anchors": ["Featureless black monolith"],
                "integration": "The translated threat becomes Kirby's mission.",
            },
            "native_shots": [
                {"action": "A featureless black monolith knocks Kirby backward."},
                {"action": "The featureless black monolith lashes out with dark energy."},
                {"action": "Kirby destroys the monolith and a stable orb remains."},
            ],
        }

        repaired = repair_native_h3_news_trace_integration(
            story,
            {"title": "AI agent attack from abroad", "keyword": "AI自主攻擊", "category": "news"},
        )

        self.assertIsNotNone(repaired)
        integration = repaired["news_trace"]["integration"]
        self.assertIn("AI自主攻擊", integration)
        self.assertIn("Featureless black monolith", integration)
        self.assertTrue(
            evaluate_native_h3_news_grounding(
                repaired,
                {"title": "AI agent attack from abroad", "keyword": "AI自主攻擊", "category": "news"},
            )["passed"]
        )

    def test_native_h3_trace_repair_adds_exact_anchor_when_story_visuals_are_grounded(self) -> None:
        story = {
            "name": "Kirby and the Typhoon Seed",
            "story_spine": {"objective": "Protect the glowing seed.", "resolution": "The seed is safe."},
            "news_trace": {
                "source_title": "Typhoon warning",
                "source_concepts": ["typhoon"],
                "visual_translation": "A typhoon vortex threatens the meadow.",
                "visual_anchors": ["typhoon vortex funnel", "glowing seed"],
                "integration": "The weather threat becomes Kirby's mission.",
            },
            "native_shots": [
                {"action": "A typhoon vortex funnel pulls at the glowing seed."},
                {"action": "The typhoon vortex funnel knocks Kirby away from the glowing seed."},
                {"action": "Kirby returns the glowing seed as the typhoon vortex funnel fades."},
            ],
        }

        repaired = repair_native_h3_news_trace_integration(
            story,
            {"title": "Typhoon warning", "keyword": "typhoon", "category": "weather"},
        )

        self.assertIsNotNone(repaired)
        self.assertIn("typhoon vortex funnel", repaired["news_trace"]["integration"])
        self.assertTrue(
            evaluate_native_h3_news_grounding(
                repaired,
                {"title": "Typhoon warning", "keyword": "typhoon", "category": "weather"},
            )["passed"]
        )

    def test_native_h3_trace_repair_bridges_source_concept_to_visual_anchor(self) -> None:
        story = {
            "name": "Kirby and the Access Key",
            "story_spine": {
                "objective": "Kirby must restore the key to the meadow pedestal.",
                "resolution": "The key restores the meadow.",
            },
            "news_trace": {
                "source_title": "頻道改密碼引發爭議",
                "source_concepts": ["改密碼"],
                "visual_translation": "The headline becomes a glowing key whose color changes when access is disrupted.",
                "visual_anchors": ["glowing shifting key"],
                "integration": "Kirby must recover the key to protect the meadow.",
            },
            "native_shots": [
                {"action": "A glowing shifting key tears away from the pedestal."},
                {"action": "Kirby is pushed back while the glowing shifting key spins away."},
                {"action": "Kirby returns the glowing shifting key to the pedestal."},
            ],
        }

        repaired = repair_native_h3_news_trace_integration(
            story,
            {"title": "頻道改密碼引發爭議", "keyword": "改密碼", "category": "司法"},
        )

        self.assertIsNotNone(repaired)
        integration = repaired["news_trace"]["integration"]
        self.assertIn("改密碼", integration)
        self.assertIn("glowing shifting key", integration)
        self.assertTrue(
            evaluate_native_h3_news_grounding(
                repaired,
                {"title": "頻道改密碼引發爭議", "keyword": "改密碼", "category": "司法"},
            )["passed"]
        )

    def test_native_h3_repair_prompt_does_not_echo_forbidden_visual_cues(self) -> None:
        storyboard = load_storyboard(self.repo_root / "configs/storyboards/kirby_native_15s.yaml")
        valid_story = {
            "name": "Kirby and the Lantern Current",
            "base_prompt": "Kirby is the only protagonist in a moonlit canal city with a runaway lantern.",
            "opening_keyframe_prompt": "Opening frame: Kirby reaches for a runaway lantern beside a canal whirlpool.",
            "ending_keyframe_prompt": "Ending frame: Kirby rests beside the calm canal as the rescued lantern rises above the water.",
            "negative_prompt": "humans, extra characters, duplicate Kirby, text, watermark, hard cut, identity drift",
            "news_trace": {
                "source_title": "city lantern outage",
                "source_concepts": ["lantern"],
                "visual_translation": "The outage becomes a runaway city lantern pulled toward a canal whirlpool.",
                "visual_anchors": ["lantern", "whirlpool"],
                "integration": "Kirby protects the lantern while the outage makes the lantern the urgent mission.",
            },
            "story_spine": {
                "premise": "A current is dragging a guiding lantern into a canal whirlpool.",
                "objective": "Kirby must redirect the lantern before the city loses its safe path.",
                "obstacle": "The current is accelerating and the cable is about to snap.",
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
                    "The current pulls from screen left toward the whirlpool on screen right.",
                ],
            },
            "native_audio": "water rush, cable tension, one bright lantern chime, then calm night ambience",
            "native_shots": [
                {"time": "0-4s", "title": "Hook - the lantern is pulled away", "action": "A lantern tears loose and drags light toward the whirlpool while Kirby reaches for its cable.", "camera": "Wide bridge reveal pushing into Kirby and the moving lantern.", "state_change": "Kirby commits to stopping the runaway lantern."},
                {"time": "4-10s", "title": "Escalation - anchor the current", "action": "Kirby slides along the wet bridge and wraps the cable around a post as the current pulls harder.", "camera": "Smooth side track with the cable and whirlpool kept in the same geography.", "state_change": "The cable holds, but the bridge begins to crack under the force."},
                {"time": "10-15s", "title": "Payoff - light returns", "action": "Kirby releases a burst of warm light through the cable, the current reverses, and the lantern rises above the canals.", "camera": "Follow the light upward, then settle into a calm wide ending frame.", "state_change": "The lanterns form a safe constellation and the danger is resolved."},
            ],
        }
        invalid_story = json.loads(json.dumps(valid_story))
        invalid_story["base_prompt"] += " A glowing document is covered in financial symbols."
        repaired_story = json.loads(json.dumps(valid_story))
        calls: list[str] = []

        def fake_chat(_manager, _system, repair_prompt, **_kwargs):
            calls.append(str(repair_prompt))
            if len(calls) == 1:
                return {"story": invalid_story, "creative_seed": "lantern-current", "source": "native_h3_llm"}
            return {"story_patch": {"base_prompt": repaired_story["base_prompt"]}}

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

        self.assertEqual(len(calls), 2)
        self.assertNotIn("Validation error:", calls[1])
        self.assertNotIn("forbidden readable-text visual cues", calls[1])
        self.assertNotIn("Do not use readable words", calls[1])
        self.assertNotIn("document", calls[1].lower())
        self.assertNotIn("financial symbols", calls[1].lower())
        self.assertIn("BEGIN PREVIOUS STORYBOARD JSON", calls[1])
        self.assertEqual(result["story"]["base_prompt"], repaired_story["base_prompt"])
        self.assertEqual(result["story"]["native_shots"], valid_story["native_shots"])

    def test_native_h3_visual_repair_replaces_information_objects_with_physical_action(self) -> None:
        repair_prompt = LLMPromptEngine._build_native_h3_repair_prompt(
            "Do not use readable words, letters, numbers, signs, labels, subtitles, headlines, or written symbols anywhere in the visuals.",
            PromptGenerationError(
                "Native H3 story contains forbidden readable-text visual cues: reads"
            ),
            expected_times=("0-4s", "4-10s", "10-15s"),
            duration_seconds=15,
        )

        lowered = repair_prompt.lower()
        self.assertIn("plain unmarked physical prop", lowered)
        self.assertIn("react physically", lowered)
        self.assertIn("do not show a map", lowered)

    def test_native_h3_visual_cue_filter_allows_unmarked_physical_surfaces(self) -> None:
        allowed = LLMPromptEngine._find_native_h3_forbidden_visual_cues(
            "no text or logos; glowing floor panels, a data-core pedestal, clear screen geography, readable cause-and-effect, and a calm light display"
        )
        self.assertEqual(allowed, [])

        rejected = LLMPromptEngine._find_native_h3_forbidden_visual_cues(
            "a floor panel displays readable text beside a glowing button"
        )
        self.assertIn("readable text", rejected)
        self.assertIn("text-bearing surface", rejected)

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
                    "source_title": "city lantern outage",
                    "source_concepts": ["lantern"],
                    "visual_translation": "The outage becomes a runaway lantern.",
                    "visual_anchors": ["lantern"],
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

    def test_native_h3_hook_repair_prompt_requires_a_visible_first_second_event(self) -> None:
        repair_prompt = LLMPromptEngine._build_native_h3_repair_prompt(
            "The first shot must show visible character or camera motion within the first second.",
            PromptGenerationError(
                "Native H3 story quality is insufficient: the hook must contain a visible disruption or motion in the opening beat"
            ),
            expected_times=("0-4s", "4-10s", "10-15s"),
            duration_seconds=15,
        )

        lowered = repair_prompt.lower()
        self.assertIn("native_shots[0].action", lowered)
        self.assertIn("native_shots[0].camera", lowered)
        self.assertIn("first second", lowered)
        self.assertIn("forms", lowered)
        self.assertIn("consequential setback", lowered)

    def test_native_h3_repair_prompt_repeats_nested_story_envelope(self) -> None:
        repair_prompt = LLMPromptEngine._build_native_h3_repair_prompt(
            "Return the native H3 story.",
            PromptGenerationError("Native H3 LLM response did not contain a story object."),
            expected_times=("0-1.5s", "1.5-4s", "4-7.5s", "7.5-11s", "11-15s"),
            duration_seconds=15,
        )

        lowered = repair_prompt.lower()
        self.assertIn('"story": {...}', repair_prompt)
        self.assertIn("all story fields belong inside story", lowered)
        self.assertIn("story.native_audio", lowered)

    def test_native_h3_normalizes_only_complete_flat_provider_story(self) -> None:
        flat_story = {
            "name": "Kirby and the Lantern Current",
            "base_prompt": "Kirby, polished 2D anime, fluid squash-and-stretch motion.",
            "opening_keyframe_prompt": "Kirby lunges as the lantern tears loose.",
            "ending_keyframe_prompt": "Kirby restores the lantern above the canal.",
            "negative_prompt": "text, humans, duplicate character",
            "news_trace": {"source_title": "city lantern outage"},
            "story_spine": {"objective": "Save the lantern."},
            "world": {"setting": "moonlit canal"},
            "native_audio": "Overall soundscape: rushing water. Non-diegetic music: heroic swell.",
            "native_shots": [{"time": "0-15s"}],
        }

        normalized = LLMPromptEngine._normalize_native_h3_story_payload(flat_story)

        self.assertEqual(normalized["story"]["name"], flat_story["name"])
        self.assertEqual(normalized["source"], "native_h3_llm")
        self.assertNotIn("story_spine", normalized)
        incomplete = {"story_spine": {}, "native_shots": []}
        self.assertIs(LLMPromptEngine._normalize_native_h3_story_payload(incomplete), incomplete)

    def test_native_h3_normalizes_mixed_provider_envelope_before_patch_mode(self) -> None:
        payload = {
            "base_prompt": "Kirby moves through a glowing meadow.",
            "native_audio": {
                "overall_soundscape": "Wind and a bright chime.",
                "non_diegetic_music": "A warm rising motif.",
            },
            "native_shots": [
                {
                    "timestamp": "0-15s",
                    "visible_action": "Kirby runs.",
                    "camera_movement": "Follow Kirby from the side.",
                }
            ],
            "story": {
                "news_trace": {
                    "source_title": "AI agent attack",
                    "source_concepts": ["AI agent"],
                },
            },
            "creative_seed": "seed",
            "source": "native_h3_llm",
        }

        normalized = LLMPromptEngine._normalize_native_h3_story_payload(payload)

        self.assertEqual(len(normalized["story"]["native_shots"]), len(payload["native_shots"]))
        self.assertIn("Overall soundscape: Wind and a bright chime.", normalized["story"]["native_audio"])
        self.assertIn("Non-diegetic music: A warm rising motif.", normalized["story"]["native_audio"])
        self.assertEqual(normalized["story"]["native_shots"][0]["time"], "0-15s")
        self.assertNotIn("timestamp", normalized["story"]["native_shots"][0])
        self.assertEqual(normalized["story"]["native_shots"][0]["action"], "Kirby runs.")
        self.assertEqual(
            normalized["story"]["native_shots"][0]["camera"],
            "Follow Kirby from the side.",
        )
        self.assertEqual(normalized["story"]["native_shots"][0]["title"], "Kirby runs")
        self.assertEqual(
            normalized["story"]["news_trace"]["source_title"],
            "AI agent attack",
        )
        patched = LLMPromptEngine._apply_native_h3_story_patch(
            normalized,
            {"story_patch": {"native_shots": [{"index": 0, "time": "0-15s"}]}},
        )
        self.assertEqual(patched["story"]["native_shots"][0]["time"], "0-15s")

    def test_native_h3_patch_normalizes_flat_story_spine_synopsis(self) -> None:
        previous = {"story": {"story_spine": {"objective": "Save the orb."}, "native_shots": []}}
        patched = LLMPromptEngine._apply_native_h3_story_patch(
            previous,
            {"story_patch": {"story_spine": "Kirby contains the unstable orb and restores the garden."}},
        )

        spine = patched["story"]["story_spine"]
        self.assertIsInstance(spine, dict)
        self.assertNotEqual(spine["climax"], spine["resolution"])
        self.assertIn("unstable orb", spine["objective"])

    def test_native_h3_normalizes_point_timestamps_to_expected_beat_ranges(self) -> None:
        payload = {
            "story": {
                "native_shots": [
                    {"timestamp": 0, "action": "Kirby moves.", "camera_movement": "Follow."},
                    {"timestamp": 1.5, "action": "Kirby turns.", "camera_movement": "Pan."},
                ]
            }
        }
        normalized = LLMPromptEngine._normalize_native_h3_story_payload(
            payload,
            expected_times=("0-1.5s", "1.5-4s"),
        )

        shots = normalized["story"]["native_shots"]
        self.assertEqual(shots[0]["time"], "0-1.5s")
        self.assertEqual(shots[1]["time"], "1.5-4s")

    def test_native_h3_patch_normalizes_flat_world_string(self) -> None:
        previous = {"story": {"native_shots": []}}
        patched = LLMPromptEngine._apply_native_h3_story_patch(
            previous,
            {"story_patch": {"world": "A glowing data center."}},
        )

        world = patched["story"]["world"]
        self.assertEqual(world["setting"], "A glowing data center.")
        self.assertTrue(world["continuity_rules"])

    def test_news_selection_rejects_placeholder_titles(self) -> None:
        self.assertFalse(NewsContextService.is_usable_selection("...", "gold;reserve"))
        self.assertFalse(NewsContextService.is_usable_selection("", "gold;reserve"))
        self.assertTrue(NewsContextService.is_usable_selection("Central bank changes reserve policy", "gold;reserve"))

    def test_native_h3_semantic_repair_retries_missing_fields_without_fallback(self) -> None:
        storyboard = load_storyboard(self.repo_root / "configs/storyboards/kirby_native_15s.yaml")
        valid_story = {
            "name": "Kirby and the Lantern Current",
            "base_prompt": "Kirby is the only protagonist in a moonlit canal city with a runaway lantern.",
            "opening_keyframe_prompt": "Opening frame: Kirby reaches for a runaway lantern beside a canal whirlpool.",
            "ending_keyframe_prompt": "Ending frame: Kirby rests beside the calm canal as the rescued lantern rises above the water.",
            "negative_prompt": "humans, extra characters, duplicate Kirby, watermark, hard cuts",
            "news_trace": {
                "source_title": "city lantern outage",
                "source_concepts": ["lantern"],
                "visual_translation": "The outage becomes a runaway city lantern pulled toward a canal whirlpool.",
                "visual_anchors": ["lantern", "whirlpool"],
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
        responses = [
            missing_stakes,
            {"story_patch": {"story_spine": {"stakes": valid_story["story_spine"]["stakes"]}}},
            {"story_patch": {"story_spine": {"climax": valid_story["story_spine"]["climax"]}}},
        ]
        calls: list[tuple[str, str]] = []

        def fake_chat(_manager, _system, prompt, **kwargs):
            calls.append((str(kwargs.get("schema_name")), str(prompt)))
            response = responses.pop(0)
            if isinstance(response, dict) and "story_patch" in response:
                return response
            return {"story": response, "creative_seed": "lantern-current", "source": "native_h3_llm"}

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
        self.assertEqual(calls[1][0], "native_h3_storyboard_repair_1")
        self.assertIn("stakes", calls[1][1])
        self.assertEqual(calls[2][0], "native_h3_storyboard_repair_2")
        self.assertEqual(result["story"]["story_spine"]["stakes"], valid_story["story_spine"]["stakes"])
        self.assertEqual(result["story"]["story_spine"]["climax"], valid_story["story_spine"]["climax"])

    def test_native_h3_motion_validator_accepts_common_inflected_motion_verbs(self) -> None:
        quality = evaluate_native_h3_story_quality(
            {
                "story_spine": {
                    "premise": "A silver car blocks Kirby's route.",
                    "objective": "Kirby must save the lantern.",
                    "obstacle": "The car cuts across the route.",
                    "stakes": "The lantern will be lost.",
                    "climax": "Kirby redirects the lantern.",
                    "resolution": "The lantern is safe.",
                },
                "native_shots": [
                    {
                        "action": "A silver car swerves across Kirby's path, pushing him backward.",
                        "camera": "The camera pans rapidly after the car.",
                        "state_change": "Kirby is knocked off balance.",
                    },
                    {
                        "action": "Kirby fails to reach the lantern as the barrier crashes down.",
                        "camera": "Track the falling barrier.",
                        "state_change": "The route is blocked.",
                    },
                    {
                        "action": "Kirby grabs the lantern and pulls it into safety.",
                        "camera": "Follow the lantern into a clear meadow.",
                        "state_change": "The lantern is safe.",
                    },
                ],
            }
        )

        self.assertTrue(quality["passed"], quality["errors"])
        self.assertTrue(quality["checks"]["hook_visible_motion"])

    def test_native_h3_risky_creative_brief_is_sanitized(self) -> None:
        sanitized = LLMPromptEngine._sanitize_native_h3_creative_brief(
            "stock ticker 810.06, chart, report, and glowing symbols"
        )
        self.assertIn("abstract atmosphere", sanitized)
        self.assertNotIn("810.06", sanitized)

        repair_prompt = LLMPromptEngine._build_native_h3_repair_prompt(
            "Creative brief: stock ticker 810.06 and chart\nNews context JSON: {\"title\": \"report\"}",
            PromptGenerationError("Native H3 story contains forbidden readable-text visual cues: ticker, chart"),
        )
        self.assertNotIn("810.06", repair_prompt)
        self.assertNotIn("stock ticker", repair_prompt.lower())

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

    def test_native_h3_story_quality_rejects_a_pose_sequence(self) -> None:
        story = {
            "story_spine": {
                "premise": "A small light is trapped at the edge of a storm.",
                "objective": "Kirby must bring the light back before the path disappears.",
                "obstacle": "The storm keeps the path hidden.",
                "stakes": "The path will vanish if the light is lost.",
                "climax": "Kirby carries the light across the broken path.",
                "resolution": "The path glows again and Kirby reaches safety.",
            },
            "native_shots": [
                {"action": "Kirby stands in a field.", "camera": "A locked wide shot.", "state_change": "Kirby is present."},
                {"action": "Kirby looks around.", "camera": "The same locked wide shot.", "state_change": "Kirby is still present."},
                {"action": "The sky becomes colorful.", "camera": "A slow static hold.", "state_change": "The scene is colorful."},
            ],
        }
        quality = evaluate_native_h3_story_quality(story)

        self.assertFalse(quality["passed"])
        self.assertIn("hook_visible_motion", quality["checks"])
        self.assertIn("escalation_or_reversal", quality["checks"])
        self.assertIn("payoff_evidence", quality["checks"])

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
