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
from agentic.skills.agent_primitives import AgentMediaSkills
from agentic.skills.longvideo import LongVideoSkills
from agentic.storyboard import (
    evaluate_native_h3_story_quality,
    format_native_h3_prompt,
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
                "native-ending-keyframe",
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
        self.assertEqual(keyframe_gate.inputs["max_regenerations"], 0)
        self.assertIn("native-opening-review", keyframe_gate.depends_on)
        self.assertIn("native-keyframe-gate", render.depends_on)
        self.assertEqual(qa.inputs["mode"], "bypass_until_final_discord_review")
        self.assertIsNone(qa.tool_name)
        self.assertEqual(plan.metadata["native_h3"]["keyframe_candidate_count"], 6)
        self.assertIn("native_h3", plan.metadata)

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
        self.assertEqual(plan.workflow_name, "minimax_h3_lowvram_t2v")
        self.assertEqual(plan.nodes[2].tool_name, "comfy.workflow.text_to_video")
        self.assertEqual(plan.metadata["render_mode"], "text_to_video")
        self.assertEqual(plan.metadata["native_h3"]["length"], 362)
        self.assertEqual(plan.metadata["native_h3"]["steps"], 16)
        self.assertNotIn("native-opening-keyframe", [node.node_id for node in plan.nodes])

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
        self.assertEqual(merged["name"], "Kirby and the Lantern Current")
        self.assertEqual(len(merged["native_shots"]), 3)
        self.assertEqual(len(merged["segments"]), 3)
        self.assertIn("lantern", prompt.lower())
        self.assertNotIn("golden star seed", prompt.lower())
        self.assertNotIn("dark sky rift", prompt.lower())

    def test_native_h3_repair_prompt_does_not_echo_forbidden_visual_cues(self) -> None:
        storyboard = load_storyboard(self.repo_root / "configs/storyboards/kirby_native_15s.yaml")
        valid_story = {
            "name": "Kirby and the Lantern Current",
            "base_prompt": "Kirby is the only protagonist in a moonlit canal city with a runaway lantern.",
            "opening_keyframe_prompt": "Opening frame: Kirby reaches for a runaway lantern beside a canal whirlpool.",
            "ending_keyframe_prompt": "Ending frame: Kirby rests beside the calm canal as the rescued lantern rises above the water.",
            "negative_prompt": "humans, extra characters, duplicate Kirby, text, watermark, hard cut, identity drift",
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
            story = invalid_story if len(calls) == 1 else repaired_story
            return {"story": story, "creative_seed": "lantern-current", "source": "native_h3_llm"}

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
        self.assertEqual(result["story"]["base_prompt"], repaired_story["base_prompt"])

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
        missing_climax = json.loads(json.dumps(valid_story))
        missing_climax["story_spine"]["climax"] = ""
        responses = [missing_stakes, missing_climax, valid_story]
        calls: list[tuple[str, str]] = []

        def fake_chat(_manager, _system, prompt, **kwargs):
            calls.append((str(kwargs.get("schema_name")), str(prompt)))
            story = responses.pop(0)
            return {"story": story, "creative_seed": "lantern-current", "source": "native_h3_llm"}

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
                raise AssertionError("technical media QA must be bypassed")

        state = RunState(
            goal={},
            metadata={},
            node_outputs={
                "native-story-prompt": {"story_quality": {"passed": False, "score": 42}},
                "native-h3-render": {"saved_files": ["clip.mp4"], "run_dir": "run"},
            },
        )
        context = SimpleNamespace(
            state=state,
            node=SimpleNamespace(inputs={"target_duration": 15, "duration_tolerance": 0.5}),
            plan=SimpleNamespace(goal=SimpleNamespace(prompt="Kirby story", duration_seconds=15)),
        )

        result = LongVideoSkills(FakeTools(), self.repo_root / ".tmp-tests" / "native-h3-qa").qa_native_h3(context)

        self.assertEqual(result.status, "success")
        self.assertTrue(result.outputs["passed"])
        self.assertEqual(result.outputs["story_quality"]["score"], 42)
        self.assertTrue(result.outputs["technical_qa"]["bypassed"])


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
