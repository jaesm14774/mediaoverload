from __future__ import annotations

import unittest
from pathlib import Path

from agentic.app.main import build_runtime
from agentic.app.character_workflow import choose_media_type
from agentic.runtime.visual_action_contract import (
    ACTION_BEATS,
    OPENING_ACTION_LOCK,
    SEMANTIC_CUE_MODE,
    default_visual_action_contract,
    enforce_opening_action_lock,
    semantic_cue_timeline,
    semantic_cue_timeline_prompt,
    visual_action_contract,
)


class VisualActionContractTests(unittest.TestCase):
    def test_user_given_short_i2v_clip_when_contract_is_compiled_then_causal_payoff_is_required(self) -> None:
        """User Given a short I2V clip When the shared contract is compiled Then it requires a visible causal payoff."""
        contract = visual_action_contract(6, media_type="text2img2video")

        self.assertIn("Short causal-action contract", contract)
        self.assertIn("one dominant physical mechanism", contract)
        self.assertIn("contact drives the mechanism", contract)
        self.assertIn("settled payoff", contract)

    def test_user_given_long_video_when_contract_is_compiled_then_each_segment_keeps_a_state_handoff(self) -> None:
        """User Given a long video When the shared contract is compiled Then each segment must hand off a settled state."""
        contract = visual_action_contract(5, media_type="long_video", segment=True)

        self.assertIn("Segment action contract", contract)
        self.assertIn("physical cause and resulting state change", contract)
        self.assertIn("hand off a settled end state", contract)

    def test_user_given_native_h3_story_when_contract_is_compiled_then_story_beats_remain_causal(self) -> None:
        """User Given a Native H3 story When the shared contract is compiled Then each beat must create a visible state change."""
        contract = visual_action_contract(15, media_type="native_h3_story")

        self.assertIn("Causal-motion contract", contract)
        self.assertIn("hook, mechanism, consequence, reaction, and payoff", contract)
        self.assertIn("each beat must create a visible state change", contract)

    def test_user_given_native_h3_render_mode_alias_when_contract_is_compiled_then_motion_contract_is_preserved(self) -> None:
        """User Given a Native H3 render-mode alias When the shared contract is compiled Then the motion contract remains active."""
        contract = visual_action_contract(15, media_type="text_to_video")

        self.assertIn("Causal-motion contract", contract)
        self.assertIn("visible state change", contract)

    def test_user_given_game_sprite_when_contract_is_compiled_then_sampled_frames_close_the_loop(self) -> None:
        """User Given a looping game sprite When the shared contract is compiled Then sampled frames must preserve the action loop."""
        contract = visual_action_contract(8, media_type="game_sprite", loop=True)

        self.assertIn("Game-sprite action contract", contract)
        self.assertIn("same silhouette, palette, and background", contract)
        self.assertIn("clean periodic loop", contract)

    def test_user_given_static_image_when_contract_is_compiled_then_no_motion_contract_is_added(self) -> None:
        """User Given a static image When the shared contract is compiled Then no video-only instruction is returned."""
        self.assertEqual(visual_action_contract(6, media_type="image"), "")
        self.assertEqual(default_visual_action_contract(6, media_type="image"), {})

    def test_user_given_removed_generation_type_when_routing_then_it_is_rejected_instead_of_falling_back(self) -> None:
        """User Given a removed generation type When routing is requested Then the obsolete path is rejected explicitly."""
        removed_generation_type = "_".join(("micro", "gag"))
        with self.assertRaisesRegex(ValueError, "Unsupported generation type"):
            choose_media_type({}, preferred_generation_type=removed_generation_type)

    def test_user_given_i2v_opening_prompt_when_contact_is_ambiguous_then_action_lock_is_appended(self) -> None:
        """User Given an I2V opening prompt When contact is ambiguous Then the prompt requires the actual causal contact point."""
        locked = enforce_opening_action_lock("Kirby reaches toward one jelly cube")

        self.assertIn(OPENING_ACTION_LOCK, locked)

    def test_user_given_video_goal_when_contract_metadata_is_created_then_route_shape_is_not_a_new_strategy(self) -> None:
        """User Given a video goal When contract metadata is created Then it records the route shape without creating a generation type."""
        metadata = default_visual_action_contract(30, media_type="long_video")

        self.assertEqual(metadata["route_shape"], "story_segment")
        self.assertEqual(metadata["beat_order"], list(ACTION_BEATS))
        self.assertNotIn("generation_type", metadata)

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
                "visual_action_profile": "text2img2video",
                "visual_action_contract": visual_action_contract(6, media_type="text2img2video"),
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

    def test_user_given_treatment_variant_when_semantic_cues_are_enabled_then_timeline_is_reproducible(self) -> None:
        """User Given a treatment variant When semantic cues are enabled Then the timeline records stable event windows."""
        timeline = semantic_cue_timeline(6, media_type="text2img2video")

        self.assertEqual(timeline["mode"], SEMANTIC_CUE_MODE)
        self.assertEqual([item["cue"] for item in timeline["cues"]], list(ACTION_BEATS))
        self.assertEqual(timeline["cues"][0]["start_seconds"], 0.0)
        self.assertEqual(timeline["cues"][-1]["end_seconds"], 6.0)
        self.assertIn("Semantic cue timeline", semantic_cue_timeline_prompt(6, media_type="text2img2video"))

    def test_user_given_control_variant_when_semantic_cues_are_disabled_then_no_timeline_is_added(self) -> None:
        """User Given the control variant When semantic cues are disabled Then the existing motion contract stays unchanged."""
        self.assertEqual(default_visual_action_contract(6, media_type="text2img2video").get("semantic_cue_timeline"), None)


if __name__ == "__main__":
    unittest.main()
