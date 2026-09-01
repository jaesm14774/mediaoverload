from __future__ import annotations

import unittest
from pathlib import Path

from agentic.runtime.contracts import GoalRequest
from agentic.minimax_prompting import short_action_contract
from agentic.runtime.prompting import (
    build_goal_brief,
    build_minimax_h3_prompt,
    build_story_segments,
    validate_story_segments,
)
from agentic.storyboard import format_native_h3_prompt, load_storyboard


class StoryboardContractTests(unittest.TestCase):

    def test_h3_prompt_exposes_story_contract(self) -> None:
        goal = GoalRequest(
            prompt="Kirby follows a mysterious light",
            media_type="long_video",
            duration_seconds=10,
            style="cinematic anime",
            constraints={"character": "Kirby"},
        )
        segment = {
            "segment_id": "discover_star_seed",
            "visual": "Kirby discovers a glowing star seed",
            "narrative_goal": "Kirby discovers the signal",
            "start_state": "Kirby stands in the meadow",
            "end_state": "Kirby holds the star seed",
            "next_hook": "The seed points to a light gate",
        }
        prompt = build_minimax_h3_prompt(goal, segment)["prompt"]
        self.assertIn("Story progression contract", prompt)
        self.assertIn("The seed points to a light gate", prompt)

    def test_h3_prompt_preserves_primary_physical_action(self) -> None:
        goal = GoalRequest(
            prompt="Kirby reaches a glowing seed",
            media_type="long_video",
            duration_seconds=10,
            style="cinematic anime",
            constraints={"character": "Kirby"},
        )
        segment = {
            "segment_id": "segment-1",
            "visual": "Kirby faces the seed in a windy meadow",
            "action": "Kirby sprints forward, skids, and snatches the seed before it blows away",
            "start_state": "the seed is loose",
            "end_state": "Kirby grips the seed",
            "cause": "the wind carries the seed away",
            "effect": "Kirby must protect it from the storm",
        }
        prompt = build_minimax_h3_prompt(goal, segment)["prompt"]
        self.assertIn("Primary physical action", prompt)
        self.assertIn("sprints forward", prompt)
        self.assertIn("Long-segment action contract", prompt)

    def test_generic_prompt_builder_uses_current_dynamic_story_contract(self) -> None:
        goal = GoalRequest(
            prompt="Kirby follows a mysterious light",
            media_type="long_video",
            duration_seconds=10,
            style="cinematic anime",
            constraints={"character": "Kirby"},
        )
        segments = build_story_segments(goal, "Kirby story brief", 2, "playful")
        self.assertEqual([segment["segment_id"] for segment in segments], ["segment-1", "segment-2"])
        self.assertNotEqual(segments[0]["stage"], segments[1]["stage"])
        self.assertIn("Kirby follows a mysterious light", segments[1]["visual"])

    def test_fallback_four_segment_story_keeps_concrete_seed_states(self) -> None:
        goal = GoalRequest(
            prompt="Kirby crosses a windy meadow, discovers a glowing seed, protects it from a storm, and reaches a warm clearing",
            media_type="long_video",
            duration_seconds=20,
            style="polished 2D anime cinematic",
            constraints={"character": "Kirby"},
        )
        segments = build_story_segments(goal, "fallback story", 4, "playful")
        states = [segments[0]["start_state"], *(segment["end_state"] for segment in segments)]
        self.assertIn("glowing seed", states[0])
        self.assertIn("grabbed the glowing seed", states[1])
        self.assertIn("shielding the glowing seed", states[2])
        self.assertIn("carried the glowing seed", states[3])
        self.assertIn("warm clearing", states[4])
        self.assertEqual(len(states), len(set(states)))

    def test_native_15s_preset_has_three_causal_shots(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        native = load_storyboard(repo_root / "configs/storyboards/kirby_native_15s.yaml")
        prompt = format_native_h3_prompt(native)
        self.assertEqual(len(native["native_shots"]), 3)
        self.assertIn("SHOT 1", prompt)
        self.assertIn("SHOT 3", prompt)
        self.assertIn("not a montage", prompt)
        self.assertLess(len(prompt.split()), 600)

    def test_five_second_brief_is_one_completed_action(self) -> None:
        goal = GoalRequest(
            prompt="Kirby swats one glowing orb into a target",
            media_type="text2img2video",
            duration_seconds=5,
            style="cinematic anime",
            constraints={"character": "Kirby"},
        )
        brief = build_goal_brief(goal, goal.style, [])
        self.assertIn("one clear physical action only", brief["prompt"])
        self.assertIn("completed end state", brief["prompt"])
        self.assertIn("opening_keyframe_prompt", brief)

    def test_short_action_contract_is_topic_neutral_and_not_aspect_specific(self) -> None:
        contract = short_action_contract(6, media_type="image_to_video")

        self.assertIn("one dominant physical mechanism", contract)
        self.assertIn("completed end state", contract)
        self.assertNotIn("Kirby", contract)
        self.assertNotIn("Qixi", contract)
        self.assertNotIn("9:16", contract)
        self.assertNotIn("vertical", contract.lower())
        self.assertEqual(short_action_contract(15, media_type="image_to_video"), "")
        self.assertEqual(short_action_contract(6, media_type="text2img"), "")

    def test_short_gag_brief_applies_reference_derived_style_contract(self) -> None:
        goal = GoalRequest(
            prompt="Kirby gets squashed by one giant mochi and bounces back",
            media_type="text2img2video",
            duration_seconds=6,
            style="polished 2D anime",
            constraints={
                "character": "Kirby",
                "visual_style_contract": (
                    "small-versus-large scale contrast, tactile prop, readable reaction, and a loopable settled ending"
                ),
            },
        )
        brief = build_goal_brief(goal, goal.style, [])
        self.assertIn("small-versus-large scale contrast", brief["prompt"])
        self.assertIn("tactile prop", brief["prompt"])

    def test_story_segment_contract_rejects_missing_physical_causality(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing: action, cause"):
            validate_story_segments(
                [
                    {
                        "segment_id": "segment-1",
                        "visual": "Kirby looks at the prop",
                        "narration": "A question appears",
                        "action": "",
                        "camera": "push in",
                        "start_state": "prop is still",
                        "end_state": "prop is still",
                        "cause": "",
                        "effect": "the next beat begins",
                    }
                ],
                1,
            )

if __name__ == "__main__":
    unittest.main()
