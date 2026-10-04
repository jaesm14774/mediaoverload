from __future__ import annotations

import unittest
from pathlib import Path

from agentic.runtime.contracts import GoalRequest
from agentic.runtime.prompting import (
    build_goal_brief,
    build_minimax_h3_prompt,
    build_story_segments,
    validate_story_segments,
)
from agentic.storyboard import format_native_h3_prompt, load_storyboard


class StoryboardContractTests(unittest.TestCase):

    def test_native_15s_preset_contains_render_defaults_without_a_fixed_story(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        storyboard_path = repo_root / "configs/storyboards/native_h3_15s.yaml"
        native = load_storyboard(storyboard_path)
        prompt = format_native_h3_prompt(native)
        self.assertEqual(native["native_duration_seconds"], 15)
        self.assertEqual(native["native_shots"], [])
        self.assertNotIn("required_shot_times", native)
        self.assertNotIn("Shot progression:", prompt)
        self.assertIn("Duration: 15 seconds", prompt)

    def test_user_given_long_video_when_story_planner_falls_back_then_it_keeps_the_brief(self) -> None:
        """User Given a long video brief When story planning falls back Then each renderer segment uses the brief without a forced plot outline."""
        goal = GoalRequest(
            prompt="Waddle Dee follows a changing signal",
            media_type="long_video",
            duration_seconds=30,
            style="cinematic animation",
            constraints={"character": "Waddle Dee"},
        )

        brief = "current news-derived delivery brief"
        segments = build_story_segments(goal, brief, 6, "curious", "text2longvideo")

        self.assertEqual(len(segments), 6)
        self.assertEqual(segments[0]["visual"], brief)
        self.assertNotIn("shots", segments[0])
        self.assertEqual(segments[-1]["segment_id"], "segment-6")
        self.assertNotIn("cause", segments[0])
        self.assertNotIn("action", segments[0])

    def test_five_second_brief_is_not_rewritten_into_a_fixed_action_contract(self) -> None:
        goal = GoalRequest(
            prompt="Kirby swats one glowing orb into a target",
            media_type="text2img2video",
            duration_seconds=5,
            style="cinematic anime",
            constraints={"character": "Kirby"},
        )
        brief = build_goal_brief(goal, goal.style, [])
        self.assertIn("Kirby swats one glowing orb into a target", brief["prompt"])
        self.assertNotIn("action contract", brief["prompt"].lower())
        self.assertIn("opening_keyframe_prompt", brief)

    def test_image_brief_preserves_requested_visual_details_without_a_fixed_visual_thesis(self) -> None:
        goal = GoalRequest(
            prompt="Kirby shelters inside one oversized paper lantern",
            media_type="image",
            style="soft storybook illustration",
            constraints={"character": "Kirby"},
        )

        brief = build_goal_brief(goal, goal.style, [])

        self.assertIn("Kirby shelters inside one oversized paper lantern", brief["prompt"])
        self.assertNotIn("one dominant visual mechanism", brief["prompt"])
        self.assertIn("style direction: soft storybook illustration", brief["prompt"])

    def test_user_given_minimal_long_video_segment_when_renderer_validates_then_missing_story_fields_are_optional(self) -> None:
        """User Given a segment with prompt text When render inputs are validated Then optional action and causal fields do not block it."""
        segments = validate_story_segments(
            [{"segment_id": "segment-1", "visual": "Kirby looks at the prop"}],
            1,
        )
        self.assertEqual(segments[0]["visual"], "Kirby looks at the prop")

if __name__ == "__main__":
    unittest.main()
