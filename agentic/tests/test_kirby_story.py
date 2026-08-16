from __future__ import annotations

import unittest
from pathlib import Path

from agentic.runtime.contracts import GoalRequest
from agentic.runtime.prompting import build_goal_brief, build_minimax_h3_prompt, build_story_segments
from agentic.storyboard import StoryboardError, build_story_plan, build_storyboard_segments, format_native_h3_prompt, load_storyboard


class KirbyStoryboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        cls.preset = load_storyboard(repo_root / "configs" / "storyboards" / "kirby_meadow_adventure.yaml")

    def test_two_segment_plan_has_distinct_story_goals_and_state_handoff(self) -> None:
        plan = build_story_plan(self.preset, duration_seconds=10)
        self.assertEqual(plan["segment_count"], 2)
        self.assertTrue(plan["progression_check"]["passed"])
        self.assertNotEqual(plan["segments"][0]["narrative_goal"], plan["segments"][1]["narrative_goal"])
        self.assertIn("glowing seed", plan["segments"][1]["start_state"])

    def test_thirty_second_plan_is_six_cards_with_a_resolved_ending(self) -> None:
        plan = build_story_plan(self.preset, duration_seconds=30)
        self.assertEqual(plan["segment_count"], 6)
        self.assertEqual(plan["planned_duration_seconds"], 30.0)
        self.assertTrue(plan["progression_check"]["passed"])
        self.assertEqual(len({segment["narrative_goal"] for segment in plan["segments"]}), 6)
        self.assertIn("no unresolved threat", plan["segments"][-1]["end_state"])
        self.assertIn("resolves here", plan["segments"][-1]["next_hook"])

    def test_storyboard_segments_are_usable_by_agentic_longvideo(self) -> None:
        segments = build_storyboard_segments(self.preset, segment_count=2, style="cinematic anime")
        self.assertEqual([segment["segment_id"] for segment in segments], ["star_seed_falls", "choose_the_mission"])
        self.assertIn("narrative goal", segments[0]["visual"])
        self.assertIn("Next story hook", segments[1]["visual"])

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

    def test_generic_prompt_builder_uses_storyboard_constraint(self) -> None:
        goal = GoalRequest(
            prompt="Kirby follows a mysterious light",
            media_type="long_video",
            duration_seconds=10,
            style="cinematic anime",
            constraints={
                "character": "Kirby",
                "storyboard_path": str(Path(__file__).resolve().parents[2] / "configs/storyboards/kirby_meadow_adventure.yaml"),
            },
        )
        segments = build_story_segments(goal, "Kirby story brief", 2, "playful")
        self.assertEqual(segments[0]["segment_id"], "star_seed_falls")
        self.assertEqual(segments[1]["segment_id"], "choose_the_mission")
        self.assertNotEqual(segments[0]["narrative_goal"], segments[1]["narrative_goal"])

    def test_thirty_second_plan_has_one_spine_across_three_acts(self) -> None:
        plan = build_story_plan(self.preset, duration_seconds=30)
        self.assertEqual(plan["story_spine"]["objective"].startswith("Kirby must"), True)
        self.assertEqual([segment["act"] for segment in plan["segments"]], [1, 1, 2, 2, 3, 3])
        self.assertTrue(all(segment["cause"] and segment["effect"] for segment in plan["segments"]))
        self.assertIn("returned star", plan["segments"][-1]["next_hook"])

    def test_story_spine_root_contract_is_required(self) -> None:
        broken = dict(self.preset)
        broken["story_spine"] = {key: value for key, value in self.preset["story_spine"].items() if key != "objective"}
        with self.assertRaises(StoryboardError):
            build_story_plan(broken, duration_seconds=30)

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


if __name__ == "__main__":
    unittest.main()
