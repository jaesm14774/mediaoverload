from __future__ import annotations

import unittest

from agentic.minimax_prompting import compose_minimax_h3_prompt
from agentic.runtime.contracts import GoalRequest
from agentic.runtime.prompting import build_goal_brief
from agentic.video_directing import (
    VIDEO_MOTION_DIRECTION_MARKER,
    apply_video_motion_direction,
    apply_video_opening_frame_direction,
)


class VideoDirectingTests(unittest.TestCase):
    def test_user_video_is_lively_all_the_way_with_a_gentle_camera(self) -> None:
        """User wants a lively charming clip. Given a drifting-camera source, When directed, Then the character moves constantly through the last frame and the camera only pushes in gently."""
        source = "Kracko slowly rotates as the camera slowly pushes in to a serene close-up."

        result = apply_video_motion_direction(source, "text2img2video")

        self.assertLess(result.index(VIDEO_MOTION_DIRECTION_MARKER), result.index(source))
        self.assertIn("constant, energetic motion for the whole clip", result)
        self.assertIn("still moving in the last frame", result)
        self.assertIn("one slow gentle push-in", result)
        self.assertIn("no pan, orbit, or cut", result)
        self.assertEqual(apply_video_motion_direction(result, "text2img2video"), result)

    def test_user_video_keeps_the_character_visible_at_one_size_while_it_hops(self) -> None:
        """User wants to see the character the whole clip. Given a hopping character, When directed, Then hops stay up-and-down or sideways at a steady size facing the camera and never run into the distance."""
        result = apply_video_motion_direction("Waddle Dee hops along the plaza.", "text2img2video").casefold()

        self.assertIn("same size", result)
        self.assertIn("never toward or away from the camera", result)
        self.assertIn("faces the camera", result)

    def test_user_video_has_no_writing_on_signs_boards_or_screens(self) -> None:
        """User wants clean pictures. Given a brief with a traffic sign, When directed, Then signs, boards, papers, and screens stay blank or show a simple icon so no garbled lettering appears."""
        result = apply_video_motion_direction("Dedede stands by a diversion sign.", "text2img2video").casefold()

        self.assertIn("sign, board, paper, or screen", result)
        self.assertIn("no writing", result)

    def test_user_video_keeps_the_illustrated_look_without_extra_story(self) -> None:
        """User wants the picture to stay itself. Given a character clip, When directed, Then the painted look holds, nothing morphs or is added, and no second subplot appears."""
        result = apply_video_motion_direction("Kirby hops through a puddle.", "text2img2video")

        self.assertIn("illustrated look of the frame stays exactly the same", result)
        self.assertIn("nothing morphs and no new objects appear", result)
        self.assertIn("Do not add a second goal, obstacle, or subplot", result)
        self.assertIn("fully inside the frame", result)

    def test_user_video_props_and_scene_respond_on_every_beat(self) -> None:
        """User wants a lively but faithful scene. Given a character clip, When directed, Then the action chains into connected moves and existing scene elements respond."""
        result = apply_video_motion_direction("Kirby carries an umbrella.", "text2img2video")

        self.assertIn("existing elements", result)
        self.assertIn("respond on every beat", result)
        self.assertIn("two or three connected moves", result)

    def test_user_video_opening_frame_leaves_room_to_move(self) -> None:
        """User wants a clean clip. Given a still opening prompt, When prepared, Then the character is small and visible and nothing crops when it jumps."""
        source = "A watercolor cloud creature centered in a pastel sky."

        result = apply_video_opening_frame_direction(source, "text2img2video")

        self.assertLess(result.index("Opening-frame action staging:"), result.index(source))
        self.assertIn("character small and clearly visible", result)
        self.assertIn("nothing crops as it jumps and moves", result)
        self.assertIn(source, result)

    def test_user_deliberately_still_video_can_keep_its_composition(self) -> None:
        """User wants a contemplative video. Given an explicit still request, When directed, Then the prompt preserves that intent."""
        source = "A contemplative locked-off shot of a candle flame in a dark room."

        result = apply_video_motion_direction(source, "image_to_video")

        self.assertIn("Honor an explicitly still or solemn brief", result)
        self.assertIn(source, result)

    def test_user_still_image_prompt_keeps_video_staging_out(self) -> None:
        """User wants a still image. Given an image-only prompt, When prepared, Then video direction is absent."""
        source = "A watercolor cloud creature centered in a pastel sky."

        self.assertEqual(apply_video_motion_direction(source, "text2image"), source)
        self.assertEqual(apply_video_opening_frame_direction(source, "text2image"), source)

    def test_user_text_to_video_brief_separates_action_from_opening_still(self) -> None:
        """User wants an action-led clip. Given a video goal, When expanded, Then motion and keyframe directions stay separate."""
        goal = GoalRequest(
            prompt="A cloud creature floats in a pastel sky.",
            media_type="text2img2video",
            style="tactile storybook watercolor",
        )

        brief = build_goal_brief(goal, goal.style, [])

        self.assertTrue(brief["prompt"].startswith(VIDEO_MOTION_DIRECTION_MARKER))
        self.assertTrue(brief["opening_keyframe_prompt"].startswith("Opening-frame action staging:"))
        self.assertNotIn(VIDEO_MOTION_DIRECTION_MARKER, brief["opening_keyframe_prompt"])

    def test_user_h3_video_gets_camera_and_performance_direction(self) -> None:
        """User Given an H3 scene, When its prompt is composed, Then it contains the scene without generic writing instructions."""
        prompt = compose_minimax_h3_prompt(
            duration_seconds=6,
            character="Kirby",
            style="tactile storybook watercolor",
            base_prompt="Kirby catches a runaway kite.",
            render_mode="image_to_video",
            prior_frame=True,
        )

        self.assertIn("Kirby catches a runaway kite", prompt)
        self.assertNotIn(VIDEO_MOTION_DIRECTION_MARKER, prompt)
        self.assertNotIn("Input relation:", prompt)

if __name__ == "__main__":
    unittest.main()
