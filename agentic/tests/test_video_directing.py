from __future__ import annotations

import unittest

from agentic.minimax_prompting import compose_minimax_h3_prompt
from agentic.runtime.contracts import GoalRequest
from agentic.runtime.prompting import build_goal_brief
from agentic.video_directing import (
    VIDEO_MOTION_DIRECTION_MARKER,
    apply_segment_video_direction,
    apply_video_motion_direction,
    apply_video_opening_frame_direction,
)


class VideoDirectingTests(unittest.TestCase):
    def test_user_action_led_video_overrides_weak_source_motion(self) -> None:
        """User wants lively video. Given a slow source description, When prepared, Then causal protagonist action leads the prompt."""
        source = "Kracko slowly rotates as the camera slowly pushes in to a serene close-up."

        result = apply_video_motion_direction(source, "text2img2video")

        self.assertLess(result.index(VIDEO_MOTION_DIRECTION_MARKER), result.index(source))
        self.assertIn("Establish a visible goal or obstacle", result)
        self.assertIn("contact or manipulation, and a visible result", result)
        self.assertIn("begin the action in the first tenth", result)
        self.assertIn("visible action or pose progression through most of the shot", result)
        self.assertIn("rather than camera drift", result)
        self.assertIn("final tenth", result)
        self.assertEqual(apply_video_motion_direction(result, "text2img2video"), result)

    def test_user_character_video_has_a_readable_charming_performance_arc(self) -> None:
        """User wants cute movement. Given a character-led clip, When directed, Then its expression and payoff stay readable."""
        result = apply_video_motion_direction("Kirby reaches for a kite.", "text2img2video")

        self.assertIn("anticipation in the eyes and body", result)
        self.assertIn("brief charming reaction", result)
        self.assertIn("surprised blink, delighted glance, or proud bounce", result)
        self.assertIn("leave a warm, satisfying afterbeat", result)
        self.assertIn("silhouette change", result)
        self.assertIn("one or two small secondary motions tied to the action", result)

    def test_user_camera_direction_explains_mechanics_and_purpose(self) -> None:
        """User wants intentional camera work. Given a moving character, When directed, Then motion serves a legible path and reveal."""
        result = apply_video_motion_direction("Kirby runs toward a kite.", "text2img2video")

        self.assertIn("one motivated camera move or intentional lock", result)
        self.assertIn("A pan or tilt rotates from a fixed position", result)
        self.assertIn("a dolly, truck, or arc travels through space and creates parallax", result)
        self.assertIn("a follow or tracking move keeps pace with the subject", result)
        self.assertIn("a zoom changes focal length", result)
        self.assertIn("what it reveals", result)
        self.assertIn("Let the camera support the action rather than compete with it", result)

    def test_user_given_a_shot_ends_at_a_new_scale_when_video_direction_is_added_then_camera_and_payoff_remain_coherent(self) -> None:
        """User wants coherent camera work and a visible cute payoff. Given a shot changes subject scale, When video direction is added, Then the named move can cause the change and the result remains readable."""
        result = apply_video_motion_direction("Kirby frees a kite and beams at it.", "text2img2video")

        self.assertIn("not subject distance or image scale", result)
        self.assertIn("A change in subject scale must come from a matching camera move, zoom, or physical subject travel", result)
        self.assertIn("one continuous camera move or a locked shot per shot", result)
        self.assertIn("the character's expressive face and the visible result together", result)
        self.assertIn("if the final beat is a reaction close-up, show the result first", result)

    def test_user_given_camera_moves_toward_a_payoff_when_video_direction_is_added_then_face_and_result_keep_safe_margins(self) -> None:
        """User: Given a camera move approaches a payoff, When video direction is added, Then the face and story result remain fully inside frame with clear margins."""
        result = apply_video_motion_direction("Kirby waters a wilted flower and smiles.", "text2img2video")

        self.assertIn("keep the face and every story-critical object within frame with clear edge margins", result.casefold())
        self.assertIn("if a close-up would crop the reaction or result, stop wider or cut to a separate shot", result.casefold())

    def test_user_given_a_single_action_brief_when_video_direction_is_added_then_story_and_style_stay_coherent(self) -> None:
        """User: Given a brief asks for one action, When video direction is added, Then it preserves one causal arc and the selected medium across the payoff."""
        result = apply_video_motion_direction("Kirby spins a top and reacts.", "text2img2video").casefold()

        self.assertIn("one visible goal, one obstacle, and one turn that changes the outcome", result)
        self.assertIn("each beat changes a visible state and causes the next", result)
        self.assertIn("do not invent a second problem or subplot", result)
        self.assertIn("stable visible rules for medium and surface, palette and light, and motion or transitions", result)

    def test_user_deliberately_still_video_can_keep_its_composition(self) -> None:
        """User wants a contemplative video. Given an explicit still request, When directed, Then the prompt preserves that intent."""
        source = "A contemplative locked-off shot of a candle flame in a dark room."

        result = apply_video_motion_direction(source, "image_to_video")

        self.assertIn("Honor an explicitly still or solemn brief", result)
        self.assertIn(source, result)

    def test_user_video_image_starts_with_action_affordance(self) -> None:
        """User wants an active video. Given a still opening prompt, When prepared, Then it stages intention and a visible goal."""
        source = "A watercolor cloud creature centered in a pastel sky."

        result = apply_video_opening_frame_direction(source, "text2img2video")

        self.assertLess(result.index("Opening-frame action staging:"), result.index(source))
        self.assertIn("visible goal or obstacle", result)
        self.assertIn("split second before movement begins", result)
        self.assertIn("clear travel lane", result)
        self.assertIn("eyes and a loaded, angled pose", result)
        self.assertIn(source, result)

    def test_user_still_image_prompt_keeps_video_staging_out(self) -> None:
        """User wants a still image. Given an image-only prompt, When prepared, Then video direction is absent."""
        source = "A watercolor cloud creature centered in a pastel sky."

        self.assertEqual(apply_video_motion_direction(source, "text2image"), source)
        self.assertEqual(apply_video_opening_frame_direction(source, "text2image"), source)

    def test_user_still_image_with_segment_metadata_keeps_video_camera_direction_out(self) -> None:
        """User wants a still image. Given a segment includes optional camera metadata, When its image prompt is built, Then time-based directions stay out."""
        prompt = apply_segment_video_direction(
            "Kirby beside a paper lantern.",
            {"camera": "track beside Kirby", "action": "Kirby hops"},
            "text2image",
        )

        self.assertEqual(prompt, "Kirby beside a paper lantern.")

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
