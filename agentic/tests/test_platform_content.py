from __future__ import annotations

import unittest

from agentic.runtime.contracts import GoalRequest
from agentic.runtime.platform_content import build_platform_bundle


class PlatformContentTests(unittest.TestCase):
    def test_video_bundle_separates_youtube_seo_from_facebook_reel_copy(self) -> None:
        goal = GoalRequest(
            prompt="neon Kirby short animation",
            media_type="publish_review",
            constraints={
                "contains_synthetic_media": False,
                "youtube_contains_synthetic_media": False,
                "originality_basis": "creator_produced_or_owned_media",
            },
        )
        bundle = build_platform_bundle(
            goal=goal,
            caption=(
                "Kirby dodges a falling neon star and turns the near miss into a bright joke.\n\n"
                "The final glow gives the short its payoff.\n\n"
                "Which moment would you replay?"
            ),
            hashtags="#Kirby #NeonAnimation #ShortFilm #OriginalArt #ExtraTag",
            platform_captions={"youtube": "YouTube version", "facebook": "Facebook version"},
            platforms=["youtube", "facebook"],
            media_paths=["neon-kirby.mp4"],
        )

        youtube = bundle["youtube"]
        facebook = bundle["facebook"]
        self.assertEqual(youtube["format"], "video")
        self.assertEqual(youtube["additional_params"]["youtube_title"], "YouTube version")
        self.assertTrue(
            youtube["additional_params"]["youtube_description"].startswith("YouTube version")
        )
        self.assertEqual(
            youtube["additional_params"]["youtube_contains_synthetic_media"],
            True,
        )
        self.assertTrue(youtube["validation"]["is_platform_eligible"])
        self.assertEqual(facebook["format"], "reel")
        self.assertEqual(facebook["additional_params"]["facebook_use_reels"], True)
        self.assertEqual(facebook["hashtags"], "#Kirby #NeonAnimation #ShortFilm")
        self.assertLessEqual(facebook["validation"]["hashtag_count"], 3)
        self.assertIn(
            "reel_duration_and_canvas_checked_at_publish",
            facebook["validation"]["warnings"],
        )

    def test_image_bundle_is_native_photo_and_not_a_reel(self) -> None:
        goal = GoalRequest(prompt="Kirby crystal key visual", media_type="publish_review")
        bundle = build_platform_bundle(
            goal=goal,
            caption="Kirby holds a blue crystal under a quiet sky.",
            hashtags="#Kirby #CrystalArt",
            platform_captions={"facebook": "Kirby holds a blue crystal under a quiet sky."},
            platforms=["facebook"],
            media_paths=["crystal.png"],
        )

        facebook = bundle["facebook"]
        self.assertEqual(facebook["format"], "native_photo")
        self.assertEqual(facebook["additional_params"], {})
        self.assertTrue(facebook["validation"]["is_platform_eligible"])

    def test_youtube_requires_video_but_keeps_structural_dispatch_ready_state(self) -> None:
        goal = GoalRequest(prompt="Kirby image post", media_type="publish_review")
        bundle = build_platform_bundle(
            goal=goal,
            caption="Kirby waves beside a blue crystal.",
            hashtags="#Kirby",
            platform_captions={"youtube": "Kirby waves beside a blue crystal."},
            platforms=["youtube"],
            media_paths=["crystal.png"],
        )

        validation = bundle["youtube"]["validation"]
        self.assertFalse(validation["is_platform_eligible"])
        self.assertTrue(validation["is_publish_ready"])
        self.assertIn("requires_video_media", validation["issues"])

    def test_public_packaging_removes_internal_hashtag_and_splits_cjk_sentences(self) -> None:
        goal = GoalRequest(prompt="Kirby neon short", media_type="publish_review")
        bundle = build_platform_bundle(
            goal=goal,
            caption="星星落下。Kirby閃身躲過，最後留下藍色光芒。",
            hashtags="#Kirby #mediaoverload #Neon",
            platform_captions={"youtube": "星星落下。Kirby閃身躲過，最後留下藍色光芒。"},
            platforms=["youtube"],
            media_paths=["clip.mp4"],
        )

        youtube = bundle["youtube"]
        self.assertEqual(youtube["hashtags"], "#Kirby #Neon")
        self.assertEqual(
            youtube["additional_params"]["youtube_title"],
            "星星落下",
        )
        self.assertNotIn("mediaoverload", youtube["additional_params"]["youtube_tags"])


if __name__ == "__main__":
    unittest.main()
