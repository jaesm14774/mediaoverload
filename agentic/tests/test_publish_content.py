from __future__ import annotations

import unittest

from agentic.runtime.contracts import GoalRequest
from agentic.runtime.platform_content import build_platform_bundle, enrich_publish_bundle
from agentic.runtime.llm_engine import LLMPromptEngine


class _FakeTextModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def chat_completion(self, messages: list[dict], **kwargs) -> str:
        return self.responses.pop(0)


class _FakeManager:
    def __init__(self, responses: list[str]) -> None:
        self.text_model = _FakeTextModel(responses)


class PublishContentTests(unittest.TestCase):



    def test_user_given_threads_caption_at_or_over_the_limit_when_preparing_publication_then_only_valid_copy_is_ready(self) -> None:
        """User Given a Threads caption near its limit When preparing publication Then copy up to 500 characters is eligible and longer copy is blocked."""
        goal = GoalRequest(prompt="publish a short clip", media_type="publish_review", style="social promo")

        accepted = build_platform_bundle(
            goal=goal,
            caption="a" * 500,
            hashtags="",
            platform_captions={},
            platforms=["threads"],
            media_paths=["clip.mp4"],
            post_strategy={"strategy_version": "test"},
        )["threads"]
        rejected = build_platform_bundle(
            goal=goal,
            caption="a" * 501,
            hashtags="",
            platform_captions={},
            platforms=["threads"],
            media_paths=["clip.mp4"],
            post_strategy={"strategy_version": "test"},
        )["threads"]

        self.assertEqual(accepted["format"], "video")
        self.assertTrue(accepted["validation"]["is_platform_publish_ready"])
        self.assertFalse(rejected["validation"]["is_platform_publish_ready"])
        self.assertIn("threads_text_exceeds_500_characters", rejected["validation"]["issues"])

    def test_user_given_generated_copy_when_preparing_publication_then_each_platform_receives_eligible_content(self) -> None:
        """User Given generated copy and image media When preparing publication Then platform captions, eligibility and dispatch readiness are preserved."""
        engine = LLMPromptEngine(mode='llm', manager=_FakeManager(['{"caption":"caption body","hashtags":"#one #two","platform_captions":{"instagram":"ig caption","youtube":"yt caption"}}']))
        goal = GoalRequest(prompt="publish kirby clip", media_type="publish_review", style="social promo")

        result = engine.prepare_publish_caption(
            goal,
            prefix="launch",
            hashtags=["one", "two"],
            platforms=["instagram", "youtube"],
            media_paths=["C:\\asset.png"],
            review_notes="stronger hook",
        )

        result = enrich_publish_bundle(result, goal, ["instagram", "youtube"], ["C:\\asset.png"])

        self.assertEqual(result["caption_strategy"], "platform_adapted")
        self.assertTrue(result["dispatch_ready"])
        self.assertTrue(result["platform_bundle"]["instagram"]["validation"]["is_publish_ready"])
        self.assertEqual(result["platform_bundle"]["instagram"]["caption"], "ig caption")
        self.assertEqual(result["platform_bundle"]["youtube"]["caption"], "yt caption")
        self.assertEqual(result["platform_bundle"]["youtube"]["format"], "video")
        self.assertEqual(
            result["platform_bundle"]["youtube"]["additional_params"]["youtube_title"],
            "yt caption",
        )
        self.assertTrue(
            result["platform_bundle"]["youtube"]["content_strategy"]["answer_first"]
        )
        self.assertFalse(
            result["platform_bundle"]["youtube"]["validation"]["is_platform_eligible"]
        )


if __name__ == "__main__":
    unittest.main()
