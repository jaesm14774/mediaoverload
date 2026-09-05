from __future__ import annotations

import unittest

from agentic.runtime.contracts import GoalRequest
from agentic.runtime.llm_engine import LLMPromptEngine
from agentic.runtime.platform_content import build_platform_bundle
from agentic.runtime.post_strategy import resolve_post_strategy


class _FakeTextModel:
    def __init__(self, response: str) -> None:
        self.response = response

    def chat_completion(self, messages: list[dict], **kwargs) -> str:
        return self.response


class _FakeManager:
    def __init__(self, response: str) -> None:
        self.text_model = _FakeTextModel(response)


class PostStrategyTests(unittest.TestCase):
    def test_strategy_is_reproducible_but_seeded_posts_can_rotate(self) -> None:
        goal = GoalRequest(
            prompt="publish a short visual story",
            media_type="publish_review",
            style="social promo",
            constraints={"post_strategy_seed": "run-a"},
        )
        first = resolve_post_strategy(goal, ["C:\\asset-a.mp4"])
        same = resolve_post_strategy(goal, ["C:\\asset-a.mp4"])
        other = resolve_post_strategy(
            GoalRequest(
                prompt=goal.prompt,
                media_type=goal.media_type,
                style=goal.style,
                constraints={"post_strategy_seed": "run-b"},
            ),
            ["C:\\asset-b.mp4"],
        )

        self.assertEqual(first, same)
        self.assertNotEqual(first["variation_key"], other["variation_key"])
        self.assertIn("editorial_question", first)
        self.assertIn("hashtag_policy", first)

    def test_news_terms_are_source_terms_and_news_variant_requires_context(self) -> None:
        plain = resolve_post_strategy(
            GoalRequest(
                prompt="publish a visual story",
                constraints={"post_strategy_seed": "seed-1"},
            ),
            ["C:\\asset.png"],
        )
        self.assertNotEqual(plain["variant_id"], "news_mechanism_bridge")
        news = resolve_post_strategy(
            GoalRequest(
                prompt="publish a visual metaphor",
                constraints={
                    "post_strategy_seed": "seed-1",
                    "news_context": {
                        "title": "Remote access security incident",
                        "topic": "identity security",
                        "keywords": ["remote access", "verification"],
                    },
                },
            ),
            ["C:\\asset.mp4"],
        )
        self.assertIn("identity security", news["discovery_terms"])
        self.assertIn("remote access", news["discovery_terms"])

    def test_empty_hashtags_are_a_valid_publish_package(self) -> None:
        bundle = build_platform_bundle(
            goal=GoalRequest(prompt="publish a quiet crystal scene", media_type="publish_review"),
            caption="Kirby rests beside a blue crystal.",
            hashtags="",
            platform_captions={"instagram": "Kirby rests beside a blue crystal."},
            platforms=["instagram"],
            media_paths=["crystal.png"],
        )

        self.assertEqual(bundle["instagram"]["hashtags"], "")
        self.assertTrue(bundle["instagram"]["validation"]["is_publish_ready"])
        self.assertIn("post_strategy", bundle["instagram"])

    def test_youtube_tags_do_not_mine_the_production_prompt(self) -> None:
        bundle = build_platform_bundle(
            goal=GoalRequest(
                prompt="publish remote access security animation with cinematic metadata",
                media_type="publish_review",
            ),
            caption="A blue crystal catches the light.",
            hashtags="",
            platform_captions={"youtube": "A blue crystal catches the light."},
            platforms=["youtube"],
            media_paths=["crystal.mp4"],
        )

        self.assertEqual(bundle["youtube"]["additional_params"]["youtube_tags"], [])

    def test_caption_model_may_choose_no_hashtags(self) -> None:
        engine = LLMPromptEngine(
            mode="llm",
            manager=_FakeManager(
                '{"caption":"A quiet blue crystal catches the light.","hashtags":"","platform_captions":{"instagram":"A quiet blue crystal catches the light."}}'
            ),
        )

        result = engine.prepare_publish_caption(
            GoalRequest(prompt="publish a quiet crystal scene", media_type="publish_review"),
            prefix="",
            hashtags=["crystal", "art"],
            platforms=["instagram"],
            media_paths=["crystal.png"],
        )

        self.assertEqual(result["hashtags"], "")
        self.assertEqual(
            LLMPromptEngine._normalize_hashtag_text("", required_hashtags=["crystal"]),
            "",
        )

    def test_generic_reach_bait_is_not_a_content_hashtag(self) -> None:
        self.assertEqual(
            LLMPromptEngine._normalize_hashtag_text("#FYP #ForYou #crystal #ExplorePage"),
            "#crystal",
        )


if __name__ == "__main__":
    unittest.main()
