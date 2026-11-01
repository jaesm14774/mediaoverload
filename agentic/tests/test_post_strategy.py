from __future__ import annotations

from agentic.runtime.prompt_tasks.publishing import normalize_hashtag_text

import unittest

from agentic.runtime.contracts import GoalRequest
from agentic.runtime.llm_engine import LLMPromptEngine
from agentic.runtime.platform_content import build_platform_bundle
from agentic.runtime.post_strategy import resolve_post_strategy
from agentic.skills.agent_social import AgentSocialSkills


class _FakeTextModel:
    def __init__(self, response: str) -> None:
        self.response = response

    def chat_completion(self, messages: list[dict], **kwargs) -> str:
        return self.response


class _FakeManager:
    def __init__(self, response: str) -> None:
        self.text_model = _FakeTextModel(response)


class PostStrategyTests(unittest.TestCase):
    def test_user_given_publish_caption_when_review_post_is_formatted_then_only_the_caption_and_hashtags_are_shown(self) -> None:
        """User Given a publish caption and hashtags When Discord review text is formatted Then it contains only publish-ready content."""
        caption = "A comet crosses the pastel sky and glows above the hill."
        review = AgentSocialSkills._format_review_post(
            caption=caption,
            hashtags="#Kirby",
        )

        self.assertEqual(review, f"{caption}\n\n#Kirby")

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

    def test_user_given_clear_article_topics_when_preparing_publish_copy_then_five_relevant_hashtags_are_preserved(self) -> None:
        """User Given a source article supports several discovery terms When preparing publish copy Then five relevant hashtags can reach the platform bundle."""
        engine = LLMPromptEngine(
            mode="llm",
            manager=_FakeManager(
                '{"caption":"這則消息真正值得留意的，是道路安全如何影響每天通勤的人。","hashtags":"#交通安全 #道路事故 #國道一號 #行車安全 #通勤生活","platform_captions":{"instagram_graph":"塞車畫面不只是在說車流變慢，也提醒我們事故會把用路人的時間一起卡住。"}}'
            ),
        )

        result = engine.prepare_publish_caption(
            GoalRequest(
                prompt="publish a visual story about traffic safety",
                media_type="publish_review",
                constraints={
                    "news_grounding_required": True,
                    "news_context": {
                        "title": "國道交通事故與通勤安全",
                        "topic": "道路安全",
                        "content": "新聞來源提供事故與通勤影響的背景。",
                    },
                },
            ),
            prefix="",
            hashtags=["交通安全", "道路事故", "國道一號", "行車安全", "通勤生活"],
            platforms=["instagram_graph"],
            media_paths=["road-safety.png"],
        )

        self.assertEqual(
            result["hashtags"],
            "#交通安全 #道路事故 #國道一號 #行車安全 #通勤生活",
        )
        self.assertEqual(
            normalize_hashtag_text("", required_hashtags=["交通安全"]),
            "",
        )

    def test_user_given_no_supported_discovery_terms_when_preparing_publish_copy_then_empty_hashtags_remain_valid(self) -> None:
        """User Given the media has no supported discovery terms When preparing publish copy Then irrelevant hashtags are not invented to meet a quota."""
        self.assertEqual(normalize_hashtag_text(""), "")

    def test_generic_reach_bait_is_not_a_content_hashtag(self) -> None:
        self.assertEqual(
            normalize_hashtag_text("#FYP #ForYou #crystal #ExplorePage"),
            "#crystal",
        )


if __name__ == "__main__":
    unittest.main()
