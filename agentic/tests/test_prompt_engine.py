from __future__ import annotations

import unittest

from agentic.runtime.contracts import GoalRequest
from agentic.runtime.llm_engine import LLMPromptEngine
from agentic.runtime.prompt_engine import PromptEngine


class _FakeTextModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def chat_completion(self, messages: list[dict], **kwargs) -> str:
        return self.responses.pop(0)


class _FakeManager:
    def __init__(self, responses: list[str]) -> None:
        self.text_model = _FakeTextModel(responses)


class PromptEngineTests(unittest.TestCase):
    def test_compose_prompt_returns_llm_bundle(self) -> None:
        engine = PromptEngine(
            LLMPromptEngine(
                mode="llm",
                manager=_FakeManager(
                    [
                        '{"prompt":"llm composed prompt","negative_prompt":"llm negative"}',
                    ]
                ),
            )
        )
        goal = GoalRequest(prompt="kirby key visual", media_type="image", style="anime key visual")

        result = engine.compose_prompt(goal, "kirby key visual", "anime key visual", prefix="hero shot")

        self.assertEqual(result["prompt"], "llm composed prompt")
        self.assertEqual(result["negative_prompt"], "llm negative")
        self.assertEqual(result["prompt_mode"], "llm")

    def test_build_sticker_prompt_set_returns_llm_bundle(self) -> None:
        engine = PromptEngine(
            LLMPromptEngine(
                mode="llm",
                manager=_FakeManager(
                    [
                        '[{"label":"sticker_01","expression":"joy","prompt":"joy prompt"},{"label":"sticker_02","expression":"sleepy","prompt":"sleepy prompt"}]',
                    ]
                ),
            )
        )
        goal = GoalRequest(prompt="kirby stickers", media_type="sticker_pack", style="LINE sticker")

        result = engine.build_sticker_prompt_set(
            goal,
            expressions=["joy", "sleepy"],
            character="Kirby",
            prompt_prefix="ramen shop",
            style="LINE sticker",
        )

        self.assertEqual(result["prompt_sets"][0]["prompt"], "joy prompt")
        self.assertEqual(result["prompt_count"], 2)
        self.assertEqual(result["prompt_mode"], "llm")

    def test_review_asset_candidates_adds_retry_metadata(self) -> None:
        engine = PromptEngine(
            LLMPromptEngine(
                mode="llm",
                manager=_FakeManager(
                    [
                        '{"selected_assets":["C:\\\\clip_b.mp4"],"ranked_candidates":[{"media_path":"C:\\\\clip_b.mp4","score":96,"rationale":"best motion"},{"media_path":"C:\\\\clip_a.png","score":80,"rationale":"weak still"}],"selection_rationale":"Best matches stronger motion.","regeneration_notes":"Push action harder."}',
                    ]
                ),
            )
        )
        goal = GoalRequest(prompt="review kirby assets", media_type="publish_review", style="social promo")

        result = engine.review_asset_candidates(
            goal,
            media_paths=["C:\\clip_a.png", "C:\\clip_b.mp4"],
            review_notes="needs stronger motion and tighter framing",
            selection_limit=1,
        )

        self.assertEqual(result["rejected_assets"], ["C:\\clip_a.png"])
        self.assertIn("motion_weak", result["failure_tags"])
        self.assertIn("composition_weak", result["failure_tags"])
        self.assertEqual(result["retry_intensity"], "high")
        self.assertTrue(result["publish_ready"])

    def test_prepare_publish_caption_adds_platform_bundle_and_dispatch_ready(self) -> None:
        engine = PromptEngine(
            LLMPromptEngine(
                mode="llm",
                manager=_FakeManager(
                    [
                        '{"caption":"caption body","hashtags":"#one #two","platform_captions":{"instagram":"ig caption","youtube":"yt caption"}}',
                    ]
                ),
            )
        )
        goal = GoalRequest(prompt="publish kirby clip", media_type="publish_review", style="social promo")

        result = engine.prepare_publish_caption(
            goal,
            prefix="launch",
            hashtags=["one", "two"],
            platforms=["instagram", "youtube"],
            media_paths=["C:\\asset.png"],
            review_notes="stronger hook",
        )

        self.assertEqual(result["caption_strategy"], "platform_adapted")
        self.assertTrue(result["dispatch_ready"])
        self.assertTrue(result["platform_bundle"]["instagram"]["validation"]["is_publish_ready"])
        self.assertEqual(result["platform_bundle"]["instagram"]["caption"], "ig caption")
        self.assertEqual(result["platform_bundle"]["youtube"]["caption"], "yt caption")
        self.assertNotIn("additional_params", result["platform_bundle"]["youtube"])


if __name__ == "__main__":
    unittest.main()
