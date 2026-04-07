from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from agentic.runtime.contracts import GoalRequest
from agentic.runtime.llm_engine import LLMPromptEngine


class _FakeTextModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def chat_completion(self, messages: list[dict], **kwargs) -> str:
        return self.responses.pop(0)


class _FakeManager:
    def __init__(self, responses: list[str]) -> None:
        self.text_model = _FakeTextModel(responses)


class LLMEngineTests(unittest.TestCase):
    def test_expand_goal_uses_llm_json_when_available(self) -> None:
        engine = LLMPromptEngine(
            mode="llm",
            manager=_FakeManager(
                [
                    '{"creative_brief":"llm brief","prompt":"llm prompt","negative_prompt":"llm negative"}',
                ]
            ),
        )
        goal = GoalRequest(prompt="kirby runs", media_type="long_video", style="anime")

        result = engine.expand_goal(goal, "anime", [])

        self.assertEqual(result["creative_brief"], "llm brief")
        self.assertEqual(result["prompt"], "llm prompt")
        self.assertEqual(result["negative_prompt"], "llm negative")
        self.assertEqual(result["prompt_mode"], "llm")

    def test_segment_story_uses_llm_array_when_available(self) -> None:
        engine = LLMPromptEngine(
            mode="llm",
            manager=_FakeManager(
                [
                    '[{"segment_id":"segment-1","visual":"shot one","narration":"line one"},{"segment_id":"segment-2","visual":"shot two","narration":"line two"}]',
                ]
            ),
        )
        goal = GoalRequest(prompt="kirby runs", media_type="long_video", style="anime")

        segments = engine.segment_story(goal, "brief", 2, "playful")

        self.assertEqual(segments[0]["visual"], "shot one")
        self.assertEqual(segments[1]["narration"], "line two")

    def test_compose_prompt_uses_llm_json_when_available(self) -> None:
        engine = LLMPromptEngine(
            mode="llm",
            manager=_FakeManager(
                [
                    '{"prompt":"llm composed prompt","negative_prompt":"llm negative"}',
                ]
            ),
        )
        goal = GoalRequest(prompt="kirby runs", media_type="image", style="anime")

        result = engine.compose_prompt(goal, "kirby runs", "anime", prefix="hero shot")

        self.assertEqual(result["prompt"], "llm composed prompt")
        self.assertEqual(result["negative_prompt"], "llm negative")
        self.assertEqual(result["prompt_mode"], "llm")

    def test_route_generation_strategy_uses_llm_json_when_available(self) -> None:
        engine = LLMPromptEngine(
            mode="llm",
            manager=_FakeManager(
                [
                    '{"generation_type":"sticker_pack","workflow_plan":{"image_workflow_name":"nova-anime-xl","video_workflow_name":"wan2.2_gguf_i2v","refine_workflow_name":"","transition_workflow_name":"","upscale_workflow_name":""},"count_plan":{"image_count":1,"video_count":1,"segment_count":1,"review_selection_limit":3,"sticker_expression_count":8,"images_per_prompt":2},"reason":"Sticker prompt with clean outline fits sticker_pack best."}',
                ]
            ),
        )

        result = engine.route_generation_strategy(
            prompt="Kirby sticker emotions: happy, angry, crying, sleepy",
            character="Kirby",
            style="LINE sticker",
            generation_type_candidates=["text2img", "sticker_pack"],
            workflow_stage_candidates={
                "text2img": {"image_workflow_name": ["nova_model_plus_z_image_anime"]},
                "sticker_pack": {
                    "image_workflow_name": ["nova-anime-xl", "nova_model_plus_z_image_anime"],
                    "video_workflow_name": ["wan2.2_gguf_i2v"],
                },
            },
            count_policies={
                "text2img": {"image_count": {"min": 1, "max": 4}},
                "sticker_pack": {
                    "image_count": {"min": 1, "max": 1},
                    "video_count": {"min": 1, "max": 2},
                    "segment_count": {"min": 1, "max": 1},
                    "review_selection_limit": {"min": 1, "max": 6},
                    "sticker_expression_count": {"min": 4, "max": 12},
                    "images_per_prompt": {"min": 1, "max": 4},
                },
            },
            routing_hints={"strategy_preferences": {"sticker_pack": ["sticker", "reaction"]}},
        )

        self.assertEqual(result["generation_type"], "sticker_pack")
        self.assertEqual(result["workflow_plan"]["image_workflow_name"], "nova-anime-xl")
        self.assertEqual(result["count_plan"]["images_per_prompt"], 2)
        self.assertEqual(result["count_plan"]["review_selection_limit"], 3)
        self.assertEqual(result["prompt_mode"], "llm")

    def test_route_generation_strategy_raises_when_manager_unavailable(self) -> None:
        engine = LLMPromptEngine(mode="llm", manager=None)

        with patch.object(engine, "_manager_or_none", return_value=None):
            engine._manager_error = "ValueError: OpenRouter API key missing"
            with self.assertRaises(Exception):
                engine.route_generation_strategy(
                    prompt="Kirby sticker emotions",
                    character="Kirby",
                    style="LINE sticker",
                    generation_type_candidates=["sticker_pack", "text2img"],
                    workflow_stage_candidates={"sticker_pack": {"image_workflow_name": ["nova-anime-xl"]}},
                    count_policies={
                        "sticker_pack": {
                            "image_count": {"min": 1, "max": 1},
                            "video_count": {"min": 1, "max": 1},
                            "segment_count": {"min": 1, "max": 1},
                            "review_selection_limit": {"min": 1, "max": 2},
                            "sticker_expression_count": {"min": 4, "max": 12},
                            "images_per_prompt": {"min": 1, "max": 4},
                        }
                    },
                )

    def test_generate_autonomous_scene_prompt_uses_llm_when_available(self) -> None:
        engine = LLMPromptEngine(
            mode="llm",
            manager=_FakeManager(
                [
                    '{"prompt":"kirby turns a headline into a whimsical alley chase","creative_seed":"headline seed","source":"autonomous_llm"}',
                ]
            ),
        )

        result = engine.generate_autonomous_scene_prompt(
            character="Kirby",
            style="anime",
            media_type="text2video",
            news_context={"title": "headline"},
        )

        self.assertEqual(result["prompt"], "kirby turns a headline into a whimsical alley chase")
        self.assertEqual(result["creative_seed"], "headline seed")
        self.assertEqual(result["source"], "autonomous_llm")
        self.assertEqual(result["prompt_mode"], "llm")

    def test_sticker_expressions_uses_llm_json_when_available(self) -> None:
        engine = LLMPromptEngine(
            mode="llm",
            manager=_FakeManager(
                [
                    '["joyful leap","sleepy yawn","angry steam","big surprise","heart eyes","confused tilt","celebration pose","dramatic sob"]',
                ]
            ),
        )
        goal = GoalRequest(prompt="ramen shop", media_type="sticker_pack", style="sticker")

        expressions = engine.sticker_expressions(goal, "ramen shop", "Kirby", 8)

        self.assertEqual(len(expressions), 8)
        self.assertEqual(expressions[0], "joyful leap")

    def test_build_sticker_prompt_set_uses_llm_json_when_available(self) -> None:
        engine = LLMPromptEngine(
            mode="llm",
            manager=_FakeManager(
                [
                    '[{"label":"sticker_01","expression":"joyful leap","prompt":"kirby joyful prompt"},{"label":"sticker_02","expression":"sleepy yawn","prompt":"kirby sleepy prompt"}]',
                ]
            ),
        )
        goal = GoalRequest(prompt="ramen shop", media_type="sticker_pack", style="sticker")

        result = engine.build_sticker_prompt_set(
            goal,
            expressions=["joyful leap", "sleepy yawn"],
            character="Kirby",
            prompt_prefix="ramen shop",
            style="LINE sticker",
        )

        self.assertEqual(result["prompt_sets"][0]["prompt"], "kirby joyful prompt")
        self.assertEqual(result["prompt_sets"][1]["expression"], "sleepy yawn")
        self.assertEqual(result["prompt_mode"], "llm")

    def test_prepare_segment_uses_llm_json_when_available(self) -> None:
        engine = LLMPromptEngine(
            mode="llm",
            manager=_FakeManager(
                [
                    '{"prompt":"segment llm prompt","narration":"segment llm narration"}',
                ]
            ),
        )
        goal = GoalRequest(prompt="kirby runs", media_type="long_video", style="anime")

        result = engine.prepare_segment(
            goal,
            segment={"segment_id": "segment-1", "visual": "base visual", "narration": "base narration"},
            negative_prompt="bad anatomy",
        )

        self.assertEqual(result["prompt"], "segment llm prompt")
        self.assertEqual(result["narration"], "segment llm narration")
        self.assertEqual(result["prompt_mode"], "llm")

    def test_refine_prompt_from_review_uses_llm_json_when_available(self) -> None:
        engine = LLMPromptEngine(
            mode="llm",
            manager=_FakeManager(
                [
                    '{"prompt":"revised prompt","negative_prompt":"revised negative"}',
                ]
            ),
        )
        goal = GoalRequest(prompt="kirby runs", media_type="image", style="anime")

        result = engine.refine_prompt_from_review(goal, "old prompt", "need stronger action")

        self.assertEqual(result["prompt"], "revised prompt")
        self.assertEqual(result["negative_prompt"], "revised negative")
        self.assertEqual(result["prompt_mode"], "llm")

    def test_build_sticker_motion_prompt_uses_llm_json_when_available(self) -> None:
        engine = LLMPromptEngine(
            mode="llm",
            manager=_FakeManager(
                [
                    '{"prompt":"looping bounce with readable silhouette"}',
                ]
            ),
        )
        goal = GoalRequest(prompt="kirby cheers", media_type="animated_sticker", style="LINE sticker")

        result = engine.build_sticker_motion_prompt(goal, "base sticker prompt", "Kirby", "cheering")

        self.assertEqual(result["prompt"], "looping bounce with readable silhouette")
        self.assertEqual(result["prompt_mode"], "llm")

    def test_build_carousel_prompt_set_uses_llm_json_when_available(self) -> None:
        engine = LLMPromptEngine(
            mode="llm",
            manager=_FakeManager(
                [
                    '[{"label":"slide_01","prompt":"shot one","narration":"line one"},{"label":"slide_02","prompt":"shot two","narration":"line two"}]',
                ]
            ),
        )
        goal = GoalRequest(prompt="travel diary", media_type="carousel", style="editorial")

        result = engine.build_carousel_prompt_set(
            goal,
            segments=[
                {"segment_id": "segment-1", "visual": "visual one", "narration": "base one"},
                {"segment_id": "segment-2", "visual": "visual two", "narration": "base two"},
            ],
            style="editorial",
        )

        self.assertEqual(result["prompt_sets"][0]["prompt"], "shot one")
        self.assertEqual(result["prompt_sets"][1]["narration"], "line two")
        self.assertEqual(result["prompt_mode"], "llm")

    def test_prepare_publish_caption_uses_llm_json_when_available(self) -> None:
        engine = LLMPromptEngine(
            mode="llm",
            manager=_FakeManager(
                [
                    '{"caption":"caption body","hashtags":"#one #two","platform_captions":{"instagram":"ig caption"}}',
                ]
            ),
        )
        goal = GoalRequest(prompt="publish kirby clip", media_type="publish_review", style="social promo")

        result = engine.prepare_publish_caption(
            goal,
            prefix="launch",
            hashtags=["one", "two"],
            platforms=["instagram"],
            media_paths=["C:\\asset.png"],
            review_notes="stronger hook",
        )

        self.assertEqual(result["caption"], "caption body")
        self.assertEqual(result["hashtags"], "#one #two")
        self.assertEqual(result["platform_captions"]["instagram"], "ig caption")
        self.assertEqual(result["prompt_mode"], "llm")

    def test_review_asset_candidates_uses_llm_json_when_available(self) -> None:
        engine = LLMPromptEngine(
            mode="llm",
            manager=_FakeManager(
                [
                    '{"selected_assets":["C:\\\\clip_b.mp4"],"ranked_candidates":[{"media_path":"C:\\\\clip_b.mp4","score":96,"rationale":"best motion"},{"media_path":"C:\\\\clip_a.png","score":80,"rationale":"usable still"}],"selection_rationale":"Best matches stronger motion.","regeneration_notes":"If rerendering, push action harder."}',
                ]
            ),
        )
        goal = GoalRequest(prompt="review kirby assets", media_type="publish_review", style="social promo")

        result = engine.review_asset_candidates(
            goal,
            media_paths=["C:\\clip_a.png", "C:\\clip_b.mp4"],
            review_notes="stronger motion",
            selection_limit=1,
        )

        self.assertEqual(result["selected_assets"], ["C:\\clip_b.mp4"])
        self.assertEqual(result["ranked_candidates"][0]["score"], 96)
        self.assertEqual(result["selection_rationale"], "Best matches stronger motion.")
        self.assertEqual(result["prompt_mode"], "llm")

    def test_expand_goal_falls_back_when_manager_unavailable(self) -> None:
        engine = LLMPromptEngine(mode="llm", manager=None)
        goal = GoalRequest(prompt="kirby runs", media_type="long_video", style="anime")

        with patch.object(engine, "_manager_or_none", return_value=None):
            engine._manager_error = "ValueError: OpenRouter API key missing"
            result = engine.expand_goal(goal, "anime", [])

        self.assertEqual(result["prompt_mode"], "template")
        self.assertEqual(result["fallback_reason"], "manager_unavailable")
        self.assertIn("OpenRouter API key missing", result["manager_error"])
        self.assertEqual(result["llm_backend"]["mode"], "llm")

    def test_expand_goal_surfaces_manager_initialization_error(self) -> None:
        engine = LLMPromptEngine(mode="llm", manager=None)
        goal = GoalRequest(prompt="kirby runs", media_type="long_video", style="anime")

        with patch.object(engine, "_manager_or_none", return_value=None):
            engine._manager_error = "KeyError: gemini_api_token"
            result = engine.expand_goal(goal, "anime", [])

        self.assertEqual(result["prompt_mode"], "template")
        self.assertEqual(result["fallback_reason"], "manager_unavailable")
        self.assertIn("KeyError: gemini_api_token", result["manager_error"])

    def test_refine_prompt_falls_back_on_json_parse_failure(self) -> None:
        engine = LLMPromptEngine(mode="llm", manager=_FakeManager(["not json"]))
        goal = GoalRequest(prompt="kirby runs", media_type="image", style="anime")

        result = engine.refine_prompt_from_review(goal, "old prompt", "need stronger action")

        self.assertEqual(result["prompt_mode"], "template")
        self.assertEqual(result["fallback_reason"], "json_parse_failed")
        self.assertIn("JSONDecodeError", result["manager_error"])
        self.assertIn("old prompt", result["prompt"])

    def test_parse_json_extracts_array_from_surrounding_text(self) -> None:
        payload = LLMPromptEngine._parse_json(
            'Here is the result:\n["happy","angry","sleepy"]\nUse it well.'
        )

        self.assertEqual(payload, ["happy", "angry", "sleepy"])

    def test_sticker_expressions_fall_back_when_llm_returns_empty(self) -> None:
        engine = LLMPromptEngine(mode="llm", manager=_FakeManager([""]))
        goal = GoalRequest(prompt="Kirby sticker emotions: happy, angry, crying, sleepy", media_type="sticker_pack")

        expressions = engine.sticker_expressions(goal, goal.prompt, "Kirby", 8)

        self.assertEqual(expressions[:4], ["happy", "angry", "crying", "sleepy"])
        self.assertEqual(len(expressions), 8)

    def test_build_sticker_prompt_set_falls_back_on_json_parse_failure(self) -> None:
        engine = LLMPromptEngine(mode="llm", manager=_FakeManager(["not json"]))
        goal = GoalRequest(prompt="ramen shop", media_type="sticker_pack", style="sticker")

        result = engine.build_sticker_prompt_set(
            goal,
            expressions=["joyful leap", "sleepy yawn"],
            character="Kirby",
            prompt_prefix="ramen shop",
            style="LINE sticker",
        )

        self.assertEqual(result["prompt_mode"], "template")
        self.assertEqual(result["fallback_reason"], "json_parse_failed")
        self.assertEqual(result["prompt_count"], 2)
        self.assertEqual(result["prompt_sets"][0]["expression"], "joyful leap")



    def test_backend_info_defaults_to_openrouter(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AGENTIC_TEXT_MODEL_PROVIDER": "",
                "AGENTIC_TEXT_MODEL": "",
                "AGENTIC_VISION_MODEL_PROVIDER": "",
                "AGENTIC_VISION_MODEL": "",
                "AGENTIC_RANDOM_MODELS": "",
            },
            clear=False,
        ):
            for key in (
                "AGENTIC_TEXT_MODEL_PROVIDER",
                "AGENTIC_TEXT_MODEL",
                "AGENTIC_VISION_MODEL_PROVIDER",
                "AGENTIC_VISION_MODEL",
                "AGENTIC_RANDOM_MODELS",
            ):
                os.environ.pop(key, None)
            engine = LLMPromptEngine(mode="llm")
            backend = engine.backend_info()

        self.assertEqual(backend["text_provider"], "openrouter")
        self.assertEqual(backend["text_model"], "qwen/qwen3.6-plus:free")
        self.assertEqual(backend["vision_provider"], "openrouter")
        self.assertEqual(backend["vision_model"], "qwen/qwen3.6-plus:free")
        self.assertFalse(backend["random_models"])

    def test_manager_creation_goes_through_agentic_adapter(self) -> None:
        engine = LLMPromptEngine(mode="llm", manager=None)
        fake_manager = _FakeManager(['{"creative_brief":"ok","prompt":"ok","negative_prompt":"ok"}'])

        with patch("agentic.runtime.llm_engine.build_llm_manager", return_value=fake_manager) as build_mock:
            manager = engine._manager_or_none()

        self.assertIs(manager, fake_manager)
        build_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
