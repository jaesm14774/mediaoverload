from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentic.runtime.contracts import GoalRequest
from agentic.runtime.llm_engine import LLMPromptEngine, PromptGenerationError
from agentic.runtime.observability import RunRecorder
from agentic.runtime.video_quality import normalize_video_semantic_qa


class _FakeTextModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def chat_completion(self, messages: list[dict], **kwargs) -> str:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return self.responses.pop(0)


class _FakeManager:
    def __init__(self, responses: list[str], vision_responses: list[str] | None = None) -> None:
        self.text_model = _FakeTextModel(responses)
        self.vision_model = _FakeTextModel(list(vision_responses or []))


class LLMEngineTests(unittest.TestCase):
    def test_run_recorder_sanitizes_run_id_before_creating_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = RunRecorder(Path(temp_dir), "../escape/run")

            self.assertEqual(recorder.run_id, "escape_run")
            self.assertEqual(recorder.run_dir.parent, Path(temp_dir))

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
                    '{"generation_type":"sticker_pack","workflow_plan":{"image_workflow_name":"nova-anime-xl","video_workflow_name":"minimax_h3_lowvram_i2v","refine_workflow_name":"","transition_workflow_name":"","upscale_workflow_name":""},"count_plan":{"image_count":1,"video_count":1,"segment_count":1,"review_selection_limit":3,"sticker_expression_count":8,"images_per_prompt":2},"reason":"Sticker prompt with clean outline fits sticker_pack best."}',
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
                    "video_workflow_name": ["minimax_h3_lowvram_i2v"],
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

    def test_route_generation_strategy_schema_locks_unavailable_sticker_stages(self) -> None:
        manager = _FakeManager(
            [
                '{"generation_type":"sticker_pack","workflow_plan":{"image_workflow_name":"nova-anime-xl","video_workflow_name":"minimax_h3_lowvram_i2v","refine_workflow_name":"","transition_workflow_name":"","upscale_workflow_name":""},"count_plan":{"image_count":1,"video_count":1,"segment_count":1,"review_selection_limit":3,"sticker_expression_count":8,"images_per_prompt":2},"reason":"Sticker prompt with clean outline fits sticker_pack best."}',
            ]
        )
        engine = LLMPromptEngine(mode="llm", manager=manager)

        engine.route_generation_strategy(
            prompt="Kirby sticker emotions: happy, angry, crying, sleepy",
            character="Kirby",
            style="LINE sticker",
            generation_type_candidates=["text2img", "sticker_pack"],
            workflow_stage_candidates={
                "text2img": {"image_workflow_name": ["nova_model_plus_z_image_anime"]},
                "sticker_pack": {
                    "image_workflow_name": ["nova-anime-xl", "nova_model_plus_z_image_anime"],
                    "video_workflow_name": ["minimax_h3_lowvram_i2v"],
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
        )

        schema = manager.text_model.calls[0]["kwargs"]["response_format"]["json_schema"]["schema"]
        sticker_branch = next(
            branch for branch in schema["oneOf"] if branch["properties"]["generation_type"]["const"] == "sticker_pack"
        )

        self.assertEqual(
            sticker_branch["properties"]["workflow_plan"]["properties"]["refine_workflow_name"],
            {"const": ""},
        )
        self.assertEqual(
            sticker_branch["properties"]["workflow_plan"]["properties"]["transition_workflow_name"],
            {"const": ""},
        )
        self.assertEqual(
            sticker_branch["properties"]["workflow_plan"]["properties"]["upscale_workflow_name"],
            {"const": ""},
        )

    def test_route_generation_strategy_defaults_an_omitted_policy_count(self) -> None:
        engine = LLMPromptEngine(
            mode="llm",
            manager=_FakeManager(
                [
                    '{"generation_type":"text2image2video","workflow_plan":{"image_workflow_name":"anima_anime","video_workflow_name":"minimax_h3_lowvram_i2v","refine_workflow_name":"","transition_workflow_name":"","upscale_workflow_name":""},"count_plan":{"image_count":2},"reason":"Short character clip."}',
                ]
            ),
        )

        result = engine.route_generation_strategy(
            prompt="Kirby makes a short animated clip",
            character="Kirby",
            style="anime",
            generation_type_candidates=["text2image2video"],
            workflow_stage_candidates={
                "text2image2video": {
                    "image_workflow_name": ["anima_anime"],
                    "video_workflow_name": ["minimax_h3_lowvram_i2v"],
                }
            },
            count_policies={
                "text2image2video": {
                    "image_count": {"min": 1, "max": 4},
                    "video_count": {"min": 1, "max": 4},
                }
            },
        )

        self.assertEqual(result["count_plan"], {"image_count": 2, "video_count": 1})

    def test_route_generation_strategy_ignores_legacy_irrelevant_counts(self) -> None:
        engine = LLMPromptEngine(
            mode="llm",
            manager=_FakeManager(
                [
                    '{"generation_type":"text2img","workflow_plan":{"image_workflow_name":"nova-anime-xl","video_workflow_name":"","refine_workflow_name":"","transition_workflow_name":"","upscale_workflow_name":""},"count_plan":{"image_count":1,"video_count":0,"segment_count":0,"review_selection_limit":1,"sticker_expression_count":0,"images_per_prompt":0},"reason":"Image-only request."}',
                ]
            ),
        )

        result = engine.route_generation_strategy(
            prompt="Kirby portrait",
            character="Kirby",
            style="anime",
            generation_type_candidates=["text2img"],
            workflow_stage_candidates={
                "text2img": {"image_workflow_name": ["nova-anime-xl"]}
            },
            count_policies={
                "text2img": {
                    "image_count": {"min": 1, "max": 4},
                    "video_count": {"min": 1, "max": 1},
                    "segment_count": {"min": 1, "max": 1},
                    "review_selection_limit": {"min": 1, "max": 4},
                    "sticker_expression_count": {"min": 1, "max": 1},
                    "images_per_prompt": {"min": 1, "max": 1},
                }
            },
        )

        self.assertEqual(result["count_plan"], {"image_count": 1, "review_selection_limit": 1})

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
        goal = GoalRequest(
            prompt="publish kirby clip",
            media_type="publish_review",
            style="social promo",
            constraints={
                "visual_grounding": {
                    "status": "fail",
                    "observed_story": "Kirby crosses a storm-lit meadow.",
                    "caption_guidance": "Mention only Kirby and the meadow.",
                    "issues": ["The news anchor is not visible."],
                }
            },
        )

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
        self.assertIn("Mention only Kirby and the meadow.", engine._manager.text_model.calls[0]["messages"][1]["content"])
        self.assertEqual(engine._manager.text_model.calls[0]["kwargs"]["max_retries"], 2)
        self.assertIsNone(engine._manager.text_model.calls[0]["kwargs"].get("max_models_per_call"))
        self.assertEqual(engine._manager.text_model.calls[0]["kwargs"]["request_timeout"], 60.0)

    def test_prepare_publish_caption_removes_internal_labels_and_keeps_long_form_post(self) -> None:
        engine = LLMPromptEngine(
            mode="llm",
            manager=_FakeManager(
                [
                    '{"caption":"Caption: The purple orb flickers above the grass.\\n\\n1️⃣ Kirby faces the energy.\\n2️⃣ A golden shard changes the outcome.\\n\\nWhich moment stayed with you?\\n\\n#kirby #mediaoverload","hashtags":"#kirby #mediaoverload","platform_captions":{"facebook":"Caption: The purple orb flickers above the grass.\\n\\nWhich moment stayed with you?"}}',
                ]
            ),
        )

        result = engine.prepare_publish_caption(
            GoalRequest(prompt="publish Kirby story", media_type="publish_review", style="social promo"),
            prefix="",
            hashtags=["kirby", "mediaoverload"],
            platforms=["facebook"],
            media_paths=["C:\\kirby.mp4"],
        )

        self.assertTrue(result["caption"].startswith("The purple orb flickers"))
        self.assertIn("Kirby faces the energy.", result["caption"])
        self.assertNotIn("Caption:", result["caption"])
        self.assertNotIn("Hashtags:", result["caption"])
        self.assertNotIn("Caption:", result["platform_captions"]["facebook"])

    def test_publish_article_format_retries_a_short_draft(self) -> None:
        engine = LLMPromptEngine(
            mode="llm",
            manager=_FakeManager(
                [
                    '{"caption":"A short sentence.","hashtags":"#kirby #mediaoverload","platform_captions":{"facebook":"A short sentence."}}',
                    '{"caption":"A strong hook opens the story.\\n\\nThe visible conflict gives the moment meaning.\\n\\n1. Kirby faces the orb.\\n2. The shard changes the outcome.\\n3. The crystal shows the payoff.\\n\\nWhich beat would you remember? Save this idea for later.","hashtags":"#kirby #mediaoverload","platform_captions":{"facebook":"A strong hook opens the story.\\n\\nThe visible conflict gives the moment meaning.\\n\\nWhich beat would you remember? Save this idea for later."}}',
                ]
            ),
        )

        result = engine.prepare_publish_caption(
            GoalRequest(
                prompt="publish Kirby story",
                media_type="publish_review",
                style="social promo",
                constraints={"social_post_format": True},
            ),
            prefix="",
            hashtags=["kirby", "mediaoverload"],
            platforms=["facebook"],
            media_paths=["C:\\kirby.mp4"],
        )

        self.assertIn("Which beat would you remember?", result["caption"])
        self.assertIn("Which beat would you remember?", result["platform_captions"]["facebook"])
        self.assertEqual(len(engine._manager.text_model.calls), 2)
        self.assertNotIn("Caption:", result["caption"])
        self.assertNotIn("Hashtags:", result["caption"])

    def test_prepare_publish_caption_fails_when_caption_provider_fails(self) -> None:
        engine = LLMPromptEngine(
            mode="llm",
            manager=_FakeManager(["not json"]),
        )
        goal = GoalRequest(
            prompt="publish Kirby image",
            media_type="publish_review",
            style="anime",
        )

        with self.assertRaises(PromptGenerationError):
            engine.prepare_publish_caption(
                goal,
                prefix="",
                hashtags=["kirby"],
                platforms=["instagram", "facebook"],
                media_paths=["C:\\kirby.png"],
            )

    def test_prepare_publish_caption_rejects_placeholder_caption(self) -> None:
        engine = LLMPromptEngine(
            mode="llm",
            manager=_FakeManager(
                [
                    '{"caption":"None","hashtags":"#kirby #mediaoverload","platform_captions":{"facebook":"None"}}',
                ]
            ),
        )

        with self.assertRaises(PromptGenerationError):
            engine.prepare_publish_caption(
                GoalRequest(prompt="publish Kirby baseball clip", media_type="publish_review", style="anime"),
                prefix="",
                hashtags=["kirby", "mediaoverload"],
                platforms=["facebook"],
                media_paths=["C:\\kirby.mp4"],
            )

    def test_prepare_publish_caption_uses_attached_visual_evidence(self) -> None:
        engine = LLMPromptEngine(
            mode="llm",
            manager=_FakeManager(
                [],
                vision_responses=[
                    '{"caption":"Kirby stands beside a glowing blue crystal.","hashtags":"#kirby #mediaoverload","platform_captions":{"instagram_graph":"Kirby stands beside a glowing blue crystal."}}',
                ],
            ),
        )
        result = engine.prepare_publish_caption(
            GoalRequest(prompt="The prompt claims a city battle.", media_type="publish_review", style="anime"),
            prefix="",
            hashtags=["kirby", "mediaoverload"],
            platforms=["instagram_graph"],
            media_paths=["D:\\kirby.png"],
            visual_paths=["D:\\kirby.png"],
        )

        self.assertEqual(result["caption"], "Kirby stands beside a glowing blue crystal.")
        self.assertEqual(engine._manager.vision_model.calls[0]["kwargs"]["images"], ["D:\\kirby.png"])
        self.assertEqual(engine._manager.text_model.calls, [])

    def test_prepare_publish_caption_closes_platform_contract_and_normalizes_tags(self) -> None:
        engine = LLMPromptEngine(
            mode="llm",
            manager=_FakeManager(
                [
                    '{"caption":"caption body","hashtags":"#one","platform_captions":{"instagram":"ig caption","youtube":"yt caption","youtube_title":"leaked metadata","youtube_tags":"leaked tags"},"youtube_title":"Kirby Clip","youtube_tags":"kirby, #anime, kirby"}',
                ]
            ),
        )
        goal = GoalRequest(prompt="publish kirby clip", media_type="publish_review", style="social promo")

        result = engine.prepare_publish_caption(
            goal,
            prefix="",
            hashtags=[],
            platforms=["instagram", "youtube"],
            media_paths=["C:\\clip.mp4"],
            review_notes="",
        )

        self.assertEqual(set(result["platform_captions"]), {"instagram", "youtube"})
        self.assertEqual(result["youtube_title"], "Kirby Clip")
        self.assertEqual(result["youtube_tags"], ["kirby", "anime"])

    def test_normalize_hashtags_blocks_internal_project_tag(self) -> None:
        result = LLMPromptEngine._normalize_hashtag_text(
            "#mediaoverload #RainyNeon #kirby",
            required_hashtags=["mediaoverload", "kirby"],
        )

        self.assertEqual(result, "#kirby #RainyNeon")
        self.assertNotIn("mediaoverload", result.casefold())

    def test_prepare_publish_caption_reuses_main_article_when_platform_caption_is_missing(self) -> None:
        engine = LLMPromptEngine(
            mode="llm",
            manager=_FakeManager(
                [
                    '{"caption":"caption body","hashtags":"#kirby","platform_captions":{"instagram":"ig caption"}}',
                ]
            ),
        )

        result = engine.prepare_publish_caption(
            GoalRequest(prompt="publish Kirby", media_type="publish_review", style="anime"),
            prefix="",
            hashtags=["kirby"],
            platforms=["instagram", "facebook"],
        )

        self.assertEqual(result["platform_captions"]["instagram"], "ig caption")
        self.assertEqual(result["platform_captions"]["facebook"], "caption body")

    def test_caption_result_and_recorder_include_concrete_model_id(self) -> None:
        manager = _FakeManager(
            [
                '{"caption":"short caption","hashtags":"#kirby","platform_captions":{"instagram_graph":"short caption"}}',
            ]
        )
        manager.text_model.last_success_model = "test/caption-model"
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = RunRecorder(Path(temp_dir), "model-id")
            engine = LLMPromptEngine(mode="llm", manager=manager, recorder=recorder)
            result = engine.prepare_publish_caption(
                GoalRequest(prompt="publish Kirby", media_type="publish_review", style="anime"),
                prefix="",
                hashtags=["kirby"],
                platforms=["instagram_graph"],
                media_paths=["D:\\kirby.png"],
            )
            record = json.loads(next((Path(temp_dir) / "model-id" / "llm").glob("*.json")).read_text(encoding="utf-8"))

        self.assertEqual(result["llm_model"], "test/caption-model")
        self.assertEqual(record["model_id"], "test/caption-model")

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

    def test_review_asset_candidates_vision_prefilters_to_top_ten(self) -> None:
        temp_dir = Path(__file__).resolve().parents[1] / ".tmp-tests" / "vision-prefilter"
        temp_dir.mkdir(parents=True, exist_ok=True)
        media_paths = []
        created_paths: list[Path] = []
        for index in range(12):
            path = temp_dir / f"asset_{index:02d}.png"
            path.write_bytes(b"fake")
            created_paths.append(path)
            media_paths.append(str(path))

        try:
            vision_responses = [
                f'{{"score": {100 - index}, "rationale": "match {index}"}}'
                for index in range(12)
            ]
            engine = LLMPromptEngine(
                mode="llm",
                manager=_FakeManager(
                    [
                        '{"selected_assets":[],"ranked_candidates":[],"selection_rationale":"Fallback ranking used.","regeneration_notes":"None."}',
                    ],
                    vision_responses=vision_responses,
                ),
            )
            goal = GoalRequest(prompt="review kirby assets", media_type="publish_review", style="social promo")

            result = engine.review_asset_candidates(
                goal,
                media_paths=media_paths,
                review_notes="match the prompt closely",
                selection_limit=10,
            )
        finally:
            for path in created_paths:
                path.unlink(missing_ok=True)

        self.assertEqual(len(result["selected_assets"]), 10)
        self.assertNotIn(media_paths[10], result["selected_assets"])
        self.assertNotIn(media_paths[11], result["selected_assets"])
        self.assertEqual(result["selected_assets"][0], media_paths[0])

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

    def test_chat_json_repairs_invalid_json_with_stricter_prompt(self) -> None:
        fake_manager = _FakeManager(["not json", '```json\n{"ok":true}\n```'])

        result = LLMPromptEngine._chat_json(
            fake_manager,
            "You are a story planner.",
            "Return the plan.",
            "plan",
            {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
        )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(len(fake_manager.text_model.calls), 2)
        self.assertIn("JSON REPAIR MODE", fake_manager.text_model.calls[1]["messages"][0]["content"])
        self.assertNotIn("response_format", fake_manager.text_model.calls[1]["kwargs"])

    def test_chat_json_uses_configurable_request_timeout(self) -> None:
        fake_manager = _FakeManager(['{"ok":true}'])

        with patch.dict(os.environ, {"AGENTIC_LLM_REQUEST_TIMEOUT_SECONDS": "7.5"}):
            result = LLMPromptEngine._chat_json(
                fake_manager,
                "You are a story planner.",
                "Return the plan.",
                "plan",
                {"type": "object", "properties": {"ok": {"type": "boolean"}}},
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(fake_manager.text_model.calls[0]["kwargs"]["request_timeout"], 7.5)

    def test_chat_json_records_each_request_and_response_for_debugging(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = RunRecorder(Path(temp_dir), "run-debug")
            manager = _FakeManager(["not json", '{"ok":true}'])
            engine = LLMPromptEngine(mode="llm", manager=manager, recorder=recorder)
            engine._require_manager()

            result = LLMPromptEngine._chat_json(
                manager,
                "You are a story planner.",
                "Return the plan.",
                "plan",
                {"type": "object", "properties": {"ok": {"type": "boolean"}}},
            )

            self.assertEqual(result, {"ok": True})
            call_files = sorted((Path(temp_dir) / "run-debug" / "llm").glob("*.json"))
            self.assertEqual(len(call_files), 2)
            first = json.loads(call_files[0].read_text(encoding="utf-8"))
            second = json.loads(call_files[1].read_text(encoding="utf-8"))
            self.assertEqual(first["status"], "failed")
            self.assertEqual(second["status"], "success")
            self.assertIn("You are a story planner.", second["messages"][0]["content"])
            self.assertEqual(second["parsed_payload"], {"ok": True})

    def test_engine_chat_wrapper_records_calls_for_secondary_llm_flows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = RunRecorder(Path(temp_dir), "run-secondary")
            manager = _FakeManager(['{"ok":true}'])
            engine = LLMPromptEngine(mode="llm", manager=manager, recorder=recorder)

            result = engine._chat_json_with_recorder(
                manager,
                "You are a caption planner.",
                "Return the plan.",
                schema_name="publish_caption",
                schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
            )

            self.assertEqual(result, {"ok": True})
            call_files = sorted((Path(temp_dir) / "run-secondary" / "llm").glob("*.json"))
            self.assertEqual(len(call_files), 1)
            self.assertEqual(json.loads(call_files[0].read_text(encoding="utf-8"))["status"], "success")

    def test_chat_json_uses_array_contract_for_array_schema(self) -> None:
        fake_manager = _FakeManager(['[{"ok":true}]'])

        result = LLMPromptEngine._chat_json(
            fake_manager,
            "You are a story planner.",
            "Return the list.",
            "items",
            {"type": "array", "items": {"type": "object"}},
        )

        self.assertEqual(result, [{"ok": True}])
        self.assertIn("JSON array", fake_manager.text_model.calls[0]["messages"][0]["content"])

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
                "AGENTIC_RANDOM_MODELS": "false",
                "AGENTIC_TEXT_FALLBACK_PROVIDER": "",
                "AGENTIC_TEXT_FALLBACK_MODEL": "",
                "AGENTIC_OPENROUTER_TEXT_MODEL_STRATEGY": "",
                "AGENTIC_OPENROUTER_VISION_MODEL_STRATEGY": "",
            },
            clear=False,
        ):
            engine = LLMPromptEngine(mode="llm")
            backend = engine.backend_info()

        self.assertEqual(backend["text_provider"], "openrouter")
        self.assertEqual(backend["text_model"], "free_pool")
        self.assertEqual(backend["text_model_raw"], "")
        self.assertTrue(backend["openrouter_text_pool_mode"])
        self.assertEqual(backend["vision_provider"], "openrouter")
        self.assertEqual(backend["vision_model"], "free_pool")
        self.assertEqual(backend["vision_model_raw"], "")
        self.assertTrue(backend["openrouter_vision_pool_mode"])
        self.assertFalse(backend["random_models"])
        self.assertEqual(backend.get("text_fallback_provider"), "")

    def test_backend_info_openrouter_free_pool_when_text_model_empty(self) -> None:
        with patch.dict(
            os.environ,
            {
                "AGENTIC_TEXT_MODEL_PROVIDER": "openrouter",
                "AGENTIC_TEXT_MODEL": "",
                "AGENTIC_VISION_MODEL_PROVIDER": "openrouter",
                "AGENTIC_VISION_MODEL": "google/gemma-4-31b-it:free",
                "AGENTIC_OPENROUTER_TEXT_MODEL_STRATEGY": "free_pool",
            },
            clear=False,
        ):
            engine = LLMPromptEngine(mode="llm")
            backend = engine.backend_info()
        self.assertTrue(backend["openrouter_text_pool_mode"])
        self.assertEqual(backend["text_model"], "free_pool")

    def test_manager_creation_goes_through_agentic_adapter(self) -> None:
        engine = LLMPromptEngine(mode="llm", manager=None)
        fake_manager = _FakeManager(['{"creative_brief":"ok","prompt":"ok","negative_prompt":"ok"}'])

        with patch("agentic.runtime.llm_engine.build_llm_manager", return_value=fake_manager) as build_mock:
            manager = engine._manager_or_none()

        self.assertIs(manager, fake_manager)
        build_mock.assert_called_once()

    def test_video_semantic_qa_uses_contact_sheet_and_normalizes_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            contact_sheet = Path(temp_dir) / "contact_sheet.jpg"
            contact_sheet.write_bytes(b"jpg")
            engine = LLMPromptEngine(
                mode="llm",
                manager=_FakeManager(
                    [],
                    vision_responses=[
                        json.dumps(
                            {
                                "status": "pass",
                                "score": 92,
                                "checks": {
                                    "protagonist_clear": True,
                                    "primary_action_visible": True,
                                    "news_anchor_visible": True,
                                    "progression_visible": True,
                                    "unwanted_extra_characters": False,
                                },
                                "observed_story": "Kirby carries the glowing seed through the storm.",
                                "issues": [],
                                "caption_guidance": "Claim only Kirby, the seed, and the storm-lit meadow.",
                            }
                        )
                    ],
                ),
            )

            result = engine.evaluate_video_contact_sheet(
                contact_sheet_path=str(contact_sheet),
                character="Kirby",
                story_spine={"hook": "seed falls", "payoff": "meadow restored"},
                native_shots=[{"action": "Kirby carries the seed"}],
                news_context={"title": "storm warning"},
                rendered_prompt="one continuous Kirby story",
            )

            self.assertTrue(result["passed"])
            self.assertEqual(result["status"], "pass")
            vision_call = engine._manager.vision_model.calls[0]
            self.assertEqual(vision_call["kwargs"]["images"], [str(contact_sheet)])

    def test_video_semantic_qa_does_not_pass_without_vision_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            contact_sheet = Path(temp_dir) / "contact_sheet.jpg"
            contact_sheet.write_bytes(b"jpg")
            result = LLMPromptEngine(mode="template").evaluate_video_contact_sheet(
                contact_sheet_path=str(contact_sheet),
                character="Kirby",
                story_spine={},
                native_shots=[],
                news_context={},
                rendered_prompt="story",
            )

            self.assertIsNone(result["passed"])
            self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["reason"], "vision model unavailable")

    def test_video_semantic_qa_flags_unwanted_characters_as_hard_failure(self) -> None:
        result = normalize_video_semantic_qa(
            {
                "status": "pass",
                "score": 100,
                "checks": {
                    "protagonist_clear": True,
                    "primary_action_visible": True,
                    "news_anchor_visible": True,
                    "progression_visible": True,
                    "unwanted_extra_characters": True,
                },
                "issues": [],
            },
            contact_sheet_path="contact_sheet.jpg",
            prompt_mode="llm",
            llm_backend={},
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "fail")

    def test_video_semantic_qa_maps_concrete_anchor_when_model_uses_legacy_news_anchor_false(self) -> None:
        result = normalize_video_semantic_qa(
            {
                "status": "fail",
                "score": 98,
                "checks": {
                    "protagonist_clear": True,
                    "primary_action_visible": True,
                    "news_anchor_visible": False,
                    "progression_visible": True,
                    "unwanted_extra_characters": False,
                },
                "observed_story": "Kirby protects a glowing golden orb from shadowy tendrils and restores it.",
                "issues": [],
                "caption_guidance": "The orb is visible.",
            },
            contact_sheet_path="contact_sheet.jpg",
            prompt_mode="llm",
            llm_backend={},
            news_anchor_terms=["glowing golden orb", "shadowy tendrils attempting to steal the orb"],
        )

        self.assertTrue(result["passed"])
        self.assertTrue(result["checks"]["news_anchor_visible"])

    def test_video_semantic_qa_blocks_multi_panel_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            contact_sheet = Path(temp_dir) / "contact_sheet.jpg"
            contact_sheet.write_bytes(b"jpg")
            engine = LLMPromptEngine(
                mode="llm",
                manager=_FakeManager(
                    [],
                    vision_responses=[
                        json.dumps(
                            {
                                "status": "fail",
                                "score": 88,
                                "checks": {
                                    "protagonist_clear": True,
                                    "primary_action_visible": True,
                                    "news_anchor_visible": True,
                                    "progression_visible": True,
                                    "unwanted_extra_characters": False,
                                },
                                "observed_story": "Kirby advances through a baseball story in a stylized panel layout.",
                                "issues": ["stylized logo and split-panel composition"],
                                "caption_guidance": "Describe only the visible Kirby baseball action.",
                            }
                        )
                    ],
                ),
            )

            result = engine.evaluate_video_contact_sheet(
                contact_sheet_path=str(contact_sheet),
                character="Kirby",
                story_spine={"objective": "reach the baseball field"},
                native_shots=[{"action": "Kirby runs"}],
                news_context={"title": "baseball reunion"},
                rendered_prompt="one continuous Kirby baseball story",
            )

            self.assertEqual(result["status"], "fail")
            self.assertFalse(result["passed"])
            self.assertFalse(result["advisory_only"])


if __name__ == "__main__":
    unittest.main()
