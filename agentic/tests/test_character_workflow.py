from __future__ import annotations

import os
import json
import random
import logging
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from agentic.app.character_workflow import (
    _extract_failure_details,
    _extract_publish_visual_grounding,
    _collect_count_policies,
    _route_generation_from_character_config,
    _select_fresh_news,
    _resolve_publish_prompt,
    build_goal_payload_from_character_config,
    collect_media_paths_from_run_result,
    load_character_config,
    run_character_workflow,
)
from agentic.tools.context_services import NewsContextService, NewsSelection
from agentic.app.main import _allow_runtime_output_for_visual_evidence


class CharacterWorkflowRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.kirby_config = cls.repo_root / "configs" / "characters" / "kirby.yaml"

    def test_runtime_output_root_is_allowed_for_vision_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            os.environ,
            {"AGENTIC_ALLOWED_IMAGE_ROOTS": ""},
            clear=False,
        ):
            output_root = Path(temp_dir) / "generated"
            _allow_runtime_output_for_visual_evidence(output_root)
            self.assertIn(str(output_root.resolve()), os.environ["AGENTIC_ALLOWED_IMAGE_ROOTS"])

    def test_duration_policy_selects_single_action_or_native_story(self) -> None:
        short = build_goal_payload_from_character_config(
            self.repo_root,
            self.kirby_config,
            prompt="Kirby swats one glowing orb into a target",
            duration_seconds=5,
            publish_after_generate=False,
        )
        self.assertEqual(short["source_generation_type"], "text2image2video")
        self.assertEqual(short["media_type"], "text2img2video")
        self.assertEqual(short["duration_seconds"], 5)
        self.assertEqual(short["constraints"]["duration_profile"], "single_action")
        self.assertEqual(short["constraints"]["duration_override_seconds"], 5)

        native = build_goal_payload_from_character_config(
            self.repo_root,
            self.kirby_config,
            prompt="Kirby protects one glowing orb from a sudden gust",
            duration_seconds=15,
            publish_after_generate=False,
        )
        self.assertEqual(native["source_generation_type"], "native_h3_story")
        self.assertEqual(native["duration_seconds"], 15)
        self.assertTrue(native["constraints"]["native_h3_storyboard_path"].endswith("kirby_native_15s.yaml"))
        self.assertEqual(native["constraints"]["duration_profile"], "compact_story")

    def test_weighted_route_selects_strategy_before_content_prompt(self) -> None:
        payload = build_goal_payload_from_character_config(
            self.repo_root,
            self.kirby_config,
            prompt="Kirby protects a glowing orb in a clear three-beat story",
            rng=random.Random(0),
            publish_after_generate=False,
        )

        self.assertEqual(payload["source_generation_type"], "sticker_pack")
        self.assertEqual(payload["constraints"]["routing_selection_source"], "weighted_random")
        self.assertEqual(payload["constraints"]["routing_prompt_mode"], "weighted_random")
        self.assertEqual(payload["prompt"], "Kirby protects a glowing orb in a clear three-beat story")

    def test_weighted_route_can_select_sticker_without_llm_strategy_routing(self) -> None:
        config = load_character_config(self.kirby_config)
        config["generation"]["generation_type_weights"] = {"sticker_pack": 1}

        with patch("agentic.app.character_workflow.LLMPromptEngine.route_generation_strategy") as route:
            result = _route_generation_from_character_config(
                self.repo_root,
                config,
                character_name="Kirby",
                style="polished 2D anime",
                prompt="Kirby chat sticker reactions",
                preferred_generation_type=None,
                requested_duration_seconds=None,
                rng=random.Random(0),
            )

        route.assert_not_called()
        self.assertEqual(result["generation_type"], "sticker_pack")
        self.assertEqual(result["selection_source"], "weighted_random")

    def test_native_h3_rejects_unsupported_duration_override(self) -> None:
        with self.assertRaisesRegex(ValueError, "supports duration_seconds=15"):
            build_goal_payload_from_character_config(
                self.repo_root,
                self.kirby_config,
                prompt="Kirby swats one glowing orb into a target",
                preferred_generation_type="native_h3_story",
                duration_seconds=5,
                publish_after_generate=False,
            )

    def test_extract_failure_details_promotes_node_log(self) -> None:
        details = _extract_failure_details(
            {
                "status": "failed",
                "records": [
                    {
                        "node_id": "native-story-prompt",
                        "skill_name": "longvideo.prepare_native_h3_story",
                        "status": "failed",
                        "logs": [
                            "PromptGenerationError: Native H3 story contains forbidden readable-text visual cues: reads"
                        ],
                    }
                ],
            }
        )

        self.assertEqual(
            details["failure_reason"],
            "PromptGenerationError: Native H3 story contains forbidden readable-text visual cues: reads",
        )
        self.assertEqual(details["failure_node"], "native-story-prompt")
        self.assertEqual(details["failure_skill"], "longvideo.prepare_native_h3_story")

    def test_count_policies_ignore_stages_not_used_by_text2img(self) -> None:
        policies = _collect_count_policies(
            {
                "count_policies": {
                    "text2img": {
                        "image_count": {"min": 1, "max": 4},
                        "video_count": {"min": 1, "max": 1},
                        "segment_count": {"min": 1, "max": 1},
                        "review_selection_limit": {"min": 1, "max": 4},
                        "sticker_expression_count": {"min": 1, "max": 1},
                        "images_per_prompt": {"min": 1, "max": 1},
                    }
                }
            },
            ["text2img"],
            workflow_stage_candidates={
                "text2img": {"image_workflow_name": ["nova-anime-xl"]}
            },
        )

        self.assertEqual(set(policies["text2img"]), {"image_count", "review_selection_limit"})

    def test_run_manifest_exposes_failure_reason_at_top_level(self) -> None:
        payload = {
            "prompt": "Kirby story",
            "duration_seconds": 15,
            "style": "polished 2D anime",
            "source_generation_type": "native_h3_story",
            "media_type": "native_h3_story",
            "selected_workflow_name": "native-h3-test",
            "character_name": "Kirby",
            "resolved_output_dir": "output",
            "constraints": {},
            "routing_summary": {},
            "routing": {},
            "character_config_summary": {},
        }

        class FakePlan:
            workflow_name = "native-h3-test"
            nodes: list[object] = []

            def to_dict(self) -> dict[str, object]:
                return {"workflow_name": self.workflow_name, "nodes": []}

        class FakePlanner:
            def create_goal(self, **_kwargs: object) -> object:
                return object()

            def build_plan(self, _goal: object) -> FakePlan:
                return FakePlan()

        class FakeRunner:
            def run(self, _plan: FakePlan) -> SimpleNamespace:
                return SimpleNamespace(
                    status="failed",
                    to_dict=lambda: {
                        "workflow_name": "native-h3-test",
                        "status": "failed",
                        "records": [
                            {
                                "node_id": "native-story-prompt",
                                "skill_name": "longvideo.prepare_native_h3_story",
                                "status": "failed",
                                "logs": ["PromptGenerationError: hook motion missing"],
                            }
                        ],
                        "state": {},
                    },
                )

        class FakeRecorder:
            def __init__(self, *_args: object, **_kwargs: object) -> None:
                self.finalized: dict[str, object] | None = None

            def finalize(self, result: dict[str, object]) -> None:
                self.finalized = result

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch(
                "agentic.app.character_workflow.build_goal_payload_from_character_config",
                return_value=payload,
            ), patch(
                "agentic.app.character_workflow.RunRecorder",
                FakeRecorder,
            ), patch(
                "agentic.app.character_workflow.create_run_logger",
                return_value=(logging.getLogger("test-character-workflow"), root / "lifecycle.log"),
            ), patch(
                "agentic.app.character_workflow.build_runtime",
                return_value=(FakePlanner(), FakeRunner(), SimpleNamespace(as_serializable=lambda: {})),
            ):
                result = run_character_workflow(root, root / "kirby.yaml", publish_after_generate=False)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failure_reason"], "PromptGenerationError: hook motion missing")
        self.assertEqual(result["failure_node"], "native-story-prompt")
        self.assertEqual(result["failure_skill"], "longvideo.prepare_native_h3_story")

    def test_build_goal_payload_uses_llm_routed_strategy_and_workflow(self) -> None:
        with patch(
            "agentic.app.character_workflow.LLMPromptEngine.route_generation_strategy",
            return_value={
                "generation_type": "sticker_pack",
                "workflow_plan": {
                    "image_workflow_name": "nova-anime-xl",
                    "video_workflow_name": "minimax_h3_lowvram_i2v",
                    "refine_workflow_name": "",
                    "transition_workflow_name": "",
                    "upscale_workflow_name": "",
                },
                "count_plan": {
                    "image_count": 1,
                    "video_count": 2,
                    "segment_count": 1,
                    "review_selection_limit": 5,
                    "sticker_expression_count": 8,
                    "images_per_prompt": 2,
                },
                "reason": "Sticker wording and clean outline fit sticker_pack best.",
                "prompt_mode": "llm",
            },
        ):
            payload = build_goal_payload_from_character_config(
                self.repo_root,
                self.kirby_config,
                prompt="Kirby sticker emotions: happy, angry, crying, sleepy",
            )

        self.assertEqual(payload["source_generation_type"], "sticker_pack")
        self.assertEqual(payload["media_type"], "sticker_pack")
        self.assertEqual(payload["selected_workflow_name"], "nova-anime-xl")
        self.assertEqual(payload["constraints"]["workflow_name"], "nova-anime-xl")
        self.assertEqual(payload["constraints"]["video_workflow_name"], "minimax_h3_lowvram_i2v")
        self.assertEqual(payload["constraints"]["video_count"], 2)
        self.assertEqual(payload["constraints"]["selection_limit"], 5)
        self.assertEqual(payload["constraints"]["images_per_prompt"], 2)
        self.assertEqual(payload["constraints"]["routing_prompt_mode"], "llm")
        self.assertEqual(payload["routing_summary"]["strategy"], "sticker_pack")
        self.assertEqual(payload["routing_summary"]["primary_workflow"], "nova-anime-xl")
        self.assertEqual(payload["routing_summary"]["workflow_plan"]["video_workflow_name"], "minimax_h3_lowvram_i2v")
        self.assertEqual(payload["routing_summary"]["count_plan"]["review_selection_limit"], 5)

    def test_publish_prompt_inherits_rendered_native_h3_story(self) -> None:
        prompt, source = _resolve_publish_prompt(
            {
                "state": {
                    "node_outputs": {
                        "native-story-prompt": {
                            "prompt": "Kirby reaches the blue orb, survives the energy surge, and restores balance."
                        }
                    }
                }
            },
            fallback_prompt="Original news brief",
        )

        self.assertEqual(
            prompt,
            "Kirby reaches the blue orb, survives the energy surge, and restores balance.",
        )
        self.assertEqual(source, "native_h3_story")

    def test_publish_visual_grounding_keeps_caption_claims_inside_video_evidence(self) -> None:
        grounding = _extract_publish_visual_grounding(
            {
                "state": {
                    "node_outputs": {
                        "native-h3-qa": {
                            "semantic_qa": {
                                "enabled": True,
                                "status": "fail",
                                "passed": False,
                                "observed_story": "Kirby crosses a storm-lit meadow.",
                                "caption_guidance": "Mention only Kirby, the meadow, and the storm light.",
                                "issues": ["The news anchor is not visible."],
                                "checks": {"news_anchor_visible": False},
                            }
                        }
                    }
                }
            }
        )

        self.assertEqual(grounding["status"], "fail")
        self.assertFalse(grounding["passed"])
        self.assertEqual(grounding["issues"], ["The news anchor is not visible."])

    def test_publish_prompt_falls_back_when_generation_has_no_native_story(self) -> None:
        prompt, source = _resolve_publish_prompt({"state": {"node_outputs": {}}}, fallback_prompt="News brief")

        self.assertEqual(prompt, "News brief")
        self.assertEqual(source, "goal_prompt")

    def test_publish_prompt_uses_compact_story_and_news_context(self) -> None:
        prompt, source = _resolve_publish_prompt(
            {
                "state": {
                    "node_outputs": {
                        "native-story-prompt": {
                            "news_context": {"title": "AI companion robot arrives"},
                            "generated_storyboard": {
                                "name": "Kirby and the Robot Seed",
                                "story_spine": {
                                    "premise": "A robot reaches for Kirby's seed.",
                                    "objective": "Kirby must keep the seed safe.",
                                    "resolution": "The seed is planted safely.",
                                },
                                "news_trace": {
                                    "source_title": "AI companion robot arrives",
                                    "visual_translation": "a gentle AI companion robot follows the seed",
                                    "visual_anchors": ["AI companion robot", "glowing seed"],
                                },
                            },
                            "prompt": "the full production prompt should not be used for captions",
                        }
                    }
                }
            },
            fallback_prompt="fallback",
        )

        self.assertEqual(source, "native_h3_story")
        self.assertIn("News headline: AI companion robot arrives", prompt)
        self.assertIn("News visual translation:", prompt)
        self.assertNotIn("the full production prompt should not be used", prompt)

    def test_publish_review_collects_only_packaged_final_video(self) -> None:
        video_path = r"C:\runs\Kirby_H3.mp4"
        result = collect_media_paths_from_run_result(
            {
                "state": {
                    "node_outputs": {
                        "native-opening-keyframe": {
                            "saved_files": [r"C:\frames\opening_1.png", r"C:\frames\opening_2.png"]
                        },
                        "native-ending-keyframe": {"saved_files": [r"C:\frames\ending.png"]},
                        "native-h3-render": {
                            "saved_files": [video_path],
                            "gif_path": r"C:\preview\preview.gif",
                        },
                        "native-h3-package": {
                            "video_path": video_path,
                            "saved_files": [video_path, r"C:\frames\ending.png"],
                        },
                    }
                }
            }
        )

        self.assertEqual(result, [video_path])

    def test_publish_review_collects_final_images_from_image_generation(self) -> None:
        image_paths = [r"C:\runs\Kirby_1.png", r"C:\runs\Kirby_2.png"]
        result = collect_media_paths_from_run_result(
            {
                "state": {
                    "node_outputs": {
                        "render-image": {"saved_files": image_paths},
                    }
                }
            }
        )

        self.assertEqual(result, image_paths)

    def test_publish_review_excludes_intermediate_frames_and_gif_preview_when_video_exists(self) -> None:
        video_path = r"C:\runs\Kirby_H3.mp4"
        result = collect_media_paths_from_run_result(
            {
                "state": {
                    "node_outputs": {
                        "render-image": {"saved_files": [r"C:\frames\opening.png"]},
                        "animate-video": {"video_path": video_path, "saved_files": [video_path]},
                        "gif-preview": {"gif_path": r"C:\preview\preview.gif", "saved_files": [r"C:\preview\preview.gif"]},
                    }
                }
            }
        )

        self.assertEqual(result, [video_path])

    def test_build_goal_payload_raises_on_invalid_routed_workflow(self) -> None:
        with patch(
            "agentic.app.character_workflow.LLMPromptEngine.route_generation_strategy",
            side_effect=ValueError("invalid route"),
        ):
            with self.assertRaises(ValueError):
                build_goal_payload_from_character_config(
                    self.repo_root,
                    self.kirby_config,
                    prompt="Kirby sticker emotions: happy, angry, crying, sleepy",
                )

    def test_build_goal_payload_generates_prompt_from_news_when_prompt_missing(self) -> None:
        with patch(
            "agentic.app.character_workflow.LLMPromptEngine.route_generation_strategy",
            return_value={
                "generation_type": "text2video",
                "workflow_plan": {
                    "image_workflow_name": "nova-anime-xl",
                    "video_workflow_name": "minimax_h3_lowvram_t2v",
                    "refine_workflow_name": "",
                    "transition_workflow_name": "",
                    "upscale_workflow_name": "",
                },
                "count_plan": {
                    "image_count": 2,
                    "video_count": 1,
                    "segment_count": 1,
                    "review_selection_limit": 3,
                    "sticker_expression_count": 1,
                    "images_per_prompt": 1,
                },
                "reason": "Short animated scene fits text2video.",
                "prompt_mode": "llm",
            },
        ), patch(
            "agentic.app.character_workflow.NewsContextService.get_random_news",
            return_value=type(
                "_News",
                (),
                {"to_dict": lambda self: {"title": "Taipei panda steals zongzi", "keyword": "panda"}}  # noqa: ARG005
            )(),
        ), patch(
            "agentic.app.character_workflow.LLMPromptEngine.generate_autonomous_scene_prompt",
            return_value={
                "prompt": "Kirby turns a panda zongzi headline into a playful rainy-night chase scene",
                "source": "autonomous_llm",
                "prompt_mode": "llm",
                "creative_seed": "Taipei panda steals zongzi",
                "news_context": {"title": "Taipei panda steals zongzi", "keyword": "panda"},
            },
        ):
            payload = build_goal_payload_from_character_config(
                self.repo_root,
                self.kirby_config,
                prompt="",
            )

        self.assertEqual(
            payload["prompt"],
            "Kirby turns a panda zongzi headline into a playful rainy-night chase scene",
        )
        self.assertEqual(payload["constraints"]["prompt_source"], "autonomous_llm")
        self.assertEqual(payload["constraints"]["creative_seed"], "Taipei panda steals zongzi")
        self.assertEqual(payload["constraints"]["news_context"]["keyword"], "panda")

    def test_news_selection_rejects_brand_unsafe_titles(self) -> None:
        self.assertTrue(NewsContextService.is_brand_safe_selection("Cloudflare launches WebMCP", "AI agent"))
        self.assertFalse(NewsContextService.is_brand_safe_selection("AI性愛機器人新產品", "AI;性愛機器人"))
        self.assertFalse(NewsContextService.is_brand_safe_selection("牙醫猝逝後愛貓也離世", "生活焦點"))

    def test_native_h3_news_mode_does_not_create_a_second_autonomous_prompt(self) -> None:
        news = type(
            "_News",
            (),
            {"to_dict": lambda self: {"title": "Taipei panda steals zongzi", "keyword": "panda"}},  # noqa: ARG005
        )()
        with patch(
            "agentic.app.character_workflow.NewsContextService.get_random_news",
            return_value=news,
        ), patch(
            "agentic.app.character_workflow.LLMPromptEngine.generate_autonomous_scene_prompt",
            side_effect=AssertionError("native H3 must not create an autonomous scene prompt"),
        ):
            payload = build_goal_payload_from_character_config(
                self.repo_root,
                self.kirby_config,
                prompt="",
                preferred_generation_type="native_h3_story",
                publish_after_generate=False,
            )

        self.assertEqual(payload["prompt"], "")
        self.assertEqual(payload["constraints"]["prompt_source"], "news")
        self.assertEqual(payload["constraints"]["prompt_mode"], "news")
        self.assertEqual(payload["constraints"]["news_context"]["keyword"], "panda")
        self.assertIn("cute micro-gag", payload["constraints"]["native_h3_creative_brief"])
        self.assertTrue(payload["constraints"]["native_h3_semantic_qa_blocking"])

    def test_news_driven_random_mode_overrides_generic_prompt_and_persists_selection(self) -> None:
        news = NewsSelection(
            title="Taipei panda steals zongzi",
            keyword="panda",
            category="technology",
        )
        history_payload: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "agentic.app.character_workflow.NewsContextService.get_random_news",
            return_value=news,
        ), patch(
            "agentic.app.character_workflow.LLMPromptEngine.generate_autonomous_scene_prompt",
            return_value={
                "prompt": "Kirby turns the panda news into a playful rescue scene",
                "source": "autonomous_llm",
                "prompt_mode": "llm",
                "creative_seed": "Taipei panda steals zongzi",
                "news_context": news.to_dict(),
            },
        ) as generate_prompt:
            history_path = Path(temp_dir) / "news-history.json"
            payload = build_goal_payload_from_character_config(
                self.repo_root,
                self.kirby_config,
                prompt="隨機產生一個有角色動作的作品",
                preferred_generation_type="text2video",
                news_driven=True,
                news_history_path=str(history_path),
                output_dir=str(Path(temp_dir) / "output"),
                publish_after_generate=False,
            )
            history_payload = json.loads(history_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["constraints"]["prompt_source"], "autonomous_llm")
        self.assertEqual(payload["constraints"]["news_context"]["title"], news.title)
        self.assertEqual(generate_prompt.call_args.kwargs["news_context"]["title"], news.title)
        self.assertEqual(history_payload[0]["title"], news.title)

    def test_weighted_news_run_selects_strategy_before_news_content_generation(self) -> None:
        news = NewsSelection(
            title="Taipei panda steals zongzi",
            keyword="panda",
            category="technology",
        )
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "agentic.app.character_workflow.NewsContextService.get_random_news",
            return_value=news,
        ), patch(
            "agentic.app.character_workflow.LLMPromptEngine.generate_autonomous_scene_prompt",
            return_value={
                "prompt": "Kirby turns the panda news into a playful multi-scene story",
                "source": "autonomous_llm",
                "prompt_mode": "llm",
                "creative_seed": news.title,
                "news_context": news.to_dict(),
            },
        ) as generate_prompt:
            payload = build_goal_payload_from_character_config(
                self.repo_root,
                self.kirby_config,
                prompt="",
                news_driven=True,
                news_history_path=str(Path(temp_dir) / "news-history.json"),
                rng=random.Random(1),
                output_dir=str(Path(temp_dir) / "output"),
                publish_after_generate=False,
            )

        self.assertEqual(payload["source_generation_type"], "text2longvideo")
        self.assertEqual(payload["constraints"]["routing_selection_source"], "weighted_random")
        self.assertEqual(payload["prompt"], "Kirby turns the panda news into a playful multi-scene story")
        self.assertEqual(generate_prompt.call_args.kwargs["news_context"]["title"], news.title)

    def test_select_fresh_news_excludes_previous_keys_and_records_each_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = Path(temp_dir) / "news-history.json"
            calls: list[set[str]] = []
            selections = [
                NewsSelection(title="First headline", keyword="first"),
                NewsSelection(title="Second headline", keyword="second"),
            ]

            class FakeNewsService:
                def get_random_news(self, *, exclude_keys: set[str] | None = None, **_kwargs):
                    calls.append(set(exclude_keys or set()))
                    return selections[len(calls) - 1]

            first = _select_fresh_news(FakeNewsService(), history_path)
            second = _select_fresh_news(FakeNewsService(), history_path)

        self.assertEqual(first.title, "First headline")
        self.assertEqual(second.title, "Second headline")
        self.assertEqual(calls[0], set())
        self.assertEqual(calls[1], {"first headline\u001ffirst"})

    def test_build_goal_payload_merges_global_social_config_and_surfaces_summary(self) -> None:
        with patch(
            "agentic.app.character_workflow.load_global_social_config",
            return_value={
                "social_media": {
                    "default_hashtags": ["global_tag"],
                    "platforms": {
                        "youtube": {
                            "config_folder_path": "/app/configs/social_media/credentials/kirby",
                            "enabled": True,
                        },
                        "twitter": {
                            "config_folder_path": "/app/configs/social_media/credentials/kirby",
                            "enabled": True,
                        }
                    },
                }
            },
        ), patch(
            "agentic.app.character_workflow.LLMPromptEngine.route_generation_strategy",
            return_value={
                "generation_type": "text2video",
                "workflow_plan": {
                    "image_workflow_name": "nova-anime-xl",
                    "video_workflow_name": "minimax_h3_lowvram_t2v",
                    "refine_workflow_name": "",
                    "transition_workflow_name": "",
                    "upscale_workflow_name": "",
                },
                "count_plan": {
                    "image_count": 1,
                    "video_count": 1,
                    "segment_count": 1,
                    "review_selection_limit": 3,
                    "sticker_expression_count": 1,
                    "images_per_prompt": 1,
                },
                "reason": "text2video",
                "prompt_mode": "llm",
            },
        ), patch(
            "agentic.app.character_workflow.LLMPromptEngine.generate_autonomous_scene_prompt",
            return_value={"prompt": "kirby prompt", "source": "autonomous_llm", "prompt_mode": "llm", "news_context": {}},
        ):
            payload = build_goal_payload_from_character_config(self.repo_root, self.kirby_config)

        self.assertNotIn("twitter", payload["constraints"]["platforms"])
        self.assertIn("instagram_graph", payload["constraints"]["platforms"])
        self.assertIn("youtube", payload["constraints"]["platforms"])
        self.assertEqual(payload["character_config_summary"]["character_name"], "Kirby")
        self.assertNotIn("twitter", payload["character_config_summary"]["enabled_platforms"])

    def test_build_goal_payload_skips_disabled_and_unsupported_platforms(self) -> None:
        temp_dir = self.repo_root / ".tmp-tests" / "character-workflow"
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        config_path = temp_dir / "character.yaml"
        try:
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "character": {"name": "Kirby"},
                        "generation": {"output_dir": "/app/output_media"},
                        "social_media": {
                            "platforms": {
                                "instagram": {
                                    "config_folder_path": "configs/social_media/credentials/kirby",
                                    "enabled": True,
                                },
                                "facebook": {
                                    "config_folder_path": "configs/social_media/credentials/kirby",
                                    "enabled": False,
                                },
                                "mastodon": {
                                    "config_folder_path": "configs/social_media/credentials/kirby",
                                    "enabled": True,
                                },
                                "twitter": {
                                    "config_folder_path": "/app/configs/social_media/credentials/kirby",
                                    "enabled": True,
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "agentic.app.character_workflow.LLMPromptEngine.route_generation_strategy",
                return_value={
                    "generation_type": "text2video",
                    "workflow_plan": {
                        "image_workflow_name": "nova-anime-xl",
                        "video_workflow_name": "minimax_h3_lowvram_t2v",
                        "refine_workflow_name": "",
                        "transition_workflow_name": "",
                        "upscale_workflow_name": "",
                    },
                    "count_plan": {
                        "image_count": 1,
                        "video_count": 1,
                        "segment_count": 1,
                        "review_selection_limit": 3,
                        "sticker_expression_count": 1,
                        "images_per_prompt": 1,
                    },
                    "reason": "text2video",
                    "prompt_mode": "llm",
                },
            ), patch("agentic.app.character_workflow.load_global_social_config", return_value={}):
                payload = build_goal_payload_from_character_config(
                    self.repo_root,
                    config_path,
                    prompt="kirby social clip",
                )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertEqual(payload["constraints"]["platforms"], ["instagram_graph", "twitter"])
        self.assertEqual(payload["constraints"]["platform_aliases"]["instagram"], "instagram_graph")
        self.assertEqual(payload["constraints"]["skipped_platforms"], ["mastodon"])
        self.assertNotIn("facebook", payload["constraints"]["platform_configs"])
        self.assertEqual(
            Path(payload["constraints"]["platform_configs"]["instagram_graph"]["config_folder_path"]).parts[-4:],
            ("configs", "social_media", "credentials", "kirby"),
        )

    def test_build_goal_payload_enables_stage_review_when_discord_env_is_present(self) -> None:
        with patch(
            "agentic.app.character_workflow.LLMPromptEngine.route_generation_strategy",
            return_value={
                "generation_type": "text2image2video",
                "workflow_plan": {
                    "image_workflow_name": "nova-anime-xl",
                        "video_workflow_name": "minimax_h3_lowvram_i2v",
                    "refine_workflow_name": "",
                    "transition_workflow_name": "",
                    "upscale_workflow_name": "Tile Upscaler SDXL",
                },
                "count_plan": {
                    "image_count": 1,
                    "video_count": 1,
                    "segment_count": 1,
                    "review_selection_limit": 3,
                    "sticker_expression_count": 1,
                    "images_per_prompt": 1,
                },
                "reason": "text2image2video",
                "prompt_mode": "llm",
            },
        ), patch.dict(
            os.environ,
            {"discord_review_bot_token": "token", "discord_review_channel_id": "123"},
            clear=False,
        ):
            automatic_payload = build_goal_payload_from_character_config(
                self.repo_root,
                self.kirby_config,
                prompt="Kirby short rainy alley clip",
            )
            reviewed_payload = build_goal_payload_from_character_config(
                self.repo_root,
                self.kirby_config,
                prompt="Kirby short rainy alley clip",
                enable_review_loop=True,
            )

        self.assertEqual(automatic_payload["media_type"], "text2img2video")
        self.assertFalse(automatic_payload["constraints"]["enable_stage_review"])
        self.assertTrue(reviewed_payload["constraints"]["enable_stage_review"])


if __name__ == "__main__":
    unittest.main()
