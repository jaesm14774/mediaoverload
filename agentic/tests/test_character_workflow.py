from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import yaml

from agentic.app.character_workflow import build_goal_payload_from_character_config


class CharacterWorkflowRoutingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.kirby_config = cls.repo_root / "configs" / "characters" / "kirby.yaml"

    def test_build_goal_payload_uses_llm_routed_strategy_and_workflow(self) -> None:
        with patch(
            "agentic.app.character_workflow.LLMPromptEngine.route_generation_strategy",
            return_value={
                "generation_type": "sticker_pack",
                "workflow_plan": {
                    "image_workflow_name": "nova-anime-xl",
                    "video_workflow_name": "wan2.2_gguf_i2v",
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
        self.assertEqual(payload["constraints"]["video_workflow_name"], "wan2.2_gguf_i2v")
        self.assertEqual(payload["constraints"]["video_count"], 2)
        self.assertEqual(payload["constraints"]["selection_limit"], 5)
        self.assertEqual(payload["constraints"]["images_per_prompt"], 2)
        self.assertEqual(payload["constraints"]["routing_prompt_mode"], "llm")
        self.assertEqual(payload["routing_summary"]["strategy"], "sticker_pack")
        self.assertEqual(payload["routing_summary"]["primary_workflow"], "nova-anime-xl")
        self.assertEqual(payload["routing_summary"]["workflow_plan"]["video_workflow_name"], "wan2.2_gguf_i2v")
        self.assertEqual(payload["routing_summary"]["count_plan"]["review_selection_limit"], 5)

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
                    "video_workflow_name": "wan2.2_gguf_i2v",
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

    def test_build_goal_payload_merges_global_social_config_and_surfaces_summary(self) -> None:
        with patch(
            "agentic.app.character_workflow.load_global_social_config",
            return_value={
                "social_media": {
                    "default_hashtags": ["global_tag"],
                    "platforms": {
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
                    "video_workflow_name": "wan2.2_gguf_i2v",
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

        self.assertIn("twitter", payload["constraints"]["platforms"])
        self.assertIn("instagram_graph", payload["constraints"]["platforms"])
        self.assertEqual(payload["character_config_summary"]["character_name"], "Kirby")
        self.assertIn("twitter", payload["character_config_summary"]["enabled_platforms"])

    def test_build_goal_payload_skips_disabled_and_unsupported_platforms(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "character.yaml"
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
                        "video_workflow_name": "wan2.2_gguf_i2v",
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

        self.assertEqual(payload["constraints"]["platforms"], ["instagram_graph", "twitter"])
        self.assertEqual(payload["constraints"]["platform_aliases"]["instagram"], "instagram_graph")
        self.assertEqual(payload["constraints"]["skipped_platforms"], ["mastodon"])
        self.assertNotIn("facebook", payload["constraints"]["platform_configs"])
        self.assertTrue(
            payload["constraints"]["platform_configs"]["instagram_graph"]["config_folder_path"].endswith(
                "configs\\social_media\\credentials\\kirby"
            )
        )

    def test_build_goal_payload_enables_stage_review_when_discord_env_is_present(self) -> None:
        with patch(
            "agentic.app.character_workflow.LLMPromptEngine.route_generation_strategy",
            return_value={
                "generation_type": "text2image2video",
                "workflow_plan": {
                    "image_workflow_name": "nova-anime-xl",
                    "video_workflow_name": "wan2.2_gguf_i2v",
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
            payload = build_goal_payload_from_character_config(
                self.repo_root,
                self.kirby_config,
                prompt="Kirby short rainy alley clip",
            )

        self.assertEqual(payload["media_type"], "text2img2video")
        self.assertTrue(payload["constraints"]["enable_stage_review"])


if __name__ == "__main__":
    unittest.main()
