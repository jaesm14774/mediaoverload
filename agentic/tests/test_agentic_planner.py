from __future__ import annotations

import unittest
from pathlib import Path

from agentic.assets.registry import AssetRegistry
from agentic.runtime.planner import TaskPlanner


class AgenticPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.asset_registry = AssetRegistry(
            cls.project_root / "configs" / "workflow_manifests",
            cls.project_root,
            asset_root=cls.project_root,
        )
        cls.planner = TaskPlanner(asset_registry=cls.asset_registry)

    def test_image_refine_uses_agentic_skill_names(self) -> None:
        goal = self.planner.create_goal(
            prompt="refine portrait lighting",
            media_type="image_refine",
            duration_seconds=30,
            style="cinematic surreal",
            auto_download_assets=False,
            constraints={"input_image_path": "C:\\stub.png"},
        )
        plan = self.planner.build_plan(goal)
        skill_names = [node.skill_name for node in plan.nodes]

        self.assertEqual(skill_names, ["agent.prompt.compose", "media.ensure_workflow", "media.image.refine"])

    def test_video_narrate_uses_agentic_media_skills(self) -> None:
        goal = self.planner.create_goal(
            prompt="narrate a clip",
            media_type="video_narrate",
            duration_seconds=30,
            style="cinematic surreal",
            auto_download_assets=False,
            constraints={"input_video_path": "C:\\clip.mp4", "text": "Hello world"},
        )
        plan = self.planner.build_plan(goal)
        skill_names = [node.skill_name for node in plan.nodes]

        self.assertEqual(
            skill_names,
            ["media.audio.narrate", "media.video.merge_audio", "media.video.gif_preview"],
        )

    def test_text2video_plan_is_available(self) -> None:
        goal = self.planner.create_goal(
            prompt="robot chef in rainy alley",
            media_type="text2video",
            duration_seconds=30,
            style="anime key visual",
            auto_download_assets=False,
            constraints={},
        )
        plan = self.planner.build_plan(goal)
        self.assertEqual(plan.workflow_name, "text2video_v1")
        self.assertIn("animate-video", [node.node_id for node in plan.nodes])

    def test_text2img2img_plan_is_available(self) -> None:
        goal = self.planner.create_goal(
            prompt="refine character art from scratch",
            media_type="text2img2img",
            duration_seconds=30,
            style="anime key visual",
            auto_download_assets=False,
            constraints={},
        )
        plan = self.planner.build_plan(goal)
        self.assertEqual(plan.workflow_name, "text2img2img_v1")
        self.assertIn("refine-image", [node.node_id for node in plan.nodes])

    def test_sticker_pack_plan_is_available(self) -> None:
        goal = self.planner.create_goal(
            prompt="cute reaction pack for rainy ramen shop",
            media_type="sticker_pack",
            duration_seconds=30,
            style="sticker illustration",
            auto_download_assets=False,
            constraints={"character": "Kirby"},
        )
        plan = self.planner.build_plan(goal)
        self.assertEqual(plan.workflow_name, "nova_model_plus_z_image_anime")
        self.assertEqual(
            [node.skill_name for node in plan.nodes],
            [
                "agent.sticker.expressions",
                "agent.sticker.prompt_set",
                "image.ensure_workflow",
                "media.image.render_batch",
                "agent.sticker.package",
            ],
        )

    def test_sticker_pack_plan_respects_routed_workflow_constraint(self) -> None:
        goal = self.planner.create_goal(
            prompt="Kirby LINE sticker reactions with clean white outline",
            media_type="sticker_pack",
            duration_seconds=30,
            style="LINE sticker",
            auto_download_assets=False,
            constraints={
                "character": "Kirby",
                "workflow_name": "nova-anime-xl",
                "image_workflow_name": "nova-anime-xl",
            },
        )
        plan = self.planner.build_plan(goal)
        render_node = next(node for node in plan.nodes if node.node_id == "render-stickers")

        self.assertEqual(plan.workflow_name, "nova-anime-xl")
        self.assertEqual(plan.metadata["selected_workflow"], "nova-anime-xl")
        self.assertEqual(render_node.inputs["workflow_name"], "nova-anime-xl")

    def test_animated_sticker_plan_is_available(self) -> None:
        goal = self.planner.create_goal(
            prompt="kirby reaction sticker dancing in place",
            media_type="animated_sticker",
            duration_seconds=8,
            style="LINE sticker illustration",
            auto_download_assets=False,
            constraints={"character": "Kirby"},
        )
        plan = self.planner.build_plan(goal)
        self.assertEqual(plan.workflow_name, "animated_sticker_v1")
        self.assertEqual(
            [node.skill_name for node in plan.nodes],
            [
                "agent.sticker.expressions",
                "agent.sticker.prompt_set",
                "image.ensure_workflow",
                "media.image.render_batch",
                "agent.sticker.motion_prompt",
                "media.ensure_workflow",
                "media.image.animate",
                "media.video.gif_preview",
                "agent.sticker.animate.package",
            ],
        )

    def test_animated_sticker_review_loop_adds_retry_branch(self) -> None:
        goal = self.planner.create_goal(
            prompt="kirby reaction sticker dancing in place",
            media_type="animated_sticker",
            duration_seconds=8,
            style="LINE sticker illustration",
            auto_download_assets=False,
            constraints={
                "character": "Kirby",
                "enable_review_loop": True,
                "review_notes": "needs stronger bounce and cleaner silhouette",
            },
        )
        plan = self.planner.build_plan(goal)
        node_ids = [node.node_id for node in plan.nodes]

        self.assertIn("review-select", node_ids)
        self.assertIn("review-refine-prompt", node_ids)
        self.assertIn("review-animate-sticker", node_ids)
        self.assertIn("review-final-select", node_ids)
        self.assertTrue(plan.metadata["review_loop_enabled"])

    def test_long_video_uses_registered_agentic_skills(self) -> None:
        goal = self.planner.create_goal(
            prompt="kirby explores a rainy cyberpunk alley",
            media_type="long_video",
            duration_seconds=20,
            style="anime cinematic travel film",
            auto_download_assets=False,
            constraints={"use_tts": True},
        )
        plan = self.planner.build_plan(goal)
        skill_names = {node.skill_name for node in plan.nodes}

        self.assertIn("agent.goal.expand", skill_names)
        self.assertIn("agent.story.segment", skill_names)
        self.assertIn("agent.segment.prepare", skill_names)
        self.assertIn("media.image.generate_keyframe", skill_names)
        self.assertIn("media.image.animate", skill_names)
        self.assertIn("media.video.extract_last_frame", skill_names)
        self.assertIn("media.audio.narrate", skill_names)
        self.assertIn("media.video.concat", skill_names)
        self.assertIn("media.audio.concat", skill_names)
        self.assertIn("media.video.merge_audio", skill_names)
        self.assertIn("media.video.gif_preview", skill_names)

    def test_long_video_review_loop_adds_regeneration_path(self) -> None:
        goal = self.planner.create_goal(
            prompt="kirby explores a rainy cyberpunk alley",
            media_type="long_video",
            duration_seconds=20,
            style="anime cinematic travel film",
            auto_download_assets=False,
            constraints={
                "use_tts": False,
                "enable_review_loop": True,
                "review_notes": "second pass should push stronger action and cleaner framing",
            },
        )
        plan = self.planner.build_plan(goal)
        node_ids = [node.node_id for node in plan.nodes]

        self.assertIn("review-select", node_ids)
        self.assertIn("review-refine-prompt", node_ids)
        self.assertIn("review-segment-prompt-01", node_ids)
        self.assertIn("review-concat-final-video", node_ids)
        self.assertIn("review-final-select", node_ids)
        self.assertIn("persist-longvideo-summary", node_ids)

    def test_carousel_plan_is_available(self) -> None:
        goal = self.planner.create_goal(
            prompt="robot chef travel diary in rainy taipei",
            media_type="carousel",
            duration_seconds=30,
            style="editorial anime postcard",
            auto_download_assets=False,
            constraints={},
        )
        plan = self.planner.build_plan(goal)
        self.assertEqual(plan.workflow_name, "carousel_v1")
        self.assertEqual(
            [node.skill_name for node in plan.nodes],
            [
                "agent.goal.expand",
                "agent.story.segment",
                "agent.carousel.prompt_set",
                "image.ensure_workflow",
                "media.image.render_batch",
                "agent.carousel.package",
            ],
        )

    def test_publish_review_plan_is_available(self) -> None:
        goal = self.planner.create_goal(
            prompt="review and prep kirby assets for instagram",
            media_type="publish_review",
            duration_seconds=30,
            style="social promo",
            auto_download_assets=False,
            constraints={
                "media_paths": ["C:\\asset_1.png", "C:\\asset_2.gif"],
                "platforms": ["instagram"],
                "dry_run": True,
            },
        )
        plan = self.planner.build_plan(goal)
        self.assertEqual(plan.workflow_name, "publish_review_v1")
        self.assertEqual(
            [node.skill_name for node in plan.nodes],
            [
                "publish.media.ingest",
                "review.assets.select",
                "publish.media.process",
                "publish.caption.prepare",
                "publish.social.dispatch",
                "agent.summary.persist",
            ],
        )
        prepare_caption = next(node for node in plan.nodes if node.node_id == "prepare-caption")
        self.assertEqual(prepare_caption.depends_on, ["review-select", "process-media"])
        persist_summary = next(node for node in plan.nodes if node.node_id == "persist-publish-review-summary")
        self.assertEqual(persist_summary.depends_on, ["review-select", "process-media", "prepare-caption", "dispatch-publish"])

    def test_text2video_review_loop_adds_retry_branch(self) -> None:
        goal = self.planner.create_goal(
            prompt="robot chef in rainy alley",
            media_type="text2video",
            duration_seconds=30,
            style="anime key visual",
            auto_download_assets=False,
            constraints={"review_notes": "needs stronger motion and less empty space"},
        )
        plan = self.planner.build_plan(goal)
        node_ids = [node.node_id for node in plan.nodes]

        self.assertIn("review-refine-prompt", node_ids)
        self.assertIn("review-render-image", node_ids)
        self.assertIn("review-animate-video", node_ids)
        self.assertIn("review-final-select", node_ids)
        self.assertIn("persist-text2video-summary", node_ids)

    def test_text2img2video_plan_persists_summary(self) -> None:
        goal = self.planner.create_goal(
            prompt="kirby rainy ramen alley short",
            media_type="text2img2video",
            duration_seconds=20,
            style="anime key visual",
            auto_download_assets=False,
            constraints={},
        )
        plan = self.planner.build_plan(goal)

        self.assertIn("persist-text2img2video-summary", [node.node_id for node in plan.nodes])

    def test_text2img2video_stage_review_gates_video_generation(self) -> None:
        goal = self.planner.create_goal(
            prompt="kirby rainy ramen alley short",
            media_type="text2img2video",
            duration_seconds=20,
            style="anime key visual",
            auto_download_assets=False,
            constraints={"enable_stage_review": True},
        )
        plan = self.planner.build_plan(goal)
        stage_review = next(node for node in plan.nodes if node.node_id == "stage-review-select")
        animate_node = next(node for node in plan.nodes if node.node_id == "animate-video")

        self.assertEqual(stage_review.depends_on, ["upscale-image"])
        self.assertIn("stage-review-select", animate_node.depends_on)

    def test_text2video_stage_review_gates_video_generation(self) -> None:
        goal = self.planner.create_goal(
            prompt="robot chef in rainy alley",
            media_type="text2video",
            duration_seconds=30,
            style="anime key visual",
            auto_download_assets=False,
            constraints={"enable_stage_review": True},
        )
        plan = self.planner.build_plan(goal)
        stage_review = next(node for node in plan.nodes if node.node_id == "stage-review-select")
        animate_node = next(node for node in plan.nodes if node.node_id == "animate-video")

        self.assertEqual(stage_review.depends_on, ["render-image"])
        self.assertIn("stage-review-select", animate_node.depends_on)

    def test_long_video_stage_review_gates_first_segment_video(self) -> None:
        goal = self.planner.create_goal(
            prompt="kirby explores a rainy cyberpunk alley",
            media_type="long_video",
            duration_seconds=20,
            style="anime cinematic travel film",
            auto_download_assets=False,
            constraints={"enable_stage_review": True},
        )
        plan = self.planner.build_plan(goal)
        stage_review = next(node for node in plan.nodes if node.node_id == "stage-review-01")
        first_segment_video = next(node for node in plan.nodes if node.node_id == "segment-video-01")

        self.assertEqual(stage_review.depends_on, ["segment-frame-01"])
        self.assertIn("stage-review-01", first_segment_video.depends_on)

    def test_text2img2video_plan_respects_stage_workflows_and_image_count(self) -> None:
        goal = self.planner.create_goal(
            prompt="kirby rainy ramen alley short",
            media_type="text2img2video",
            duration_seconds=20,
            style="anime key visual",
            auto_download_assets=False,
            constraints={
                "image_workflow_name": "nova-anime-xl",
                "video_workflow_name": "wan2.2_gguf_i2v",
                "upscale_workflow_name": "Tile Upscaler SDXL",
                "image_count": 3,
            },
        )
        plan = self.planner.build_plan(goal)
        render_node = next(node for node in plan.nodes if node.node_id == "render-image")
        video_check_node = next(node for node in plan.nodes if node.node_id == "video-asset-check")

        self.assertEqual(render_node.inputs["workflow_name"], "nova-anime-xl")
        self.assertEqual(render_node.inputs["image_count"], 3)
        self.assertEqual(video_check_node.inputs["workflow_name"], "wan2.2_gguf_i2v")

    def test_image_plan_render_node_depends_on_idea_brief(self) -> None:
        # Regression test: render-image must depend on idea-brief so that the
        # LLM-generated prompt is passed to ComfyUI instead of the raw user prompt.
        goal = self.planner.create_goal(
            prompt="kirby on a floating cloud",
            media_type="image",
            duration_seconds=0,
            style="anime key visual",
            auto_download_assets=False,
        )
        plan = self.planner.build_plan(goal)
        render_node = next(node for node in plan.nodes if node.node_id == "render-image")
        self.assertIn("idea-brief", render_node.depends_on)

    def test_long_video_plan_respects_segment_count_and_review_selection_limit(self) -> None:
        goal = self.planner.create_goal(
            prompt="kirby explores a rainy cyberpunk alley",
            media_type="long_video",
            duration_seconds=20,
            style="anime cinematic travel film",
            auto_download_assets=False,
            constraints={
                "segment_count": 4,
                "review_selection_limit": 5,
                "review_notes": "stronger motion and cleaner framing",
                "video_workflow_name": "wan2.2_gguf_i2v",
            },
        )
        plan = self.planner.build_plan(goal)
        script_plan = next(node for node in plan.nodes if node.node_id == "script-plan")
        review_select = next(node for node in plan.nodes if node.node_id == "review-select")
        segment_video = next(node for node in plan.nodes if node.node_id == "segment-video-01")

        self.assertEqual(script_plan.inputs["segment_count"], 4)
        self.assertEqual(review_select.inputs["limit"], 5)
        self.assertEqual(segment_video.inputs["workflow_name"], "wan2.2_gguf_i2v")

    def test_text2video_plan_respects_video_count(self) -> None:
        goal = self.planner.create_goal(
            prompt="robot chef in rainy alley",
            media_type="text2video",
            duration_seconds=30,
            style="anime key visual",
            auto_download_assets=False,
            constraints={
                "image_workflow_name": "nova-anime-xl",
                "video_workflow_name": "wan2.2_gguf_i2v",
                "video_count": 3,
            },
        )
        plan = self.planner.build_plan(goal)
        animate_node = next(node for node in plan.nodes if node.node_id == "animate-video")

        self.assertEqual(animate_node.inputs["workflow_name"], "wan2.2_gguf_i2v")
        self.assertEqual(animate_node.inputs["video_count"], 3)


if __name__ == "__main__":
    unittest.main()
