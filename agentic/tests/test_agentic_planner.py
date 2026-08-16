from __future__ import annotations

import unittest
from pathlib import Path

from agentic.assets.registry import AssetRegistry
from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest, RunState, SkillContext
from agentic.runtime.planner import TaskPlanner
from agentic.skills.agent_primitives import AgentPlanningSkills


class AgenticPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.asset_registry = AssetRegistry(
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
        video_check = next(node for node in plan.nodes if node.node_id == "video-asset-check")
        self.assertEqual(video_check.inputs["workflow_name"], "minimax_h3_lowvram_t2v")

    def test_segment_prepare_does_not_append_ordinary_idea_prompt_as_review_direction(self) -> None:
        class FakePromptEngine:
            def prepare_segment(self, *args, **kwargs):
                del args, kwargs
                return {"prompt": "single-frame segment", "negative_prompt": "bad anatomy"}

        goal = GoalRequest(prompt="kirby runs", media_type="long_video", style="anime")
        plan = ExecutionPlan(goal=goal, workflow_name="long_video", nodes=[])
        node = ExecutionNode(
            node_id="segment-prompt-01",
            skill_name="agent.segment.prepare",
            inputs={"segment_index": 0},
            depends_on=["script-plan", "idea-brief"],
        )
        state = RunState(
            goal={},
            metadata={},
            node_outputs={
                "script-plan": {"segments": [{"segment_id": "segment-1", "visual": "one scene"}]},
                "idea-brief": {"prompt": "the full expanded brief", "creative_brief": "the brief"},
            },
        )

        result = AgentPlanningSkills(prompt_engine=FakePromptEngine()).prepare_segment(SkillContext(plan, node, state))

        self.assertEqual(result.outputs["prompt"], "single-frame segment")

    def test_segment_prepare_appends_only_explicit_review_revision(self) -> None:
        class FakePromptEngine:
            def prepare_segment(self, *args, **kwargs):
                del args, kwargs
                return {"prompt": "single-frame segment", "negative_prompt": "bad anatomy"}

        goal = GoalRequest(prompt="kirby runs", media_type="long_video", style="anime")
        plan = ExecutionPlan(goal=goal, workflow_name="long_video", nodes=[])
        node = ExecutionNode(
            node_id="review-segment-prompt-01",
            skill_name="agent.segment.prepare",
            inputs={"segment_index": 0},
            depends_on=["script-plan", "idea-brief", "review-refine-prompt"],
        )
        state = RunState(
            goal={},
            metadata={},
            node_outputs={
                "script-plan": {"segments": [{"segment_id": "segment-1", "visual": "one scene"}]},
                "idea-brief": {"prompt": "the full expanded brief"},
                "review-refine-prompt": {"revised_prompt": "make the action clearer"},
            },
        )

        result = AgentPlanningSkills(prompt_engine=FakePromptEngine()).prepare_segment(SkillContext(plan, node, state))

        self.assertIn("revision direction: make the action clearer", result.outputs["prompt"])

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
        self.assertIn("longvideo.render_segment_video", skill_names)
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
        review_select = next(node for node in plan.nodes if node.node_id == "review-select")
        self.assertEqual(review_select.inputs["limit"], 10)
        self.assertEqual(review_select.inputs["review_scope"], "final_media")
        self.assertFalse(review_select.inputs["review_all_candidates"])
        prepare_caption = next(node for node in plan.nodes if node.node_id == "prepare-caption")
        self.assertEqual(prepare_caption.depends_on, ["review-select", "process-media"])
        persist_summary = next(node for node in plan.nodes if node.node_id == "persist-publish-review-summary")
        self.assertEqual(persist_summary.depends_on, ["review-select", "process-media", "prepare-caption", "dispatch-publish"])

    def test_publish_review_plan_supports_final_media_review(self) -> None:
        goal = self.planner.create_goal(
            prompt="review Kirby images for Instagram and Facebook",
            media_type="publish_review",
            duration_seconds=0,
            style="polished 2D anime",
            auto_download_assets=False,
            constraints={
                "media_paths": ["C:\\asset_1.png", "C:\\asset_2.png"],
                "platforms": ["instagram_graph", "facebook"],
                "review_scope": "final_media",
                "review_all_candidates": True,
                "require_human_review": True,
            },
        )

        plan = self.planner.build_plan(goal)
        review_select = next(node for node in plan.nodes if node.node_id == "review-select")
        self.assertEqual(review_select.inputs["review_scope"], "final_media")
        self.assertTrue(review_select.inputs["review_all_candidates"])

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

    def test_five_second_text2img2video_uses_exact_frame_override(self) -> None:
        goal = self.planner.create_goal(
            prompt="Kirby swats one glowing orb into a target",
            media_type="text2img2video",
            duration_seconds=5,
            style="anime key visual",
            auto_download_assets=False,
            constraints={"duration_override_seconds": 5, "video_frame_rate": 24},
        )
        plan = self.planner.build_plan(goal)
        animate = next(node for node in plan.nodes if node.node_id == "animate-video")
        self.assertEqual(animate.inputs["length"], 120)

    def test_native_h3_story_prompt_declares_first_frame_i2v(self) -> None:
        goal = self.planner.create_goal(
            prompt="Kirby protects one glowing orb from a sudden gust",
            media_type="native_h3_story",
            duration_seconds=15,
            style="anime key visual",
            auto_download_assets=False,
            constraints={"native_h3_storyboard_path": "configs/storyboards/kirby_native_15s.yaml"},
        )
        plan = self.planner.build_plan(goal)
        prompt_node = next(node for node in plan.nodes if node.node_id == "native-story-prompt")
        self.assertEqual(prompt_node.inputs["render_mode"], "image_to_video")

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
        self.assertEqual(stage_review.inputs["review_scope"], "first_frame")
        self.assertEqual(stage_review.inputs["review_phase"], "opening_frame")
        self.assertEqual(stage_review.inputs["limit"], 1)
        self.assertTrue(stage_review.inputs["review_all_candidates"])
        self.assertIn("stage-review-select", animate_node.depends_on)

    def test_pre_video_text2img2video_binds_the_selected_single_frame(self) -> None:
        goal = self.planner.create_goal(
            prompt="Kirby runs through a neon ramen alley",
            media_type="text2img2video",
            duration_seconds=15,
            style="anime key visual",
            auto_download_assets=False,
            constraints={
                "pre_video_review_enabled": True,
                "pre_video_candidate_count": 6,
                "pre_video_review_require_human": True,
            },
        )
        plan = self.planner.build_plan(goal)
        render = next(node for node in plan.nodes if node.node_id == "render-image")
        review = next(node for node in plan.nodes if node.node_id == "stage-review-select")
        animate = next(node for node in plan.nodes if node.node_id == "animate-video")

        self.assertEqual(render.inputs["image_count"], 6)
        self.assertEqual(review.depends_on, ["render-image"])
        self.assertEqual(review.inputs["limit"], 1)
        self.assertTrue(review.inputs["require_human_review"])
        self.assertIn("stage-review-select", animate.depends_on)

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
        self.assertEqual(stage_review.inputs["review_scope"], "first_frame")
        self.assertEqual(stage_review.inputs["review_phase"], "opening_frame")
        self.assertEqual(stage_review.inputs["limit"], 1)
        self.assertTrue(stage_review.inputs["review_all_candidates"])
        self.assertIn("stage-review-select", animate_node.depends_on)

    def test_shared_pre_video_gate_does_not_change_t2v_conditioning(self) -> None:
        goal = self.planner.create_goal(
            prompt="Kirby protects a glowing seed",
            media_type="text2video",
            duration_seconds=15,
            style="anime key visual",
            auto_download_assets=False,
            constraints={
                "pre_video_review_enabled": True,
                "pre_video_candidate_count": 6,
                "pre_video_review_require_human": True,
                "video_workflow_name": "minimax_h3_lowvram_t2v",
            },
        )
        plan = self.planner.build_plan(goal)
        render = next(node for node in plan.nodes if node.node_id == "render-image")
        animate = next(node for node in plan.nodes if node.node_id == "animate-video")

        self.assertEqual(render.inputs["image_count"], 1)
        self.assertNotIn("stage-review-select", [node.node_id for node in plan.nodes])
        self.assertEqual(animate.inputs["workflow_name"], "minimax_h3_lowvram_t2v")
        self.assertNotIn("stage-review-select", animate.depends_on)

    def test_pre_video_gate_only_expands_first_long_video_segment(self) -> None:
        goal = self.planner.create_goal(
            prompt="Kirby crosses a stormy meadow",
            media_type="long_video",
            duration_seconds=20,
            style="anime cinematic travel film",
            auto_download_assets=False,
            constraints={
                "pre_video_review_enabled": True,
                "pre_video_candidate_count": 6,
                "pre_video_review_require_human": True,
                "segment_count": 2,
                "longvideo_mix_weights": {"anchor_first": 1},
            },
        )
        plan = self.planner.build_plan(goal)
        first_frame = next(node for node in plan.nodes if node.node_id == "segment-frame-01")
        review = next(node for node in plan.nodes if node.node_id == "stage-review-01")
        first_video = next(node for node in plan.nodes if node.node_id == "segment-video-01")
        second_video = next(node for node in plan.nodes if node.node_id == "segment-video-02")

        self.assertEqual(first_frame.inputs["image_count"], 6)
        self.assertEqual(review.inputs["limit"], 1)
        self.assertTrue(review.inputs["require_human_review"])
        self.assertIn("stage-review-01", first_video.depends_on)
        self.assertEqual(second_video.inputs["conditioning_plan"]["anchors"]["first"], "segment-tail-01")

    def test_long_video_stage_review_gates_first_segment_video(self) -> None:
        goal = self.planner.create_goal(
            prompt="kirby explores a rainy cyberpunk alley",
            media_type="long_video",
            duration_seconds=20,
            style="anime cinematic travel film",
            auto_download_assets=False,
            constraints={
                "enable_stage_review": True,
                "longvideo_mix_weights": {"anchor_first": 1},
            },
        )
        plan = self.planner.build_plan(goal)
        stage_review = next(node for node in plan.nodes if node.node_id == "stage-review-01")
        first_segment_video = next(node for node in plan.nodes if node.node_id == "segment-video-01")

        self.assertEqual(stage_review.depends_on, ["segment-frame-01"])
        self.assertEqual(stage_review.inputs["review_scope"], "first_frame")
        self.assertEqual(stage_review.inputs["review_phase"], "opening_frame")
        self.assertEqual(stage_review.inputs["limit"], 1)
        self.assertTrue(stage_review.inputs["review_all_candidates"])
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
                "video_workflow_name": "minimax_h3_lowvram_i2v",
                "upscale_workflow_name": "Tile Upscaler SDXL",
                "image_count": 3,
            },
        )
        plan = self.planner.build_plan(goal)
        render_node = next(node for node in plan.nodes if node.node_id == "render-image")
        video_check_node = next(node for node in plan.nodes if node.node_id == "video-asset-check")

        self.assertEqual(render_node.inputs["workflow_name"], "nova-anime-xl")
        self.assertEqual(render_node.inputs["image_count"], 3)
        self.assertEqual(video_check_node.inputs["workflow_name"], "minimax_h3_lowvram_i2v")

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
                "video_workflow_name": "minimax_h3_lowvram_i2v",
                "longvideo_mix_weights": {"anchor_first": 1},
            },
        )
        plan = self.planner.build_plan(goal)
        script_plan = next(node for node in plan.nodes if node.node_id == "script-plan")
        review_select = next(node for node in plan.nodes if node.node_id == "review-select")
        segment_video = next(node for node in plan.nodes if node.node_id == "segment-video-01")

        self.assertEqual(script_plan.inputs["segment_count"], 4)
        self.assertEqual(review_select.inputs["limit"], 5)
        self.assertEqual(segment_video.inputs["workflow_name"], "minimax_h3_lowvram_i2v")

    def test_long_video_every_segment_review_generates_multi_anchor_candidates(self) -> None:
        goal = self.planner.create_goal(
            prompt="kirby explores a rainy cyberpunk alley",
            media_type="long_video",
            duration_seconds=20,
            style="anime cinematic travel film",
            auto_download_assets=False,
            constraints={
                "segment_count": 2,
                "require_human_review": True,
                "longvideo_review_policy": "every_segment",
                "longvideo_frame_candidate_count": 4,
                "longvideo_mix_weights": {"anchor_first": 1},
            },
        )
        plan = self.planner.build_plan(goal)

        second_first = next(node for node in plan.nodes if node.node_id == "segment-frame-02")
        second_first_review = next(node for node in plan.nodes if node.node_id == "segment-anchor-first-select-02")

        self.assertEqual(second_first.inputs["image_count"], 4)
        self.assertIn("segment-tail-01", second_first.depends_on)
        self.assertEqual(second_first_review.inputs["review_scope"], "first_frame")
        self.assertTrue(second_first_review.inputs["require_human_review"])

    def test_text2video_plan_respects_video_count(self) -> None:
        goal = self.planner.create_goal(
            prompt="robot chef in rainy alley",
            media_type="text2video",
            duration_seconds=30,
            style="anime key visual",
            auto_download_assets=False,
            constraints={
                "image_workflow_name": "nova-anime-xl",
                "video_workflow_name": "minimax_h3_lowvram_t2v",
                "video_count": 3,
            },
        )
        plan = self.planner.build_plan(goal)
        animate_node = next(node for node in plan.nodes if node.node_id == "animate-video")

        self.assertEqual(animate_node.inputs["workflow_name"], "minimax_h3_lowvram_t2v")
        self.assertEqual(animate_node.inputs["video_count"], 3)


if __name__ == "__main__":
    unittest.main()
