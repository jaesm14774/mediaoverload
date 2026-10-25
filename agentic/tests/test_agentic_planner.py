from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from agentic.assets.registry import AssetRegistry
from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest, RunState, SkillContext
from agentic.runtime.planner import TaskPlanner
from agentic.runtime.registry import ToolRegistry
from agentic.skills.agent_social import AgentSocialSkills
from agentic.skills.agent_primitives import AgentMediaSkills, AgentPlanningSkills


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
            constraints={"input_video_path": "C:\\clip.mp4", "text": "Hello world"},
        )
        plan = self.planner.build_plan(goal)
        skill_names = [node.skill_name for node in plan.nodes]

        self.assertEqual(
            skill_names,
            ["media.audio.narrate", "media.video.merge_audio", "media.video.gif_preview"],
        )

    def test_image_sequence_edit_is_a_first_class_agent_plan(self) -> None:
        goal = self.planner.create_goal(
            prompt="edit three generated stills into a fashion reel",
            media_type="image_sequence_edit",
            duration_seconds=9,
            style="editorial surreal",
            constraints={
                "edit_input_paths": ["/tmp/one.png", "/tmp/two.png", "/tmp/three.png"],
                "edit_profile": "editorial_kinetic_v1",
                "edit_variant_seed": 4,
            },
        )

        plan = self.planner.build_plan(goal)

        self.assertEqual(plan.workflow_name, "image_sequence_edit_v1")
        compose = plan.nodes[0]
        self.assertEqual(compose.skill_name, "media.video.compose_timeline")
        self.assertEqual(compose.inputs["profile"], "editorial_kinetic_v1")
        self.assertEqual(compose.inputs["variant_seed"], 4)
        self.assertEqual(plan.nodes[1].skill_name, "media.video.inspect")

    def test_image_sequence_edit_defaults_to_motion_cut_for_stills(self) -> None:
        goal = self.planner.create_goal(
            prompt="animate generated stills into a reel",
            media_type="image_sequence_edit",
            duration_seconds=9,
            style="playful",
            constraints={
                "edit_input_paths": ["/tmp/one.png", "/tmp/two.png", "/tmp/three.png"],
            },
        )

        plan = self.planner.build_plan(goal)

        self.assertEqual(plan.nodes[0].inputs["profile"], "motion_cut_v1")
        self.assertNotIn("creative_review", plan.nodes[0].inputs)

    def test_image_sequence_edit_defaults_to_hard_cut_for_video_segments(self) -> None:
        goal = self.planner.create_goal(
            prompt="join generated video segments",
            media_type="image_sequence_edit",
            duration_seconds=15,
            style="cinematic",
            constraints={
                "edit_input_paths": ["/tmp/one.mp4", "/tmp/two.mp4", "/tmp/three.mp4"],
            },
        )

        plan = self.planner.build_plan(goal)

        self.assertEqual(plan.nodes[0].inputs["profile"], "baseline_concat")
        self.assertNotIn("creative_review", plan.nodes[0].inputs)

    def test_explicit_edit_plan_drives_render_and_qa_contract(self) -> None:
        goal = self.planner.create_goal(
            prompt="render an explicit edit plan",
            media_type="image_sequence_edit",
            duration_seconds=9,
            style="editorial",
            constraints={
                "edit_plan": {
                    "clips": [{"path": "/tmp/one.png"}, {"path": "/tmp/two.png"}],
                    "transitions": [{"name": "fade", "duration_seconds": 0.1}],
                    "output_width": 720,
                    "output_height": 1280,
                    "fps": 30,
                    "target_duration_seconds": 4,
                    "profile": "xfade_clean_v1",
                }
            },
        )

        plan = self.planner.build_plan(goal)
        compose = plan.nodes[0]
        qa = plan.nodes[1]

        self.assertEqual(compose.inputs["edit_plan"]["output_width"], 720)
        self.assertEqual(qa.inputs["expected_fps"], 30.0)
        self.assertEqual(qa.inputs["target_duration"], 4.0)
        self.assertTrue(qa.inputs["require_audio"])
        self.assertTrue(qa.inputs["require_stereo_audio"])

    def test_drama_plan_compiles_into_existing_edit_timeline_node(self) -> None:
        drama_plan = {
            "plan_id": "episode-001",
            "title": "The runaway prop",
            "premise": "A small mistake creates a physical comedy payoff.",
            "objective": "recover the prop",
            "character_ids": ["hero"],
            "prop_ids": ["prop"],
            "scenes": [
                {
                    "scene_id": "S01",
                    "beat": "HOOK",
                    "duration_seconds": 2,
                    "objective": "reach the prop",
                    "start_state": "the hero reaches",
                    "action_beats": ["the prop slips"],
                    "end_state": "the hero loses balance",
                    "next_hook": "the hero must recover it",
                    "cause": "the prop slips",
                    "effect": "the hero stumbles",
                    "character_ids": ["hero"],
                    "prop_ids": ["prop"],
                    "selected_source_path": "/tmp/scene-01.mp4",
                },
                {
                    "scene_id": "S02",
                    "beat": "PAYOFF",
                    "duration_seconds": 2,
                    "objective": "recover the prop",
                    "start_state": "the hero is off balance",
                    "action_beats": ["the hero catches it"],
                    "end_state": "the hero holds it proudly",
                    "next_hook": "loop back to the opening reach",
                    "cause": "the hero reacts",
                    "effect": "the prop is recovered",
                    "character_ids": ["hero"],
                    "prop_ids": ["prop"],
                    "selected_source_path": "/tmp/scene-02.mp4",
                },
            ],
        }
        goal = self.planner.create_goal(
            prompt="compile this short drama",
            media_type="image_sequence_edit",
            duration_seconds=4,
            style="playful",
            constraints={"drama_plan": drama_plan},
        )

        plan = self.planner.build_plan(goal)

        compose = plan.nodes[0]
        self.assertTrue(plan.metadata["explicit_drama_plan"])
        self.assertEqual(compose.inputs["profile"], "motion_cut_v1")
        self.assertEqual(compose.inputs["drama_plan"]["plan_id"], "episode-001")
        self.assertEqual(compose.inputs["output_width"], 576)
        self.assertEqual(compose.inputs["output_height"], 1024)
        self.assertEqual(plan.nodes[1].inputs["target_duration"], 4.0)

    def test_text2video_plan_is_available(self) -> None:
        goal = self.planner.create_goal(
            prompt="robot chef in rainy alley",
            media_type="text2video",
            duration_seconds=30,
            style="anime key visual",
            constraints={},
        )
        plan = self.planner.build_plan(goal)
        self.assertEqual(plan.workflow_name, "text2video_v1")
        self.assertIn("animate-video", [node.node_id for node in plan.nodes])
        video_check = next(node for node in plan.nodes if node.node_id == "video-asset-check")
        self.assertEqual(video_check.inputs["workflow_name"], "minimax_h3_t2va")

    def test_stage_probe_text2video_reviews_cover_without_conditioning_video(self) -> None:
        goal = self.planner.create_goal(
            prompt="Kirby crosses one visible news-derived obstacle",
            media_type="text2video",
            duration_seconds=30,
            style="anime key visual",
            constraints={
                "image_workflow_name": "krea2_turbo",
                "video_workflow_name": "minimax_h3_t2va",
                "image_count": 6,
                "enable_stage_review": True,
                "stage_probe_auto_select": True,
            },
        )
        plan = self.planner.build_plan(goal)
        preview = next(node for node in plan.nodes if node.node_id == "stage-review-select")
        animate = next(node for node in plan.nodes if node.node_id == "animate-video")

        self.assertEqual(preview.inputs["review_scope"], "first_frame")
        self.assertTrue(preview.inputs["auto_select_for_probe"])
        self.assertIn("stage-review-select", animate.depends_on)

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

    def test_user_given_first_video_segment_when_prepared_then_opening_keyframe_stages_its_upcoming_action(self) -> None:
        """User: Given the first video segment has an upcoming action, When its frame prompt is prepared, Then the opening keyframe shows anticipation before that action."""
        class FakePromptEngine:
            def prepare_segment(self, *args, **kwargs):
                del args, kwargs
                return {"prompt": "the hero hops across the puddle", "negative_prompt": ""}

        goal = GoalRequest(prompt="the hero crosses a puddle", media_type="long_video")
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
                "script-plan": {
                    "segments": [
                        {
                            "segment_id": "segment-1",
                            "visual": "the hero faces a puddle",
                            "action": "leans forward, then springs over the puddle",
                        }
                    ]
                },
                "idea-brief": {
                    "opening_keyframe_prompt": "The hero stands before a puddle with room to jump.",
                },
            },
        )

        result = AgentPlanningSkills(prompt_engine=FakePromptEngine()).prepare_segment(SkillContext(plan, node, state))

        self.assertIn("Opening-frame action staging:", result.outputs["opening_keyframe_prompt"])
        self.assertIn("split second before movement begins", result.outputs["opening_keyframe_prompt"])
        self.assertIn("leans forward, then springs over the puddle", result.outputs["opening_keyframe_prompt"])

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
            constraints={},
        )
        plan = self.planner.build_plan(goal)
        self.assertEqual(plan.workflow_name, "text2img2img_v1")
        self.assertIn("refine-image", [node.node_id for node in plan.nodes])

    def test_user_given_a_text_to_image_edit_when_planned_then_flux_is_the_selected_refiner_after_krea2(self) -> None:
        """User Given a request to change a generated still When the image-edit plan is built Then Krea2 creates the source and FLUX performs the requested reference edit."""
        goal = self.planner.create_goal(
            prompt="Keep Kirby's identity and paper-book style; open the closed seed pod into a bright flower.",
            media_type="text2img2img",
            duration_seconds=30,
            style="hand-drawn paper storybook animation",
            constraints={
                "image_workflow_name": "krea2_turbo",
                "refine_workflow_name": "flux3_image_klein_img2img",
            },
        )

        plan = self.planner.build_plan(goal)
        source = next(node for node in plan.nodes if node.node_id == "render-image")
        refine = next(node for node in plan.nodes if node.node_id == "refine-image")

        self.assertEqual(source.inputs["workflow_name"], "krea2_turbo")
        self.assertEqual(refine.inputs["workflow_name"], "flux3_image_klein_img2img")
        self.assertIn("render-image", refine.depends_on)


    def test_stage_probe_image_plan_keeps_candidates_and_auto_selects_one(self) -> None:
        goal = self.planner.create_goal(
            prompt="Kirby catches one visible news-derived object",
            media_type="image",
            duration_seconds=30,
            style="anime key visual",
            constraints={
                "workflow_name": "krea2_turbo",
                "image_count": 4,
                "review_selection_limit": 1,
                "stage_probe_auto_select": True,
            },
        )
        plan = self.planner.build_plan(goal)
        preview = next(node for node in plan.nodes if node.node_id == "stage-preview-select")

        self.assertEqual(preview.inputs["limit"], 1)
        self.assertTrue(preview.inputs["auto_select_for_probe"])
        self.assertFalse(preview.inputs["require_human_review"])
        self.assertEqual(preview.depends_on, ["render-image"])

    def test_stage_probe_text2img2img_refines_selected_candidate(self) -> None:
        goal = self.planner.create_goal(
            prompt="Kirby catches one visible news-derived object",
            media_type="text2img2img",
            duration_seconds=30,
            style="anime key visual",
            constraints={
                "image_workflow_name": "krea2_turbo",
                "refine_workflow_name": "krea2_turbo_img2img",
                "image_count": 4,
                "stage_probe_auto_select": True,
            },
        )
        plan = self.planner.build_plan(goal)
        preview = next(node for node in plan.nodes if node.node_id == "stage-preview-select")
        refine = next(node for node in plan.nodes if node.node_id == "refine-image")

        self.assertTrue(preview.inputs["auto_select_for_probe"])
        self.assertIn("stage-preview-select", refine.depends_on)

    def test_sticker_pack_plan_is_available(self) -> None:
        goal = self.planner.create_goal(
            prompt="cute reaction pack for rainy ramen shop",
            media_type="sticker_pack",
            duration_seconds=30,
            style="sticker illustration",
            constraints={"character": "Kirby"},
        )
        plan = self.planner.build_plan(goal)
        self.assertEqual(plan.workflow_name, "krea2_turbo")
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
            constraints={
                "character": "Kirby",
                "workflow_name": "krea2_turbo",
                "image_workflow_name": "krea2_turbo",
            },
        )
        plan = self.planner.build_plan(goal)
        render_node = next(node for node in plan.nodes if node.node_id == "render-stickers")

        self.assertEqual(plan.workflow_name, "krea2_turbo")
        self.assertEqual(plan.metadata["selected_workflow"], "krea2_turbo")
        self.assertEqual(render_node.inputs["workflow_name"], "krea2_turbo")

    def test_animated_sticker_plan_is_available(self) -> None:
        goal = self.planner.create_goal(
            prompt="kirby reaction sticker dancing in place",
            media_type="animated_sticker",
            duration_seconds=8,
            style="LINE sticker illustration",
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





    def test_carousel_plan_is_available(self) -> None:
        goal = self.planner.create_goal(
            prompt="robot chef travel diary in rainy taipei",
            media_type="carousel",
            duration_seconds=30,
            style="editorial anime postcard",
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
            constraints={},
        )
        plan = self.planner.build_plan(goal)

        self.assertIn("persist-text2img2video-summary", [node.node_id for node in plan.nodes])

    def test_user_given_a_five_second_i2v_request_when_planned_then_delivery_keeps_the_ten_second_minimum(self) -> None:
        """User Given a five-second I2V request When its plan is built Then the public duration is clamped to the ten-second delivery minimum."""
        goal = self.planner.create_goal(
            prompt="Kirby swats one glowing orb into a target",
            media_type="text2img2video",
            duration_seconds=5,
            style="anime key visual",
            constraints={"duration_override_seconds": 5, "video_frame_rate": 24},
        )
        plan = self.planner.build_plan(goal)
        animate = next(node for node in plan.nodes if node.node_id == "animate-video")
        self.assertEqual(goal.duration_seconds, 10)
        self.assertEqual(goal.constraints["duration_override_seconds"], 10)
        self.assertGreaterEqual(animate.inputs["length"] / 24, 10)

    def test_user_given_default_i2v_when_speed_is_2x_then_source_is_10_seconds_and_final_target_is_5(self) -> None:
        """User Given the default I2V route When speed is 2x Then it renders 10s before speed and targets 5s after."""
        goal = self.planner.create_goal(
            prompt="Kirby deflects one runaway toy scooter with a broom",
            media_type="text2img2video",
            duration_seconds=10,
            style="anime key visual",
            constraints={"video_speed": {"enabled": True, "factor": 2}},
        )

        plan = self.planner.build_plan(goal)

        animate = next(node for node in plan.nodes if node.node_id == "animate-video")
        speed = next(node for node in plan.nodes if node.node_id == "video-speed")
        qa = next(node for node in plan.nodes if node.node_id == "video-qa")
        self.assertEqual(animate.inputs["length"], 240)
        self.assertEqual(speed.inputs["speed"], 2.0)
        self.assertEqual(qa.inputs["target_duration"], 5.0)

    def test_native_h3_story_prompt_declares_first_frame_i2v(self) -> None:
        goal = self.planner.create_goal(
            prompt="Kirby protects one glowing orb from a sudden gust",
            media_type="native_h3_story",
            duration_seconds=15,
            style="anime key visual",
            constraints={"native_h3_storyboard_path": "configs/storyboards/native_h3_15s.yaml"},
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

    def test_pre_video_text2img2video_can_skip_unconsumed_upscale(self) -> None:
        goal = self.planner.create_goal(
            prompt="Kirby taps one glowing jelly cube",
            media_type="text2img2video",
            duration_seconds=5,
            style="anime key visual",
            constraints={
                "pre_video_review_enabled": True,
                "pre_video_candidate_count": 6,
                "pre_video_review_require_human": True,
                "skip_upscale_for_i2v": True,
            },
        )
        plan = self.planner.build_plan(goal)
        node_ids = [node.node_id for node in plan.nodes]
        review = next(node for node in plan.nodes if node.node_id == "stage-review-select")
        video_check = next(node for node in plan.nodes if node.node_id == "video-asset-check")

        self.assertNotIn("upscale-image", node_ids)
        self.assertEqual(review.depends_on, ["render-image"])
        self.assertEqual(video_check.depends_on, ["render-image"])

    def test_text2video_stage_review_gates_video_generation(self) -> None:
        goal = self.planner.create_goal(
            prompt="robot chef in rainy alley",
            media_type="text2video",
            duration_seconds=30,
            style="anime key visual",
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
            constraints={
                "pre_video_review_enabled": True,
                "pre_video_candidate_count": 6,
                "pre_video_review_require_human": True,
                "video_workflow_name": "minimax_h3_t2va",
            },
        )
        plan = self.planner.build_plan(goal)
        render = next(node for node in plan.nodes if node.node_id == "render-image")
        animate = next(node for node in plan.nodes if node.node_id == "animate-video")

        self.assertEqual(render.inputs["image_count"], 1)
        self.assertNotIn("stage-review-select", [node.node_id for node in plan.nodes])
        self.assertEqual(animate.inputs["workflow_name"], "minimax_h3_t2va")
        self.assertNotIn("stage-review-select", animate.depends_on)



    def test_text2img2video_plan_respects_stage_workflows_and_image_count(self) -> None:
        goal = self.planner.create_goal(
            prompt="kirby rainy ramen alley short",
            media_type="text2img2video",
            duration_seconds=20,
            style="anime key visual",
            constraints={
                "image_workflow_name": "krea2_turbo",
                "video_workflow_name": "minimax_h3_i2va",
                "upscale_workflow_name": "Tile Upscaler SDXL",
                "image_count": 3,
            },
        )
        plan = self.planner.build_plan(goal)
        render_node = next(node for node in plan.nodes if node.node_id == "render-image")
        video_check_node = next(node for node in plan.nodes if node.node_id == "video-asset-check")

        self.assertEqual(render_node.inputs["workflow_name"], "krea2_turbo")
        self.assertEqual(render_node.inputs["image_count"], 3)
        self.assertEqual(video_check_node.inputs["workflow_name"], "minimax_h3_i2va")

    def test_text2img2video_without_pre_video_review_renders_one_keyframe(self) -> None:
        goal = self.planner.create_goal(
            prompt="kirby rainy ramen alley short",
            media_type="text2img2video",
            duration_seconds=5,
            style="anime key visual",
            constraints={
                "pre_video_review_enabled": False,
                "image_count": 1,
            },
        )
        plan = self.planner.build_plan(goal)
        render_node = next(node for node in plan.nodes if node.node_id == "render-image")

        self.assertEqual(render_node.inputs["image_count"], 1)
        self.assertNotIn("stage-review-select", [node.node_id for node in plan.nodes])

    def test_text2img2video_can_skip_unconsumed_upscale_for_raw_i2v_frame(self) -> None:
        goal = self.planner.create_goal(
            prompt="kirby taps one jelly cube",
            media_type="text2img2video",
            duration_seconds=5,
            style="anime key visual",
            constraints={
                "pre_video_review_enabled": False,
                "image_count": 1,
                "skip_upscale_for_i2v": True,
            },
        )
        plan = self.planner.build_plan(goal)
        node_ids = [node.node_id for node in plan.nodes]
        video_check = next(node for node in plan.nodes if node.node_id == "video-asset-check")
        animate = next(node for node in plan.nodes if node.node_id == "animate-video")

        self.assertNotIn("upscale-image", node_ids)
        self.assertEqual(video_check.depends_on, ["render-image"])
        self.assertNotIn("upscale-image", animate.depends_on)

    def test_text2img2video_review_loop_can_skip_unconsumed_upscale(self) -> None:
        """User Given the route skips an unused upscale stage and review is enabled, When the graph is validated, Then all review and retry dependencies exist and retry I2V consumes the raw frame."""
        goal = self.planner.create_goal(
            prompt="Kirby taps one jelly cube",
            media_type="text2img2video",
            duration_seconds=5,
            style="anime key visual",
            constraints={
                "enable_review_loop": True,
                "skip_upscale_for_i2v": True,
            },
        )
        plan = self.planner.build_plan(goal)
        node_ids = {node.node_id for node in plan.nodes}
        retry_animate = next(node for node in plan.nodes if node.node_id == "review-animate-video")

        plan.as_graph().topological_order()

        self.assertNotIn("upscale-image", node_ids)
        self.assertNotIn("review-upscale-image", node_ids)
        self.assertIn("review-render-image", retry_animate.depends_on)

    def test_image_plan_render_node_depends_on_idea_brief(self) -> None:
        # Regression test: render-image must depend on idea-brief so that the
        # LLM-generated prompt is passed to ComfyUI instead of the raw user prompt.
        goal = self.planner.create_goal(
            prompt="kirby on a floating cloud",
            media_type="image",
            duration_seconds=0,
            style="anime key visual",
        )
        plan = self.planner.build_plan(goal)
        render_node = next(node for node in plan.nodes if node.node_id == "render-image")
        self.assertIn("idea-brief", render_node.depends_on)










    def test_user_given_an_accepted_segment_review_when_its_tail_is_extracted_then_the_accepted_video_is_used(self) -> None:
        """User Given Discord accepted a segment and exposed its media path, When the tail frame is extracted, Then the accepted video is sent to the media extraction tool."""
        accepted_video_path = r"C:\runs\segment_01.mp4"
        goal = GoalRequest(
            prompt="Continue the meadow story",
            media_type="long_video",
            style="anime",
        )
        review_node = ExecutionNode(
            node_id="segment-review-01",
            skill_name="review.assets.select",
            inputs={},
        )
        tail_node = ExecutionNode(
            node_id="segment-tail-01",
            skill_name="media.video.extract_last_frame",
            depends_on=[review_node.node_id],
            inputs={},
        )
        plan = ExecutionPlan(goal=goal, workflow_name="long_video", nodes=[review_node, tail_node])
        state = RunState(
            goal={"prompt": goal.prompt},
            metadata={},
            node_outputs={review_node.node_id: {"media_paths": [accepted_video_path]}},
        )
        context = SkillContext(plan=plan, node=tail_node, state=state)
        captured_payload: dict[str, object] = {}
        tools = ToolRegistry()

        def extract_last_frame(payload: dict[str, object]) -> dict[str, object]:
            captured_payload.update(payload)
            return {"frame_path": str(payload["output_path"])}

        tools.register(
            "media.extract_last_frame",
            extract_last_frame,
            "Extract the last frame from a video",
        )
        with TemporaryDirectory() as temp_dir:
            skills = AgentMediaSkills(tools, Path(temp_dir))
            result = skills.extract_last_frame(context)

        self.assertEqual(captured_payload["video_path"], accepted_video_path)
        self.assertEqual(result.outputs["frame_path"], captured_payload["output_path"])

    def test_user_given_a_segment_acceptance_review_when_its_message_is_prepared_then_the_tail_consequence_is_clear(self) -> None:
        """User Given a rendered segment is awaiting human acceptance, When its review message is prepared, Then the reviewer is told that acceptance enables continuation from the observed tail."""
        guidance = (
            "Accept only if this segment is suitable for the next segment to continue "
            "from its observed final frame. Reject to stop the sequence."
        )

        review_text = AgentSocialSkills._build_final_publish_review_text(
            draft_caption="Overall story brief",
            draft_hashtags="",
            review_guidance=guidance,
        )

        self.assertIn(guidance, review_text)
        self.assertIn("Overall story brief", review_text)

    def test_text2video_plan_respects_video_count(self) -> None:
        goal = self.planner.create_goal(
            prompt="robot chef in rainy alley",
            media_type="text2video",
            duration_seconds=30,
            style="anime key visual",
            constraints={
                "image_workflow_name": "nova-anime-xl",
                "video_workflow_name": "minimax_h3_t2va",
                "video_count": 3,
            },
        )
        plan = self.planner.build_plan(goal)
        animate_node = next(node for node in plan.nodes if node.node_id == "animate-video")

        self.assertEqual(animate_node.inputs["workflow_name"], "minimax_h3_t2va")
        self.assertEqual(animate_node.inputs["video_count"], 3)


if __name__ == "__main__":
    unittest.main()
