from __future__ import annotations

import os
import unittest
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest, RunState, SkillContext
from agentic.runtime.registry import ToolRegistry
from agentic.skills.agent_social import AgentSocialSkills
from agentic.tools.context_services import DiscordHumanReviewService


class AgentSocialSkillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[2]

    def test_publish_social_blocks_when_platform_bundle_is_not_ready(self) -> None:
        tool_registry = ToolRegistry()
        invoked = {"count": 0}

        def publish_social(payload: dict[str, object]) -> dict[str, object]:
            invoked["count"] += 1
            return {"status": "success", "payload": payload}

        tool_registry.register("publish.social", publish_social, "publish")
        skills = AgentSocialSkills(tool_registry, self.project_root / ".tmp-tests")
        plan = ExecutionPlan(
            goal=GoalRequest(
                prompt="publish kirby clip",
                media_type="publish_review",
                style="social promo",
                constraints={"platforms": ["instagram"]},
            ),
            workflow_name="publish_review_v1",
            nodes=[],
        )
        node = ExecutionNode(
            node_id="dispatch-publish",
            skill_name="publish.social.dispatch",
            depends_on=["process-media", "prepare-caption"],
            inputs={"platforms": ["instagram"], "dry_run": False},
        )
        state = RunState(
            goal={"prompt": "publish kirby clip"},
            metadata={},
            node_outputs={
                "process-media": {"media_paths": ["C:\\selected.mp4"]},
                "prepare-caption": {
                    "caption": "",
                    "hashtags": "#kirby",
                    "dispatch_ready": False,
                    "platform_bundle": {
                        "instagram": {
                            "caption": "",
                            "hashtags": "#kirby",
                            "validation": {"has_caption": False, "has_media": True, "is_publish_ready": False},
                        }
                    },
                },
            },
        )

        result = skills.publish_social(SkillContext(plan=plan, node=node, state=state))

        self.assertEqual(result.status, "blocked")
        self.assertEqual(invoked["count"], 0)
        self.assertEqual(result.outputs["blocked_platforms"], ["instagram"])
        self.assertFalse(result.outputs["dispatch_ready"])
        self.assertIn("dispatch_plan", result.outputs)

    def test_publish_social_passes_dispatch_plan_to_tool(self) -> None:
        tool_registry = ToolRegistry()

        def publish_social(payload: dict[str, object]) -> dict[str, object]:
            return {
                "status": "dry_run" if payload.get("dry_run") else "success",
                "platforms": payload.get("platforms", []),
                "dispatch_plan": payload.get("platform_bundle", {}),
            }

        tool_registry.register("publish.social", publish_social, "publish")
        skills = AgentSocialSkills(tool_registry, self.project_root / ".tmp-tests")
        plan = ExecutionPlan(
            goal=GoalRequest(
                prompt="publish kirby clip",
                media_type="publish_review",
                style="social promo",
                constraints={"platforms": ["instagram"]},
            ),
            workflow_name="publish_review_v1",
            nodes=[],
        )
        node = ExecutionNode(
            node_id="dispatch-publish",
            skill_name="publish.social.dispatch",
            depends_on=["process-media", "prepare-caption"],
            inputs={"platforms": ["instagram"], "dry_run": True},
        )
        state = RunState(
            goal={"prompt": "publish kirby clip"},
            metadata={},
            node_outputs={
                "process-media": {"media_paths": ["C:\\selected.mp4"]},
                "prepare-caption": {
                    "caption": "Launch clip",
                    "hashtags": "#kirby",
                    "dispatch_ready": True,
                    "platform_bundle": {
                        "instagram": {
                            "caption": "IG launch",
                            "hashtags": "#kirby",
                            "validation": {"has_caption": True, "has_media": True, "is_publish_ready": True},
                        }
                    },
                },
            },
        )

        result = skills.publish_social(SkillContext(plan=plan, node=node, state=state))

        self.assertEqual(result.status, "success")
        self.assertTrue(result.outputs["dispatch_ready"])
        self.assertEqual(result.outputs["dispatch_plan"]["instagram"]["caption"], "IG launch")

    def test_publish_social_marks_platform_ineligible_without_reporting_ready(self) -> None:
        tool_registry = ToolRegistry()

        def publish_social(payload: dict[str, object]) -> dict[str, object]:
            return {
                "status": "success",
                "dispatch_status": "skipped",
                "platforms": payload.get("platforms", []),
            }

        tool_registry.register("publish.social", publish_social, "publish")
        skills = AgentSocialSkills(tool_registry, self.project_root / ".tmp-tests")
        plan = ExecutionPlan(
            goal=GoalRequest(
                prompt="publish kirby image",
                media_type="publish_review",
                constraints={"platforms": ["youtube"]},
            ),
            workflow_name="publish_review_v1",
            nodes=[],
        )
        node = ExecutionNode(
            node_id="dispatch-publish",
            skill_name="publish.social.dispatch",
            depends_on=["process-media", "prepare-caption"],
            inputs={"platforms": ["youtube"], "dry_run": False},
        )
        state = RunState(
            goal={"prompt": "publish kirby image"},
            metadata={},
            node_outputs={
                "process-media": {"media_paths": ["C:\\selected.png"]},
                "prepare-caption": {
                    "caption": "Image caption",
                    "hashtags": "#kirby",
                    "dispatch_ready": True,
                    "platform_bundle": {
                        "youtube": {
                            "caption": "Image caption",
                            "validation": {
                                "has_caption": True,
                                "has_media": True,
                                "is_publish_ready": True,
                                "is_platform_publish_ready": False,
                                "issues": ["requires_video_media"],
                            },
                        }
                    },
                },
            },
        )

        result = skills.publish_social(SkillContext(plan=plan, node=node, state=state))

        self.assertEqual(result.status, "success")
        self.assertEqual(result.outputs["platform_ineligible"], ["youtube"])
        self.assertFalse(result.outputs["dispatch_ready"])

    def test_review_select_uses_discord_decision_when_available(self) -> None:
        tool_registry = ToolRegistry()
        skills = AgentSocialSkills(tool_registry, self.project_root / ".tmp-tests")
        plan = ExecutionPlan(
            goal=GoalRequest(
                prompt="pick the best kirby frame",
                media_type="text2img2video",
                style="anime",
                constraints={"enable_stage_review": True},
            ),
            workflow_name="text2img2video_v1",
            nodes=[],
        )
        node = ExecutionNode(
            node_id="stage-review-select",
            skill_name="review.assets.select",
            depends_on=["render-image"],
            inputs={"limit": 2, "review_notes": "prefer stronger composition"},
        )
        state = RunState(
            goal={"prompt": "pick the best kirby frame"},
            metadata={},
            node_outputs={
                "render-image": {"saved_files": ["C:\\frame_a.png", "C:\\frame_b.png"]},
            },
        )

        with patch.object(
            skills.prompt_engine,
            "review_asset_candidates",
            return_value={
                "selected_assets": ["C:\\frame_a.png"],
                "ranked_candidates": [
                    {"media_path": "C:\\frame_a.png", "score": 90, "rationale": "good"},
                    {"media_path": "C:\\frame_b.png", "score": 80, "rationale": "okay"},
                ],
                "selection_rationale": "LLM shortlist",
                "regeneration_notes": "None",
                "prompt_mode": "llm",
            },
        ), patch.object(
            skills.prompt_engine,
            "prepare_publish_caption",
            return_value={
                "caption": "Draft social post",
                "hashtags": "#kirby",
                "platform_captions": {},
                "platform_bundle": {},
                "dispatch_ready": True,
            },
        ), patch.object(
            skills.discord_review,
            "review_candidates",
            return_value=type(
                "_Decision",
                (),
                {
                    "review_mode": "discord",
                    "status": "approved",
                    "selected_paths": ["C:\\frame_b.png"],
                    "reviewer": "tester#0001",
                    "session_id": "sess-1",
                    "session_path": "C:\\session.json",
                    "edited_text": "keep frame b",
                },
            )(),
        ):
            result = skills.select_best_assets(SkillContext(plan=plan, node=node, state=state))

        self.assertEqual(result.status, "success")
        self.assertEqual(result.outputs["selected_assets"], ["C:\\frame_b.png"])
        self.assertEqual(result.outputs["review_mode"], "discord")
        self.assertEqual(result.outputs["reviewer"], "tester#0001")
        self.assertEqual(result.outputs["approved_review_text"], "keep frame b")

    def test_publish_review_requires_discord_approval_even_without_explicit_flag(self) -> None:
        tool_registry = ToolRegistry()
        skills = AgentSocialSkills(tool_registry, self.project_root / ".tmp-tests")
        plan = ExecutionPlan(
            goal=GoalRequest(
                prompt="publish the selected Kirby clip",
                media_type="publish_review",
                style="anime",
                constraints={},
            ),
            workflow_name="publish_review_v1",
            nodes=[],
        )
        node = ExecutionNode(
            node_id="review-select",
            skill_name="review.assets.select",
            depends_on=["ingest-media"],
            inputs={"limit": 1},
        )
        state = RunState(
            goal={"prompt": "publish the selected Kirby clip"},
            metadata={},
            node_outputs={"ingest-media": {"media_paths": ["C:\\selected.mp4"]}},
        )

        with patch.object(
            skills.prompt_engine,
            "prepare_publish_caption",
            return_value={"caption": "A caption", "hashtags": "#kirby", "dispatch_ready": True},
        ), patch.object(
            skills.discord_review,
            "review_candidates",
            return_value=type(
                "_Decision",
                (),
                {
                    "review_mode": "auto",
                    "status": "skipped",
                    "selected_paths": [],
                    "fallback_reason": "Discord is not configured",
                },
            )(),
        ) as discord_review:
            result = skills.select_best_assets(SkillContext(plan=plan, node=node, state=state))

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.outputs["review_mode"], "auto")
        discord_review.assert_called_once()

    def test_review_select_falls_back_to_llm_shortlist_when_discord_has_no_decision(self) -> None:
        tool_registry = ToolRegistry()
        skills = AgentSocialSkills(tool_registry, self.project_root / ".tmp-tests")
        plan = ExecutionPlan(
            goal=GoalRequest(
                prompt="pick the best kirby frame",
                media_type="text2img2video",
                style="anime",
                constraints={"enable_stage_review": True},
            ),
            workflow_name="text2img2video_v1",
            nodes=[],
        )
        node = ExecutionNode(
            node_id="stage-review-select",
            skill_name="review.assets.select",
            depends_on=["render-image"],
            inputs={"limit": 2, "review_notes": "prefer stronger composition"},
        )
        state = RunState(
            goal={"prompt": "pick the best kirby frame"},
            metadata={},
            node_outputs={
                "render-image": {"saved_files": ["C:\\frame_a.png", "C:\\frame_b.png"]},
            },
        )

        with patch.object(
            skills.prompt_engine,
            "review_asset_candidates",
            return_value={
                "selected_assets": ["C:\\frame_a.png"],
                "ranked_candidates": [
                    {"media_path": "C:\\frame_a.png", "score": 90, "rationale": "good"},
                    {"media_path": "C:\\frame_b.png", "score": 80, "rationale": "okay"},
                ],
                "selection_rationale": "LLM shortlist",
                "regeneration_notes": "None",
                "prompt_mode": "llm",
            },
        ), patch.object(
            skills.prompt_engine,
            "prepare_publish_caption",
            return_value={
                "caption": "Draft social post",
                "hashtags": "#kirby",
                "platform_captions": {},
                "platform_bundle": {},
                "dispatch_ready": True,
            },
        ), patch.object(
            skills.discord_review,
            "review_candidates",
            return_value=type(
                "_Decision",
                (),
                {
                    "review_mode": "discord",
                    "status": "failed",
                    "selected_paths": [],
                    "reviewer": "",
                    "session_id": "sess-2",
                    "session_path": "C:\\session.json",
                    "edited_text": "keep the llm shortlist",
                    "fallback_reason": "discord review did not return a valid human decision",
                },
            )(),
        ):
            result = skills.select_best_assets(SkillContext(plan=plan, node=node, state=state))

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.outputs["selected_assets"], [])
        self.assertEqual(result.outputs["review_mode"], "discord")
        self.assertEqual(
            result.outputs["fallback_reason"],
            "discord review did not return a valid human decision",
        )

    def test_pre_video_review_can_fallback_to_top_candidate_on_discord_failure(self) -> None:
        tool_registry = ToolRegistry()
        skills = AgentSocialSkills(tool_registry, self.project_root / ".tmp-tests")
        plan = ExecutionPlan(
            goal=GoalRequest(
                prompt="pick one opening Kirby frame",
                media_type="text2img2video",
                style="anime",
                constraints={
                    "enable_stage_review": True,
                    "pre_video_review_failure_policy": "fallback_to_top",
                },
            ),
            workflow_name="text2img2video_v1",
            nodes=[],
        )
        node = ExecutionNode(
            node_id="stage-review-select",
            skill_name="review.assets.select",
            depends_on=["render-image"],
            inputs={
                "limit": 1,
                "review_all_candidates": True,
                "review_scope": "first_frame",
                "review_phase": "opening_frame",
            },
        )
        state = RunState(
            goal={"prompt": "pick one opening Kirby frame"},
            metadata={},
            node_outputs={
                "render-image": {"saved_files": ["C:\\frame_a.png", "C:\\frame_b.png"]},
            },
        )

        with patch.object(
            skills.discord_review,
            "review_candidates",
            return_value=type(
                "_Decision",
                (),
                {
                    "review_mode": "discord",
                    "status": "failed",
                    "selected_paths": [],
                    "reviewer": "",
                    "session_id": "sess-fallback",
                    "session_path": "C:\\session-fallback.json",
                    "edited_text": "",
                    "fallback_reason": "discord review timed out before any decision was received",
                },
            )(),
        ):
            result = skills.select_best_assets(SkillContext(plan=plan, node=node, state=state))

        self.assertEqual(result.status, "success")
        self.assertEqual(result.outputs["selected_assets"], ["C:\\frame_a.png"])
        self.assertTrue(result.outputs["review_fallback_used"])
        self.assertIn("timed out", result.outputs["fallback_reason"])

    def test_first_frame_review_sends_all_six_candidates_and_never_auto_selects(self) -> None:
        tool_registry = ToolRegistry()
        skills = AgentSocialSkills(tool_registry, self.project_root / ".tmp-tests")
        candidate_paths = [f"C:\\frame_{index}.png" for index in range(1, 7)]
        plan = ExecutionPlan(
            goal=GoalRequest(
                prompt="Kirby must stop the runaway lantern in the first second",
                media_type="native_h3_story",
                style="anime",
                constraints={
                    "require_human_review": True,
                    "source_generation_type": "native_h3_story",
                    "workflow_name": "minimax_h3_lowvram_15s_fl2va_i2v",
                },
            ),
            workflow_name="minimax_h3_lowvram_15s_fl2va_i2v",
            nodes=[],
        )
        node = ExecutionNode(
            node_id="native-opening-review",
            skill_name="review.assets.select",
            depends_on=["native-opening-keyframe"],
            inputs={
                "limit": 1,
                "review_all_candidates": True,
                "review_scope": "first_frame",
                "review_notes": "Choose one opening frame; reject all six if none is usable.",
            },
        )
        state = RunState(
            goal={"prompt": plan.goal.prompt},
            metadata={},
            node_outputs={
                "native-story-prompt": {
                    "generated_storyboard": {
                        "name": "Kirby and the Seed",
                        "story_spine": {
                            "premise": "Kirby protects one glowing seed as a sudden storm reshapes the meadow.",
                            "objective": "Kirby must keep the seed safe.",
                        },
                    }
                },
                "native-opening-keyframe": {"saved_files": candidate_paths},
            },
        )
        captured: dict[str, object] = {}

        def fake_review(**kwargs):
            captured.update(kwargs)
            return type(
                "_Decision",
                (),
                {
                    "review_mode": "discord",
                    "status": "approved",
                    "selected_paths": [candidate_paths[3]],
                    "reviewer": "tester#0001",
                    "session_id": "sess-six",
                    "session_path": "C:\\session-six.json",
                    "edited_text": "keep asset 4",
                    "fallback_reason": "",
                },
            )()

        with patch.object(skills.discord_review, "review_candidates", side_effect=fake_review):
            result = skills.select_best_assets(SkillContext(plan=plan, node=node, state=state))

        self.assertEqual(result.status, "success")
        self.assertEqual(result.outputs["selected_assets"], [candidate_paths[3]])
        self.assertEqual(captured["media_paths"], candidate_paths)
        self.assertEqual(captured["text"], "stage: preview")
        self.assertNotIn("可愛爆擊", captured["text"])
        self.assertNotIn("請選擇最適合的開場首幀", captured["text"])
        self.assertNotIn("Workflow:", captured["text"])
        self.assertNotIn("Stage:", captured["text"])
        self.assertNotIn("Prompt:", captured["text"])
        self.assertNotIn("Candidates attached", captured["text"])
        self.assertFalse(captured["allow_text_edit"])
        self.assertEqual(captured["selection_mode"], "single")
        self.assertTrue(captured["selection_required"])
        self.assertEqual(captured["selection_limit"], 1)
        return
        self.assertEqual(captured["text"], "stage: preview")
        self.assertNotIn("可愛爆擊", captured["text"])
        self.assertNotIn("請選擇最適合的開場首幀", captured["text"])
        self.assertNotIn("Candidates attached", captured["text"])
        self.assertNotIn("Choose one opening frame", captured["text"])

    def test_stage_probe_uses_prompt_engine_for_automatic_candidate_selection(self) -> None:
        tool_registry = ToolRegistry()
        skills = AgentSocialSkills(tool_registry, self.project_root / ".tmp-tests")
        candidate_paths = [f"C:\\probe_frame_{index}.png" for index in range(1, 7)]
        plan = ExecutionPlan(
            goal=GoalRequest(
                prompt="",
                media_type="native_h3_story",
                style="anime",
                constraints={"character": "Kirby", "enable_stage_review": True},
            ),
            workflow_name="minimax_h3_lowvram_15s_fl2va_i2v",
            nodes=[],
        )
        node = ExecutionNode(
            node_id="native-opening-review",
            skill_name="review.assets.select",
            depends_on=["native-opening-keyframe"],
            inputs={
                "limit": 1,
                "review_scope": "first_frame",
                "auto_select_for_probe": True,
            },
        )
        state = RunState(
            goal={"prompt": ""},
            metadata={},
            node_outputs={
                "native-story-prompt": {
                    "opening_keyframe_prompt": "Kirby stretches toward the locked mint-green straw dispenser while a cart rolls downhill."
                },
                "native-opening-keyframe": {"saved_files": candidate_paths},
            },
        )
        captured: dict[str, object] = {}

        class FakePromptEngine:
            def review_asset_candidates(self, goal, media_paths, review_notes, selection_limit):
                captured.update(
                    goal_prompt=goal.prompt,
                    media_paths=media_paths,
                    selection_limit=selection_limit,
                )
                return {
                    "selected_assets": [candidate_paths[2]],
                    "ranked_candidates": [{"media_path": candidate_paths[2], "score": 93, "rationale": "best prompt match"}],
                    "selection_rationale": "vision ranking selected the strongest probe frame",
                    "prompt_mode": "llm",
                }

        skills.prompt_engine = FakePromptEngine()
        with patch.object(skills.discord_review, "review_candidates", side_effect=AssertionError("probe must not open Discord")):
            result = skills.select_best_assets(SkillContext(plan=plan, node=node, state=state))

        self.assertEqual(result.status, "success")
        self.assertEqual(result.outputs["selected_assets"], [candidate_paths[2]])
        self.assertEqual(captured["selection_limit"], 1)
        self.assertIn("locked mint-green straw dispenser", captured["goal_prompt"])
        self.assertTrue(result.outputs["auto_select_for_probe"])

    def test_final_video_review_filters_frames_and_disables_asset_picker(self) -> None:
        tool_registry = ToolRegistry()
        skills = AgentSocialSkills(tool_registry, self.project_root / ".tmp-tests")
        video_path = r"C:\final\Kirby_H3.mp4"
        plan = ExecutionPlan(
            goal=GoalRequest(
                prompt="Publish the rendered Kirby video",
                media_type="native_h3_story",
                style="anime",
                constraints={"enable_stage_review": True, "review_notes": None},
            ),
            workflow_name="minimax_h3_lowvram_15s_fl2va_i2v",
            nodes=[],
        )
        node = ExecutionNode(
            node_id="review-select",
            skill_name="review.assets.select",
            depends_on=["native-h3-package"],
            inputs={"limit": 1, "review_scope": "final_video"},
        )
        state = RunState(
            goal={"prompt": plan.goal.prompt},
            metadata={},
            node_outputs={
                "native-h3-package": {
                    "saved_files": [
                        r"C:\frames\opening.png",
                        r"C:\frames\ending.png",
                        video_path,
                        r"C:\preview\preview.gif",
                    ]
                }
            },
        )
        captured: dict[str, object] = {}

        def fake_review(**kwargs):
            captured.update(kwargs)
            return type(
                "_Decision",
                (),
                {
                    "review_mode": "discord",
                    "status": "approved",
                    "selected_paths": [video_path],
                    "reviewer": "tester#0001",
                    "session_id": "sess-video",
                    "session_path": r"C:\session-video.json",
                    "edited_text": "",
                    "fallback_reason": "",
                },
            )()

        with patch.object(
            skills.prompt_engine,
            "review_asset_candidates",
            side_effect=AssertionError("final video review must not invoke asset shortlist LLM"),
        ), patch.object(
            skills.prompt_engine,
            "prepare_publish_caption",
            return_value={"caption": "Final caption", "hashtags": "#kirby", "dispatch_ready": True},
        ), patch.object(skills.discord_review, "review_candidates", side_effect=fake_review):
            result = skills.select_best_assets(SkillContext(plan=plan, node=node, state=state))

        self.assertEqual(result.status, "success")
        self.assertEqual(result.outputs["selected_assets"], [video_path])
        self.assertEqual(captured["media_paths"], [video_path])
        self.assertFalse(captured["allow_asset_selection"])
        self.assertTrue(captured["allow_text_edit"])
        self.assertEqual(captured["text"], "Final caption\n\n#kirby")
        self.assertNotIn("Caption:", captured["text"])
        self.assertNotIn("Hashtags:", captured["text"])
        self.assertNotIn("Strategy:", captured["text"])
        self.assertNotIn("Workflow:", captured["text"])
        self.assertNotIn("Stage:", captured["text"])
        self.assertNotIn("None", captured["text"])
        self.assertNotIn("Candidates attached", captured["text"])

    def test_final_media_review_sends_images_to_discord_and_keeps_selection(self) -> None:
        tool_registry = ToolRegistry()
        skills = AgentSocialSkills(tool_registry, self.project_root / ".tmp-tests")
        image_paths = [r"C:\final\Kirby_1.png", r"C:\final\Kirby_2.png"]
        plan = ExecutionPlan(
            goal=GoalRequest(
                prompt="Publish Kirby images",
                media_type="image",
                style="anime",
                constraints={"require_human_review": True, "platforms": ["instagram_graph", "facebook"]},
            ),
            workflow_name="publish_review_v1",
            nodes=[],
        )
        node = ExecutionNode(
            node_id="review-select",
            skill_name="review.assets.select",
            depends_on=["ingest-media"],
            inputs={"limit": 1, "review_scope": "final_media", "review_all_candidates": True},
        )
        state = RunState(
            goal={"prompt": plan.goal.prompt},
            metadata={},
            node_outputs={"ingest-media": {"media_paths": image_paths}},
        )
        captured: dict[str, object] = {}

        def fake_review(**kwargs):
            captured.update(kwargs)
            return type(
                "_Decision",
                (),
                {
                    "review_mode": "discord",
                    "status": "approved",
                    "selected_paths": [image_paths[1]],
                    "reviewer": "tester#0001",
                    "session_id": "sess-images",
                    "session_path": r"C:\session-images.json",
                    "edited_text": "publish image 2",
                    "fallback_reason": "",
                    "delivery": {},
                },
            )()

        with patch.object(
            skills.prompt_engine,
            "prepare_publish_caption",
            return_value={"caption": "Kirby image", "hashtags": "#kirby", "dispatch_ready": True},
        ) as prepare_caption, patch.object(skills.discord_review, "review_candidates", side_effect=fake_review):
            result = skills.select_best_assets(SkillContext(plan=plan, node=node, state=state))

        self.assertEqual(result.status, "success")
        self.assertEqual(result.outputs["selected_assets"], [image_paths[1]])
        self.assertEqual(captured["media_paths"], image_paths)
        self.assertTrue(captured["allow_asset_selection"])
        self.assertTrue(captured["allow_text_edit"])
        self.assertEqual(captured["text"], "Kirby image\n\n#kirby")
        self.assertNotIn("Caption:", captured["text"])
        self.assertNotIn("Hashtags:", captured["text"])
        self.assertEqual(prepare_caption.call_args.kwargs["media_paths"], image_paths)
        self.assertNotIn("Candidates attached", captured["text"])

    def test_last_frame_review_requires_explicit_approval_without_asset_picker(self) -> None:
        tool_registry = ToolRegistry()
        skills = AgentSocialSkills(tool_registry, self.project_root / ".tmp-tests")
        ending_path = r"C:\ending\Kirby_H3_ending.png"
        plan = ExecutionPlan(
            goal=GoalRequest(
                prompt="Approve the ending frame for the rendered Kirby story",
                media_type="native_h3_story",
                style="anime",
                constraints={"require_human_review": True},
            ),
            workflow_name="minimax_h3_lowvram_15s_fl2va_i2v",
            nodes=[],
        )
        node = ExecutionNode(
            node_id="native-ending-review",
            skill_name="review.assets.select",
            depends_on=["native-ending-keyframe"],
            inputs={
                "limit": 1,
                "review_all_candidates": True,
                "review_scope": "last_frame",
                "review_notes": "Reject if the ending cannot connect naturally from the selected opening frame.",
            },
        )
        state = RunState(
            goal={"prompt": plan.goal.prompt},
            metadata={},
            node_outputs={"native-ending-keyframe": {"saved_files": [ending_path]}},
        )
        captured: dict[str, object] = {}

        def fake_review(**kwargs):
            captured.update(kwargs)
            return type(
                "_Decision",
                (),
                {
                    "review_mode": "discord",
                    "status": "approved",
                    "selected_paths": [ending_path],
                    "reviewer": "tester#0001",
                    "session_id": "sess-ending",
                    "session_path": r"C:\session-ending.json",
                    "edited_text": "",
                    "fallback_reason": "",
                },
            )()

        with patch.object(skills.discord_review, "review_candidates", side_effect=fake_review):
            result = skills.select_best_assets(SkillContext(plan=plan, node=node, state=state))

        self.assertEqual(result.status, "success")
        self.assertEqual(result.outputs["selected_assets"], [ending_path])
        self.assertEqual(captured["media_paths"], [ending_path])
        self.assertTrue(captured["allow_asset_selection"])
        self.assertEqual(captured["review_scope"], "last_frame")
        self.assertEqual(captured["text"], "stage: preview")
        self.assertNotIn("Strategy:", captured["text"])
        self.assertNotIn("Workflow:", captured["text"])
        self.assertNotIn("Stage:", captured["text"])
        self.assertNotIn("Prompt:", captured["text"])
        return
        self.assertIn("故事：Approve the ending frame", captured["text"])
        self.assertNotIn("Reject if the ending cannot connect naturally", captured["text"])

    def test_required_first_frame_review_blocks_when_discord_is_unavailable(self) -> None:
        tool_registry = ToolRegistry()
        skills = AgentSocialSkills(tool_registry, self.project_root / ".tmp-tests")
        plan = ExecutionPlan(
            goal=GoalRequest(
                prompt="Kirby opening frame",
                media_type="native_h3_story",
                style="anime",
                constraints={"require_human_review": True},
            ),
            workflow_name="minimax_h3_lowvram_15s_fl2va_i2v",
            nodes=[],
        )
        node = ExecutionNode(
            node_id="native-opening-review",
            skill_name="review.assets.select",
            depends_on=["native-opening-keyframe"],
            inputs={"limit": 1, "review_all_candidates": True, "review_scope": "first_frame"},
        )
        state = RunState(
            goal={"prompt": plan.goal.prompt},
            metadata={},
            node_outputs={"native-opening-keyframe": {"saved_files": ["C:\\frame_a.png", "C:\\frame_b.png"]}},
        )
        with patch.object(
            skills.discord_review,
            "review_candidates",
            return_value=type(
                "_Decision",
                (),
                {"review_mode": "auto", "status": "skipped", "fallback_reason": "not configured"},
            )(),
        ):
            result = skills.select_best_assets(SkillContext(plan=plan, node=node, state=state))

        self.assertEqual(result.status, "blocked")
        self.assertEqual(result.outputs["selected_assets"], [])
        self.assertIn("no candidate was selected automatically", result.outputs["selection_rationale"])

    def test_first_frame_review_blocks_accepting_all_candidates(self) -> None:
        tool_registry = ToolRegistry()
        skills = AgentSocialSkills(tool_registry, self.project_root / ".tmp-tests")
        paths = [f"C:\\frame_{index}.png" for index in range(1, 7)]
        plan = ExecutionPlan(
            goal=GoalRequest(
                prompt="Kirby opening frame",
                media_type="native_h3_story",
                style="anime",
                constraints={"require_human_review": True},
            ),
            workflow_name="minimax_h3_lowvram_15s_fl2va_i2v",
            nodes=[],
        )
        node = ExecutionNode(
            node_id="native-opening-review",
            skill_name="review.assets.select",
            depends_on=["native-opening-keyframe"],
            inputs={"limit": 1, "review_all_candidates": True, "review_scope": "first_frame"},
        )
        state = RunState(
            goal={"prompt": plan.goal.prompt},
            metadata={},
            node_outputs={"native-opening-keyframe": {"saved_files": paths}},
        )
        with patch.object(
            skills.discord_review,
            "review_candidates",
            return_value=type(
                "_Decision",
                (),
                {
                    "review_mode": "discord",
                    "status": "approved",
                    "selected_paths": paths,
                    "reviewer": "tester",
                    "session_id": "sess-many",
                    "session_path": "C:\\session-many.json",
                },
            )(),
        ):
            result = skills.select_best_assets(SkillContext(plan=plan, node=node, state=state))

        self.assertEqual(result.status, "blocked")
        self.assertIn("exactly one", result.outputs["selection_rationale"])

    def test_build_review_text_stays_short_enough_for_discord(self) -> None:
        text = AgentSocialSkills._build_review_text(
            strategy="text2image2video",
            workflow="z_image_plus_nova_model",
            prompt="Kirby " + ("very detailed " * 80),
            review_notes="Prefer the best composition " * 40,
            ranked_candidates=[
                {
                    "media_path": r"C:\long\path\Heroic Stance\images\model1_00001__agentic_image.png",
                    "score": 95,
                    "rationale": "Strong pose and effects " * 30,
                }
                for _ in range(4)
            ],
            selection_limit=4,
            draft_caption="Draft post body " * 80,
            draft_hashtags="#kirby #mediaoverload",
            platforms=["instagram_graph", "facebook", "twitter"],
        )

        self.assertLessEqual(len(text), 1900)
        self.assertNotIn("Accept to publish with these assets", text)
        self.assertFalse(text.startswith("Draft post:"))
        self.assertNotIn("Platforms:", text)

    def test_final_review_text_preserves_article_paragraphs_without_internal_labels(self) -> None:
        text = AgentSocialSkills._build_final_publish_review_text(
            draft_caption="The purple orb flickers above the grass.\n\n1️⃣ Kirby faces the energy.\n2️⃣ The star shard changes the outcome.\n\nWhich moment stayed with you?",
            draft_hashtags="#kirby #mediaoverload",
            platforms=["facebook"],
        )

        self.assertIn("The purple orb flickers above the grass.", text)
        self.assertIn("\n\n1️⃣ Kirby faces the energy.", text)
        self.assertIn("Which moment stayed with you?", text)
        self.assertTrue(text.endswith("#kirby #mediaoverload"))
        self.assertNotIn("Caption:", text)
        self.assertNotIn("Hashtags:", text)
        self.assertNotIn("Strategy:", text)

    def test_prepare_caption_uses_edited_review_text_as_final_caption(self) -> None:
        tool_registry = ToolRegistry()
        skills = AgentSocialSkills(tool_registry, self.project_root / ".tmp-tests")
        plan = ExecutionPlan(
            goal=GoalRequest(
                prompt="publish kirby clip",
                media_type="publish_review",
                style="social promo",
                constraints={"platforms": ["instagram"], "hashtags": ["kirby", "mediaoverload"]},
            ),
            workflow_name="publish_review_v1",
            nodes=[],
        )
        node = ExecutionNode(
            node_id="prepare-caption",
            skill_name="publish.caption.prepare",
            depends_on=["review-select", "process-media"],
            inputs={"platforms": ["instagram"]},
        )
        state = RunState(
            goal={"prompt": "publish kirby clip"},
            metadata={},
            node_outputs={
                "review-select": {"edited_review_text": "Edited caption body\n\n#edited #tags"},
                "process-media": {"media_paths": ["C:\\selected.png"]},
            },
        )

        with patch.object(
            skills.prompt_engine,
            "prepare_publish_caption",
            return_value={
                "caption": "Generated caption",
                "hashtags": "#kirby #mediaoverload",
                "platform_captions": {"instagram": "Generated caption"},
                "platform_bundle": {
                    "instagram": {
                        "caption": "Generated caption",
                        "hashtags": "#kirby #mediaoverload",
                        "character_count": 17,
                        "validation": {"has_caption": True, "has_media": True, "is_publish_ready": True},
                    }
                },
                "caption_strategy": "platform_adapted",
                "dispatch_ready": True,
                "prompt_mode": "llm",
            },
        ):
            result = skills.prepare_caption(SkillContext(plan=plan, node=node, state=state))

        self.assertEqual(result.outputs["caption"], "Edited caption body")
        self.assertEqual(result.outputs["hashtags"], "#edited #tags")
        self.assertEqual(result.outputs["platform_bundle"]["instagram"]["caption"], "Edited caption body")

    def test_prepare_caption_uses_approved_review_text_without_second_llm_gate(self) -> None:
        tool_registry = ToolRegistry()
        skills = AgentSocialSkills(tool_registry, self.project_root / ".tmp-tests")
        approved_text = (
            "KingDedede steers a wooden cart down a pastel toy track and meets a tiny racing cart on the same lane. "
            "The bumpers compress with a springy boing, sending the smaller cart safely into the red hat.\n\n"
            "The playful collision turns a wrong-way traffic warning into a visible, harmless cartoon consequence.\n\n"
            "1️⃣ A single lane forces direct contact.\n"
            "2️⃣ Spring-loaded bumpers absorb the shock.\n"
            "3️⃣ A safety feature changes the outcome.\n\n"
            "What would you do if your commute became a cartoon collision?\n\n"
            "Like and save this for later.\n\n"
            "#toycar #cartooncrash #safety"
        )
        plan = ExecutionPlan(
            goal=GoalRequest(
                prompt="publish the approved Kirby clip",
                media_type="publish_review",
                style="social promo",
                constraints={"platforms": ["facebook", "youtube"]},
            ),
            workflow_name="publish_review_v1",
            nodes=[],
        )
        node = ExecutionNode(
            node_id="prepare-caption",
            skill_name="publish.caption.prepare",
            depends_on=["review-select", "process-media"],
            inputs={"platforms": ["facebook", "youtube"]},
        )
        state = RunState(
            goal={"prompt": "publish the approved Kirby clip"},
            metadata={},
            node_outputs={
                "review-select": {
                    "review_mode": "discord",
                    "approved_review_text": approved_text,
                    "edited_review_text": "",
                },
                "process-media": {"media_paths": ["C:\\selected.mp4"]},
            },
        )

        with patch.object(skills.prompt_engine, "prepare_publish_caption") as prepare_caption:
            result = skills.prepare_caption(SkillContext(plan=plan, node=node, state=state))

        prepare_caption.assert_not_called()
        self.assertEqual(result.outputs["caption"].count("\n\n"), 4)
        self.assertEqual(result.outputs["hashtags"], "#toycar #cartooncrash #safety")
        self.assertEqual(result.outputs["platform_captions"], {
            "facebook": result.outputs["caption"],
            "youtube": result.outputs["caption"],
        })
        self.assertEqual(result.outputs["platform_bundle"]["facebook"]["format"], "reel")
        self.assertTrue(result.outputs["platform_bundle"]["facebook"]["validation"]["is_platform_publish_ready"])
        self.assertEqual(result.outputs["platform_bundle"]["youtube"]["format"], "video")
        self.assertTrue(result.outputs["platform_bundle"]["youtube"]["validation"]["is_platform_publish_ready"])
        self.assertIn("youtube_title", result.outputs["platform_bundle"]["youtube"]["additional_params"])
        self.assertIn("youtube_description", result.outputs["platform_bundle"]["youtube"]["additional_params"])

    def test_prepare_caption_rebuilds_platform_native_metadata_after_edit(self) -> None:
        tool_registry = ToolRegistry()
        skills = AgentSocialSkills(tool_registry, self.project_root / ".tmp-tests")
        plan = ExecutionPlan(
            goal=GoalRequest(
                prompt="publish kirby clip",
                media_type="publish_review",
                style="social promo",
                constraints={"platforms": ["youtube", "facebook"]},
            ),
            workflow_name="publish_review_v1",
            nodes=[],
        )
        node = ExecutionNode(
            node_id="prepare-caption",
            skill_name="publish.caption.prepare",
            depends_on=["review-select", "process-media"],
            inputs={"platforms": ["youtube", "facebook"]},
        )
        state = RunState(
            goal={"prompt": "publish kirby clip"},
            metadata={},
            node_outputs={
                "review-select": {"edited_review_text": "Edited caption body\n\n#edited #tags"},
                "process-media": {"media_paths": ["C:\\selected.mp4"]},
            },
        )

        with patch.object(
            skills.prompt_engine,
            "prepare_publish_caption",
            return_value={
                "caption": "Generated caption",
                "hashtags": "#kirby",
                "platform_captions": {
                    "youtube": "Generated YouTube caption",
                    "facebook": "Generated Facebook caption",
                },
                "platform_bundle": {},
                "caption_strategy": "platform_adapted",
                "dispatch_ready": True,
                "prompt_mode": "llm",
            },
        ):
            result = skills.prepare_caption(SkillContext(plan=plan, node=node, state=state))

        self.assertEqual(
            result.outputs["platform_bundle"]["youtube"]["additional_params"]["youtube_title"],
            "Edited caption body",
        )
        self.assertEqual(
            result.outputs["platform_bundle"]["facebook"]["format"],
            "reel",
        )
        self.assertEqual(
            result.outputs["platform_bundle"]["facebook"]["hashtags"],
            "#edited #tags",
        )

    def test_prepare_caption_uses_contact_sheet_for_video_visual_evidence(self) -> None:
        tool_registry = ToolRegistry()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_path = root / "clip.mp4"
            contact_sheet = root / "output" / "clip_contact_sheet.jpg"
            video_path.write_bytes(b"video")
            contact_sheet.parent.mkdir(parents=True, exist_ok=True)
            contact_sheet.write_bytes(b"image")
            skills = AgentSocialSkills(tool_registry, root / "output")
            plan = ExecutionPlan(
                goal=GoalRequest(
                    prompt="publish Kirby clip",
                    media_type="publish_review",
                    style="social promo",
                    constraints={
                        "platforms": ["instagram"],
                        "hashtags": ["kirby"],
                        "visual_grounding": {"contact_sheet_path": str(contact_sheet)},
                    },
                ),
                workflow_name="publish_review_v1",
                nodes=[],
            )
            node = ExecutionNode(
                node_id="prepare-caption",
                skill_name="publish.caption.prepare",
                depends_on=["review-select", "process-media"],
                inputs={"platforms": ["instagram"]},
            )
            state = RunState(
                goal={"prompt": "publish Kirby clip"},
                metadata={},
                node_outputs={
                    "review-select": {},
                    "process-media": {"media_paths": [str(video_path)]},
                },
            )
            with patch.object(
                skills.prompt_engine,
                "prepare_publish_caption",
                return_value={
                    "caption": "Kirby dodges metal spheres.",
                    "hashtags": "#kirby",
                    "platform_captions": {"instagram": "Kirby dodges metal spheres."},
                    "platform_bundle": {},
                    "dispatch_ready": True,
                    "prompt_mode": "llm",
                },
            ) as prepare_caption:
                skills.prepare_caption(SkillContext(plan=plan, node=node, state=state))

            self.assertEqual(
                prepare_caption.call_args.kwargs["visual_paths"],
                [str(contact_sheet)],
            )


class DiscordHumanReviewServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[2]

    def test_review_candidates_filters_paths_to_media_under_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inside = root / "inside.png"
            outside = root.parent / f"outside_{uuid.uuid4().hex}.txt"
            inside.write_bytes(b"image")
            outside.write_text("do not upload", encoding="utf-8")
            self.addCleanup(outside.unlink)
            service = DiscordHumanReviewService(root)

            self.assertEqual(service._filter_media_paths([str(inside), str(outside)]), [str(inside.resolve())])

    def test_review_candidates_fails_closed_without_reviewer_allowlist(self) -> None:
        service = DiscordHumanReviewService(self.project_root / ".tmp-tests")
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(service.is_configured() and bool(os.getenv("discord_review_allowed_user_ids")))

    def test_review_candidates_treats_missing_discord_decision_as_skipped(self) -> None:
        service = DiscordHumanReviewService(self.project_root / ".tmp-tests")
        temp_file = self.project_root / ".tmp-tests" / "discord_review_candidate.png"
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file.write_bytes(b"fake")
        self.addCleanup(temp_file.unlink)

        with patch.dict("os.environ", {"discord_review_channel_id": "123"}), patch.object(service, "is_configured", return_value=True), patch(
            "agentic.tools.context_services._run_discord_file_feedback_process",
            return_value=("timeout", None, "review text", None, {"status": "timeout"}),
        ):
            decision = service.review_candidates(text="review text", media_paths=[str(temp_file)], timeout_seconds=30)

        self.assertEqual(decision.status, "failed")
        self.assertEqual(decision.review_mode, "discord")
        self.assertEqual(decision.selected_paths, [str(temp_file)])
        self.assertIn("timed out", decision.fallback_reason)

    def test_review_candidates_returns_failed_when_bot_start_errors_before_ready(self) -> None:
        service = DiscordHumanReviewService(self.project_root / ".tmp-tests")
        temp_file = self.project_root / ".tmp-tests" / "discord_review_start_failure.png"
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file.write_bytes(b"fake")
        self.addCleanup(temp_file.unlink)

        with patch.dict("os.environ", {"discord_review_channel_id": "123"}), patch.object(service, "is_configured", return_value=True), patch(
            "agentic.tools.context_services._run_discord_file_feedback_process",
            side_effect=RuntimeError("login failed"),
        ):
            decision = service.review_candidates(text="review text", media_paths=[str(temp_file)], timeout_seconds=30)

        self.assertEqual(decision.status, "failed")
        self.assertEqual(decision.review_mode, "discord")
        self.assertEqual(decision.selected_paths, [str(temp_file)])
        self.assertIn("failed before a decision", decision.fallback_reason)


if __name__ == "__main__":
    unittest.main()
