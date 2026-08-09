from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest, RunState, SkillContext
from agentic.runtime.registry import ToolRegistry
from agentic.skills.agent_social import AgentSocialSkills
from agentic.tools.context_services import DiscordHumanReviewService


class AgentSocialSkillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]

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

    def test_review_select_is_automatic_without_explicit_review_flag(self) -> None:
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
            "review_asset_candidates",
            return_value={"selected_assets": ["C:\\selected.mp4"], "ranked_candidates": []},
        ), patch.object(
            skills.prompt_engine,
            "prepare_publish_caption",
            return_value={"caption": "A caption", "hashtags": "#kirby", "dispatch_ready": True},
        ), patch.object(skills.discord_review, "review_candidates") as discord_review:
            result = skills.select_best_assets(SkillContext(plan=plan, node=node, state=state))

        self.assertEqual(result.status, "success")
        self.assertEqual(result.outputs["review_mode"], "automatic")
        discord_review.assert_not_called()

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

    def test_first_frame_review_sends_all_six_candidates_and_never_auto_selects(self) -> None:
        tool_registry = ToolRegistry()
        skills = AgentSocialSkills(tool_registry, self.project_root / ".tmp-tests")
        candidate_paths = [f"C:\\frame_{index}.png" for index in range(1, 7)]
        plan = ExecutionPlan(
            goal=GoalRequest(
                prompt="Kirby must stop the runaway lantern in the first second",
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
            node_outputs={"native-opening-keyframe": {"saved_files": candidate_paths}},
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
        self.assertIn("Asset 6", captured["text"])

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


class DiscordHumanReviewServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]

    def test_review_candidates_treats_missing_discord_decision_as_skipped(self) -> None:
        service = DiscordHumanReviewService(self.project_root / ".tmp-tests")
        temp_file = self.project_root / ".tmp-tests" / "discord_review_candidate.png"
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file.write_bytes(b"fake")
        self.addCleanup(temp_file.unlink)

        with patch.object(service, "is_configured", return_value=True), patch(
            "agentic.tools.context_services._run_discord_file_feedback_process",
            return_value=("timeout", None, "review text", None),
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

        with patch.object(service, "is_configured", return_value=True), patch(
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
