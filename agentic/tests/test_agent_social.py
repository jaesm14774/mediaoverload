from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest, RunState, SkillContext
from agentic.runtime.registry import ToolRegistry
from agentic.skills.agent_social import AgentSocialSkills


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
                constraints={},
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


if __name__ == "__main__":
    unittest.main()
