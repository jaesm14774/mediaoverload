from __future__ import annotations

import unittest
import uuid
import shutil
from pathlib import Path

from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest, SkillContext, SkillResult
from agentic.runtime.registry import SkillRegistry
from agentic.runtime.runner import WorkflowRunner
from agentic.skills.agent_primitives import AgentMediaSkills


class WorkflowRunnerTests(unittest.TestCase):
    def make_workspace_tempdir(self) -> Path:
        base_dir = Path(__file__).resolve().parents[1] / ".tmp-tests"
        base_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = base_dir / f"workflow-runner-{uuid.uuid4().hex}"
        temp_dir.mkdir()
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        return temp_dir

    def test_runner_records_prompt_modes_and_lineage(self) -> None:
        registry = SkillRegistry()

        def idea_brief(context: SkillContext) -> SkillResult:
            return SkillResult(
                status="success",
                outputs={"prompt": "base prompt", "prompt_mode": "llm"},
            )

        def refine_prompt(context: SkillContext) -> SkillResult:
            return SkillResult(
                status="success",
                outputs={
                    "prompt": "revised prompt",
                    "prompt_mode": "llm",
                    "original_prompt": "base prompt",
                    "revised_prompt": "revised prompt",
                    "review_notes": "stronger action",
                    "selected_assets": ["C:\\asset.png"],
                },
            )

        registry.register("agent.goal.expand", idea_brief, "idea")
        registry.register("agent.review.refine_prompt", refine_prompt, "refine")

        runner = WorkflowRunner(skill_registry=registry)
        plan = ExecutionPlan(
            goal=GoalRequest(prompt="test prompt"),
            workflow_name="test_workflow",
            nodes=[
                ExecutionNode(node_id="idea-brief", skill_name="agent.goal.expand"),
                ExecutionNode(
                    node_id="review-refine-prompt",
                    skill_name="agent.review.refine_prompt",
                    depends_on=["idea-brief"],
                ),
            ],
        )

        result = runner.run(plan)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.state.node_prompt_modes["idea-brief"], "llm")
        self.assertEqual(result.state.node_prompt_modes["review-refine-prompt"], "llm")
        self.assertEqual(len(result.state.prompt_lineage), 1)
        self.assertEqual(result.state.prompt_lineage[0]["original_prompt"], "base prompt")
        self.assertEqual(result.state.prompt_lineage[0]["revised_prompt"], "revised prompt")
        self.assertEqual(result.state.prompt_lineage[0]["selected_assets"], ["C:\\asset.png"])

    def test_runner_records_review_selection_lineage(self) -> None:
        registry = SkillRegistry()

        def review_select(context: SkillContext) -> SkillResult:
            return SkillResult(
                status="success",
                outputs={
                    "selected_assets": ["C:\\clip.mp4"],
                    "selected_count": 1,
                    "rejected_assets": ["C:\\still.png"],
                    "selection_rationale": "Clip has stronger motion.",
                    "regeneration_notes": "If rerendering, tighten framing.",
                    "prompt_mode": "llm",
                },
            )

        registry.register("review.assets.select", review_select, "review")
        runner = WorkflowRunner(skill_registry=registry)
        plan = ExecutionPlan(
            goal=GoalRequest(prompt="test prompt"),
            workflow_name="test_workflow",
            nodes=[ExecutionNode(node_id="review-select", skill_name="review.assets.select")],
        )

        result = runner.run(plan)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.state.prompt_lineage[0]["selection_rationale"], "Clip has stronger motion.")
        self.assertEqual(result.state.prompt_lineage[0]["rejected_assets"], ["C:\\still.png"])

    def test_runner_records_fallback_reason_and_backend_in_lineage(self) -> None:
        registry = SkillRegistry()

        def idea_brief(context: SkillContext) -> SkillResult:
            return SkillResult(
                status="success",
                outputs={
                    "prompt": "base prompt",
                    "prompt_mode": "template",
                    "fallback_reason": "manager_unavailable",
                    "manager_error": "KeyError: gemini_api_token",
                    "llm_backend": {"text_provider": "gemini"},
                    "original_prompt": "base prompt",
                },
            )

        registry.register("agent.goal.expand", idea_brief, "idea")
        runner = WorkflowRunner(skill_registry=registry)
        plan = ExecutionPlan(
            goal=GoalRequest(prompt="test prompt"),
            workflow_name="test_workflow",
            nodes=[ExecutionNode(node_id="idea-brief", skill_name="agent.goal.expand")],
        )

        result = runner.run(plan)

        self.assertEqual(result.state.prompt_lineage[0]["fallback_reason"], "manager_unavailable")
        self.assertEqual(result.state.prompt_lineage[0]["manager_error"], "KeyError: gemini_api_token")
        self.assertEqual(result.state.prompt_lineage[0]["llm_backend"]["text_provider"], "gemini")

    def test_runner_converts_skill_exception_into_failed_result(self) -> None:
        registry = SkillRegistry()

        def broken_skill(context: SkillContext) -> SkillResult:
            raise RuntimeError("llm unavailable")

        registry.register("agent.goal.expand", broken_skill, "idea")
        runner = WorkflowRunner(skill_registry=registry)
        plan = ExecutionPlan(
            goal=GoalRequest(prompt="test prompt"),
            workflow_name="test_workflow",
            nodes=[ExecutionNode(node_id="idea-brief", skill_name="agent.goal.expand")],
        )

        result = runner.run(plan)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.records[0].status, "failed")
        self.assertIn("RuntimeError: llm unavailable", result.records[0].logs[0])

    def test_agent_media_asset_check_fails_when_required_assets_are_missing(self) -> None:
        class FakeTools:
            @staticmethod
            def call(name: str, payload: dict[str, object]) -> dict[str, object]:
                self.assertEqual(name, "asset.ensure_workflow_ready")
                self.assertEqual(payload["workflow_name"], "minimax_h3_lowvram_i2v")
                return {
                    "workflow_name": "minimax_h3_lowvram_i2v",
                    "asset_status": [
                        {"asset": "minimax_h3_fl2va_pruned_fp8_Q4_0.gguf", "status": "missing", "action": "manual_setup"},
                        {"asset": "qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf", "status": "missing", "action": "manual_setup"},
                    ],
                }

        skills = AgentMediaSkills(FakeTools(), self.make_workspace_tempdir())
        context = SkillContext(
            plan=ExecutionPlan(
                goal=GoalRequest(prompt="kirby push-in shot"),
                workflow_name="text2img2video_v1",
                nodes=[],
            ),
            node=ExecutionNode(
                node_id="video-asset-check",
                skill_name="media.ensure_workflow",
                inputs={"workflow_name": "minimax_h3_lowvram_i2v", "auto_download": False},
            ),
            state={},
        )

        result = skills.ensure_workflow(context)

        self.assertEqual(result.status, "failed")
        self.assertIn("minimax_h3_fl2va_pruned_fp8_Q4_0.gguf", result.logs[0])
        self.assertIn("qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf", result.logs[0])


if __name__ == "__main__":
    unittest.main()
