from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from agentic.app.main import build_runtime
from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest, RunState, SkillContext


class AuthoringToolTests(unittest.TestCase):
    def make_workspace_tempdir(self) -> Path:
        base_dir = Path(__file__).resolve().parents[1] / ".tmp-tests"
        base_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = base_dir / f"authoring-tools-{uuid.uuid4().hex}"
        temp_dir.mkdir()
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        return temp_dir

    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.planner, cls.runner, cls.run_memory = build_runtime(cls.project_root)

    def test_asset_plan_and_acquire_contracts_return_expected_shape(self) -> None:
        tool_registry = self.runner.tool_registry
        acquisition_plan = tool_registry.call("asset.plan_acquisition", {"workflow_name": "nova_model_plus_z_image_anime"})
        acquisition_result = tool_registry.call("asset.acquire_missing", {"workflow_name": "nova_model_plus_z_image_anime"})

        self.assertIn("missing_assets", acquisition_plan)
        self.assertIn("missing_count", acquisition_plan)
        self.assertIn("prepared_assets", acquisition_result)
        self.assertIn("prepared_count", acquisition_result)

    def test_workflow_recommend_and_validate_contracts_return_expected_shape(self) -> None:
        tool_registry = self.runner.tool_registry
        recommendation = tool_registry.call(
            "workflow.recommend",
            {"media_type": "image", "style": "cinematic anime", "prompt": "robot chef", "limit": 1},
        )
        validation = tool_registry.call("workflow.validate_manifest", {"workflow_name": "nova_model_plus_z_image_anime"})

        self.assertEqual(recommendation["recommendation_count"], 1)
        self.assertIn("workflow_name", recommendation["recommendations"][0])
        self.assertIn("score", recommendation["recommendations"][0])
        self.assertIn("valid", validation)
        self.assertIn("issues", validation)
        self.assertIn("workflow_path", validation)

    def test_workflow_authoring_tools_create_and_patch_draft(self) -> None:
        temp_dir = self.make_workspace_tempdir()
        _, runner, _ = build_runtime(self.project_root)
        runner.tool_registry._tools["workflow.author.create_draft"].handler.__self__.root = temp_dir
        runner.tool_registry._tools["workflow.author.patch_draft"].handler.__self__.root = temp_dir

        created = runner.tool_registry.call(
            "workflow.author.create_draft",
            {
                "workflow_name": "nova_model_plus_z_image_anime",
                "variant_name": "legacy_image_test_draft",
                "summary": "test draft",
            },
        )
        patched = runner.tool_registry.call(
            "workflow.author.patch_draft",
            {
                "draft_path": created["draft_path"],
                "patches": [{"node_id": "1", "input_key": "value", "value": "patched prompt"}],
            },
        )

        draft_path = Path(str(created["draft_path"]))
        payload = draft_path.read_text(encoding="utf-8")

        self.assertTrue(draft_path.exists())
        self.assertEqual(patched["patch_count"], 1)
        self.assertIn("patched prompt", payload)

    def test_packaged_outputs_include_prompt_metadata(self) -> None:
        output_root = self.make_workspace_tempdir()
        _, runner, _ = build_runtime(self.project_root, output_root=output_root)
        skill = runner.skill_registry.get("agent.sticker.animate.package").handler
        plan = ExecutionPlan(
            goal=GoalRequest(prompt="kirby cheers", media_type="animated_sticker", style="LINE sticker"),
            workflow_name="animated_sticker_v1",
            nodes=[],
        )
        node = ExecutionNode(
            node_id="package-animated-sticker",
            skill_name="agent.sticker.animate.package",
            depends_on=["animate-sticker", "gif-preview"],
        )
        state = RunState(
            goal={"prompt": "kirby cheers"},
            metadata={},
            node_outputs={
                "animate-sticker": {
                    "video_path": "C:\\artifact.mp4",
                    "saved_files": ["C:\\artifact.mp4"],
                },
                "gif-preview": {
                    "gif_path": "C:\\artifact.gif",
                },
            },
            prompt_lineage=[
                {
                    "node_id": "animate-sticker",
                    "original_prompt": "base prompt",
                    "revised_prompt": "revised prompt",
                    "prompt_mode": "llm",
                }
            ],
            node_prompt_modes={"animate-sticker": "llm", "gif-preview": "template"},
        )

        result = skill(SkillContext(plan=plan, node=node, state=state))
        summary_path = Path(str(result.outputs["summary_path"]))
        payload = summary_path.read_text(encoding="utf-8")

        self.assertTrue(summary_path.exists())
        self.assertIn("\"prompt_lineage\"", payload)
        self.assertIn("\"node_prompt_modes\"", payload)
        self.assertEqual(result.outputs["prompt_lineage"][0]["node_id"], "animate-sticker")
        self.assertEqual(result.outputs["node_prompt_modes"]["animate-sticker"], "llm")

    def test_persisted_summary_includes_review_and_prompt_metadata(self) -> None:
        output_root = self.make_workspace_tempdir()
        _, runner, _ = build_runtime(self.project_root, output_root=output_root)
        skill = runner.skill_registry.get("agent.summary.persist").handler
        plan = ExecutionPlan(
            goal=GoalRequest(prompt="publish kirby clip", media_type="publish_review", style="social promo"),
            workflow_name="publish_review_v1",
            nodes=[],
        )
        node = ExecutionNode(
            node_id="persist-publish-review-summary",
            skill_name="agent.summary.persist",
            depends_on=["review-select", "prepare-caption", "dispatch-publish"],
            inputs={"summary_name": "publish_review_summary.json", "summary_scope": "publish_review"},
        )
        state = RunState(
            goal={"prompt": "publish kirby clip"},
            metadata={},
            node_outputs={
                "review-select": {
                    "selected_assets": ["C:\\selected.mp4"],
                    "rejected_assets": ["C:\\rejected.png"],
                    "rejected_asset_details": [{"media_path": "C:\\rejected.png", "reason": "weak still"}],
                    "selection_rationale": "Video has stronger motion.",
                    "failure_tags": ["motion_weak"],
                    "retry_direction": "increase visible action",
                    "retry_intensity": "high",
                },
                "prepare-caption": {
                    "caption": "Launch clip",
                    "platform_bundle": {"instagram": {"caption": "IG launch", "validation": {"is_publish_ready": True}}},
                },
                "dispatch-publish": {
                    "status": "dry_run",
                    "media_paths": ["C:\\selected.mp4"],
                },
            },
            prompt_lineage=[
                {
                    "node_id": "prepare-caption",
                    "caption": "Launch clip",
                    "platform_bundle": {"instagram": {"caption": "IG launch"}},
                    "prompt_mode": "llm",
                }
            ],
            node_prompt_modes={"prepare-caption": "llm", "review-select": "llm"},
        )

        result = skill(SkillContext(plan=plan, node=node, state=state))
        summary_path = Path(str(result.outputs["summary_path"]))
        payload = summary_path.read_text(encoding="utf-8")

        self.assertTrue(summary_path.exists())
        self.assertIn("\"review_summary\"", payload)
        self.assertIn("\"retry_direction\": \"increase visible action\"", payload)
        self.assertIn("\"platform_bundle\"", payload)
