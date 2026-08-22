from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest, RunState, SkillContext
from agentic.runtime.story_service import NativeH3StoryService
from agentic.app.main import build_runtime
from agentic.skills.shared import (
    asset_check_result,
    build_run_dir,
    collect_output_values,
    resolve_dependency_value,
    safe_path_component,
    slug_path_component,
)
from agentic.tools.publishing_adapter import MediaPost as AdapterMediaPost
from agentic.tools.publishing_adapter import build_dispatch_plan
from agentic.tools.social_native import MediaPost as NativeMediaPost


class SharedSkillHelperTests(unittest.TestCase):
    def _context(self, outputs: dict[str, object], *, inputs: dict[str, object] | None = None) -> SkillContext:
        plan = ExecutionPlan(goal=GoalRequest(prompt="shared helper test"), workflow_name="test", nodes=[])
        node = ExecutionNode(
            node_id="consumer",
            skill_name="test.consumer",
            inputs=inputs or {},
            depends_on=list(outputs),
        )
        state = RunState(goal={}, metadata={}, node_outputs=outputs)
        return SkillContext(plan=plan, node=node, state=state)

    def test_run_directory_helper_preserves_canonical_variants(self) -> None:
        with patch("agentic.skills.shared.datetime") as datetime_class:
            datetime_class.now.return_value.strftime.return_value = "20260809_120000"
            image_path = build_run_dir(Path("output"), "A rainy Kirby scene", default_slug="comfy-image", max_slug_length=40)
            workflow_path = build_run_dir(
                Path("output"), "A rainy Kirby scene", "img2img", default_slug="workflow", suffix_first=True
            )

        self.assertEqual(image_path, Path("output/20260809_120000_a-rainy-kirby-scene"))
        self.assertEqual(workflow_path, Path("output/20260809_120000_img2img_a-rainy-kirby-scene"))

    def test_dependency_collector_can_preserve_first_matching_key_contract(self) -> None:
        context = self._context({"render": {"saved_files": ["first.mp4"], "video_path": "fallback.mp4"}})

        collected = collect_output_values(context, ("saved_files", "video_path"), first_key_only=True)

        self.assertEqual(collected, ["first.mp4"])

    def test_dependency_collector_preserves_dependency_order_across_candidate_keys(self) -> None:
        context = self._context(
            {
                "first": {"video_path": "first.mp4"},
                "second": {"saved_files": ["second.mp4"]},
            }
        )

        collected = collect_output_values(context, ("saved_files", "video_path"), first_key_only=True)

        self.assertEqual(collected, ["first.mp4", "second.mp4"])

    def test_dependency_value_returns_the_first_scalar_candidate(self) -> None:
        context = self._context({"prompt": {"prompt": "generated prompt", "creative_brief": "brief"}})

        self.assertEqual(resolve_dependency_value(context, ("prompt", "creative_brief")), "generated prompt")

    def test_asset_check_helper_reports_missing_assets(self) -> None:
        result = asset_check_result(
            {"workflow_name": "native_h3", "asset_status": [{"asset": "model.safetensors", "status": "missing"}]},
            "checked",
        )

        self.assertEqual(result.status, "failed")
        self.assertIn("model.safetensors", result.logs[0])

    def test_path_component_helper_rejects_traversal_without_changing_valid_names(self) -> None:
        self.assertEqual(safe_path_component("segment-01_image"), "segment-01_image")
        with self.assertRaises(ValueError):
            safe_path_component("..\\escape", field_name="run suffix")
        with self.assertRaises(ValueError):
            build_run_dir(Path("output"), "prompt", "../escape")

    def test_slug_path_component_handles_free_form_model_labels(self) -> None:
        self.assertEqual(
            slug_path_component("Kirby mid-sprint: 'oof' / surprise?"),
            "kirby-mid-sprint-oof-surprise",
        )

    def test_dispatch_plan_is_one_shared_publish_contract(self) -> None:
        plan = build_dispatch_plan(
            media_paths=["clip.mp4"],
            caption="default caption",
            hashtags="#kirby",
            platforms=["instagram"],
            platform_bundle={
                "instagram": {
                    "caption": "IG caption",
                    "media_paths": ["ig.mp4"],
                    "validation": {"is_publish_ready": True},
                }
            },
        )

        self.assertEqual(plan["instagram"]["caption"], "IG caption")
        self.assertEqual(plan["instagram"]["media_paths"], ["ig.mp4"])
        self.assertTrue(plan["instagram"]["validation"]["is_publish_ready"])
        self.assertIs(AdapterMediaPost, NativeMediaPost)

class NativeH3StoryServiceTests(unittest.TestCase):
    def test_service_injects_news_and_llm_once_without_storyboard_import_cycle(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeNewsService:
            def get_random_news(self) -> SimpleNamespace:
                return SimpleNamespace(to_dict=lambda: {"title": "rain warning", "keyword": "weather"})

        class FakeLLM:
            def generate_native_h3_storyboard(self, **kwargs: object) -> dict[str, object]:
                calls.append(kwargs)
                return {"story": {"name": "generated"}}

        with patch(
            "agentic.runtime.story_service.merge_native_h3_storyboard",
            return_value={"name": "merged"},
        ) as merge_story:
            service = NativeH3StoryService(llm_engine=FakeLLM(), news_service=FakeNewsService())  # type: ignore[arg-type]
            merged, payload = service.resolve(
                {"name": "base"},
                character="Kirby",
                style="anime",
                duration_seconds=15,
                news_context={},
            )

        self.assertEqual(merged, {"name": "merged"})
        self.assertEqual(payload["story"], {"name": "generated"})
        self.assertEqual(calls[0]["news_context"], {"title": "rain warning", "keyword": "weather"})
        merge_story.assert_called_once_with({"name": "base"}, {"name": "generated"})

    def test_service_replaces_brand_unsafe_injected_news_context(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeNewsService:
            def get_random_news(self) -> SimpleNamespace:
                return SimpleNamespace(to_dict=lambda: {"title": "rain warning", "keyword": "weather"})

        class FakeLLM:
            def generate_native_h3_storyboard(self, **kwargs: object) -> dict[str, object]:
                calls.append(kwargs)
                return {"story": {"name": "generated"}}

        with patch(
            "agentic.runtime.story_service.merge_native_h3_storyboard",
            return_value={"name": "merged"},
        ):
            service = NativeH3StoryService(llm_engine=FakeLLM(), news_service=FakeNewsService())  # type: ignore[arg-type]
            service.resolve(
                {"name": "base"},
                character="Kirby",
                style="anime",
                duration_seconds=15,
                news_context={"title": "AI性愛機器人新產品", "keyword": "AI;性愛機器人"},
            )

        self.assertEqual(calls[0]["news_context"], {"title": "rain warning", "keyword": "weather"})


if __name__ == "__main__":
    unittest.main()
