from __future__ import annotations

import unittest
import os
from pathlib import Path
from unittest.mock import patch

from agentic.app.main import build_runtime


class RuntimeRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.project_root = Path(__file__).resolve().parents[1]
        cls.planner, cls.runner, cls.run_memory = build_runtime(cls.project_root)

    def test_publish_and_review_skills_are_registered(self) -> None:
        skill_names = set(self.runner.skill_registry.list_names())
        self.assertIn("publish.media.ingest", skill_names)
        self.assertIn("publish.caption.prepare", skill_names)
        self.assertIn("publish.media.collect", skill_names)
        self.assertIn("publish.media.process", skill_names)
        self.assertIn("publish.social.dispatch", skill_names)
        self.assertIn("review.assets.select", skill_names)

    def test_authoring_tools_are_registered(self) -> None:
        tool_names = set(self.runner.tool_registry.list_names())
        self.assertIn("asset.plan_acquisition", tool_names)
        self.assertIn("asset.acquire_missing", tool_names)
        self.assertIn("workflow.recommend", tool_names)
        self.assertIn("workflow.validate_manifest", tool_names)
        self.assertIn("workflow.author.create_draft", tool_names)
        self.assertIn("workflow.author.patch_draft", tool_names)

    def test_runtime_uses_comfyui_root_environment_for_asset_checks(self) -> None:
        configured_root = self.project_root / ".tmp-tests" / "configured-comfy-root"
        with patch.dict(os.environ, {"COMFYUI_ROOT": str(configured_root)}, clear=False):
            planner, _runner, _memory = build_runtime(self.project_root)

        self.assertEqual(planner.asset_registry.asset_root, configured_root.resolve())


if __name__ == "__main__":
    unittest.main()
