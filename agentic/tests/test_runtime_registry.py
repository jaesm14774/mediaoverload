from __future__ import annotations

import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
