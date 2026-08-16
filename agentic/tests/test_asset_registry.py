from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path

from agentic.assets.registry import AssetRegistry


class AssetRegistryTests(unittest.TestCase):
    def make_workspace_tempdir(self) -> Path:
        base_dir = Path(__file__).resolve().parents[1] / ".tmp-tests"
        base_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = base_dir / f"asset-registry-{uuid.uuid4().hex}"
        temp_dir.mkdir()
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        return temp_dir

    def _write_workflow(self, root: Path, stem: str, workflow: dict) -> None:
        wf_dir = root / "configs" / "workflow"
        wf_dir.mkdir(parents=True)
        (wf_dir / f"{stem}.json").write_text(json.dumps(workflow), encoding="utf-8")

    def test_materialize_resolves_under_project_root(self) -> None:
        root = self.make_workspace_tempdir()
        self._write_workflow(root, "z_stub", {"1": {"class_type": "SaveImage", "inputs": {}}})
        registry = AssetRegistry(root, asset_root=root)
        manifest = registry.get_manifest("z_stub")
        path = registry.materialize_workflow(manifest)
        self.assertTrue(path.exists())
        self.assertEqual(path.suffix, ".json")

    def test_ensure_workflow_ready_has_empty_asset_status(self) -> None:
        root = self.make_workspace_tempdir()
        self._write_workflow(root, "wf1", {"1": {"class_type": "SaveImage", "inputs": {}}})
        registry = AssetRegistry(root, asset_root=root)
        result = registry.ensure_workflow_ready("wf1", auto_download=False)
        self.assertEqual(result["asset_status"], [])

    def test_recommend_workflows_orders_by_style_token_in_name(self) -> None:
        root = self.make_workspace_tempdir()
        wf_dir = root / "configs" / "workflow"
        wf_dir.mkdir(parents=True)
        (wf_dir / "anime_cinematic_ready.json").write_text(json.dumps({"1": {"class_type": "SaveImage", "inputs": {}}}), encoding="utf-8")
        (wf_dir / "generic_plain.json").write_text(json.dumps({"1": {"class_type": "SaveImage", "inputs": {}}}), encoding="utf-8")

        registry = AssetRegistry(root, asset_root=root)
        recommendations = registry.recommend_workflows("image", style="anime cinematic", limit=2)

        self.assertGreaterEqual(len(recommendations), 2)
        self.assertEqual(recommendations[0]["workflow_name"], "anime_cinematic_ready")
        self.assertGreater(recommendations[0]["score"], recommendations[1]["score"])
        self.assertIn("anime", recommendations[0]["style_matches"])

    def test_validate_workflow_reports_missing_dependency(self) -> None:
        root = self.make_workspace_tempdir()
        self._write_workflow(
            root,
            "broken_image",
            {
                "1": {"class_type": "KSampler", "inputs": {"model": ["999", 0]}, "_meta": {"alias": "sampler"}},
                "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
            },
        )
        registry = AssetRegistry(root, asset_root=root)
        validation = registry.validate_workflow("broken_image")

        self.assertFalse(validation["valid"])
        self.assertEqual(validation["aliases"], ["sampler"])
        self.assertEqual(validation["output_nodes"], ["2"])
        self.assertEqual(validation["unresolved_inputs"][0]["missing_node"], "999")


if __name__ == "__main__":
    unittest.main()
