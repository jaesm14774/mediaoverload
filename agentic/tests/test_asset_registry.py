from __future__ import annotations

import json
import unittest
import uuid
import shutil
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

    def _write_manifest(self, manifest_dir: Path) -> None:
        manifest = {
            "name": "comfy_image_v1",
            "media_types": ["image"],
            "workflow_path": "configs/workflow/z_image.json",
            "summary": "test manifest",
            "required_assets": [
                {
                    "name": "model.safetensors",
                    "kind": "checkpoint",
                    "target_dir": "ComfyUI/models/checkpoints/sdxl",
                    "source": "test",
                }
            ],
        }
        (manifest_dir / "comfy_image_v1.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_asset_check_accepts_parent_of_comfy_root(self) -> None:
        root = self.make_workspace_tempdir()
        manifest_dir = root / "manifests"
        manifest_dir.mkdir(parents=True)
        self._write_manifest(manifest_dir)
        asset_path = root / "ComfyUI" / "models" / "checkpoints" / "sdxl" / "model.safetensors"
        asset_path.parent.mkdir(parents=True)
        asset_path.write_text("stub", encoding="utf-8")

        registry = AssetRegistry(manifest_dir, project_root=root, asset_root=root)
        status = registry.ensure_workflow_ready("comfy_image_v1", auto_download=False)["asset_status"][0]

        self.assertEqual(status["status"], "ready")
        self.assertEqual(Path(status["target_path"]), asset_path)

    def test_asset_check_accepts_comfy_root_directly(self) -> None:
        root = self.make_workspace_tempdir()
        manifest_dir = root / "manifests"
        manifest_dir.mkdir(parents=True)
        self._write_manifest(manifest_dir)
        comfy_root = root / "ComfyUI"
        asset_path = comfy_root / "models" / "checkpoints" / "sdxl" / "model.safetensors"
        asset_path.parent.mkdir(parents=True)
        asset_path.write_text("stub", encoding="utf-8")

        registry = AssetRegistry(manifest_dir, project_root=root, asset_root=comfy_root)
        status = registry.ensure_workflow_ready("comfy_image_v1", auto_download=False)["asset_status"][0]

        self.assertEqual(status["status"], "ready")
        self.assertEqual(Path(status["target_path"]), asset_path)

    def test_recommend_workflows_prefers_ready_and_style_matched_manifest(self) -> None:
        root = self.make_workspace_tempdir()
        manifest_dir = root / "manifests"
        manifest_dir.mkdir(parents=True)
        workflow_dir = root / "configs" / "workflow"
        workflow_dir.mkdir(parents=True)
        (workflow_dir / "ready.json").write_text(json.dumps({"1": {"class_type": "SaveImage", "inputs": {}}}), encoding="utf-8")
        (workflow_dir / "missing.json").write_text(json.dumps({"1": {"class_type": "SaveImage", "inputs": {}}}), encoding="utf-8")

        ready_manifest = {
            "name": "anime_ready_v1",
            "media_types": ["image"],
            "workflow_path": "configs/workflow/ready.json",
            "summary": "anime cinematic workflow",
            "required_assets": [{"name": "ready.safetensors", "kind": "checkpoint", "target_dir": "ComfyUI/models/checkpoints"}],
        }
        missing_manifest = {
            "name": "generic_missing_v1",
            "media_types": ["image"],
            "workflow_path": "configs/workflow/missing.json",
            "summary": "generic workflow",
            "required_assets": [{"name": "missing.safetensors", "kind": "checkpoint", "target_dir": "ComfyUI/models/checkpoints"}],
        }
        (manifest_dir / "anime_ready_v1.json").write_text(json.dumps(ready_manifest), encoding="utf-8")
        (manifest_dir / "generic_missing_v1.json").write_text(json.dumps(missing_manifest), encoding="utf-8")
        ready_asset = root / "ComfyUI" / "models" / "checkpoints" / "ready.safetensors"
        ready_asset.parent.mkdir(parents=True)
        ready_asset.write_text("stub", encoding="utf-8")

        registry = AssetRegistry(manifest_dir, project_root=root, asset_root=root)
        recommendations = registry.recommend_workflows("image", style="anime cinematic", limit=2)

        self.assertEqual(recommendations[0]["workflow_name"], "anime_ready_v1")
        self.assertGreater(recommendations[0]["score"], recommendations[1]["score"])
        self.assertIn("anime", recommendations[0]["style_matches"])

    def test_validate_workflow_reports_missing_dependency(self) -> None:
        root = self.make_workspace_tempdir()
        manifest_dir = root / "manifests"
        manifest_dir.mkdir(parents=True)
        workflow_dir = root / "configs" / "workflow"
        workflow_dir.mkdir(parents=True)
        (workflow_dir / "broken.json").write_text(
            json.dumps(
                {
                    "1": {"class_type": "KSampler", "inputs": {"model": ["999", 0]}, "_meta": {"alias": "sampler"}},
                    "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
                }
            ),
            encoding="utf-8",
        )
        manifest = {
            "name": "broken_image_v1",
            "media_types": ["image"],
            "workflow_path": "configs/workflow/broken.json",
            "summary": "broken workflow",
            "required_assets": [],
        }
        (manifest_dir / "broken_image_v1.json").write_text(json.dumps(manifest), encoding="utf-8")

        registry = AssetRegistry(manifest_dir, project_root=root, asset_root=root)
        validation = registry.validate_workflow("broken_image_v1")

        self.assertFalse(validation["valid"])
        self.assertEqual(validation["aliases"], ["sampler"])
        self.assertEqual(validation["output_nodes"], ["2"])
        self.assertEqual(validation["unresolved_inputs"][0]["missing_node"], "999")


if __name__ == "__main__":
    unittest.main()
