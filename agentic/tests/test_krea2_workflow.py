from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml

from agentic.assets.registry import AssetRegistry


class Krea2WorkflowContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.workflow_dir = cls.repo_root / "configs" / "workflow"
        cls.routing = yaml.safe_load((cls.repo_root / "configs" / "routing.yaml").read_text(encoding="utf-8"))["routing"]

    def test_active_routes_use_krea2_and_not_retired_image_workflows(self) -> None:
        retired = {
            "z_image",
            "z_image_i2i_anime",
        }
        active = {
            workflow_name
            for stages in self.routing["workflow_stage_candidates"].values()
            for workflow_names in stages.values()
            for workflow_name in workflow_names
        }
        self.assertTrue(active & {"krea2_turbo", "krea2_turbo_img2img"})
        self.assertEqual(active & retired, set())

    def test_text2img_route_is_active_and_uses_canonical_names(self) -> None:
        self.assertIn("text2img", self.routing["strategy_candidates"])
        self.assertNotIn("strategy_aliases", self.routing)
        self.assertEqual(
            self.routing["workflow_stage_candidates"]["text2img"]["image_workflow_name"][0],
            "krea2_turbo",
        )

    def test_every_text_to_image_video_stage_includes_krea2_and_requires_an_image_stage(self) -> None:
        image_stages = {
            "text2video",
            "text2image2video",
            "text2longvideo",
            "native_h3_story",
            "native_h3_fl2va_story",
            "native_h3_l2va_story",
            "text2image2native_h3_ref2va",
            "text2image2image",
            "sticker_pack",
        }
        for strategy in image_stages:
            candidates = self.routing["workflow_stage_candidates"][strategy]["image_workflow_name"]
            self.assertIn("krea2_turbo", candidates, strategy)
            self.assertTrue(candidates, strategy)

    def test_retired_image_workflow_files_are_removed(self) -> None:
        for workflow_name in (
            "z_image",
            "z_image_i2i_anime",
        ):
            self.assertFalse((self.workflow_dir / f"{workflow_name}.json").exists(), workflow_name)

    def test_krea2_turbo_is_official_low_vram_sampler_shape(self) -> None:
        graph = json.loads((self.workflow_dir / "krea2_turbo.json").read_text(encoding="utf-8"))
        classes = {node["class_type"] for node in graph.values()}
        self.assertIn("UnetLoaderGGUF", classes)
        self.assertIn("CLIPLoaderGGUF", classes)
        self.assertIn("ConditioningZeroOut", classes)
        sampler = next(node for node in graph.values() if node["class_type"] == "KSampler")["inputs"]
        self.assertEqual(sampler["steps"], 8)
        self.assertEqual(sampler["cfg"], 1.0)
        self.assertEqual(sampler["sampler_name"], "euler")
        self.assertEqual(sampler["scheduler"], "simple")
        self.assertNotIn("TextGenerate", classes)

    def test_krea2_img2img_keeps_the_same_model_family(self) -> None:
        graph = json.loads((self.workflow_dir / "krea2_turbo_img2img.json").read_text(encoding="utf-8"))
        sampler = next(node for node in graph.values() if node["class_type"] == "KSampler")["inputs"]
        self.assertEqual(sampler["denoise"], 0.25)
        self.assertEqual(graph["2"]["inputs"]["unet_name"], "krea2_turbo_bf16-Q4_0.gguf")
        self.assertEqual(graph["3"]["inputs"]["type"], "krea2")
        self.assertEqual(graph["4"]["inputs"]["vae_name"], "qwen_image_vae.safetensors")

    def test_asset_registry_reports_the_three_missing_local_files(self) -> None:
        registry = AssetRegistry(self.repo_root / "agentic", asset_root=self.repo_root)
        manifest = registry.get_manifest("krea2_turbo")
        self.assertEqual(
            [asset.name for asset in manifest.required_assets],
            [
                "krea2_turbo_bf16-Q4_0.gguf",
                "Qwen3VL-4B-Instruct-Q4_K_M.gguf",
                "qwen_image_vae.safetensors",
            ],
        )
        statuses = registry.ensure_workflow_ready("krea2_turbo", auto_download=False)["asset_status"]
        self.assertEqual([item["status"] for item in statuses], ["missing", "missing", "missing"])


if __name__ == "__main__":
    unittest.main()
