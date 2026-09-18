from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from agentic.assets.fasth3 import FASTH3_ASSETS
from agentic.assets.registry import AssetRegistry


class FastH3BenchmarkContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]
        scripts_dir = cls.repo_root / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import run_fasth3_benchmark

        cls.benchmark = run_fasth3_benchmark

    def test_official_asset_manifest_has_unique_verified_sizes_and_targets(self) -> None:
        self.assertEqual(len({asset.name for asset in FASTH3_ASSETS}), len(FASTH3_ASSETS))
        self.assertTrue(all(len(asset.sha256) == 64 for asset in FASTH3_ASSETS))
        with self.subTest("target containment"):
            root = Path("E:/comfyui/_extra")
            for asset in FASTH3_ASSETS:
                target = asset.target_path(root)
                self.assertEqual(target.parts[-2], asset.model_dir)
                self.assertEqual(target.parts[-1], asset.name)

    def test_native_and_fasth3_workflows_are_valid_api_graphs(self) -> None:
        registry = AssetRegistry(self.repo_root / "agentic", asset_root=self.repo_root)
        for workflow_name in ("minimax_h3_native_int8_t2v", "minimax_h3_fasth3_v2_t2v"):
            self.assertIn("E:/comfyui/_extra", registry.get_manifest(workflow_name).asset_extra_roots)
            validation = registry.validate_workflow(workflow_name)
            self.assertTrue(validation["valid"], validation)
            workflow = json.loads((self.repo_root / "configs" / "workflow" / f"{workflow_name}.json").read_text(encoding="utf-8"))
            self.assertEqual(sum(node.get("class_type") == "MiniMaxH3ImageToVideo" for node in workflow.values()), 1)
            self.assertEqual(sum(node.get("class_type") == "SaveVideo" for node in workflow.values()), 1)

    def test_fasth3_workflow_uses_trained_v2_contract(self) -> None:
        workflow = json.loads((self.repo_root / "configs" / "workflow" / "minimax_h3_fasth3_v2_t2v.json").read_text(encoding="utf-8"))
        self.assertEqual(workflow["6"]["inputs"]["shift_video"], 10.0)
        self.assertEqual(workflow["7"]["inputs"]["steps"], 8)
        self.assertEqual(workflow["17"]["inputs"]["selection"], "vsa")
        self.assertEqual(workflow["17"]["inputs"]["selection.keep_percent"], 10.0)
        self.assertEqual(workflow["17"]["inputs"]["sink_conditioning"], "exact_kv_and_rows")

    def test_benchmark_preparation_keeps_same_prompt_and_seed_contract(self) -> None:
        for workflow_name in ("minimax_h3_lowvram_t2v", "minimax_h3_native_int8_t2v", "minimax_h3_fasth3_v2_t2v"):
            workflow = json.loads((self.repo_root / "configs" / "workflow" / f"{workflow_name}.json").read_text(encoding="utf-8"))
            prepared = self.benchmark._prepare_workflow(workflow, seed=123, run_prefix="test/run", steps=8 if "fasth3" in workflow_name else 20)
            prompt_nodes = [node for node in prepared.values() if node.get("class_type") == "MiniMaxH3ImageToVideo"]
            self.assertEqual(len(prompt_nodes), 1)
            self.assertEqual(prompt_nodes[0]["inputs"]["prompt"], self.benchmark.PROMPT)
            self.assertEqual(prompt_nodes[0]["inputs"]["width"], 608)
            self.assertEqual(prompt_nodes[0]["inputs"]["height"], 352)
            self.assertEqual(prompt_nodes[0]["inputs"]["length"], 124)
            self.assertEqual(next(node for node in prepared.values() if node.get("class_type") == "RandomNoise")["inputs"]["noise_seed"], 123)

    def test_benchmark_preparation_accepts_prompt_override(self) -> None:
        workflow = json.loads((self.repo_root / "configs" / "workflow" / "minimax_h3_fasth3_v2_t2v.json").read_text(encoding="utf-8"))
        prompt = "A different test story with a blue robot and a red flower."
        prepared = self.benchmark._prepare_workflow(workflow, seed=456, run_prefix="test/override", steps=8, prompt=prompt)
        prompt_node = next(node for node in prepared.values() if node.get("class_type") == "MiniMaxH3ImageToVideo")
        self.assertEqual(prompt_node["inputs"]["prompt"], prompt)


if __name__ == "__main__":
    unittest.main()
