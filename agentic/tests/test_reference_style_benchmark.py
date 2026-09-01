from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from agentic.runtime.reference_style_benchmark import (
    collect_reference_items,
    effective_k_sampler_seed,
    select_reference_items,
    stable_seed,
)
from agentic.runtime.llm_engine import compute_reference_style_score


class ReferenceStyleBenchmarkTests(unittest.TestCase):
    def test_collects_images_and_marks_screenshot_conditioning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (32, 24), "pink").save(root / "clean.jpg")
            Image.new("RGB", (24, 32), "white").save(root / "螢幕擷取畫面 01.png")
            (root / "clip.mp4").write_bytes(b"placeholder")

            items, videos = collect_reference_items(root)

            self.assertEqual(len(items), 2)
            self.assertEqual(len(videos), 1)
            self.assertFalse(items[0]["likely_screenshot"])
            self.assertTrue(items[1]["likely_screenshot"])
            self.assertTrue(items[0]["img2img_eligible"])
            self.assertFalse(items[1]["img2img_eligible"])

    def test_selection_is_even_and_seed_is_stable(self) -> None:
        items = [{"item_id": f"image-{index}"} for index in range(10)]
        selected = select_reference_items(items, 4)

        self.assertEqual([item["item_id"] for item in selected], ["image-0", "image-3", "image-6", "image-9"])
        self.assertEqual(stable_seed("image-0", 20260830), stable_seed("image-0", 20260830))
        self.assertNotEqual(stable_seed("image-0", 20260830), stable_seed("image-1", 20260830))

    def test_effective_seed_uses_matching_node_index_not_comfy_node_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workflow.json"
            path.write_text(json.dumps({"10": {"class_type": "KSampler", "inputs": {"seed": 1}}}), encoding="utf-8")
            self.assertEqual(effective_k_sampler_seed(path, 42), 42)

    def test_reference_style_score_is_computed_from_dimensions(self) -> None:
        score, weights = compute_reference_style_score(
            {
                "style_grammar": 100,
                "palette_lighting": 0,
                "composition": 0,
                "subject_clarity": 0,
                "creative_beat": 0,
            }
        )
        self.assertEqual(score, 30)
        self.assertEqual(sum(weights.values()), 100)


if __name__ == "__main__":
    unittest.main()
