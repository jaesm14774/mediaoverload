from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from agentic.app.character_workflow import (
    _build_h3_reference_runtime_context,
    build_goal_payload_from_character_config,
)
from character_workflow_helpers import make_character_workflow_request


class RoutingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.routing_path = cls.repo_root / "configs" / "routing.yaml"
        cls.character_path = cls.repo_root / "configs" / "characters" / "kirby.yaml"
        cls.routing = yaml.safe_load(cls.routing_path.read_text(encoding="utf-8"))["routing"]
        cls.hints = cls.routing["routing_hints"]

    def test_every_candidate_workflow_has_preference_and_description(self) -> None:
        workflow_names = {
            workflow_name
            for stages in self.routing["workflow_stage_candidates"].values()
            for workflow_names in stages.values()
            for workflow_name in workflow_names
        }
        self.assertTrue(workflow_names)
        self.assertEqual(
            workflow_names - set(self.hints["workflow_preferences"]),
            set(),
        )
        self.assertEqual(
            workflow_names - set(self.hints["workflow_descriptions"]),
            set(),
        )

    def test_h3_strategy_contracts_match_routing_semantics(self) -> None:
        descriptions = self.hints["strategy_descriptions"]
        native_story = descriptions["native_h3_story"]
        self.assertIn("First Frame I2VA", native_story["summary"])
        self.assertNotIn("First+Last Frame", native_story["summary"])
        self.assertTrue(any("first frame" in rule.lower() for rule in native_story["hard_rules"]))
        self.assertIn("First+Last Frame", descriptions["native_h3_fl2va_story"]["summary"])
        self.assertIn("last frame", descriptions["native_h3_l2va_story"]["summary"].lower())
        self.assertIn("T2V", descriptions["text2video"]["summary"])
        self.assertIn("video conditioning", " ".join(descriptions["text2video"]["hard_rules"]).lower())
        self.assertEqual(
            self.routing["workflow_stage_candidates"]["text2image2video"]["video_workflow_name"][0],
            "minimax_h3_lowvram_i2v",
        )
        self.assertEqual(
            self.routing["workflow_stage_candidates"]["text2longvideo"]["video_workflow_name"][0],
            "minimax_h3_lowvram_i2v",
        )

        stage_contracts = self.hints["workflow_stage_contracts"]
        self.assertIn("previous segment's rendered tail", stage_contracts["text2longvideo"]["video_workflow_name"])
        for strategy in self.routing["strategy_candidates"]:
            self.assertIn(strategy, stage_contracts)
        self.assertIn("not connected as video conditioning", stage_contracts["text2video"]["image_workflow_name"])
        self.assertIn("first_frame", stage_contracts["native_h3_story"]["video_workflow_name"])
        self.assertIn("T2V", stage_contracts["native_h3_t2v_story"]["video_workflow_name"])
        self.assertIn("last_frame", stage_contracts["native_h3_fl2va_story"]["video_workflow_name"])
        self.assertIn("reference images/videos", stage_contracts["native_h3_ref2va"]["video_workflow_name"])
        self.assertIn("segment", stage_contracts["text2longvideo"]["video_workflow_name"])
        self.assertEqual(self.routing["count_policies"]["text2longvideo"]["segment_count"], {"min": 4, "max": 8})
        longvideo_config = self.routing["longvideo_config"]
        self.assertEqual(longvideo_config["default_duration_seconds"], 30)
        self.assertEqual(longvideo_config["segment_duration"], 5)
        self.assertEqual(longvideo_config["storyboard_path"], "configs/storyboards/text2longvideo_story.yaml")
        auto_contract = stage_contracts["text2image2native_h3_ref2va"]
        self.assertIn("six", auto_contract["image_workflow_name"])
        self.assertIn("explicitly selected", auto_contract["video_workflow_name"])

    def test_shared_pre_video_review_contract_is_fixed_six_and_bounded(self) -> None:
        pre_video = self.routing["pre_video_review"]
        self.assertTrue(pre_video["enabled"])
        self.assertEqual(pre_video["candidate_count"], 6)
        self.assertEqual(pre_video["default_selection_limit"], 1)
        self.assertEqual(pre_video["ref2va_selection_limit"], 4)
        for strategy in (
            "text2video",
            "text2image2video",
            "text2longvideo",
            "native_h3_story",
            "native_h3_t2v_story",
            "native_h3_fl2va_story",
            "native_h3_l2va_story",
            "text2image2native_h3_ref2va",
        ):
            self.assertEqual(self.routing["count_policies"][strategy]["image_count"], {"min": 6, "max": 6})

    def test_kirby_weighted_pool_covers_all_routes_with_requested_weights(self) -> None:
        config = yaml.safe_load(self.character_path.read_text(encoding="utf-8"))
        self.assertEqual(
            config["generation"]["generation_type_weights"],
            {
                "text2img": 1,
                "text2image2video": 1,
                "text2longvideo": 2,
                "native_h3_story": 1,
                "native_h3_t2v_story": 1,
                "native_h3_fl2va_story": 1,
                "native_h3_l2va_story": 1,
                "native_h3_ref2va": 1,
                "text2image2native_h3_ref2va": 1,
                "sticker_pack": 2,
            },
        )

    def test_workflow_graph_inputs_match_h3_stage_contracts(self) -> None:
        def graph_inputs(name: str) -> dict[str, object]:
            graph = json.loads(
                (self.repo_root / "configs" / "workflow" / f"{name}.json").read_text(encoding="utf-8")
            )
            node = next(
                node
                for node in graph.values()
                if node.get("class_type") in {"MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo"}
            )
            return node["inputs"]

        fl2va = graph_inputs("minimax_h3_lowvram_15s_fl2va_i2v")
        self.assertIn("first_frame", fl2va)
        self.assertIn("last_frame", fl2va)

        t2v = graph_inputs("minimax_h3_native_t2v")
        self.assertNotIn("first_frame", t2v)
        self.assertNotIn("last_frame", t2v)

        i2v = graph_inputs("minimax_h3_lowvram_i2v")
        self.assertIn("first_frame", i2v)
        self.assertNotIn("last_frame", i2v)

        ref2va = graph_inputs("minimax_h3_ref2va")
        ref_node = next(
            node
            for node in json.loads(
                (self.repo_root / "configs" / "workflow" / "minimax_h3_ref2va.json").read_text(encoding="utf-8")
            ).values()
            if node.get("class_type") == "MiniMaxH3ReferenceToVideo"
        )
        self.assertEqual(ref_node["class_type"], "MiniMaxH3ReferenceToVideo")
        self.assertNotIn("ref_audios", ref2va)

    def test_automatic_routing_keeps_ref2va_available_for_candidate_generation(self) -> None:
        route_result = {
            "generation_type": "text2video",
            "workflow_plan": {
                "image_workflow_name": "",
                "video_workflow_name": "minimax_h3_lowvram_t2v",
                "refine_workflow_name": "",
                "transition_workflow_name": "",
                "upscale_workflow_name": "",
            },
            "count_plan": {
                "image_count": 1,
                "video_count": 1,
                "segment_count": 1,
                "review_selection_limit": 1,
                "sticker_expression_count": 1,
                "images_per_prompt": 1,
            },
            "reason": "T2V is the best available route.",
            "prompt_mode": "llm",
        }
        with patch(
            "agentic.app.character_workflow.LLMPromptEngine.route_generation_strategy",
            return_value=route_result,
        ) as route:
            payload = build_goal_payload_from_character_config(make_character_workflow_request(
                self.repo_root,
                self.character_path,
                prompt="A free single-shot motion test with no reference assets",
                publish_after_generate=False,
            ))

        request = route.call_args.args[0]
        self.assertIn("native_h3_ref2va", request.generation_type_candidates)
        self.assertIn("text2image2native_h3_ref2va", request.generation_type_candidates)
        context = request.routing_hints["runtime_context"]
        self.assertFalse(context["reference_manifest_available"])
        self.assertTrue(context["automatic_ref2va_eligible"])
        self.assertEqual(context["reference_manifest_error_code"], "auto_generation_available")
        self.assertEqual(payload["constraints"]["routing_runtime_context"], context)

    def test_valid_configured_reference_manifest_enables_automatic_ref2va(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "routing-reference.png"
            reference.write_bytes(b"test")
            with self.subTest("validated image reference"):
                context = _build_h3_reference_runtime_context(
                    self.repo_root,
                    {"native_h3_ref2va": {"reference_manifest": [str(reference)]}},
                )

        self.assertTrue(context["reference_manifest_available"])
        self.assertTrue(context["automatic_ref2va_eligible"])
        self.assertEqual(context["reference_image_count"], 1)
        self.assertEqual(context["reference_video_count"], 0)
        self.assertEqual(context["reference_manifest_error_code"], "")

    def test_manual_ref2va_override_remains_available_for_later_review_binding(self) -> None:
        payload = build_goal_payload_from_character_config(make_character_workflow_request(
            self.repo_root,
            self.character_path,
            prompt="Use the reviewed character and motion references",
            preferred_generation_type="native_h3_ref2va",
            publish_after_generate=False,
        ))
        self.assertEqual(payload["source_generation_type"], "native_h3_ref2va")
        self.assertTrue(payload["constraints"]["routing_runtime_context"]["automatic_ref2va_eligible"])
        self.assertTrue(payload["constraints"]["auto_reference_generation"])
        self.assertEqual(payload["constraints"]["generation_type_candidates"], ["native_h3_ref2va"])



if __name__ == "__main__":
    unittest.main()
