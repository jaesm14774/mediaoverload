from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agentic.app.character_workflow import build_goal_payload_from_character_config
from agentic.app.main import build_runtime
from agentic.runtime.contracts import RunState
from agentic.skills.agent_primitives import AgentMediaSkills
from agentic.skills.longvideo import LongVideoSkills
from agentic.tools.comfy_workflow_tool import ComfyWorkflowToolset


class H3ModePlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.config_path = cls.repo_root / "configs" / "characters" / "kirby.yaml"

    def _plan(self, generation_type: str, *, no_review: bool = False):
        payload = build_goal_payload_from_character_config(
            self.repo_root,
            self.config_path,
            prompt="Kirby protects a glowing seed while a storm crosses the meadow",
            preferred_generation_type=generation_type,
            publish_after_generate=False,
            no_review=no_review,
        )
        planner, _runner, _memory = build_runtime(
            self.repo_root / "agentic",
            output_root=self.repo_root / ".tmp-tests" / generation_type,
            comfy_host="127.0.0.1",
            comfy_port=8188,
        )
        goal = planner.create_goal(
            prompt=payload["prompt"],
            media_type=payload["media_type"],
            duration_seconds=payload["duration_seconds"],
            style=payload["style"],
            auto_download_assets=False,
            constraints=payload["constraints"],
        )
        return planner.build_plan(goal)

    def test_l2va_plan_has_last_frame_only_gate(self) -> None:
        plan = self._plan("native_h3_l2va_story")
        node_ids = [node.node_id for node in plan.nodes]
        self.assertIn("native-l2va-ending-review", node_ids)
        self.assertIn("native-l2va-frame-gate", node_ids)
        self.assertNotIn("native-opening-keyframe", node_ids)
        gate = next(node for node in plan.nodes if node.node_id == "native-l2va-frame-gate")
        render = next(node for node in plan.nodes if node.node_id == "native-h3-render")
        self.assertTrue(gate.inputs["preserve_last_frame"])
        self.assertEqual(render.skill_name, "longvideo.render_native_h3_l2va")
        self.assertEqual(plan.metadata["render_mode"], "last_frame_to_video")
        self.assertEqual(render.inputs["width"], 512)
        self.assertEqual(render.inputs["height"], 288)
        self.assertEqual(render.inputs["length"], 124)
        self.assertEqual(render.inputs["model_profile"], "q2")
        self.assertTrue(plan.metadata["native_h3"]["lowvram_preview"])

    def test_fl2va_plan_is_first_and_last_frame_workflow(self) -> None:
        plan = self._plan("native_h3_fl2va_story", no_review=True)
        node_ids = [node.node_id for node in plan.nodes]
        self.assertIn("native-opening-keyframe", node_ids)
        self.assertIn("native-ending-keyframe", node_ids)
        self.assertNotIn("native-opening-review", node_ids)
        self.assertNotIn("native-ending-review", node_ids)
        render = next(node for node in plan.nodes if node.node_id == "native-h3-render")
        self.assertEqual(render.inputs["h3_mode"], "fl2va")
        self.assertTrue(render.inputs["use_last_frame"])
        self.assertEqual(plan.workflow_name, "minimax_h3_lowvram_15s_fl2va_i2v")
        self.assertEqual(plan.metadata["recipe"], "native_h3_fl2va_story")

    def test_ref2va_plan_records_manifest_and_disables_reference_audio(self) -> None:
        plan = self._plan("native_h3_ref2va")
        reference = next(node for node in plan.nodes if node.node_id == "native-ref2va-reference-check")
        render = next(node for node in plan.nodes if node.node_id == "native-h3-render")
        self.assertEqual(plan.workflow_name, "minimax_h3_ref2va")
        self.assertEqual(reference.skill_name, "longvideo.validate_native_h3_references")
        self.assertEqual(render.skill_name, "longvideo.render_native_h3_ref2va")
        self.assertFalse(plan.metadata["reference_audio_enabled"])
        self.assertNotIn("audio", str(plan.metadata["reference_manifest"]).lower())
        self.assertEqual(render.inputs["model_profile"], "q4")

    def test_auto_ref2va_plan_generates_candidates_then_uses_selected_references(self) -> None:
        plan = self._plan("text2image2native_h3_ref2va", no_review=False)
        node_ids = [node.node_id for node in plan.nodes]
        self.assertIn("native-image-asset-check", node_ids)
        self.assertIn("native-ref2va-reference-candidates", node_ids)
        self.assertIn("native-ref2va-reference-review", node_ids)
        review = next(node for node in plan.nodes if node.node_id == "native-ref2va-reference-review")
        self.assertTrue(review.inputs["require_human_review"])
        self.assertEqual(review.inputs["limit"], 4)
        candidates = next(node for node in plan.nodes if node.node_id == "native-ref2va-reference-candidates")
        self.assertEqual(candidates.inputs["image_count"], 6)
        reference_check = next(node for node in plan.nodes if node.node_id == "native-ref2va-reference-check")
        self.assertEqual(reference_check.depends_on, ["native-story-prompt", "native-ref2va-reference-review"])
        render = next(node for node in plan.nodes if node.node_id == "native-h3-render")
        self.assertEqual(render.skill_name, "longvideo.render_native_h3_ref2va")
        self.assertEqual(render.inputs["length"], 362)
        self.assertEqual(plan.metadata["recipe"], "text2image2native_h3_ref2va")
        self.assertEqual(
            plan.metadata["selected_workflows"],
            {"image": "kirby_keyframe_anima", "video": "minimax_h3_ref2va"},
        )

    def test_pre_video_native_t2v_uses_reviewed_first_frame_i2v(self) -> None:
        plan = self._plan("native_h3_t2v_story")
        opening = next(node for node in plan.nodes if node.node_id == "native-opening-keyframe")
        review = next(node for node in plan.nodes if node.node_id == "native-opening-review")
        render = next(node for node in plan.nodes if node.node_id == "native-h3-render")

        self.assertEqual(opening.inputs["image_count"], 6)
        self.assertEqual(review.inputs["limit"], 1)
        self.assertTrue(review.inputs["require_human_review"])
        self.assertEqual(render.skill_name, "longvideo.render_native_h3")
        self.assertEqual(plan.workflow_name, "minimax_h3_lowvram_i2v")

    def test_pre_video_l2va_uses_six_candidates_and_one_approved_last_frame(self) -> None:
        plan = self._plan("native_h3_l2va_story")
        ending = next(node for node in plan.nodes if node.node_id == "native-l2va-ending-keyframe")
        review = next(node for node in plan.nodes if node.node_id == "native-l2va-ending-review")
        gate = next(node for node in plan.nodes if node.node_id == "native-l2va-frame-gate")

        self.assertEqual(ending.inputs["image_count"], 6)
        self.assertEqual(review.inputs["limit"], 1)
        self.assertTrue(review.inputs["require_human_review"])
        self.assertEqual(gate.inputs["frame_node"], "native-l2va-ending-review")

    def test_auto_ref2va_enforces_routing_candidate_and_selection_bounds(self) -> None:
        payload = build_goal_payload_from_character_config(
            self.repo_root,
            self.config_path,
            prompt="Kirby protects a glowing seed",
            preferred_generation_type="text2image2native_h3_ref2va",
            publish_after_generate=False,
        )
        payload["constraints"]["image_count"] = 6
        payload["constraints"]["native_h3_reference_candidate_count"] = 1
        payload["constraints"]["native_h3_reference_selection_limit"] = 12
        planner, _runner, _memory = build_runtime(
            self.repo_root / "agentic",
            output_root=self.repo_root / ".tmp-tests" / "auto-ref2va-bounds",
            comfy_host="127.0.0.1",
            comfy_port=8188,
        )
        goal = planner.create_goal(
            prompt=payload["prompt"],
            media_type=payload["media_type"],
            duration_seconds=payload["duration_seconds"],
            style=payload["style"],
            auto_download_assets=False,
            constraints=payload["constraints"],
        )
        plan = planner.build_plan(goal)
        candidates = next(node for node in plan.nodes if node.node_id == "native-ref2va-reference-candidates")
        review = next(node for node in plan.nodes if node.node_id == "native-ref2va-reference-review")
        self.assertEqual(candidates.inputs["image_count"], 6)
        self.assertEqual(review.inputs["limit"], 4)

    def test_direct_ref2va_no_review_does_not_create_a_review_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "identity.png"
            image.write_bytes(b"image")
            payload = build_goal_payload_from_character_config(
                self.repo_root,
                self.config_path,
                prompt="Use the reviewed character reference",
                preferred_generation_type="native_h3_ref2va",
                publish_after_generate=False,
                no_review=True,
            )
            payload["constraints"]["media_paths"] = [str(image)]
            planner, _runner, _memory = build_runtime(
                self.repo_root / "agentic",
                output_root=self.repo_root / ".tmp-tests" / "direct-ref2va-no-review",
                comfy_host="127.0.0.1",
                comfy_port=8188,
            )
            goal = planner.create_goal(
                prompt=payload["prompt"],
                media_type=payload["media_type"],
                duration_seconds=payload["duration_seconds"],
                style=payload["style"],
                auto_download_assets=False,
                constraints=payload["constraints"],
            )
            plan = planner.build_plan(goal)
        node_ids = [node.node_id for node in plan.nodes]
        self.assertIn("native-ref2va-reference-candidates", node_ids)
        self.assertNotIn("native-ref2va-reference-review", node_ids)
        reference_check = next(node for node in plan.nodes if node.node_id == "native-ref2va-reference-check")
        self.assertEqual(reference_check.depends_on, ["native-story-prompt", "native-ref2va-reference-candidates"])

    def test_auto_ref2va_no_review_uses_one_deterministic_reference(self) -> None:
        plan = self._plan("text2image2native_h3_ref2va", no_review=True)
        node_ids = [node.node_id for node in plan.nodes]
        self.assertNotIn("native-ref2va-reference-review", node_ids)
        reference_check = next(node for node in plan.nodes if node.node_id == "native-ref2va-reference-check")
        self.assertIn("native-ref2va-reference-candidates", reference_check.depends_on)
        self.assertEqual(reference_check.inputs["selection_limit"], 1)
        self.assertTrue(reference_check.inputs["auto_reference_generation"])

    def test_ref2va_candidate_media_goes_through_multi_asset_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "identity.png"
            video = root / "motion.mp4"
            image.write_bytes(b"image")
            video.write_bytes(b"video")
            payload = build_goal_payload_from_character_config(
                self.repo_root,
                self.config_path,
                prompt="Kirby protects a glowing seed",
                preferred_generation_type="native_h3_ref2va",
                publish_after_generate=False,
            )
            payload["constraints"]["media_paths"] = [str(image), str(video)]
            payload["constraints"]["native_h3_reference_selection_limit"] = 4
            planner, _runner, _memory = build_runtime(
                self.repo_root / "agentic",
                output_root=self.repo_root / ".tmp-tests" / "ref2va-review",
                comfy_host="127.0.0.1",
                comfy_port=8188,
            )
            goal = planner.create_goal(
                prompt=payload["prompt"],
                media_type=payload["media_type"],
                duration_seconds=payload["duration_seconds"],
                style=payload["style"],
                auto_download_assets=False,
                constraints=payload["constraints"],
            )
            plan = planner.build_plan(goal)

        node_ids = [node.node_id for node in plan.nodes]
        self.assertIn("native-ref2va-reference-candidates", node_ids)
        self.assertIn("native-ref2va-reference-review", node_ids)
        reference_check = next(node for node in plan.nodes if node.node_id == "native-ref2va-reference-check")
        self.assertIn("native-ref2va-reference-review", reference_check.depends_on)
        review = next(node for node in plan.nodes if node.node_id == "native-ref2va-reference-review")
        self.assertEqual(review.inputs["limit"], 4)
        self.assertTrue(review.inputs["review_all_candidates"])

    def test_ref2va_only_collects_review_selected_assets(self) -> None:
        context = SimpleNamespace(
            node=SimpleNamespace(depends_on=["reference-review"]),
            state=RunState(
                goal={},
                metadata={},
                node_outputs={
                    "reference-review": {
                        "selected_assets": ["identity.png", "motion.mp4"],
                        "media_paths": ["identity.png"],
                    }
                },
            ),
        )
        self.assertEqual(
            LongVideoSkills._collect_accepted_reference_assets(context),
            ["identity.png", "motion.mp4"],
        )

    def test_auto_ref2va_limits_generated_saved_files_when_review_is_disabled(self) -> None:
        context = SimpleNamespace(
            node=SimpleNamespace(
                depends_on=["reference-candidates"],
                inputs={"auto_reference_generation": True, "selection_limit": 1},
            ),
            state=RunState(
                goal={},
                metadata={},
                node_outputs={
                    "reference-candidates": {
                        "saved_files": ["one.png", "two.png", "three.png"],
                    }
                },
            ),
        )
        self.assertEqual(
            LongVideoSkills._collect_accepted_reference_assets(context),
            ["one.png"],
        )

    def test_kirby_batch_keyframe_validates_every_candidate(self) -> None:
        class FakeTools:
            def call(self, _name, _payload):
                return {"saved_files": ["one.png", "two.png"]}

        context = SimpleNamespace(
            node=SimpleNamespace(
                inputs={
                    "workflow_name": "kirby_keyframe_anima",
                    "image_count": 2,
                    "max_regenerations": 0,
                },
                depends_on=[],
            ),
            plan=SimpleNamespace(
                goal=SimpleNamespace(
                    prompt="Kirby protects a glowing seed",
                    style="anime",
                    constraints={"character": "kirby"},
                )
            ),
            state=SimpleNamespace(node_outputs={}),
        )
        report = SimpleNamespace(passed=True, reasons=[])
        with patch("agentic.skills.agent_primitives.inspect_kirby_input", return_value=report) as inspect:
            result = AgentMediaSkills(FakeTools(), self.repo_root / ".tmp-tests" / "batch-qa").generate_keyframe(context)
        self.assertEqual(result.status, "success")
        self.assertEqual(inspect.call_count, 2)

    def test_ref2va_deduplicates_selected_asset_against_string_manifest_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "identity.png"
            image.write_bytes(b"image")
            context = SimpleNamespace(
                node=SimpleNamespace(
                    inputs={"reference_manifest": [str(image)]},
                    depends_on=["reference-review"],
                ),
                state=RunState(
                    goal={},
                    metadata={},
                    node_outputs={"reference-review": {"selected_assets": [str(image)]}},
                ),
                plan=SimpleNamespace(goal=SimpleNamespace(constraints={})),
            )
            result = LongVideoSkills.__new__(LongVideoSkills).validate_native_h3_references(context)
        self.assertEqual(len(result.outputs["reference_manifest"]), 1)
        self.assertEqual(result.outputs["reference_manifest"][0]["path"], str(image))

    def test_ordinary_i2v_resolves_only_the_first_approved_image(self) -> None:
        context = SimpleNamespace(
            node=SimpleNamespace(inputs={}, depends_on=["image-review"]),
            state=RunState(
                goal={},
                metadata={},
                node_outputs={
                    "image-review": {
                        "selected_assets": ["first.png", "second.png"],
                    }
                },
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            resolved = AgentMediaSkills(SimpleNamespace(), Path(directory))._resolve_image_path(context)
        self.assertEqual(resolved, "first.png")

    def test_last_frame_validator_does_not_regenerate_approved_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "last.png"
            frame.write_bytes(b"approved")
            context = SimpleNamespace(
                state=SimpleNamespace(node_outputs={"ending": {"saved_files": [str(frame)]}}),
                node=SimpleNamespace(
                    inputs={"frame_node": "ending", "preserve_last_frame": True},
                    depends_on=["ending"],
                ),
                plan=SimpleNamespace(goal=SimpleNamespace(prompt="test", constraints={})),
            )
            result = AgentMediaSkills(SimpleNamespace(), Path(directory)).validate_last_frame(context)
            self.assertEqual(result.outputs["last_frame_path"], str(frame))
            self.assertEqual(result.outputs["identity_reports"][0]["validation"], "human_selected_immutable")
            self.assertEqual(result.outputs["regenerated_count"], 0)

    def test_ref2va_graph_is_empty_until_runtime_references_are_bound(self) -> None:
        workflow_path = self.repo_root / "configs" / "workflow" / "minimax_h3_ref2va.json"
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))

        self.assertEqual(workflow["31"]["class_type"], "SamplerCustomAdvanced")
        self.assertEqual(workflow["32"]["inputs"]["samples"], ["31", 0])
        self.assertEqual(workflow["33"]["inputs"]["samples"], ["31", 0])
        self.assertEqual(workflow["34"]["inputs"]["images"], ["32", 0])
        self.assertEqual(workflow["34"]["inputs"]["audio"], ["33", 0])
        self.assertEqual(workflow["35"]["inputs"]["video"], ["34", 0])
        self.assertFalse(any(node.get("class_type") == "LoadImage" for node in workflow.values()))
        self.assertFalse(any(node.get("class_type") == "VHS_LoadVideoPath" for node in workflow.values()))
        self.assertFalse(
            any(
                key.startswith(("ref_images.", "ref_videos.", "ref_audios.", "ref_video_audios."))
                for key in workflow["5"]["inputs"]
            )
        )

        toolset = ComfyWorkflowToolset.__new__(ComfyWorkflowToolset)
        image_only = toolset._build_runtime_reference_workflow(
            workflow,
            [{"path": "identity.png", "type": "image"}],
            {"width": 608, "height": 352},
        )
        video_only = toolset._build_runtime_reference_workflow(
            workflow,
            [{"path": "motion.mp4", "type": "video"}],
            {"width": 608, "height": 352},
        )
        mixed = toolset._build_runtime_reference_workflow(
            workflow,
            [
                {"path": "identity.png", "type": "image"},
                {"path": "motion.mp4", "type": "video"},
            ],
            {"width": 608, "height": 352},
        )

        self.assertEqual(sum(node.get("class_type") == "LoadImage" for node in image_only.values()), 1)
        self.assertEqual(sum(node.get("class_type") == "VHS_LoadVideoPath" for node in image_only.values()), 0)
        self.assertEqual(list(image_only["5"]["inputs"]), ["clip", "vae", "audio_vae", "prompt", "width", "height", "length", "ref_image_size", "ref_images.ref_image_0"])
        self.assertEqual(sum(node.get("class_type") == "LoadImage" for node in video_only.values()), 0)
        self.assertEqual(sum(node.get("class_type") == "VHS_LoadVideoPath" for node in video_only.values()), 1)
        self.assertNotIn("ref_images.ref_image_0", video_only["5"]["inputs"])
        self.assertEqual(sum(node.get("class_type") == "LoadImage" for node in mixed.values()), 1)
        self.assertEqual(sum(node.get("class_type") == "VHS_LoadVideoPath" for node in mixed.values()), 1)
        self.assertEqual(mixed["5"]["inputs"]["ref_images.ref_image_0"][1], 0)
        self.assertEqual(mixed["5"]["inputs"]["ref_videos.ref_video_0"][1], 0)

        for node in workflow.values():
            if node.get("class_type") == "LoadImage":
                self.assertNotIn("__unused_", str(node["inputs"].get("image")))
            if node.get("class_type") == "VHS_LoadVideoPath":
                self.assertNotIn("__unused_", str(node["inputs"].get("video")))


if __name__ == "__main__":
    unittest.main()
