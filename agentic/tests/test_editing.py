from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from agentic.runtime.editing import EditClip, EditPlan, EditPlanError, EditTransition, build_edit_plan
from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest, RunState, SkillContext
from agentic.runtime.registry import ToolRegistry
from agentic.skills.editing import EditingSkills
from agentic.tools.editing_adapter import EditRenderError, OpenCutEditAdapter


class EditPlanTests(unittest.TestCase):
    def test_editorial_profile_is_deterministic_and_varied(self) -> None:
        paths = ["/tmp/one.mp4", "/tmp/two.mp4", "/tmp/three.mp4", "/tmp/four.mp4"]
        first = build_edit_plan(paths, profile="editorial_kinetic_v1", variant_seed=7)
        second = build_edit_plan(paths, profile="editorial_kinetic_v1", variant_seed=7)

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(len(first.transitions), 3)
        self.assertGreaterEqual(len({transition.name for transition in first.transitions}), 2)

    def test_image_motion_is_selected_only_for_stills(self) -> None:
        plan = build_edit_plan(
            ["/tmp/one.png", "/tmp/two.mp4"],
            profile="editorial_kinetic_v1",
            variant_seed=2,
        )

        self.assertNotEqual(plan.clips[0].motion, "none")
        self.assertEqual(plan.clips[1].motion, "none")

    def test_motion_cut_keeps_hard_cuts_and_rotates_bounded_still_motion(self) -> None:
        plan = build_edit_plan(
            ["/tmp/one.png", "/tmp/two.png", "/tmp/three.png"],
            profile="motion_cut_v1",
            variant_seed=1,
        )

        self.assertEqual(plan.transitions, ())
        self.assertEqual(
            [clip.motion for clip in plan.clips],
            ["pan_left", "slow_zoom_out", "pan_right"],
        )

    def test_creative_variant_adds_bounded_motion_to_video_clips(self) -> None:
        plan = build_edit_plan(
            ["/tmp/one.mp4", "/tmp/two.mp4", "/tmp/three.mp4"],
            profile="xfade_clean_v1",
        )
        variant = EditingSkills._transition_plan(
            plan,
            profile="baseline_concat",
            duration=0.0,
            seed=3,
        )

        self.assertEqual([clip.motion for clip in variant.clips], ["pan_right", "drift_up", "drift_down"])
        self.assertEqual(variant.transitions, ())

    def test_round_trip_preserves_timeline_contract(self) -> None:
        plan = EditPlan(
            clips=(EditClip("/tmp/one.png", duration_seconds=2.0, motion="pan_left"), EditClip("/tmp/two.mp4")),
            transitions=(EditTransition("fade", 0.2),),
            output_width=576,
            output_height=1024,
            target_duration_seconds=3.5,
            variant_seed=11,
        )

        restored = EditPlan.from_dict(plan.to_dict())
        self.assertEqual(restored, plan)

    def test_rejects_unsafe_timeline_shapes(self) -> None:
        with self.assertRaises(EditPlanError):
            EditPlan(clips=(EditClip("/tmp/one.mp4"),), output_width=577).validate()
        with self.assertRaises(EditPlanError):
            EditPlan(
                clips=(EditClip("/tmp/one.mp4"), EditClip("/tmp/two.mp4")),
                transitions=(EditTransition("fade", 0.2), EditTransition("fade", 0.2)),
            ).validate()
        with self.assertRaises(EditPlanError):
            EditPlan(clips=(EditClip("/tmp/one.mp4", motion="freeform_ffmpeg"),)).validate()
        with self.assertRaises(EditPlanError):
            EditPlan(clips=(EditClip("/tmp/one.mp4"),), fps=float("nan")).validate()
        with self.assertRaises(EditPlanError):
            EditPlan.from_dict({"clips": None, "transitions": []})
        with self.assertRaises(EditPlanError):
            EditPlan.from_dict({"clips": [{"path": "/tmp/one.mp4"}], "transitions": [None]})
        with self.assertRaises(EditPlanError):
            EditPlan.from_dict({"clips": [{"path": "/tmp/one.mp4"}], "audio_crossfade": "false"})
        with self.assertRaises(EditPlanError):
            EditPlan(
                clips=(EditClip("/tmp/one.mp4"), EditClip("/tmp/two.mp4")),
                profile="xfade_clean_v1",
                transitions=(),
            ).validate()
        with self.assertRaises(EditPlanError):
            EditPlan(clips=(EditClip("/tmp/one.mp4"),), output_width=8192).validate()

    def test_skill_keeps_outputs_under_configured_root(self) -> None:
        with TemporaryDirectory() as temp_dir:
            skills = EditingSkills(ToolRegistry(), Path(temp_dir))
            with self.assertRaisesRegex(ValueError, "configured output root"):
                skills._resolve_output_path(
                    Path(temp_dir).parent / "escape.mp4",
                    Path(temp_dir) / "editing" / "compose" / "edited.mp4",
                )

    def test_dependency_inputs_include_stills_and_video_in_order(self) -> None:
        context = SimpleNamespace(
            node=SimpleNamespace(inputs={}, depends_on=["render"]),
            state={
                "render": {
                    "saved_files": ["/tmp/still.png", "/tmp/clip.mp4", "/tmp/notes.json"],
                    "video_path": "/tmp/clip.mp4",
                }
            },
        )

        self.assertEqual(
            EditingSkills._input_paths(context),
            ["/tmp/still.png", "/tmp/clip.mp4"],
        )

    def test_adapter_rejects_unapproved_input_and_source_collision(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            source.write_bytes(b"not a real video")
            outside = root.parent / "outside.mp4"
            outside.write_bytes(b"not a real video")
            adapter = OpenCutEditAdapter(output_root=root, input_roots=[root])
            with self.assertRaisesRegex(EditRenderError, "approved media roots"):
                adapter._validate_input_path(outside.resolve())
            with self.assertRaisesRegex(EditRenderError, "cannot overwrite source"):
                adapter._reject_artifact_collisions({source.resolve()}, {source.resolve()})

    def test_materialize_rejects_final_path_matching_original_source(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original = root / "original.mp4"
            original.write_bytes(b"original")
            candidate = root / "candidate.mp4"
            candidate.write_bytes(b"candidate")
            candidate_manifest = root / "candidate.edit_manifest.json"
            candidate_manifest.write_text(
                '{"sources":[{"path":"' + str(original).replace("\\", "\\\\") + '"}]}',
                encoding="utf-8",
            )
            adapter = OpenCutEditAdapter(output_root=root, input_roots=[root])

            with self.assertRaisesRegex(EditRenderError, "cannot overwrite source"):
                adapter.materialize_result(
                    {"video_path": str(candidate), "manifest_path": str(candidate_manifest)},
                    output_path=str(original),
                )

    def test_adapter_rejects_symlink_input_before_resolution(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            source.write_bytes(b"not a real video")
            link = root / "link.mp4"
            try:
                link.symlink_to(source)
            except (OSError, NotImplementedError):
                self.skipTest("Symlinks are unavailable in this environment")
            adapter = OpenCutEditAdapter(output_root=root, input_roots=[root])
            with self.assertRaisesRegex(EditRenderError, "symlink path"):
                adapter._validate_input_path(link)

    def test_review_evidence_includes_hard_cut_boundaries(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            class FakeFFmpeg:
                def extract_frame_at(self, video_path: str, output_path: str, timestamp_seconds: float) -> str:
                    Path(output_path).write_bytes(f"{video_path}:{timestamp_seconds}".encode())
                    return output_path

            adapter = OpenCutEditAdapter.__new__(OpenCutEditAdapter)
            adapter._ffmpeg = FakeFFmpeg()
            evidence = adapter._build_review_evidence(
                root / "final.mp4",
                EditPlan(clips=(EditClip("/tmp/one.mp4"), EditClip("/tmp/two.mp4"), EditClip("/tmp/three.mp4"))),
                [2.0, 2.0, 2.0],
                {"duration": 6.0, "video_duration": 6.0, "frame_rate": 24.0},
                root / "review_frames",
            )

            self.assertEqual(len(evidence), 8)
            self.assertTrue(any("boundary_01_join_2.000s.jpg" in path for path in evidence))
            self.assertTrue(any("boundary_02_join_4.000s.jpg" in path for path in evidence))

    def test_creative_review_loop_uses_llm_recommendation_and_materializes_best_candidate(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            class FakeTools:
                def __init__(self) -> None:
                    self.calls: list[tuple[str, dict[str, object]]] = []

                def call(self, name: str, payload: dict[str, object]) -> dict[str, object]:
                    self.calls.append((name, payload))
                    if name == "media.compose_edit":
                        output = Path(str(payload["output_path"]))
                        output.parent.mkdir(parents=True, exist_ok=True)
                        return {
                            "video_path": str(output),
                            "manifest_path": str(payload["manifest_path"]),
                            "contact_sheet_path": str(payload["contact_sheet_path"]),
                            "review_evidence_paths": [str(payload.get("review_evidence_dir") or "")],
                        }
                    if name == "media.video_qa":
                        return {"passed": True}
                    return {
                        "video_path": str(payload["output_path"]),
                        "manifest_path": str(payload["manifest_path"]),
                        "contact_sheet_path": str(payload["contact_sheet_path"]),
                    }

            class FakePromptEngine:
                def __init__(self) -> None:
                    self.reviews = [
                        {
                            "enabled": True,
                            "required": True,
                            "passed": False,
                            "status": "fail",
                            "score": 40,
                            "next_change": "hard_cut",
                            "issues": ["crossfade ghosting"],
                        },
                        {
                            "enabled": True,
                            "required": True,
                            "passed": True,
                            "status": "pass",
                            "score": 86,
                            "next_change": "keep",
                            "issues": [],
                        },
                    ]
                    self.calls: list[dict[str, object]] = []

                def evaluate_edit_contact_sheet(self, **kwargs: object) -> dict[str, object]:
                    self.calls.append(kwargs)
                    return self.reviews.pop(0)

            fake_tools = FakeTools()
            fake_prompt = FakePromptEngine()
            skills = EditingSkills(fake_tools, root, prompt_engine=fake_prompt)  # type: ignore[arg-type]
            goal = GoalRequest(
                prompt="turn connected shots into a fashion reel",
                media_type="image_sequence_edit",
                duration_seconds=3,
                style="editorial",
            )
            node = ExecutionNode(
                node_id="compose-edit",
                skill_name="media.video.compose_timeline",
                inputs={
                    "input_paths": ["/tmp/one.mp4", "/tmp/two.mp4"],
                    "profile": "xfade_clean_v1",
                    "creative_review": True,
                    "creative_review_max_attempts": 3,
                },
            )
            context = SkillContext(
                plan=ExecutionPlan(goal=goal, workflow_name="image_sequence_edit_v1", nodes=[node]),
                node=node,
                state=RunState(goal={}, metadata={}),
            )

            result = skills.compose_timeline(context)

            self.assertEqual(result.status, "success")
            self.assertEqual(len(fake_prompt.calls), 2)
            self.assertEqual(fake_prompt.calls[1]["plan"]["profile"], "baseline_concat")
            self.assertEqual(result.outputs["creative_review"]["selected"]["attempt"], 2)
            self.assertTrue(Path(result.outputs["creative_review_path"]).is_file())
            self.assertEqual(
                [name for name, _ in fake_tools.calls],
                [
                    "media.compose_edit",
                    "media.video_qa",
                    "media.compose_edit",
                    "media.video_qa",
                    "media.materialize_edit",
                ],
            )

    def test_creative_review_loop_fails_closed_and_persists_rejection_receipt(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            class FakeTools:
                def call(self, name: str, payload: dict[str, object]) -> dict[str, object]:
                    if name == "media.video_qa":
                        return {"passed": True}
                    output = Path(str(payload.get("output_path") or root / "candidate.mp4"))
                    output.parent.mkdir(parents=True, exist_ok=True)
                    return {
                        "video_path": str(output),
                        "manifest_path": str(payload.get("manifest_path") or output.with_suffix(".json")),
                        "contact_sheet_path": str(payload.get("contact_sheet_path") or output.with_suffix(".jpg")),
                        "review_evidence_paths": [],
                    }

            class RejectingPromptEngine:
                def evaluate_edit_contact_sheet(self, **kwargs: object) -> dict[str, object]:
                    return {"enabled": True, "required": True, "passed": False, "status": "fail", "score": 20, "next_change": "hard_cut"}

            skills = EditingSkills(FakeTools(), root, prompt_engine=RejectingPromptEngine())  # type: ignore[arg-type]
            goal = GoalRequest(prompt="reject weak edit", media_type="image_sequence_edit", duration_seconds=3)
            node = ExecutionNode(
                node_id="compose-edit",
                skill_name="media.video.compose_timeline",
                inputs={"input_paths": ["/tmp/one.mp4", "/tmp/two.mp4"], "creative_review": True, "creative_review_max_attempts": 2},
            )
            context = SkillContext(
                plan=ExecutionPlan(goal=goal, workflow_name="image_sequence_edit_v1", nodes=[node]),
                node=node,
                state=RunState(goal={}, metadata={}),
            )

            result = skills.compose_timeline(context)

            self.assertEqual(result.status, "failed")
            self.assertEqual(result.outputs["creative_review"]["status"], "rejected")
            self.assertTrue(Path(result.outputs["creative_review_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
