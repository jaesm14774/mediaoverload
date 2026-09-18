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




if __name__ == "__main__":
    unittest.main()
