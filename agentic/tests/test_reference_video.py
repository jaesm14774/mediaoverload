from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from agentic.app.main import build_runtime
from agentic.runtime.reference_video import ReferenceVideoAnalyzer, ReferenceVideoError
from agentic.runtime.reference_video import format_reference_video_directive
from character_workflow_helpers import make_character_workflow_request
from agentic.app.character_workflow import build_goal_payload_from_character_config


class ReferenceVideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.config_path = cls.repo_root / "configs" / "characters" / "kirby.yaml"

    def test_reference_video_node_is_added_only_when_source_is_present(self) -> None:
        planner, _runner, _memory = build_runtime(
            self.repo_root,
            output_root=self.repo_root / ".tmp-tests" / "reference-video-plan",
            comfy_host="127.0.0.1",
            comfy_port=8188,
        )
        base_constraints = {
            "character": "Kirby",
            "native_h3_storyboard_path": str(self.repo_root / "configs" / "storyboards" / "native_h3_15s.yaml"),
            "native_h3_workflow_name": "minimax_h3_lowvram_15s_fl2va_i2v",
            "native_h3_keyframe_workflow_name": "krea2_turbo",
            "native_h3_refine_workflow_name": "krea2_turbo_img2img",
            "pre_video_review_enabled": False,
            "require_human_review": False,
        }

        baseline = planner.build_plan(
            planner.create_goal(
                prompt="A tiny storm creates a tactile Kirby gag",
                media_type="native_h3_story",
                duration_seconds=15,
                style="tactile pastel 2D anime",
                auto_download_assets=False,
                constraints=base_constraints,
            )
        )
        self.assertNotIn("reference-video-analysis", [node.node_id for node in baseline.nodes])

        with_reference = dict(base_constraints)
        with_reference.update(
            {
                "reference_video_source": "C:/references/clip.mp4",
                "reference_video_depth": "deep",
                "reference_video_max_keyframes": 16,
            }
        )
        plan = planner.build_plan(
            planner.create_goal(
                prompt="A tiny storm creates a tactile Kirby gag",
                media_type="native_h3_story",
                duration_seconds=15,
                style="tactile pastel 2D anime",
                auto_download_assets=False,
                constraints=with_reference,
            )
        )
        node_ids = [node.node_id for node in plan.nodes]
        self.assertEqual(node_ids[0], "reference-video-analysis")
        story = next(node for node in plan.nodes if node.node_id == "native-story-prompt")
        self.assertEqual(story.depends_on, ["reference-video-analysis"])
        self.assertEqual(plan.metadata["reference_video"]["analysis_depth"], "deep")

    def test_text2img2video_reference_profile_reuses_existing_i2v_graph(self) -> None:
        planner, _runner, _memory = build_runtime(
            self.repo_root,
            output_root=self.repo_root / ".tmp-tests" / "reference-video-i2v-plan",
            comfy_host="127.0.0.1",
            comfy_port=8188,
        )
        plan = planner.build_plan(
            planner.create_goal(
                prompt="Kirby taps a jelly cube and gets surprised by the wobble",
                media_type="text2img2video",
                duration_seconds=6,
                style="polished 2D anime",
                auto_download_assets=False,
                constraints={
                    "reference_video_source": "C:/references/clip.mp4",
                    "reference_video_depth": "deep",
                    "reference_video_max_keyframes": 12,
                    "reference_micro_gag_profile": "reference_micro_gag_v1",
                    "seed": 1234,
                    "duration_override_seconds": 6,
                    "video_frame_rate": 24,
                    "skip_upscale_for_i2v": True,
                },
            )
        )
        node_ids = [node.node_id for node in plan.nodes]
        self.assertEqual(node_ids[0], "reference-video-analysis")
        self.assertEqual(plan.workflow_name, "text2img2video_v1")
        idea = next(node for node in plan.nodes if node.node_id == "idea-brief")
        image = next(node for node in plan.nodes if node.node_id == "render-image")
        animate = next(node for node in plan.nodes if node.node_id == "animate-video")
        qa = next(node for node in plan.nodes if node.node_id == "video-qa")
        self.assertEqual(idea.depends_on, ["reference-video-analysis"])
        self.assertEqual(image.inputs["seed"], 1234)
        self.assertEqual(animate.inputs["seed"], 1234)
        self.assertEqual(qa.inputs["semantic_qa_profile"], "reference_micro_gag_v1")
        self.assertIn("reference-video-analysis", qa.depends_on)

    def test_reference_analyzer_writes_structural_brief_and_visual_evidence(self) -> None:
        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            self.skipTest("ffmpeg and ffprobe are required for the reference-video evidence test")
        with tempfile.TemporaryDirectory(prefix="mediaoverload-reference-test-") as temp_dir:
            root = Path(temp_dir)
            source = root / "source.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=orange:s=320x180:r=12:d=2",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-y",
                    str(source),
                ],
                check=True,
                timeout=60,
            )
            brief = ReferenceVideoAnalyzer().analyze(str(source), output_root=root / "output", max_keyframes=4)

            self.assertEqual(brief["analysis_mode"], "structural_ffmpeg")
            self.assertEqual(brief["source"]["type"], "local_file")
            self.assertEqual(len(brief["keyframes"]), 4)
            self.assertTrue(Path(brief["contact_sheet_path"]).is_file())
            brief_path = Path(brief["brief_path"])
            self.assertTrue(brief_path.is_file())
            persisted = json.loads(brief_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["structure_analysis"]["scene_count"], brief["structure_analysis"]["scene_count"])
            directive = format_reference_video_directive(brief)
            self.assertIn("borrow_grammar_not_assets", directive)
            self.assertIn("duration_seconds", directive)
            self.assertNotIn(str(source), directive)

    def test_url_reference_fails_explicitly_without_downloader(self) -> None:
        analyzer = ReferenceVideoAnalyzer(yt_dlp="missing-yt-dlp-for-test")
        with self.assertRaisesRegex(ReferenceVideoError, "yt-dlp"):
            analyzer.analyze(
                "https://www.youtube.com/watch?v=example",
                output_root=Path(tempfile.gettempdir()) / "mediaoverload-reference-url-test",
            )


if __name__ == "__main__":
    unittest.main()
