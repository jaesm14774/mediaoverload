from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from agentic.assets.registry import AssetRegistry
from agentic.app.character_workflow import build_goal_payload_from_character_config, collect_media_paths_from_run_result
from agentic.runtime.contracts import ExecutionNode, ExecutionPlan, GoalRequest, RunState, SkillContext
from agentic.runtime.motion_graphics import MotionGraphicsPlan
from agentic.runtime.planner import TaskPlanner
from agentic.runtime.registry import ToolRegistry
from agentic.skills.motion_graphics import MotionGraphicsSkills
from agentic.tools.media_services import register_media_service_tools
from character_workflow_helpers import make_character_workflow_request


class ScriptedMotionGraphicsPlanner:
    """Local provider boundary used to exercise review orchestration without an external LLM."""

    def __init__(self) -> None:
        self.review_count = 0

    def build_motion_graphics_plan(self, goal, *, duration_seconds: float, fps: float) -> dict[str, object]:
        return MotionGraphicsPlan.from_dict(
            {
                "duration_seconds": duration_seconds,
                "cues": [
                {
                    "kind": "burst",
                    "start_seconds": 0.25,
                    "end_seconds": min(1.5, duration_seconds),
                    "x": 0.5,
                    "y": 0.45,
                    "text": "",
                    "color": "#FFE066",
                    "accent": "#FF5D8F",
                    "size": 1.0,
                }
                ],
            }
        ).to_dict()

    def review_motion_graphics_plan(self, goal, *, plan, contact_sheet_path: str, round_number: int) -> dict[str, object]:
        if not Path(contact_sheet_path).is_file():
            raise FileNotFoundError("visual review requires the renderer's sampled contact sheet")
        self.review_count += 1
        revised_plan = MotionGraphicsPlan.from_dict(plan).to_dict()
        if self.review_count == 1:
            revised_plan["cues"][0]["color"] = "#00D9FF"
            revised_plan["cues"][0]["size"] = 1.2
        return {"satisfied": False, "critique": f"review {round_number}", "plan": revised_plan}


class MotionGraphicsTests(unittest.TestCase):
    @staticmethod
    def _make_source(path: Path) -> None:
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=c=0x17233b:s=320x240:r=24:d=2",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
                "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-y", str(path),
            ],
            check=True,
            capture_output=True,
        )

    def test_user_given_h3_source_when_motion_graphics_are_composed_then_video_contract_and_audio_are_preserved(self) -> None:
        """User Given a completed H3 clip When program synthesis is enabled Then its motion, timing, canvas, and audio remain intact."""
        from agentic.tools.motion_graphics_adapter import MotionGraphicsAdapter

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mp4"
            output = root / "composed.mp4"
            self._make_source(source)
            plan = MotionGraphicsPlan.from_dict(
                {
                    "duration_seconds": 2.0,
                    "cues": [
                    {
                        "kind": "caption",
                        "start_seconds": 0.2,
                        "end_seconds": 1.8,
                        "x": 0.5,
                        "y": 0.72,
                        "text": "一擊命中",
                        "color": "#FFFFFF",
                        "accent": "#FF5D8F",
                        "size": 1.0,
                    },
                    {
                        "kind": "burst",
                        "start_seconds": 0.5,
                        "end_seconds": 1.2,
                        "x": 0.5,
                        "y": 0.42,
                        "text": "",
                        "color": "#FFE066",
                        "accent": "#FF5D8F",
                        "size": 1.0,
                    },
                    ],
                }
            )

            result = MotionGraphicsAdapter().compose(
                video_path=str(source),
                output_path=str(output),
                plan=plan,
                work_dir=root / "frames",
            )

            source_probe = MotionGraphicsAdapter().ffmpeg.probe_media(str(source))
            final_probe = result["probe"]
            self.assertTrue(final_probe["has_video"])
            self.assertTrue(final_probe["has_audio"])
            self.assertEqual(final_probe["width"], source_probe["width"])
            self.assertEqual(final_probe["height"], source_probe["height"])
            self.assertAlmostEqual(final_probe["frame_rate"], source_probe["frame_rate"], delta=0.15)
            self.assertAlmostEqual(final_probe["duration"], source_probe["duration"], delta=1 / 24)
            self.assertEqual(final_probe["audio_codec"], source_probe["audio_codec"])
            self.assertNotEqual(Path(result["video_path"]).resolve(), source.resolve())
            self.assertTrue(Path(result["contact_sheet_path"]).is_file())

    def test_user_given_validated_motion_plan_when_coordinates_leave_safe_area_then_plan_is_rejected(self) -> None:
        """User Given a declarative motion plan When a text cue leaves the subtitle-safe area Then validation rejects it."""
        raw = {
            "duration_seconds": 2.0,
            "cues": [
                {
                    "kind": "caption", "start_seconds": 0.2, "end_seconds": 1.2,
                    "x": 0.5, "y": 0.95, "text": "Too low", "color": "#FFFFFF",
                    "accent": "#FF5D8F", "size": 1.0,
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "safe area"):
            MotionGraphicsPlan.from_dict(raw)

    def test_user_given_optional_feature_when_planner_builds_route_then_only_enabled_route_adds_composition_before_qa(self) -> None:
        """User Given an optional graphics request When a short-video plan is built Then composition is opt-in and precedes technical QA."""
        project_root = Path(__file__).resolve().parents[1]
        planner = TaskPlanner(asset_registry=AssetRegistry(project_root, asset_root=project_root))

        default_plan = planner.build_plan(
            planner.create_goal(
                prompt="Kirby hits one target",
                media_type="text2img2video",
                duration_seconds=5,
                style="playful",
                auto_download_assets=False,
            )
        )
        enabled_plan = planner.build_plan(
            planner.create_goal(
                prompt="Kirby hits one target",
                media_type="text2img2video",
                duration_seconds=5,
                style="playful",
                auto_download_assets=False,
                constraints={
                    "enable_motion_graphics": True,
                    "video_speed": {"enabled": True, "factor": 2.0},
                },
            )
        )

        self.assertNotIn("motion-graphics-review", [node.node_id for node in default_plan.nodes])
        motion = next(node for node in enabled_plan.nodes if node.node_id == "motion-graphics-review")
        qa = next(node for node in enabled_plan.nodes if node.node_id == "video-qa")
        preview = next(node for node in enabled_plan.nodes if node.node_id == "gif-preview")
        self.assertEqual(motion.depends_on, ["video-speed"])
        self.assertIn("motion-graphics-review", qa.depends_on)
        self.assertEqual(preview.depends_on, ["motion-graphics-review"])

    def test_user_given_character_short_video_when_motion_graphics_are_requested_then_cli_option_reaches_plan_constraints(self) -> None:
        """User Given a Kirby short-video request When motion graphics are selected Then the explicit option reaches the planner."""
        repo_root = Path(__file__).resolve().parents[2]
        request = make_character_workflow_request(
            repo_root,
            repo_root / "configs" / "characters" / "kirby.yaml",
            prompt="Kirby hits one target",
            preferred_generation_type="text2image2video",
            duration_seconds=10,
            motion_graphics=True,
            publish_after_generate=False,
        )

        payload = build_goal_payload_from_character_config(request)

        self.assertEqual(payload["media_type"], "text2img2video")
        self.assertTrue(payload["constraints"]["enable_motion_graphics"])

    def test_user_given_non_i2v_route_when_motion_graphics_are_requested_then_request_is_rejected(self) -> None:
        """User Given an unsupported media route When motion graphics are requested Then the user receives a clear route error."""
        repo_root = Path(__file__).resolve().parents[2]
        request = make_character_workflow_request(
            repo_root,
            repo_root / "configs" / "characters" / "kirby.yaml",
            prompt="A Kirby story card",
            preferred_generation_type="story_card",
            motion_graphics=True,
            publish_after_generate=False,
        )

        with self.assertRaisesRegex(ValueError, "only with the text2image2video"):
            build_goal_payload_from_character_config(request)

    def test_user_given_visual_review_requests_multiple_changes_when_motion_graphics_are_rendered_then_two_rounds_reuse_the_same_original(self) -> None:
        """User Given visual critique requests further edits When the overlay is revised Then at most two renders use the same untouched source."""
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mp4"
            source.write_bytes(b"")
            self._make_source(source)
            original_digest = hashlib.sha256(source.read_bytes()).hexdigest()
            tools = ToolRegistry()
            register_media_service_tools(tools, root / "output")
            planner = ScriptedMotionGraphicsPlanner()
            skills = MotionGraphicsSkills(tools, root / "output", planner)
            node = ExecutionNode(
                node_id="motion-graphics-review",
                skill_name="media.video.motion_graphics",
                depends_on=["video-speed"],
            )
            plan = ExecutionPlan(
                goal=GoalRequest(prompt="Kirby hits one target", media_type="text2img2video", duration_seconds=2),
                workflow_name="text2img2video_v1",
                nodes=[node],
            )
            state = RunState(goal={}, metadata={}, node_outputs={"video-speed": {"video_path": str(source)}})

            result = skills.compose_and_review(SkillContext(plan, node, state))
            manifest = json.loads(Path(result.outputs["manifest_path"]).read_text(encoding="utf-8"))

            self.assertEqual(result.status, "success")
            self.assertEqual(planner.review_count, 2)
            self.assertEqual(len(manifest["rounds"]), 2)
            self.assertTrue(all(item["source_video_path"] == str(source) for item in manifest["rounds"]))
            self.assertNotEqual(
                manifest["rounds"][0]["plan"]["cues"][0]["color"],
                manifest["rounds"][1]["plan"]["cues"][0]["color"],
            )
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original_digest)
            self.assertEqual(result.outputs["video_path"], manifest["final_video_path"])

    def test_user_given_overlay_and_speed_outputs_when_media_is_collected_then_only_hybrid_video_reaches_discord(self) -> None:
        """User Given both the source and overlay outputs When final media is collected Then only the hybrid video reaches Discord."""
        run_result = {
            "state": {
                "node_outputs": {
                    "video-speed": {"video_path": "C:/output/source.mp4", "speed": 2.0},
                    "motion-graphics-review": {
                        "video_path": "C:/output/hybrid.mp4",
                        "motion_graphics_applied": True,
                    },
                }
            }
        }

        self.assertEqual(collect_media_paths_from_run_result(run_result), ["C:/output/hybrid.mp4"])


if __name__ == "__main__":
    unittest.main()
