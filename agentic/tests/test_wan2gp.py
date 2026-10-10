from __future__ import annotations

import json
import tempfile
import unittest
import subprocess
import os
import shutil
from pathlib import Path

from PIL import Image

from agentic.tools.wan2gp import Wan2GPToolset, build_h3_settings, deliver_video


class Wan2GPContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.first = self.root / "first.png"
        self.last = self.root / "last.png"
        Image.new("RGB", (608, 352), "red").save(self.first)
        Image.new("RGB", (608, 352), "blue").save(self.last)
        self.payload = {"prompt": "A small hero catches a seed.", "width": 608, "height": 352, "length": 124, "steps": 16, "seed": 42}

    def test_user_given_paragraphs_when_submitting_one_video_then_the_prompt_remains_one_task(self) -> None:
        """User Given paragraph breaks When one video is submitted Then all paragraphs remain one generation task."""
        prompt = "A hero reaches for a rolling seed.\n\nThe hero catches it and smiles."
        settings, _ = build_h3_settings("t2va", {**self.payload, "prompt": prompt})
        self.assertEqual(settings["prompt"], prompt)
        self.assertEqual(settings["multi_prompts_gen_type"], "FG")

    def test_user_given_native_frame_override_when_planning_then_qa_uses_the_delivered_duration(self) -> None:
        """User Given an overridden frame count When planning 2x native video Then QA uses its delivered duration and audio contract."""
        from agentic.app.main import build_runtime
        from agentic.runtime.contracts import GoalRequest
        repo = Path(__file__).resolve().parents[2]
        planner, _, _ = build_runtime(repo / "agentic", output_root=self.root / "runtime")
        for media_type in ("native_h3_t2v_story", "native_h3_story", "native_h3_fl2va_story", "native_h3_l2va_story", "native_h3_ref2va"):
            with self.subTest(media_type=media_type):
                goal = GoalRequest(prompt="A hero catches a seed.", media_type=media_type, duration_seconds=15, constraints={"native_h3_storyboard_path": str(repo / "configs/storyboards/native_h3_15s.yaml"), "native_h3_length": 124, "native_h3_reference_image_paths": [str(self.first)], "video_speed": {"enabled": True, "factor": 2}})
                plan = planner.build_plan(goal)
                qa = next(node for node in plan.nodes if node.node_id == "native-h3-qa")
                self.assertAlmostEqual(qa.inputs["target_duration"], 124 / 24 / 2, delta=1 / 24)
                self.assertEqual(qa.inputs["expected_fps"], 24)
                self.assertTrue(qa.inputs["require_audio"])
                self.assertTrue(qa.inputs["require_stereo_audio"])

    def test_user_given_each_anchor_strategy_when_building_settings_then_inputs_keep_their_roles(self) -> None:
        """User Given each H3 anchor strategy When settings are built Then start/end roles remain distinct."""
        for mode, flags, inputs in (
            ("t2va", "", {}),
            ("i2va", "S", {"image_path": str(self.first)}),
            ("fl2va", "SE", {"image_path": str(self.first), "last_image_path": str(self.last)}),
            ("l2va", "E", {"last_image_path": str(self.last)}),
        ):
            with self.subTest(mode=mode):
                settings, evidence = build_h3_settings(mode, {**self.payload, **inputs})
                self.assertEqual(settings["image_prompt_type"], flags)
                self.assertEqual(settings["force_fps"], "24")
                self.assertEqual(settings["video_length"], self.payload["length"])
                self.assertEqual(settings["seed"], self.payload["seed"])
                self.assertEqual(bool(settings.get("image_start")), "S" in flags)
                self.assertEqual(bool(settings.get("image_end")), "E" in flags)
                self.assertEqual(len(evidence["conditioning_files"]), len(inputs))
                self.assertTrue(all(len(item["sha256"]) == 64 for item in evidence["conditioning_files"]))

    def test_user_given_two_reference_images_when_building_settings_then_ref2va_keeps_order_and_canvas(self) -> None:
        """User Given ordered reference images When Ref2VA is built Then the dedicated model retains labels and dimensions."""
        settings, evidence = build_h3_settings("ref2va", {**self.payload, "reference_image_paths": [str(self.last), str(self.first)]})
        self.assertIn("ref2va", settings["model_type"])
        self.assertEqual(settings["image_refs"], [str(self.last.resolve()), str(self.first.resolve())])
        self.assertEqual(settings["video_prompt_type"], "I")
        self.assertEqual(settings["resolution"], "608x352")
        self.assertEqual(settings["audio_prompt_type"], "")
        self.assertEqual([r["prompt_label"] for r in evidence["reference_manifest"]], ["<Picture 1>", "<Picture 2>"])

    def test_user_given_missing_conditioning_when_rendering_then_validation_fails(self) -> None:
        """User Given missing or mismatched conditioning When rendering Then the requested mode fails without a fallback."""
        for mode, inputs in (("i2va", {}), ("fl2va", {"image_path": str(self.first)}), ("l2va", {}), ("ref2va", {}), ("t2va", {"image_path": str(self.first)})):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                build_h3_settings(mode, {**self.payload, **inputs})
        with self.assertRaises(FileNotFoundError):
            build_h3_settings("i2va", {**self.payload, "image_path": str(self.root / "missing.png")})

    def test_user_given_unaligned_output_when_building_settings_then_generation_covers_the_delivery_contract(self) -> None:
        """User Given a 640x360 240-frame delivery When settings are built Then the H3 grid covers every requested frame."""
        settings, evidence = build_h3_settings("i2va", {**self.payload, "image_path": str(self.first), "width": 640, "height": 360, "length": 240})
        self.assertEqual(settings["video_length"] % 17, 5)
        self.assertGreaterEqual(settings["video_length"], 240)
        self.assertEqual(evidence["delivery"], {"width": 640, "height": 360, "length": 240, "fps": 24})
        width, height = map(int, settings["resolution"].split("x"))
        self.assertEqual(width % 32, 0)
        self.assertEqual(height % 32, 0)

    def test_user_given_corrupt_media_when_building_settings_then_generation_is_rejected(self) -> None:
        """User Given corrupt image/video conditioning When rendering Then the input fails before GPU submission."""
        invalid_image, invalid_video = self.root / "broken.png", self.root / "broken.mp4"
        invalid_image.write_bytes(b"not an image")
        invalid_video.write_bytes(b"not a video")
        with self.assertRaises(ValueError):
            build_h3_settings("i2va", {**self.payload, "image_path": str(invalid_image)})
        with self.assertRaises(ValueError):
            build_h3_settings("ref2va", {**self.payload, "reference_video_paths": [str(invalid_video)]})

    def test_user_given_invalid_render_contract_when_building_settings_then_unsupported_values_fail(self) -> None:
        """User Given unsupported H3 settings When rendering Then the output contract fails explicitly."""
        for change in ({"width": 128}, {"length": 500}, {"fps": 30}, {"model_profile": "native"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_h3_settings("t2va", {**self.payload, **change})

    def test_user_given_reference_videos_when_building_settings_then_video_order_and_reference_flags_are_kept(self) -> None:
        """User Given one to three real reference videos When Ref2VA is built Then their order and reference roles are kept."""
        source = self.root / "motion_1.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=green:size=320x256:rate=24:duration=1", "-c:v", "libx264", str(source)], check=True, capture_output=True)
        paths = [source]
        for index in (2, 3):
            target = self.root / f"motion_{index}.mp4"
            shutil.copy2(source, target)
            paths.append(target)
        for count, flags in ((1, "V-U"), (2, "V+-U"), (3, "V+*-U")):
            settings, evidence = build_h3_settings("ref2va", {**self.payload, "reference_video_paths": [str(path) for path in paths[:count]]})
            self.assertEqual(settings["video_prompt_type"], flags)
            self.assertEqual(settings["video_guide"], str(source.resolve()))
            self.assertEqual(len(evidence["conditioning_files"]), count)
            self.assertEqual(settings["audio_prompt_type"], "")
        mixed, _ = build_h3_settings("ref2va", {**self.payload, "reference_image_paths": [str(self.first)], "reference_video_paths": [str(source)]})
        self.assertEqual(mixed["video_prompt_type"], "IV-U")

    def test_user_given_missing_wangp_installation_when_rendering_then_asset_readiness_fails_explicitly(self) -> None:
        """User Given a missing local runtime When rendering Then the asset check explains missing files before generation."""
        from agentic.assets.registry import AssetRegistry
        previous = os.environ.get("WAN2GP_ROOT")
        os.environ["WAN2GP_ROOT"] = str(self.root / "missing_install")
        try:
            registry = AssetRegistry(Path(__file__).resolve().parents[1])
            manifest = registry.get_manifest("wan2gp_h3_t2va")
            statuses = registry.ensure_requirements(manifest, auto_download=False)
            self.assertTrue(any(item["status"] == "missing" for item in statuses))
            with self.assertRaisesRegex(RuntimeError, "assets missing or corrupt"):
                Wan2GPToolset(registry, self.root / "output").execute({**self.payload, "workflow_name": manifest.name, "run_dir": str(self.root / "render")})
            self.assertFalse(any((self.root / "render").rglob("*.mp4")))
        finally:
            if previous is None:
                os.environ.pop("WAN2GP_ROOT", None)
            else:
                os.environ["WAN2GP_ROOT"] = previous

    def test_user_given_workflow_catalog_when_selecting_h3_then_only_wangp_recipes_are_available(self) -> None:
        """User Given the production catalog When selecting H3 Then the five WanGP strategies replace Comfy H3 graphs."""
        from agentic.assets.registry import AssetRegistry
        registry = AssetRegistry(Path(__file__).resolve().parents[1])
        for mode in ("t2va", "i2va", "fl2va", "l2va", "ref2va"):
            manifest = registry.get_manifest(f"wan2gp_h3_{mode}")
            self.assertEqual(manifest.conditioning["provider"], "wan2gp")
            self.assertTrue(registry.validate_workflow(manifest.name)["valid"])
            self.assertEqual(json.loads(registry.materialize_workflow(manifest).read_text("utf-8"))["h3_mode"], mode)
        self.assertFalse(any(m.name.startswith("minimax_h3_") for m in registry.all_manifests()))

    def test_user_given_native_h3_grid_when_delivering_then_exact_canvas_frames_and_audio_are_kept(self) -> None:
        """User Given an aligned H3 video When delivered on a different grid Then its requested canvas, frames and stereo audio remain usable."""
        source, target = self.root / "native.mp4", self.root / "delivery.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:size=608x352:rate=24:duration=5.125", "-f", "lavfi", "-i", "color=c=blue:size=608x352:rate=24:duration=0.04166667", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=32000", "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-map", "2:a", "-t", str(124 / 24), "-ac", "2", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(source)], check=True, capture_output=True)
        requested = {"width": 640, "height": 360, "length": 120, "fps": 24}
        actual = deliver_video(source, target, requested)
        self.assertTrue(target.is_file())
        self.assertEqual({key: actual[key] for key in requested}, requested)
        self.assertAlmostEqual(actual["duration_seconds"], 120 / 24, delta=1 / 24)
        self.assertEqual(actual["audio"]["channels"], 2)
        self.assertAlmostEqual(float(actual["audio"]["duration"]), 120 / 24, delta=1 / 24)
        final_frame = self.root / "final.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(target), "-vf", "select=eq(n\\,119)", "-frames:v", "1", str(final_frame)], check=True, capture_output=True)
        red, green, blue = Image.open(final_frame).convert("RGB").getpixel((320, 180))
        self.assertGreater(blue, red + green)
