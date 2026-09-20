from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from agentic.assets.minimax_h3 import download_profile, get_profile, inspect_profile, minimax_h3_model_overrides
from agentic.assets.registry import AssetRegistry
from agentic.app.character_workflow import _prioritize_h3_profile
from agentic.minimax_prompting import compose_minimax_h3_prompt, structured_visual_prompt, subject_identity_lock
from agentic.runtime.contracts import GoalRequest
from agentic.runtime.prompting import build_minimax_h3_prompt


class MiniMaxH3ProfileTests(unittest.TestCase):
    def test_balanced_profile_points_to_portable_comfy_directories(self) -> None:
        profile = get_profile("balanced-lowvram")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = [asset.target_path(root) for asset in profile.assets]
        self.assertTrue(
            any(
                path.parts[-4:]
                == ("ComfyUI", "models", "unet", "minimax_h3_fl2va_pruned_fp8_Q4_0.gguf")
                for path in paths
            )
        )
        self.assertTrue(
            any(
                path.parts[-4:]
                == ("ComfyUI", "models", "clip", "qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf")
                for path in paths
            )
        )
        self.assertEqual(profile.width, 608)
        self.assertEqual(profile.height, 352)
        self.assertEqual(profile.length, 240)

    def test_dry_run_does_not_create_model_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = download_profile("ultra-lowvram", root, dry_run=True)
            self.assertFalse(result["ready"])
            self.assertEqual(len(result["assets"]), 4)
            self.assertFalse(any(root.rglob("*.gguf")))

    def test_ref2va_model_profile_switches_loaders_without_second_workflow(self) -> None:
        q4 = minimax_h3_model_overrides("q4", reference_to_video=True)
        self.assertEqual(q4["1"]["class_type"], "UnetLoaderGGUF")
        self.assertEqual(q4["1"]["inputs"]["unet_name"], "MiniMax-H3-Ref2VA-Pruned-Q4_K_M.gguf")
        q2 = minimax_h3_model_overrides("q2")
        self.assertEqual(q2["2"]["class_type"], "CLIPLoaderGGUF")
        self.assertEqual(q2["2"]["inputs"]["clip_name"], "qwen3vl-32B-MiniMax-H3-Q2_K.gguf")
        native = minimax_h3_model_overrides("native", reference_to_video=True)
        self.assertEqual(native["1"]["class_type"], "UNETLoader")
        self.assertEqual(native["1"]["inputs"]["unet_name"], "minimax_h3_ref2va_pruned_int8_convrot.safetensors")
        self.assertEqual(native["2"]["class_type"], "CLIPLoader")

    def test_partial_file_is_reported_as_corrupt_or_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            profile = get_profile("balanced-lowvram")
            path = profile.assets[0].target_path(root)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"partial")
            result = inspect_profile(profile, root)
            self.assertFalse(result["ready"])
            self.assertEqual(result["assets"][0]["status"], "corrupt")

    def test_registry_materializes_h3_manifest_and_validates_graph(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        registry = AssetRegistry(repo_root / "agentic", asset_root=repo_root)
        manifest = registry.get_manifest("minimax_h3_lowvram_i2v")
        self.assertEqual(len(manifest.required_assets), 4)
        self.assertEqual(manifest.recommended_defaults["width"], 608)
        validation = registry.validate_workflow("minimax_h3_lowvram_i2v")
        self.assertTrue(validation["valid"], validation)
        workflow = json.loads(Path(validation["workflow_path"]).read_text(encoding="utf-8"))
        self.assertEqual(workflow["5"]["class_type"], "MiniMaxH3ImageToVideo")
        self.assertEqual(workflow["15"]["class_type"], "SaveVideo")

        fl2va_manifest = registry.get_manifest("minimax_h3_lowvram_15s_fl2va_i2v")
        self.assertEqual(fl2va_manifest.recommended_defaults["length"], 362)
        self.assertEqual(fl2va_manifest.recommended_defaults["steps"], 16)

    def test_ref2va_uses_one_canonical_workflow_file(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        registry = AssetRegistry(repo_root / "agentic", asset_root=repo_root)
        self.assertEqual(
            [manifest.name for manifest in registry.all_manifests() if "ref2va" in manifest.name],
            ["minimax_h3_ref2va"],
        )
        self.assertFalse(any(manifest.name.startswith("minimax_h3_ultra_lowvram") for manifest in registry.all_manifests()))
        self.assertFalse((repo_root / "configs" / "workflow" / "minimax_h3_ref2va_native.json").exists())
        self.assertFalse((repo_root / "configs" / "workflow" / "comfyui").exists() and any((repo_root / "configs" / "workflow" / "comfyui").iterdir()))

    def test_lowvram_graph_routes_spectrum_between_sigma_shift_and_sampler(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        for workflow_name in ("minimax_h3_lowvram_i2v", "minimax_h3_lowvram_t2v"):
            workflow_path = repo_root / "configs" / "workflow" / f"{workflow_name}.json"
            workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
            self.assertEqual(workflow["17"]["class_type"], "SpectrumApplyMiniMaxH3")
            self.assertEqual(workflow["17"]["inputs"]["model"], ["6", 0])
            self.assertEqual(workflow["7"]["inputs"]["model"], ["17", 0])
            self.assertEqual(workflow["10"]["inputs"]["model"], ["17", 0])

    def test_kirby_h3_profile_changes_video_candidate_priority(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        candidates = {
            "text2video": {"video_workflow_name": ["minimax_h3_lowvram_t2v", "minimax_h3_native_t2v"]},
            "text2longvideo": {"video_workflow_name": ["minimax_h3_lowvram_t2v", "minimax_h3_lowvram_i2v"]},
        }
        _prioritize_h3_profile(repo_root, {"h3_profile": "ultra-lowvram"}, candidates, list(candidates))
        self.assertEqual(candidates["text2video"]["video_workflow_name"][0], "minimax_h3_lowvram_t2v")
        self.assertEqual(candidates["text2longvideo"]["video_workflow_name"][0], "minimax_h3_lowvram_i2v")

    def test_krea2_keyframe_and_identity_workflows_are_registered(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        registry = AssetRegistry(repo_root / "agentic", asset_root=repo_root)
        for workflow_name in ("krea2_turbo", "krea2_turbo_img2img"):
            validation = registry.validate_workflow(workflow_name)
            self.assertTrue(validation["valid"], validation)
        krea2 = json.loads((repo_root / "configs" / "workflow" / "krea2_turbo.json").read_text(encoding="utf-8"))
        self.assertEqual(krea2["1"]["inputs"]["unet_name"], "krea2_turbo_bf16-Q4_0.gguf")
        continuity = json.loads((repo_root / "configs" / "workflow" / "krea2_turbo_img2img.json").read_text(encoding="utf-8"))
        self.assertEqual(continuity["9"]["inputs"]["denoise"], 0.25)
        h3_i2v = json.loads((repo_root / "configs" / "workflow" / "minimax_h3_lowvram_i2v.json").read_text(encoding="utf-8"))
        self.assertEqual(h3_i2v["5"]["inputs"]["first_frame"], ["16", 0])
        self.assertEqual(h3_i2v["5"]["inputs"]["length"], 240)








class MiniMaxH3PromptTests(unittest.TestCase):
    def test_single_subject_identity_lock_uses_canonical_role_description(self) -> None:
        lock = subject_identity_lock(
            "Waddle Dee",
            {
                "character_profile": {
                    "role_description": "Waddle Dee has a tan pear-shaped face and no mouth.",
                    "keywords": "Waddle Dee, tan, no mouth",
                }
            },
        )
        self.assertIn("Waddle Dee", lock)
        self.assertIn("tan pear-shaped face and no mouth", lock)
        self.assertIn("canonical character identity", lock.lower())
        self.assertIn("do not invent or add conflicting features", lock)

    def test_local_h3_prompt_uses_context_ir_order_and_i2v_input_relation(self) -> None:
        prompt = compose_minimax_h3_prompt(
            duration_seconds=6,
            character="Kirby",
            style="polished 2D anime",
            story_spine={"objective": "save the seed", "obstacle": "a strong gust"},
            shots=[
                {
                    "time": "0-6s",
                    "title": "Rescue",
                    "action": "Kirby runs and catches the seed",
                    "camera": "tracking shot follows the run",
                    "state_change": "the seed is safe",
                }
            ],
            prior_frame=True,
        )
        self.assertLess(len(prompt), 7000)
        self.assertIn("integrated_multimodal_description:", prompt)
        self.assertIn("[Shot 1 / SHOT 1 | 0-6s]", prompt)
        self.assertIn("overall_soundscape:", prompt)
        self.assertIn("non_diegetic_music:", prompt)
        self.assertIn("first-frame image is authoritative", prompt)

    def test_first_last_frame_prompt_carries_ending_condition(self) -> None:
        prompt = compose_minimax_h3_prompt(
            duration_seconds=15,
            character="Kirby",
            style="polished 2D anime",
            shots=[{"time": "0-15s", "action": "Kirby reaches the closing gate", "state_change": "the gate opens"}],
            render_mode="first_last_frame_to_video",
        )
        self.assertIn("first-frame image is authoritative", prompt)
        self.assertIn("supplied last-frame state", prompt)

    def test_pair_prompt_keeps_same_name_slots_and_interaction_contract(self) -> None:
        prompt = compose_minimax_h3_prompt(
            duration_seconds=6,
            character="Kirby",
            style="polished 2D anime",
            story_spine={"objective": "push the glowing mechanism together"},
            shots=[
                {
                    "time": "0-6s",
                    "action": "the two Kirby subject slots push the mechanism together",
                    "state_change": "the mechanism opens",
                }
            ],
            subject_context={
                "subjects": [
                    {"role": "primary", "name": "Kirby"},
                    {"role": "secondary", "name": "Kirby"},
                ],
                "interaction_contract": {"required": True, "same_frame": True},
            },
        )

        self.assertIn("Two required subject slots", prompt)
        self.assertIn("slots may use the same name", prompt)
        self.assertIn("unrequested third subject", prompt)
        self.assertNotIn("duplicate protagonist", prompt)

    def test_visual_prompt_has_stable_subject_action_camera_order(self) -> None:
        prompt = structured_visual_prompt(
            subject="Kirby",
            scene="a flower garden",
            action="runs toward a falling seed",
            environment="petals scatter in the wind",
            camera="tracking shot",
            style="2D anime",
            quality="clear silhouette",
        )
        self.assertLess(prompt.index("Subject:"), prompt.index("Action:"))
        self.assertLess(prompt.index("Action:"), prompt.index("Camera:"))
        self.assertLess(prompt.index("Camera:"), prompt.index("Quality:"))

    def test_kirby_prompt_contains_identity_motion_and_native_audio_contract(self) -> None:
        goal = GoalRequest(
            prompt="Kirby races through a neon night market and catches a falling star",
            media_type="long_video",
            duration_seconds=5,
            style="cinematic anime",
            constraints={"character": "Kirby"},
        )
        result = build_minimax_h3_prompt(
            goal,
            {"segment_id": "segment-1", "visual": "Kirby runs through a glowing market"},
            prior_frame="frame.png",
        )
        self.assertIn("Character lock", result["prompt"])
        self.assertIn("Kirby", result["prompt"])
        self.assertIn("Motion direction", result["prompt"])
        self.assertIn("Audio direction", result["prompt"])
        self.assertIn("native stereo audio", result["prompt"])


if __name__ == "__main__":
    unittest.main()
