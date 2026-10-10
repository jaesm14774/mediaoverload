from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.run_cinematic_directing_ab import (
    CAMERA_STYLE_STORY_TREATMENT,
    EXPERIMENT_LLM_TIMEOUT_SECONDS,
    _pin_experiment_models,
    _probe_contains_video,
    _write_blind_gallery,
)


class CinematicDirectingEvidenceTests(unittest.TestCase):
    def test_user_given_an_approaching_camera_move_when_treatment_is_written_then_face_and_payoff_keep_frame_margins(self) -> None:
        """User: Given a camera move approaches a payoff, When the benchmark treatment is written, Then it keeps the face and result inside frame or ends wider."""
        self.assertIn("within frame with clear edge margins", CAMERA_STYLE_STORY_TREATMENT)
        self.assertIn("stop wider or cut to a separate shot", CAMERA_STYLE_STORY_TREATMENT)

    def test_user_given_a_slow_local_prompt_when_benchmark_models_are_pinned_then_request_and_total_deadlines_allow_completion(self) -> None:
        """User: Given local Qwen needs more than 180 seconds for a detailed prompt, When benchmark models are pinned, Then request and total deadlines allow it without provider fallback."""
        env_keys = (
            "AGENTIC_LLM_REQUEST_TIMEOUT_SECONDS",
            "AGENTIC_OLLAMA_REQUEST_TIMEOUT_SECONDS",
            "AGENTIC_LLM_TOTAL_TIMEOUT_SECONDS",
            "AGENTIC_PROVIDER_FALLBACK_ENABLED",
            "AGENTIC_TEXT_ALLOW_FALLBACK",
            "AGENTIC_VISION_ALLOW_FALLBACK",
        )
        previous = {key: os.environ.get(key) for key in env_keys}
        try:
            _pin_experiment_models()

            self.assertGreater(EXPERIMENT_LLM_TIMEOUT_SECONDS, 180)
            self.assertEqual(os.environ["AGENTIC_LLM_REQUEST_TIMEOUT_SECONDS"], str(EXPERIMENT_LLM_TIMEOUT_SECONDS))
            self.assertEqual(os.environ["AGENTIC_OLLAMA_REQUEST_TIMEOUT_SECONDS"], str(EXPERIMENT_LLM_TIMEOUT_SECONDS))
            self.assertEqual(os.environ["AGENTIC_LLM_TOTAL_TIMEOUT_SECONDS"], str(EXPERIMENT_LLM_TIMEOUT_SECONDS))
            self.assertEqual(os.environ["AGENTIC_PROVIDER_FALLBACK_ENABLED"], "false")
            self.assertEqual(os.environ["AGENTIC_TEXT_ALLOW_FALLBACK"], "false")
            self.assertEqual(os.environ["AGENTIC_VISION_ALLOW_FALLBACK"], "false")
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_user_given_ffprobe_success_when_result_has_no_video_stream_then_benchmark_rejects_it(self) -> None:
        """User: Given ffprobe succeeds but reports no video stream, When benchmark evidence is checked, Then the render fails technical validation."""
        probe = {
            "duration_seconds": 6.0,
            "streams": [{"codec_type": "audio"}],
        }

        self.assertFalse(_probe_contains_video(probe))

    def test_user_given_valid_video_stream_when_ffprobe_result_is_checked_then_benchmark_accepts_it(self) -> None:
        """User: Given ffprobe reports a video stream with duration and dimensions, When benchmark evidence is checked, Then the render passes media-probe validation."""
        probe = {
            "duration_seconds": 6.0,
            "streams": [{"codec_type": "video", "width": 640, "height": 360}],
        }

        self.assertTrue(_probe_contains_video(probe))

    def test_user_given_a_matched_pair_when_blind_gallery_is_written_then_media_paths_hide_the_a_b_mapping(self) -> None:
        """User: Given a matched A/B pair, When the blind gallery is written, Then it serves copied X/Y assets without exposing the A/B source paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            variants: dict[str, dict[str, object]] = {}
            for variant in ("A", "B"):
                variant_dir = run_dir / "cases" / "sample" / "repeat-01" / variant
                variant_dir.mkdir(parents=True)
                video = variant_dir / f"{variant}_render.mp4"
                video.write_bytes(f"video-{variant}".encode())
                contact_sheet = variant_dir / f"{variant}_contact.png"
                Image.new("RGB", (32, 18), color=(20, 30, 40)).save(contact_sheet)
                variants[variant] = {
                    "evidence": {
                        "primary_video_path": str(video),
                        "contact_sheet_path": str(contact_sheet),
                    }
                }
            pair = {
                "case_id": "sample",
                "repeat": 1,
                "blind_order": ["B", "A"],
                "case": {"brief": "A character follows one falling leaf."},
                "variants": variants,
            }

            _write_blind_gallery(run_dir, [pair], [{"pair_id": "sample__r01", "X": "B", "Y": "A"}])

            gallery_path = run_dir / "blind_review.html"
            gallery = gallery_path.read_text(encoding="utf-8")
            self.assertIn("blind_assets/sample__r01/X_video.mp4", gallery)
            self.assertIn("blind_assets/sample__r01/Y_video.mp4", gallery)
            self.assertNotIn("/A/", gallery)
            self.assertNotIn("/B/", gallery)
            self.assertNotIn("A_render.mp4", gallery)
            self.assertNotIn("B_render.mp4", gallery)
            self.assertNotIn("unblinding_key.json", gallery)
            self.assertTrue((run_dir / "blind_assets" / "sample__r01" / "X_video.mp4").is_file())
            self.assertTrue((run_dir / "blind_assets" / "sample__r01" / "Y_contact_sheet.jpg").is_file())


if __name__ == "__main__":
    unittest.main()
