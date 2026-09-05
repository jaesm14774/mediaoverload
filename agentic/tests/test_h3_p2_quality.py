from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic.runtime.h3_modes import H3Mode, mode_contract, validate_h3_payload
from agentic.tools.media_services import MediaServiceTools


class H3ModeContractTests(unittest.TestCase):
    def test_all_public_modes_have_distinct_conditioning_contracts(self) -> None:
        self.assertEqual(mode_contract("t2va").render_mode, "text_to_video")
        self.assertTrue(mode_contract(H3Mode.I2VA).requires_first_frame)
        self.assertTrue(mode_contract("fl2va").requires_last_frame)
        self.assertTrue(mode_contract("l2va").requires_last_frame)
        self.assertTrue(mode_contract("ref2va").allows_reference_videos)

    def test_contract_rejects_reusing_the_wrong_frame_inputs(self) -> None:
        with self.assertRaises(ValueError):
            validate_h3_payload("l2va", {"image_path": "opening.png", "last_image_path": "landing.png"})
        with self.assertRaises(ValueError):
            validate_h3_payload("i2va", {"image_path": "opening.png", "last_image_path": "landing.png"})
        with self.assertRaises(ValueError):
            validate_h3_payload("ref2va", {"reference_manifest": [{"type": "image", "path": "a.png"}], "reference_audio_paths": ["voice.wav"]})

    def test_ref2va_accepts_path_lists_before_manifest_normalization(self) -> None:
        validate_h3_payload(
            "ref2va",
            {"reference_image_paths": ["identity.png"], "reference_video_paths": ["motion.mp4"]},
        )

    def test_fl2va_still_rejects_a_timeline_with_missing_endpoints(self) -> None:
        with self.assertRaises(ValueError):
            validate_h3_payload(
                "fl2va",
                {},
            )


class H3P2VideoQualityTests(unittest.TestCase):
    def _tools(self, *, has_audio: bool = True, mean_volume: float = -20.0, silence_ratio: float = 0.05) -> MediaServiceTools:
        tools = MediaServiceTools(Path(tempfile.mkdtemp()))
        video = Path(tempfile.mktemp(suffix=".mp4"))
        video.write_bytes(b"actual-test-fixture")
        self.addCleanup(lambda: video.unlink(missing_ok=True))

        class FakeFFmpeg:
            def probe_media(self, path: str) -> dict[str, object]:
                return {
                    "path": path,
                    "duration": 5.166,
                    "video_duration": 5.166,
                    "audio_duration": 5.166 if has_audio else 0.0,
                    "has_video": True,
                    "has_audio": has_audio,
                    "width": 608,
                    "height": 352,
                    "frame_rate": 24.0,
                    "channels": 2 if has_audio else 0,
                    "channel_layout": "stereo" if has_audio else "",
                }

            def analyze_audio(self, path: str, **kwargs: object) -> dict[str, object]:
                del path, kwargs
                return {
                    "mean_volume_db": mean_volume,
                    "max_volume_db": -3.0,
                    "silence_ratio": silence_ratio,
                }

            def make_contact_sheet(self, **kwargs: object) -> str:
                return str(kwargs["output_path"])

        tools._ffmpeg = FakeFFmpeg()  # type: ignore[assignment]
        tools._fixture_path = video  # type: ignore[attr-defined]
        return tools

    def test_strict_gate_checks_audio_stereo_loudness_and_alignment(self) -> None:
        tools = self._tools()
        result = tools.video_qa(
            {
                "video_path": str(tools._fixture_path),  # type: ignore[attr-defined]
                "target_duration": 5.166,
                "expected_width": 608,
                "expected_height": 352,
                "expected_fps": 24,
                "require_audio": True,
                "require_stereo_audio": True,
                "analyze_audio": True,
            }
        )
        self.assertTrue(result["passed"])
        self.assertTrue(result["checks"]["stereo"])
        self.assertTrue(result["checks"]["loudness"])
        self.assertTrue(result["checks"]["duration_alignment"])

    def test_strict_gate_rejects_missing_audio(self) -> None:
        tools = self._tools(has_audio=False)
        result = tools.video_qa(
            {
                "video_path": str(tools._fixture_path),  # type: ignore[attr-defined]
                "target_duration": 5.166,
                "require_audio": True,
                "require_stereo_audio": True,
                "analyze_audio": True,
            }
        )
        self.assertFalse(result["passed"])
        self.assertIn("audio stream is required but missing", result["errors"])


if __name__ == "__main__":
    unittest.main()
