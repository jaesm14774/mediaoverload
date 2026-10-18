from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic.h3_reference import format_ref2va_prompt, normalize_reference_manifest


class H3ReferenceTests(unittest.TestCase):
    def test_normalizes_image_and_video_with_deterministic_tags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "subject.png"
            video = root / "motion.mp4"
            image.write_bytes(b"image")
            video.write_bytes(b"video")
            refs = normalize_reference_manifest(
                [
                    {"path": str(image), "type": "image", "role": "identity"},
                    {"path": str(video), "type": "video", "role": "motion"},
                ],
                require_files=True,
            )
            self.assertEqual([ref["tag"] for ref in refs], ["reference_image_1", "reference_video_1"])
            self.assertEqual([ref["type"] for ref in refs], ["image", "video"])

    def test_rejects_reference_audio(self) -> None:
        with self.assertRaisesRegex(ValueError, "Reference audio"):
            normalize_reference_manifest([{"path": "voice.wav", "type": "audio"}])

    def test_rejects_audio_extension_without_declared_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "type=image or type=video"):
            normalize_reference_manifest([{"path": "voice.wav"}])

    def test_user_given_ref2va_references_when_prompt_is_formatted_then_all_six_sections_and_audio_are_present(self) -> None:
        """User Given a Ref2VA scene and reference image, When formatted, Then the six official sections include separate audio fields."""
        refs = [{"tag": "reference_image_1", "prompt_label": "<Picture 1>", "role": "identity", "retention": "identity_and_appearance"}]
        prompt = format_ref2va_prompt(
            "A Kirby scene", refs,
            overall_soundscape="Quiet canal water and one lantern chime.",
            non_diegetic_music="N/A",
        )
        self.assertIn("subject_definitions:", prompt)
        self.assertIn("summary:", prompt)
        self.assertIn("retention_analysis:", prompt)
        self.assertIn("detailed_description:", prompt)
        self.assertIn("overall_soundscape:", prompt)
        self.assertIn("non_diegetic_music:", prompt)
        self.assertIn("[Shot 1] A Kirby scene", prompt)
        self.assertNotIn("<Subject Definitions>", prompt)
        self.assertIn("<Picture 1>", prompt)
        self.assertNotIn("reference audio", prompt.lower())
        self.assertNotIn("<Reference Audio>", prompt)

    def test_user_given_ref2va_continuation_when_prompt_is_formatted_then_picture_is_marked_as_first_frame(self) -> None:
        """User Given a continuation frame and subject image, When formatted, Then continuation is explicitly anchored as the first frame."""
        refs = [
            {"prompt_label": "<Picture 1>", "role": "identity", "type": "image", "notes": "one Kirby"},
            {"prompt_label": "<Picture 2>", "role": "continuation", "type": "image", "notes": "lossless tail of the previous clip"},
        ]
        prompt = format_ref2va_prompt(
            "Kirby takes the next action", refs,
            overall_soundscape="Canal water moves softly.",
        )
        self.assertIn("[keyframe completion + reference generation]", prompt)
        self.assertIn("<Picture 2> is the supplied first-frame input for [Shot 1]", prompt)
        self.assertIn("<Picture 2> ([Shot 1] first frame): use as the starting image", prompt)
        self.assertIn("starts from the supplied [Shot 1] image", prompt)


if __name__ == "__main__":
    unittest.main()
