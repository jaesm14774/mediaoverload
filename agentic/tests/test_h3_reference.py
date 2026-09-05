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

    def test_ref2va_prompt_has_contract_and_no_audio_reference_input(self) -> None:
        refs = [{"tag": "reference_image_1", "prompt_label": "<Picture 1>", "role": "identity", "retention": "identity_and_appearance"}]
        prompt = format_ref2va_prompt("A Kirby scene", refs)
        self.assertIn("subject_definitions:", prompt)
        self.assertIn("summary:", prompt)
        self.assertIn("retention_analysis:", prompt)
        self.assertIn("detailed_description:", prompt)
        self.assertIn("non_diegetic_music:", prompt)
        self.assertNotIn("<Subject Definitions>", prompt)
        self.assertIn("<Picture 1>", prompt)
        self.assertIn("overall_soundscape:", prompt)
        self.assertIn("do not use reference audio", prompt.lower())
        self.assertNotIn("<Reference Audio>", prompt)

    def test_ref2va_prompt_marks_continuation_picture_as_first_frame(self) -> None:
        refs = [
            {"prompt_label": "<Picture 1>", "role": "identity", "type": "image", "notes": "one Kirby"},
            {"prompt_label": "<Picture 2>", "role": "continuation", "type": "image", "notes": "lossless tail of the previous clip"},
        ]
        prompt = format_ref2va_prompt("Kirby takes the next action", refs)
        self.assertIn("[keyframe completion + reference generation]", prompt)
        self.assertIn("<Picture 2> is the lossless first-frame continuity anchor for [Shot 1]", prompt)
        self.assertIn("<Picture 2> ([Shot 1] first frame): fully_preserved", prompt)
        self.assertIn("begins exactly from the declared continuity anchor", prompt)


if __name__ == "__main__":
    unittest.main()
